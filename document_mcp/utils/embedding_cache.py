"""Embedding cache utilities for semantic search optimization.

This module provides snapshot-style caching for paragraph embeddings,
organizing cache files in a directory structure that mirrors the document organization.
"""

import datetime
import hashlib
import json

import numpy as np

from ..helpers import _get_chapter_embeddings_path
from ..helpers import _get_chapter_path
from ..helpers import _get_embeddings_path
from ..logger_config import ErrorCategory
from ..logger_config import log_structured_error
from ..models import ChapterEmbeddingManifest
from ..models import EmbeddingCacheEntry


class EmbeddingCache:
    """Snapshot-style embedding cache manager."""

    def __init__(self, model_version: str = "models/text-embedding-004"):
        """Initialize cache with specific embedding model version."""
        self.model_version = model_version

    def get_chapter_embeddings(self, document_name: str, chapter_name: str) -> dict[int, np.ndarray]:
        """Get all cached embeddings for a chapter if cache is valid.

        Args:
            document_name: Name of the document
            chapter_name: Name of the chapter file (e.g., "01-intro.md")

        Returns:
            Dictionary mapping paragraph index to embedding array.
            Empty dict if cache is invalid or doesn't exist.
        """
        try:
            # Check if cache is valid (newer than source file)
            if not self._is_cache_valid(document_name, chapter_name):
                return {}

            chapter_embeddings_path = _get_chapter_embeddings_path(document_name, chapter_name)
            manifest_path = chapter_embeddings_path / "manifest.json"

            if not manifest_path.exists():
                return {}

            # Load manifest
            with open(manifest_path, encoding="utf-8") as f:
                manifest_data = json.load(f)

            manifest = ChapterEmbeddingManifest.model_validate(manifest_data)

            # Check model version compatibility
            if not all(entry.model_version == self.model_version for entry in manifest.cache_entries):
                return {}

            # Load embeddings
            embeddings = {}
            for entry in manifest.cache_entries:
                embedding_file = chapter_embeddings_path / f"paragraph_{entry.paragraph_index}.npy"
                if embedding_file.exists():
                    try:
                        embedding = np.load(str(embedding_file))
                        embeddings[entry.paragraph_index] = embedding
                    except Exception as e:
                        log_structured_error(
                            category=ErrorCategory.WARNING,
                            message=f"Failed to load embedding for paragraph {entry.paragraph_index}",
                            exception=e,
                            context={
                                "document_name": document_name,
                                "chapter_name": chapter_name,
                                "paragraph_index": entry.paragraph_index,
                            },
                            operation="get_chapter_embeddings",
                        )
                        continue

            return embeddings

        except Exception as e:
            log_structured_error(
                category=ErrorCategory.WARNING,
                message="Failed to get chapter embeddings",
                exception=e,
                context={"document_name": document_name, "chapter_name": chapter_name},
                operation="get_chapter_embeddings",
            )
            return {}

    def store_chapter_embeddings(
        self,
        document_name: str,
        chapter_name: str,
        paragraph_embeddings: dict[int, np.ndarray],
        paragraph_contents: dict[int, str],
    ):
        """Store embeddings for all paragraphs in a chapter.

        Args:
            document_name: Name of the document
            chapter_name: Name of the chapter file
            paragraph_embeddings: Dict mapping paragraph index to embedding
            paragraph_contents: Dict mapping paragraph index to content (for hashing)
        """
        try:
            chapter_embeddings_path = _get_chapter_embeddings_path(document_name, chapter_name)
            chapter_embeddings_path.mkdir(parents=True, exist_ok=True)

            # Get source file modification time
            chapter_path = _get_chapter_path(document_name, chapter_name)
            if chapter_path.exists():
                file_modified_time = datetime.datetime.fromtimestamp(chapter_path.stat().st_mtime)
            else:
                file_modified_time = datetime.datetime.now()

            # Create cache entries
            cache_entries = []
            for paragraph_index, embedding in paragraph_embeddings.items():
                # Generate content hash
                content = paragraph_contents.get(paragraph_index, "")
                content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()

                # Save embedding to binary file (np.save adds .npy extension automatically)
                embedding_file = chapter_embeddings_path / f"paragraph_{paragraph_index}"
                np.save(str(embedding_file), embedding)

                # Create cache entry
                entry = EmbeddingCacheEntry(
                    content_hash=content_hash,
                    paragraph_index=paragraph_index,
                    model_version=self.model_version,
                    created_at=datetime.datetime.now(),
                    file_modified_time=file_modified_time,
                )
                cache_entries.append(entry)

            # Create manifest
            manifest = ChapterEmbeddingManifest(
                chapter_name=chapter_name,
                total_paragraphs=len(paragraph_embeddings),
                cache_entries=cache_entries,
                last_updated=datetime.datetime.now(),
            )

            # Save manifest
            manifest_path = chapter_embeddings_path / "manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest.model_dump(), f, indent=2, default=str)

        except Exception as e:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message="Failed to store chapter embeddings",
                exception=e,
                context={"document_name": document_name, "chapter_name": chapter_name},
                operation="store_chapter_embeddings",
            )

    def _is_cache_valid(self, document_name: str, chapter_name: str) -> bool:
        """Check if cache is newer than source file modification time.

        Args:
            document_name: Name of the document
            chapter_name: Name of the chapter file

        Returns:
            True if cache is valid and newer than source file
        """
        try:
            chapter_embeddings_path = _get_chapter_embeddings_path(document_name, chapter_name)
            manifest_path = chapter_embeddings_path / "manifest.json"

            if not manifest_path.exists():
                return False

            # Get source file modification time
            chapter_path = _get_chapter_path(document_name, chapter_name)
            if not chapter_path.exists():
                return False

            source_mtime = datetime.datetime.fromtimestamp(chapter_path.stat().st_mtime)

            # Load manifest to check cache creation time
            with open(manifest_path, encoding="utf-8") as f:
                manifest_data = json.load(f)

            manifest = ChapterEmbeddingManifest.model_validate(manifest_data)

            # Cache is valid if it was created after (or very close to) the source file modification
            # Add small tolerance (0.1 second) to handle timing issues in tests
            time_diff = (manifest.last_updated - source_mtime).total_seconds()
            return time_diff >= -0.1  # Allow 0.1 second tolerance for timing issues

        except Exception as e:
            log_structured_error(
                category=ErrorCategory.WARNING,
                message="Failed to check cache validity",
                exception=e,
                context={"document_name": document_name, "chapter_name": chapter_name},
                operation="_is_cache_valid",
            )
            return False

    def invalidate_chapter_cache(self, document_name: str, chapter_name: str):
        """Remove cache entries for a specific chapter.

        Args:
            document_name: Name of the document
            chapter_name: Name of the chapter file
        """
        try:
            chapter_embeddings_path = _get_chapter_embeddings_path(document_name, chapter_name)

            if chapter_embeddings_path.exists():
                # Remove all files in the chapter embedding directory
                for file_path in chapter_embeddings_path.iterdir():
                    if file_path.is_file():
                        file_path.unlink()

                # Remove the directory if it's empty
                try:
                    chapter_embeddings_path.rmdir()
                except OSError:
                    # Directory not empty, that's fine
                    pass

        except Exception as e:
            log_structured_error(
                category=ErrorCategory.WARNING,
                message="Failed to invalidate chapter cache",
                exception=e,
                context={"document_name": document_name, "chapter_name": chapter_name},
                operation="invalidate_chapter_cache",
            )

    def invalidate_document_cache(self, document_name: str):
        """Remove all cache entries for a document.

        Args:
            document_name: Name of the document
        """
        try:
            embeddings_path = _get_embeddings_path(document_name)

            if embeddings_path.exists():
                # Remove all chapter embedding directories
                for chapter_dir in embeddings_path.iterdir():
                    if chapter_dir.is_dir():
                        # Remove all files in chapter directory
                        for file_path in chapter_dir.iterdir():
                            if file_path.is_file():
                                file_path.unlink()
                        chapter_dir.rmdir()

                # Remove the embeddings directory if it's empty
                try:
                    embeddings_path.rmdir()
                except OSError:
                    # Directory not empty, that's fine
                    pass

        except Exception as e:
            log_structured_error(
                category=ErrorCategory.WARNING,
                message="Failed to invalidate document cache",
                exception=e,
                context={"document_name": document_name},
                operation="invalidate_document_cache",
            )
