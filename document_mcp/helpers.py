"""Centralized helper functions for the Document MCP system.

This module contains all shared helper functions used across the modular tool files.
It provides validation, text processing, file operations, and other utility functions.
"""

import datetime
import difflib
import os
import re
from pathlib import Path
from typing import Any

from .batch import BatchOperation
from .logger_config import ErrorCategory
from .logger_config import log_structured_error
from .models import ChapterContent
from .models import ChapterMetadata

# Import DOCS_ROOT_PATH from main module to ensure environment variable updates work
from .utils.file_operations import DOCS_ROOT_PATH
from .utils.validation import CHAPTER_MANIFEST_FILE

# Document-specific constants
DOCUMENT_SUMMARY_FILE = "_SUMMARY.md"


# --- Input Validation Helpers ---


# --- Diff Generation Helper ---


def _generate_content_diff(
    original_content: str, new_content: str, filename: str = "chapter"
) -> dict[str, Any]:
    """Generate a unified diff between original and new content.

    Compares two strings and produces a diff report including a summary,
    lines added/removed, and the full unified diff text.
    """
    if original_content == new_content:
        return {"changed": False, "diff": None, "summary": "No changes made to content"}

    original_lines = original_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff_lines = list(
        difflib.unified_diff(
            original_lines,
            new_lines,
            fromfile=f"{filename} (before)",
            tofile=f"{filename} (after)",
            lineterm="",
        )
    )

    lines_added = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    lines_removed = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))

    return {
        "changed": True,
        "diff": "\n".join(diff_lines),
        "summary": f"Content changed: {lines_added} lines added, {lines_removed} lines removed",
        "stats": {"added": lines_added, "removed": lines_removed},
    }


# --- Path and File Operations ---


def _get_document_path(document_name: str) -> Path:
    """Return the full path for a given document name."""
    # Use environment variable if available for test isolation

    # Check for custom document root directory (for testing or configuration)
    docs_root_name = os.environ.get("DOCUMENT_ROOT_DIR")
    if docs_root_name:
        root_path = Path(docs_root_name)
        return root_path / document_name
    return DOCS_ROOT_PATH / document_name


def _get_chapter_path(document_name: str, chapter_filename: str) -> Path:
    """Return the full path for a given chapter file."""
    doc_path = _get_document_path(document_name)
    return doc_path / chapter_filename


def _is_valid_chapter_filename(filename: str) -> bool:
    """Check if a filename is a valid, non-reserved chapter file.

    Verifies that the filename ends with '.md' and is not a reserved name
    like the manifest or summary file.
    """
    if not filename.lower().endswith(".md"):
        return False
    if filename in (CHAPTER_MANIFEST_FILE, DOCUMENT_SUMMARY_FILE):
        return False
    return True


def _get_ordered_chapter_files(document_name: str) -> list[Path]:
    """Retrieve a sorted list of all valid chapter files in a document.

    The files are sorted alphanumerically by filename. Non-chapter files
    (like summaries or manifests) are excluded.
    """
    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        return []

    # For now, simple alphanumeric sort of .md files.
    # Future: could read CHAPTER_MANIFEST_FILE for explicit order.
    chapter_files = sorted(
        [f for f in doc_path.iterdir() if f.is_file() and _is_valid_chapter_filename(f.name)]
    )
    return chapter_files


# --- Text Processing ---


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into a list of paragraphs.

    Paragraphs are separated by one or more blank lines. Leading/trailing
    whitespace is stripped from each paragraph.
    """
    if not text:
        return []
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Split by one or more blank lines (a line with only whitespace is considered blank after strip)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized_text.strip())]
    return [p for p in paragraphs if p]


def _count_words(text: str) -> int:
    """Count the number of words in a given text string."""
    return len(text.split())


# --- Chapter Content Operations ---


def _read_chapter_content_details(document_name: str, chapter_file_path: Path) -> ChapterContent | None:
    """Read the content and metadata of a chapter from its file path."""
    if not chapter_file_path.is_file():
        return None
    try:
        content = chapter_file_path.read_text(encoding="utf-8")
        paragraphs = _split_into_paragraphs(content)
        word_count = _count_words(content)
        stat = chapter_file_path.stat()
        return ChapterContent(
            document_name=document_name,
            chapter_name=chapter_file_path.name,
            content=content,
            word_count=word_count,
            paragraph_count=len(paragraphs),
            last_modified=datetime.datetime.fromtimestamp(stat.st_mtime, tz=datetime.timezone.utc),
        )
    except Exception as e:
        log_structured_error(
            category=ErrorCategory.ERROR,
            message=f"Failed to read chapter file: {chapter_file_path.name}",
            exception=e,
            context={
                "document_name": document_name,
                "chapter_file_path": str(chapter_file_path),
                "file_exists": chapter_file_path.exists(),
            },
            operation="read_chapter_content_details",
        )
        return None


def _get_chapter_metadata(document_name: str, chapter_file_path: Path) -> ChapterMetadata | None:
    """Generate metadata for a chapter from its file path.

    This helper reads chapter content to calculate word and paragraph counts
    for the metadata object.
    """
    if not chapter_file_path.is_file():
        return None
    try:
        content = chapter_file_path.read_text(encoding="utf-8")  # Read to count words/paragraphs
        paragraphs = _split_into_paragraphs(content)
        word_count = _count_words(content)
        stat = chapter_file_path.stat()
        return ChapterMetadata(
            chapter_name=chapter_file_path.name,
            word_count=word_count,
            paragraph_count=len(paragraphs),
            last_modified=datetime.datetime.fromtimestamp(stat.st_mtime, tz=datetime.timezone.utc),
            # title can be added later if we parse H1 from content
        )
    except Exception as e:
        log_structured_error(
            category=ErrorCategory.ERROR,
            message=f"Failed to get metadata for chapter: {chapter_file_path.name}",
            exception=e,
            context={
                "document_name": document_name,
                "chapter_file_path": str(chapter_file_path),
                "file_exists": chapter_file_path.exists(),
            },
            operation="get_chapter_metadata",
        )
        return None


# --- Additional Path Operations ---


def _get_snapshots_path(document_name: str) -> Path:
    """Return the path to the snapshots directory for a document."""
    doc_path = _get_document_path(document_name)
    return doc_path / ".snapshots"


def _get_modification_history_path(document_name: str) -> Path:
    """Return the path to the modification history file for a document."""
    doc_path = _get_document_path(document_name)
    return doc_path / ".mod_history.json"


def _get_embeddings_path(document_name: str) -> Path:
    """Return the path to the embeddings cache directory for a document."""
    doc_path = _get_document_path(document_name)
    return doc_path / ".embeddings"


def _get_chapter_embeddings_path(document_name: str, chapter_name: str) -> Path:
    """Return the path to chapter-specific embeddings directory."""
    embeddings_path = _get_embeddings_path(document_name)
    return embeddings_path / chapter_name


def _resolve_operation_dependencies(
    operations: list[BatchOperation],
) -> list[BatchOperation]:
    """Resolve operation dependencies and return operations in executable order."""
    # Simple topological sort for dependency resolution
    resolved = []
    remaining = operations.copy()
    while remaining:
        # Find operations with no unresolved dependencies
        ready = []
        for op in remaining:
            if not op.depends_on:
                ready.append(op)
            else:
                # Check if all dependencies are resolved
                dependencies_met = all(
                    any(resolved_op.operation_id == dep_id for resolved_op in resolved)
                    for dep_id in op.depends_on
                )
                if dependencies_met:
                    ready.append(op)
        if not ready:
            # Detect specific type of dependency issue
            remaining_ids = [op.operation_id for op in remaining]

            # Check if any remaining operation depends on another remaining operation
            has_circular = False
            missing_deps = set()

            for op in remaining:
                for dep_id in op.depends_on:
                    # Check if dependency is in remaining operations (potential circular)
                    if any(remaining_op.operation_id == dep_id for remaining_op in remaining):
                        has_circular = True
                    # Check if dependency was never defined
                    elif not any(resolved_op.operation_id == dep_id for resolved_op in resolved):
                        missing_deps.add(dep_id)

            if has_circular:
                raise ValueError(f"Circular dependency detected among operations: {remaining_ids}")
            elif missing_deps:
                raise ValueError(f"Operations depend on unknown operation(s): {list(missing_deps)}")
            else:
                raise ValueError(f"Dependency resolution failed for operations: {remaining_ids}")
        # Sort ready operations by their order
        ready.sort(key=lambda op: op.order)
        resolved.extend(ready)
        # Remove resolved operations from remaining
        for op in ready:
            remaining.remove(op)
    return resolved
