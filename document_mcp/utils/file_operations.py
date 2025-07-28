"""File system operation utilities for the Document MCP system.

This module contains file path utilities and file system operations
used throughout the system.
"""

import os
from pathlib import Path

from ..config import get_settings

# === Configuration ===

# === Dynamic Document Root ===


def get_docs_root_path():
    """Get the document root path dynamically from current settings.

    This ensures that environment variable changes (like in tests)
    are properly reflected, rather than caching the path at module level.
    """
    settings = get_settings()
    return settings.document_root_path


# For backward compatibility, provide DOCS_ROOT_PATH as a property
# that dynamically resolves to the current document root
class _DocsRootPath:
    def __getattr__(self, name):
        # Delegate all Path operations to the dynamic root path
        return getattr(get_docs_root_path(), name)

    def __truediv__(self, other):
        return get_docs_root_path() / other

    def __str__(self):
        return str(get_docs_root_path())

    def __repr__(self):
        return repr(get_docs_root_path())

    def resolve(self):
        return get_docs_root_path().resolve()


DOCS_ROOT_PATH = _DocsRootPath()


# === Path Utilities ===


def get_document_path(document_name: str) -> Path:
    """Return the full path for a given document name."""
    return DOCS_ROOT_PATH / document_name


def get_chapter_path(document_name: str, chapter_filename: str) -> Path:
    """Return the full path for a given chapter file."""
    doc_path = get_document_path(document_name)
    return doc_path / chapter_filename


def get_operation_path(document_name: str, chapter_name: str | None) -> Path:
    """Get file path for operation based on document and chapter names."""
    if chapter_name:
        return get_chapter_path(document_name, chapter_name)
    else:
        return get_document_path(document_name)


def get_snapshots_path(document_name: str) -> Path:
    """Return the path to the snapshots directory for a document."""
    doc_path = get_document_path(document_name)
    return doc_path / ".snapshots"


def get_modification_history_path(document_name: str) -> Path:
    """Return the path to the modification history file for a document."""
    doc_path = get_document_path(document_name)
    return doc_path / ".mod_history.json"


def is_valid_chapter_filename(filename: str) -> bool:
    """Check if a filename is a valid, non-reserved chapter file.
    Verifies that the filename ends with '.md' and is not a reserved name.
    """
    if not filename.lower().endswith(".md"):
        return False

    # Reserved filenames that should not be treated as chapters
    reserved_names = {
        "_manifest.json",
        "_summary.md",
        "_index.md",
        "readme.md",
        ".gitignore",
        ".mod_history.json",
    }

    return filename.lower() not in reserved_names


def ensure_directory_exists(path: Path) -> None:
    """Ensure a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)


def get_current_user() -> str:
    """Get current user identifier for tracking modifications."""
    # In production, this would integrate with authentication system
    # For now, return a simple identifier
    return os.environ.get("USER", "system_user")


# === File Content Utilities ===


def split_content_into_paragraphs(content: str) -> list[str]:
    """Split content into paragraphs based on double newlines.

    Args:
        content: The text content to split

    Returns:
        List of paragraph strings, with empty paragraphs filtered out
    """
    if not content:
        return []

    # Split on double newlines and filter out empty paragraphs
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    return paragraphs


def join_paragraphs(paragraphs: list[str]) -> str:
    """Join paragraphs back into content with double newlines.

    Args:
        paragraphs: List of paragraph strings

    Returns:
        Joined content string
    """
    return "\n\n".join(paragraphs)


def generate_content_diff(old_content: str, new_content: str) -> dict:
    """Generate a simple diff between old and new content.

    Args:
        old_content: Original content
        new_content: Modified content

    Returns:
        Dict with diff information including line counts and change summary
    """
    old_lines = old_content.splitlines() if old_content else []
    new_lines = new_content.splitlines() if new_content else []

    return {
        "old_line_count": len(old_lines),
        "new_line_count": len(new_lines),
        "lines_added": max(0, len(new_lines) - len(old_lines)),
        "lines_removed": max(0, len(old_lines) - len(new_lines)),
        "has_changes": old_content != new_content,
        "content_size_change": len(new_content) - len(old_content),
    }
