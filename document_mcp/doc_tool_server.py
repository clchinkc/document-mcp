"""
MCP Server for Document Management.

This module provides a FastAPI-based MCP server for managing structured Markdown documents.
It exposes tools for creating, reading, updating, and deleting documents and chapters,
as well as for analyzing their content.
"""
import argparse
import datetime
import difflib  # Added for generating unified diffs
import os
import re  # Added for robust paragraph splitting
import shutil
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp.server import FastMCP
from mcp.types import Resource, TextResourceContents, ErrorData
from mcp import McpError
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import Response

# import logging # No longer needed here
# import functools # No longer needed here
from .logger_config import (  # Import the configured logger and the decorator
    log_mcp_call,
    log_structured_error,
    ErrorCategory,
    safe_operation,
)

# Import metrics functionality for the metrics endpoint
try:
    from .metrics_config import get_metrics_export
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

# --- Configuration ---
# Each "document" will be a subdirectory within DOCS_ROOT_DIR.
# Chapters will be .md files within their respective document subdirectory.
# Default for production
_DEFAULT_DOCS_ROOT = ".documents_storage"

# Check if running under pytest. If so, allow override via .env for test isolation.
# In production, this is not used and the path is fixed.
if "PYTEST_CURRENT_TEST" in os.environ:
    load_dotenv()
    DOCS_ROOT_DIR_NAME = os.environ.get("DOCUMENT_ROOT_DIR", _DEFAULT_DOCS_ROOT)
else:
    DOCS_ROOT_DIR_NAME = _DEFAULT_DOCS_ROOT

DOCS_ROOT_PATH = Path(DOCS_ROOT_DIR_NAME)
DOCS_ROOT_PATH.mkdir(parents=True, exist_ok=True)  # Ensure the root directory exists

# Manifest file name to store chapter order and metadata (optional, for future explicit ordering)
CHAPTER_MANIFEST_FILE = "_manifest.json"
# Summary file name
DOCUMENT_SUMMARY_FILE = "_SUMMARY.md"

# Validation constants
MAX_DOCUMENT_NAME_LENGTH = 100
MAX_CHAPTER_NAME_LENGTH = 100
MAX_CONTENT_LENGTH = 1_000_000  # 1MB max content
MIN_PARAGRAPH_INDEX = 0

# HTTP SSE server configuration
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 3001

mcp_server = FastMCP(
    name="DocumentManagementTools",
    capabilities=["tools", "resources"]
)

# --- Input Validation Helpers ---


def _validate_document_name(document_name: str) -> tuple[bool, str]:
    """Validate document name input."""
    if not document_name or not isinstance(document_name, str) or not document_name.strip():
        return False, "Document name cannot be empty"
    if len(document_name) > MAX_DOCUMENT_NAME_LENGTH:
        return (
            False,
            f"Document name too long (max {MAX_DOCUMENT_NAME_LENGTH} characters)",
        )
    if "/" in document_name or "\\" in document_name:
        return False, "Document name cannot contain path separators"
    if document_name.startswith("."):
        return False, "Document name cannot start with a dot"
    return True, ""


def _validate_chapter_name(chapter_name: str) -> tuple[bool, str]:
    """Validate chapter name input."""
    if not chapter_name or not isinstance(chapter_name, str) or not chapter_name.strip():
        return False, "Chapter name cannot be empty"
    if len(chapter_name) > MAX_CHAPTER_NAME_LENGTH:
        return (
            False,
            f"Chapter name too long (max {MAX_CHAPTER_NAME_LENGTH} characters)",
        )
    if chapter_name == CHAPTER_MANIFEST_FILE:
        return False, f"Chapter name cannot be reserved name '{CHAPTER_MANIFEST_FILE}'"
    if not chapter_name.lower().endswith(".md"):
        return False, "Chapter name must end with .md"
    if "/" in chapter_name or "\\" in chapter_name:
        return False, "Chapter name cannot contain path separators"
    return True, ""


def _validate_content(content: str) -> tuple[bool, str]:
    """Validate content input."""
    if content is None:
        return False, "Content cannot be None"
    if not isinstance(content, str):
        return False, "Content must be a string"
    if len(content) > MAX_CONTENT_LENGTH:
        return False, f"Content too long (max {MAX_CONTENT_LENGTH} characters)"
    return True, ""


def _validate_paragraph_index(index: int) -> tuple[bool, str]:
    """Validate paragraph index input."""
    if not isinstance(index, int):
        return False, "Paragraph index must be an integer"
    if index < MIN_PARAGRAPH_INDEX:
        return False, "Paragraph index cannot be negative"
    return True, ""


def _validate_search_query(query: str) -> tuple[bool, str]:
    """Validate search query input."""
    if query is None:
        return False, "Search query cannot be None"
    if not isinstance(query, str):
        return False, "Search query must be a string"
    if not query.strip():
        return False, "Search query cannot be empty or whitespace only"
    return True, ""


# --- Diff Generation Helper ---


def _generate_content_diff(original_content: str, new_content: str, filename: str = "chapter") -> Dict[str, Any]:
    """
    Generate a unified diff between original and new content.
    
    Compares two strings and produces a diff report including a summary,
    lines added/removed, and the full unified diff text.
    """
    if original_content == new_content:
        return {
            "changed": False,
            "diff": None,
            "summary": "No changes made to content"
        }
    
    # Split content into lines for difflib
    original_lines = original_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    
    # Generate unified diff
    diff_lines = list(difflib.unified_diff(
        original_lines,
        new_lines,
        fromfile=f"{filename} (before)",
        tofile=f"{filename} (after)",
        lineterm=""
    ))
    
    # Count changes
    added_lines = sum(1 for line in diff_lines if line.startswith('+') and not line.startswith('+++'))
    removed_lines = sum(1 for line in diff_lines if line.startswith('-') and not line.startswith('---'))
    
    return {
        "changed": True,
        "diff": "\n".join(diff_lines),
        "summary": f"Modified content: +{added_lines} lines, -{removed_lines} lines",
        "lines_added": added_lines,
        "lines_removed": removed_lines
    }


# --- Pydantic Models for Tool I/O ---


class OperationStatus(BaseModel):
    """Generic status for operations."""

    success: bool
    message: str
    details: Optional[Dict[str, Any]] = (
        None  # For extra info, e.g., created entity name
    )


class ChapterMetadata(BaseModel):
    """Metadata for a chapter within a document."""

    chapter_name: str  # File name of the chapter, e.g., "01-introduction.md"
    title: Optional[str] = None  # Optional: Could be extracted from H1 or from manifest
    word_count: int
    paragraph_count: int
    last_modified: datetime.datetime
    # chapter_index: int # Determined by order in list_chapters


class DocumentInfo(BaseModel):
    """Represents metadata for a document."""

    document_name: str  # Directory name of the document
    total_chapters: int
    total_word_count: int
    total_paragraph_count: int
    last_modified: (
        datetime.datetime
    )  # Could be latest of any chapter or document folder itself
    chapters: List[ChapterMetadata]  # Ordered list of chapter metadata
    has_summary: bool = False


class ParagraphDetail(BaseModel):
    """Detailed information about a paragraph."""

    document_name: str
    chapter_name: str
    paragraph_index_in_chapter: int  # 0-indexed within its chapter
    content: str
    word_count: int


class ChapterContent(BaseModel):
    """Content of a chapter file."""

    document_name: str
    chapter_name: str
    # chapter_index: int # Can be inferred from order if needed by agent
    content: str
    word_count: int
    paragraph_count: int
    last_modified: datetime.datetime


class FullDocumentContent(BaseModel):
    """Content of an entire document, comprising all its chapters in order."""

    document_name: str
    chapters: List[ChapterContent]  # Ordered list of chapter contents
    total_word_count: int
    total_paragraph_count: int


class DocumentSummary(BaseModel):
    """Content of a document's summary file."""

    document_name: str
    content: str


class StatisticsReport(BaseModel):
    """Report for analytical queries."""

    scope: str  # e.g., "document: my_doc", "chapter: my_doc/ch1.md"
    word_count: int
    paragraph_count: int
    chapter_count: Optional[int] = None  # Only for document-level stats


# --- Helper Functions ---


def _get_document_path(document_name: str) -> Path:
    """Return the full path for a given document name."""
    # Ensure DOCS_ROOT_PATH is a Path object (defensive programming for tests)
    root_path = Path(DOCS_ROOT_PATH) if not isinstance(DOCS_ROOT_PATH, Path) else DOCS_ROOT_PATH
    return root_path / document_name


def _get_chapter_path(document_name: str, chapter_filename: str) -> Path:
    """Return the full path for a given chapter file."""
    doc_path = _get_document_path(document_name)
    return doc_path / chapter_filename

def _is_valid_chapter_filename(filename: str) -> bool:
    """
    Check if a filename is a valid, non-reserved chapter file.
    
    Verifies that the filename ends with '.md' and is not a reserved name
    like the manifest or summary file.
    """
    if not filename.lower().endswith(".md"):
        return False
    if filename == CHAPTER_MANIFEST_FILE:
        return False
    if filename == DOCUMENT_SUMMARY_FILE: # Exclude document summary file
        return False
    return True


def _split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into a list of paragraphs.
    
    Paragraphs are separated by one or more blank lines. Leading/trailing
    whitespace is stripped from each paragraph.
    """
    if not text:
        return []
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Split by one or more blank lines (a line with only whitespace is considered blank after strip)
    # Using re.split on '\n\s*\n' (newline, optional whitespace, newline)
    # also stripping the overall text first to handle leading/trailing blank areas cleanly.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized_text.strip())]
    return [
        p for p in paragraphs if p
    ]  # Filter out any truly empty strings resulting from multiple splits or empty strip


def _count_words(text: str) -> int:
    """Count the number of words in a given text string."""
    return len(text.split())


def _get_ordered_chapter_files(document_name: str) -> List[Path]:
    """
    Retrieve a sorted list of all valid chapter files in a document.
    
    The files are sorted alphanumerically by filename. Non-chapter files
    (like summaries or manifests) are excluded.
    """
    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        return []

    # For now, simple alphanumeric sort of .md files.
    # Future: could read CHAPTER_MANIFEST_FILE for explicit order.
    chapter_files = sorted(
        [
            f
            for f in doc_path.iterdir()
            if f.is_file() and _is_valid_chapter_filename(f.name)
        ]
    )
    return chapter_files


def _read_chapter_content_details(
    document_name: str, chapter_file_path: Path
) -> Optional[ChapterContent]:
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
            last_modified=datetime.datetime.fromtimestamp(
                stat.st_mtime, tz=datetime.timezone.utc
            ),
        )
    except Exception as e:
        log_structured_error(
            category=ErrorCategory.ERROR,
            message=f"Failed to read chapter file: {chapter_file_path.name}",
            exception=e,
            context={
                "document_name": document_name,
                "chapter_file_path": str(chapter_file_path),
                "file_exists": chapter_file_path.exists()
            },
            operation="read_chapter_content"
        )
        return None


def _get_chapter_metadata(
    document_name: str, chapter_file_path: Path
) -> Optional[ChapterMetadata]:
    """
    Generate metadata for a chapter from its file path.
    
    This helper reads chapter content to calculate word and paragraph counts
    for the metadata object.
    """
    if not chapter_file_path.is_file():
        return None
    try:
        content = chapter_file_path.read_text(
            encoding="utf-8"
        )  # Read to count words/paragraphs
        paragraphs = _split_into_paragraphs(content)
        word_count = _count_words(content)
        stat = chapter_file_path.stat()
        return ChapterMetadata(
            chapter_name=chapter_file_path.name,
            word_count=word_count,
            paragraph_count=len(paragraphs),
            last_modified=datetime.datetime.fromtimestamp(
                stat.st_mtime, tz=datetime.timezone.utc
            ),
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
                "file_exists": chapter_file_path.exists()
            },
            operation="get_chapter_metadata"
        )
        return None


# ============================================================================
# DOCUMENT MANAGEMENT TOOLS
# ============================================================================
# Tools for managing document collections (directories containing chapters)

@mcp_server.tool()
@log_mcp_call
def list_documents() -> List[DocumentInfo]:
    """
    List all available document collections in the document management system.
    
    This tool retrieves metadata for all document directories, where each document
    is a collection of ordered Markdown chapter files (.md). Provides comprehensive
    information including chapter counts, word counts, and modification timestamps.
    
    Parameters:
        None
    
    Returns:
        List[DocumentInfo]: A list of document metadata objects. Each DocumentInfo contains:
            - document_name (str): Directory name of the document
            - total_chapters (int): Number of chapter files in the document
            - total_word_count (int): Sum of words across all chapters
            - total_paragraph_count (int): Sum of paragraphs across all chapters  
            - last_modified (datetime): Most recent modification time across all chapters
            - chapters (List[ChapterMetadata]): Ordered list of chapter metadata
            - has_summary (bool): Whether a _SUMMARY.md file exists
        
        Returns empty list [] if no documents exist or documents directory is not found.
    
    Example Usage:
        ```json
        {
            "name": "list_documents",
            "arguments": {}
        }
        ```
    
    Example Response:
        ```json
        [
            {
                "document_name": "user_guide",
                "total_chapters": 3,
                "total_word_count": 1250,
                "total_paragraph_count": 45,
                "last_modified": "2024-01-15T10:30:00Z",
                "chapters": [
                    {
                        "chapter_name": "01-introduction.md",
                        "word_count": 300,
                        "paragraph_count": 12,
                        "last_modified": "2024-01-15T10:30:00Z"
                    }
                ],
                "has_summary": true
            }
        ]
        ```
    """
    docs_info = []
    # Ensure DOCS_ROOT_PATH is a Path object (defensive programming for tests)
    root_path = Path(DOCS_ROOT_PATH) if not isinstance(DOCS_ROOT_PATH, Path) else DOCS_ROOT_PATH
    
    if not root_path.exists() or not root_path.is_dir():
        return []

    for doc_dir in root_path.iterdir():
        if doc_dir.is_dir():  # Each subdirectory is a potential document
            document_name = doc_dir.name
            ordered_chapter_files = _get_ordered_chapter_files(document_name)

            chapters_metadata_list = []
            doc_total_word_count = 0
            doc_total_paragraph_count = 0
            latest_mod_time = datetime.datetime.min.replace(
                tzinfo=datetime.timezone.utc
            )  # Ensure timezone aware for comparison

            for chapter_file_path in ordered_chapter_files:
                metadata = _get_chapter_metadata(document_name, chapter_file_path)
                if metadata:
                    chapters_metadata_list.append(metadata)
                    doc_total_word_count += metadata.word_count
                    doc_total_paragraph_count += metadata.paragraph_count
                    # Ensure metadata.last_modified is offset-aware before comparison
                    current_mod_time_aware = metadata.last_modified
                    if current_mod_time_aware > latest_mod_time:
                        latest_mod_time = current_mod_time_aware

            if (
                not chapters_metadata_list
            ):  # If no valid chapters, maybe don't list as a doc or list with 0s
                # Or, use directory's mtime if no chapters. For now, only list if chapters exist.
                # Or list if it's an empty initialized doc.
                # Let's list it even if empty, using the folder's mtime.
                if not ordered_chapter_files:  # No chapter files at all
                    stat_dir = doc_dir.stat()
                    latest_mod_time = datetime.datetime.fromtimestamp(
                        stat_dir.st_mtime, tz=datetime.timezone.utc
                    )

            summary_file_path = doc_dir / DOCUMENT_SUMMARY_FILE
            has_summary_file = summary_file_path.is_file()

            docs_info.append(
                DocumentInfo(
                    document_name=document_name,
                    total_chapters=len(chapters_metadata_list),
                    total_word_count=doc_total_word_count,
                    total_paragraph_count=doc_total_paragraph_count,
                    last_modified=(
                        latest_mod_time
                        if latest_mod_time
                        != datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
                        else datetime.datetime.fromtimestamp(
                            doc_dir.stat().st_mtime, tz=datetime.timezone.utc
                        )
                    ),
                    chapters=chapters_metadata_list,
                    has_summary=has_summary_file,
                )
            )
    return docs_info


# ============================================================================
# CHAPTER MANAGEMENT TOOLS  
# ============================================================================
# Tools for managing individual chapter files within documents

@mcp_server.tool()
@log_mcp_call
def list_chapters(document_name: str) -> Optional[List[ChapterMetadata]]:
    """
    List all chapter files within a specified document, ordered by filename.
    
    This tool retrieves metadata for all chapter files (.md) within a document directory.
    Chapters are automatically ordered alphanumerically by filename, which typically
    corresponds to their intended sequence (e.g., 01-intro.md, 02-setup.md).
    
    Parameters:
        document_name (str): Name of the document directory to list chapters from
    
    Returns:
        Optional[List[ChapterMetadata]]: List of chapter metadata objects if document exists,
        None if document not found. Each ChapterMetadata contains:
            - chapter_name (str): Filename of the chapter (e.g., "01-introduction.md")
            - title (Optional[str]): Chapter title (currently None, reserved for future use)
            - word_count (int): Total number of words in the chapter
            - paragraph_count (int): Total number of paragraphs in the chapter
            - last_modified (datetime): Timestamp of last file modification
        
        Returns empty list [] if document exists but contains no valid chapter files.
        Returns None if the document directory does not exist.
    
    Example Usage:
        ```json
        {
            "name": "list_chapters",
            "arguments": {
                "document_name": "user_guide"
            }
        }
        ```
    
    Example Success Response:
        ```json
        [
            {
                "chapter_name": "01-introduction.md",
                "title": null,
                "word_count": 342,
                "paragraph_count": 8,
                "last_modified": "2024-01-15T10:30:00Z"
            },
            {
                "chapter_name": "02-getting-started.md", 
                "title": null,
                "word_count": 567,
                "paragraph_count": 15,
                "last_modified": "2024-01-15T11:45:00Z"
            }
        ]
        ```
    
    Example Not Found Response:
        ```json
        null
        ```
    """
    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        print(f"Document '{document_name}' not found at {doc_path}")
        return None  # Or perhaps OperationStatus(success=False, message="Document not found")
        # For now, following Optional[List[...]] pattern for read lists

    ordered_chapter_files = _get_ordered_chapter_files(document_name)
    chapters_metadata_list = []
    for chapter_file_path in ordered_chapter_files:
        metadata = _get_chapter_metadata(document_name, chapter_file_path)
        if metadata:
            chapters_metadata_list.append(metadata)

    if not ordered_chapter_files and not chapters_metadata_list:
        # If the directory exists but has no valid chapter files, return empty list.
        return []

    return chapters_metadata_list


@mcp_server.tool()
@log_mcp_call
def read_chapter_content(
    document_name: str, chapter_name: str
) -> Optional[ChapterContent]:
    r"""
    Retrieve the complete content and metadata of a specific chapter within a document.
    
    This tool reads a chapter file (.md) from a document directory and returns both
    the raw content and associated metadata including word counts, paragraph counts,
    and modification timestamps. Returns None if the chapter or document is not found.
    
    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to read. Must be:
            - Valid .md filename (e.g., "01-introduction.md")
            - Existing file within the specified document directory
            - Not a reserved filename like "_manifest.json"
    
    Returns:
        Optional[ChapterContent]: Chapter content object if found, None if not found. 
        ChapterContent contains:
            - document_name (str): Name of the parent document
            - chapter_name (str): Filename of the chapter
            - content (str): Full raw text content of the chapter file
            - word_count (int): Total number of words in the chapter
            - paragraph_count (int): Total number of paragraphs in the chapter
            - last_modified (datetime): Timestamp of last file modification
        
        Returns None if document doesn't exist, chapter doesn't exist, or chapter
        filename is invalid.
    
    Example Usage:
        ```json
        {
            "name": "read_chapter_content",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-introduction.md"
            }
        }
        ```
    
    Example Success Response:
        ```json
        {
            "document_name": "user_guide",
            "chapter_name": "01-introduction.md",
            "content": "# Introduction\n\nWelcome to our user guide...",
            "word_count": 342,
            "paragraph_count": 8,
            "last_modified": "2024-01-15T10:30:00Z"
        }
        ```
    
    Example Not Found Response:
        ```json
        null
        ```
    """
    chapter_file_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_file_path.is_file() or not _is_valid_chapter_filename(chapter_name):
        print(
            f"Chapter '{chapter_name}' not found or invalid in document '{document_name}' at {chapter_file_path}"
        )
        return None
    return _read_chapter_content_details(document_name, chapter_file_path)


@mcp_server.tool()
@log_mcp_call
def read_document_summary(document_name: str) -> Optional[DocumentSummary]:
    r"""
    Retrieve the content of a document's summary file (_SUMMARY.md).
    
    This tool reads the special _SUMMARY.md file that can be used to provide
    an overview or table of contents for a document collection. The summary
    file is optional and may not exist for all documents.
    
    Parameters:
        document_name (str): Name of the document directory to read summary from
    
    Returns:
        Optional[DocumentSummary]: A DocumentSummary object containing the document name
        and summary content if it exists, None if the summary file doesn't exist or
        cannot be read.
        
        Returns None if:
        - Document directory doesn't exist
        - _SUMMARY.md file doesn't exist in the document
    
    Example Usage:
        ```json
        {
            "name": "read_document_summary",
            "arguments": {
                "document_name": "user_guide"
            }
        }
        ```
    
    Example Success Response:
        ```json
        {
            "document_name": "user_guide",
            "content": "# Document Summary\n\n- Chapter 1\n- Chapter 2"
        }
        ```
    
    Example Not Found Response:
        ```json
        null
        ```
    """
    # Validate document name (optional, but good practice if it can be invalid)
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        log_structured_error(
            category=ErrorCategory.WARNING,
            message="Invalid document name provided",
            context={
                "document_name": document_name,
                "validation_error": doc_error
            },
            operation="read_document_summary"
        )
        # Depending on desired strictness, could return None or raise error
        return None # For now, let's be lenient if the path check below handles it

    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        log_structured_error(
            category=ErrorCategory.INFO,
            message="Document not found",
            context={
                "document_name": document_name,
                "attempted_path": str(doc_path)
            },
            operation="read_document_summary"
        )
        return None

    summary_file_path = doc_path / DOCUMENT_SUMMARY_FILE
    if not summary_file_path.is_file():
        log_structured_error(
            category=ErrorCategory.INFO,
            message="Summary file not found in document",
            context={
                "document_name": document_name,
                "summary_file_name": DOCUMENT_SUMMARY_FILE,
                "summary_file_path": str(summary_file_path)
            },
            operation="read_document_summary"
        )
        return None

    try:
        summary_content = summary_file_path.read_text(encoding="utf-8")
        return DocumentSummary(document_name=document_name, content=summary_content)
    except Exception as e:
        log_structured_error(
            category=ErrorCategory.ERROR,
            message="Failed to read summary file",
            exception=e,
            context={
                "document_name": document_name,
                "summary_file_path": str(summary_file_path)
            },
            operation="read_document_summary"
        )
        return None


@mcp_server.tool()
@log_mcp_call
def read_paragraph_content(
    document_name: str, chapter_name: str, paragraph_index_in_chapter: int
) -> Optional[ParagraphDetail]:
    """
    Retrieve the content and metadata of a specific paragraph within a chapter.
    
    This tool extracts a single paragraph from a chapter file using zero-indexed
    positioning. Paragraphs are defined as text blocks separated by blank lines.
    Useful for targeted content retrieval and editing operations.
    
    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter containing the paragraph
        paragraph_index_in_chapter (int): Zero-indexed position of the paragraph. Must be:
            - Non-negative integer (≥0)
            - Within the valid range of existing paragraphs in the chapter
    
    Returns:
        Optional[ParagraphDetail]: Paragraph content object if found, None if not found.
        ParagraphDetail contains:
            - document_name (str): Name of the parent document
            - chapter_name (str): Name of the chapter file
            - paragraph_index_in_chapter (int): Zero-indexed position within the chapter
            - content (str): Full text content of the paragraph
            - word_count (int): Number of words in the paragraph
        
        Returns None if:
        - Document doesn't exist
        - Chapter doesn't exist or is invalid
        - Paragraph index is out of bounds (negative or exceeds paragraph count)
    
    Example Usage:
        ```json
        {
            "name": "read_paragraph_content",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-introduction.md",
                "paragraph_index_in_chapter": 2
            }
        }
        ```
    
    Example Success Response:
        ```json
        {
            "document_name": "user_guide",
            "chapter_name": "01-introduction.md", 
            "paragraph_index_in_chapter": 2,
            "content": "This comprehensive guide will walk you through all the essential features and help you become productive quickly.",
            "word_count": 18
        }
        ```
    
    Example Not Found Response:
        ```json
        null
        ```
    """
    chapter_content_obj = read_chapter_content(document_name, chapter_name)
    if not chapter_content_obj:
        return None

    paragraphs = _split_into_paragraphs(chapter_content_obj.content)
    total_paragraphs = len(paragraphs)

    if not (0 <= paragraph_index_in_chapter < total_paragraphs):
        log_structured_error(
            category=ErrorCategory.WARNING,
            message="Paragraph index out of bounds",
            context={
                "document_name": document_name,
                "chapter_name": chapter_name,
                "paragraph_index": paragraph_index_in_chapter,
                "total_paragraphs": total_paragraphs,
                "valid_range": f"0-{total_paragraphs-1}"
            },
            operation="read_paragraph_content"
        )
        return None

    paragraph_text = paragraphs[paragraph_index_in_chapter]
    return ParagraphDetail(
        document_name=document_name,
        chapter_name=chapter_name,
        paragraph_index_in_chapter=paragraph_index_in_chapter,
        content=paragraph_text,
        word_count=_count_words(paragraph_text),
    )


@mcp_server.tool()
@log_mcp_call
def read_full_document(document_name: str) -> Optional[FullDocumentContent]:
    r"""
    Retrieve the complete content of an entire document including all chapters in order.
    
    This tool reads all chapter files within a document directory and returns them
    as a single structured object. Chapters are ordered alphanumerically by filename,
    providing the complete document content for comprehensive operations.
    
    Parameters:
        document_name (str): Name of the document directory to read completely
    
    Returns:
        Optional[FullDocumentContent]: Complete document content object if found, None if not found.
        FullDocumentContent contains:
            - document_name (str): Name of the document
            - chapters (List[ChapterContent]): Ordered list of all chapter contents
            - total_word_count (int): Sum of words across all chapters
            - total_paragraph_count (int): Sum of paragraphs across all chapters
        
        Each ChapterContent within chapters contains:
            - document_name (str): Name of the parent document
            - chapter_name (str): Filename of the chapter
            - content (str): Full raw text content of the chapter
            - word_count (int): Number of words in the chapter
            - paragraph_count (int): Number of paragraphs in the chapter
            - last_modified (datetime): Timestamp of last file modification
        
        Returns None if document directory doesn't exist.
        Returns empty chapters list if document exists but has no chapter files.
    
    Example Usage:
        ```json
        {
            "name": "read_full_document",
            "arguments": {
                "document_name": "user_guide"
            }
        }
        ```
    
    Example Success Response:
        ```json
        {
            "document_name": "user_guide",
            "chapters": [
                {
                    "document_name": "user_guide",
                    "chapter_name": "01-introduction.md",
                    "content": "# Introduction\n\nWelcome to our guide...",
                    "word_count": 342,
                    "paragraph_count": 8,
                    "last_modified": "2024-01-15T10:30:00Z"
                },
                {
                    "document_name": "user_guide", 
                    "chapter_name": "02-setup.md",
                    "content": "# Setup\n\nFollow these steps...",
                    "word_count": 567,
                    "paragraph_count": 15,
                    "last_modified": "2024-01-15T11:45:00Z"
                }
            ],
            "total_word_count": 909,
            "total_paragraph_count": 23
        }
        ```
    
    Example Not Found Response:
        ```json
        null
        ```
    """
    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        log_structured_error(
            category=ErrorCategory.INFO,
            message="Document not found for full read",
            context={
                "document_name": document_name,
                "attempted_path": str(doc_path)
            },
            operation="read_full_document"
        )
        return None

    ordered_chapter_files = _get_ordered_chapter_files(document_name)
    if not ordered_chapter_files:
        # Document exists but has no chapters
        return FullDocumentContent(
            document_name=document_name,
            chapters=[],
            total_word_count=0,
            total_paragraph_count=0,
        )

    all_chapter_contents = []
    doc_total_word_count = 0
    doc_total_paragraph_count = 0

    for chapter_file_path in ordered_chapter_files:
        chapter_details = _read_chapter_content_details(
            document_name, chapter_file_path
        )
        if chapter_details:
            all_chapter_contents.append(chapter_details)
            doc_total_word_count += chapter_details.word_count
            doc_total_paragraph_count += chapter_details.paragraph_count
        else:
            # If a chapter file is listed but can't be read, this indicates an issue.
            # For now, we'll skip it, but this could also be an error condition.
            log_structured_error(
                category=ErrorCategory.WARNING,
                message="Could not read chapter file, skipping",
                context={
                    "document_name": document_name,
                    "chapter_file_name": chapter_file_path.name,
                    "chapter_file_path": str(chapter_file_path)
                },
                operation="read_full_document"
            )

    return FullDocumentContent(
        document_name=document_name,
        chapters=all_chapter_contents,
        total_word_count=doc_total_word_count,
        total_paragraph_count=doc_total_paragraph_count,
    )


@mcp_server.tool()
@log_mcp_call
def create_document(document_name: str) -> OperationStatus:
    r"""
    Create a new document collection as a directory in the document management system.
    
    This tool initializes a new document by creating a directory that will contain
    chapter files (.md). The document name must be valid for filesystem usage and
    will serve as the directory name for organizing chapters.
    
    Parameters:
        document_name (str): Name for the new document directory. Must be:
            - Non-empty string
            - ≤100 characters
            - Valid filesystem directory name
            - Cannot contain path separators (/ or \)
            - Cannot start with a dot (.)
            - Cannot conflict with existing document names
    
    Returns:
        OperationStatus: Structured result object containing:
            - success (bool): True if document was created successfully, False otherwise
            - message (str): Human-readable description of the operation result
            - details (Dict[str, Any], optional): Additional context including:
                - document_name (str): Name of the created document (on success)
    
    Example Usage:
        ```json
        {
            "name": "create_document",
            "arguments": {
                "document_name": "user_manual"
            }
        }
        ```
    
    Example Success Response:
        ```json
        {
            "success": true,
            "message": "Document 'user_manual' created successfully.",
            "details": {
                "document_name": "user_manual"
            }
        }
        ```
    
    Example Error Response:
        ```json
        {
            "success": false,
            "message": "Document 'user_manual' already exists.",
            "details": null
        }
        ```
    """
    # Validate input
    is_valid, error_msg = _validate_document_name(document_name)
    if not is_valid:
        return OperationStatus(success=False, message=error_msg)

    doc_path = _get_document_path(document_name)
    if doc_path.exists():
        return OperationStatus(
            success=False, message=f"Document '{document_name}' already exists."
        )
    try:
        doc_path.mkdir(parents=True, exist_ok=False)
        return OperationStatus(
            success=True,
            message=f"Document '{document_name}' created successfully.",
            details={"document_name": document_name},
        )
    except Exception as e:
        return OperationStatus(
            success=False, message=f"Error creating document '{document_name}': {e}"
        )


@mcp_server.tool()
@log_mcp_call
def delete_document(document_name: str) -> OperationStatus:
    """
    Permanently deletes a document directory and all its chapter files.
    
    This tool removes an entire document collection including all chapter files
    and any associated metadata. This operation is irreversible and should be
    used with caution. All content within the document directory will be lost.
    
    Parameters:
        document_name (str): Name of the document directory to delete
    
    Returns:
        OperationStatus: Structured result object containing:
            - success (bool): True if document was deleted successfully, False otherwise
            - message (str): Human-readable description of the operation result
            - details (Dict[str, Any], optional): Additional context (currently None)
    
    Example Usage:
        ```json
        {
            "name": "delete_document",
            "arguments": {
                "document_name": "old_manual"
            }
        }
        ```
    
    Example Success Response:
        ```json
        {
            "success": true,
            "message": "Document 'old_manual' and its contents deleted successfully.",
            "details": null
        }
        ```
    
    Example Error Response:
        ```json
        {
            "success": false,
            "message": "Document 'old_manual' not found.",
            "details": null
        }
        ```
    """
    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        return OperationStatus(
            success=False, message=f"Document '{document_name}' not found."
        )
    try:
        shutil.rmtree(doc_path)
        return OperationStatus(
            success=True,
            message=f"Document '{document_name}' and its contents deleted successfully.",
        )
    except Exception as e:
        return OperationStatus(
            success=False, message=f"Error deleting document '{document_name}': {e}"
        )


@mcp_server.tool()
@log_mcp_call
def create_chapter(
    document_name: str, chapter_name: str, initial_content: str = ""
) -> OperationStatus:
    r"""
    Create a new chapter file within an existing document directory.

    .. note::
       While this tool supports creating a chapter with initial content, for clarity it is recommended to create an empty chapter and then use a dedicated write/update tool to add content. This helps separate creation from modification operations.
    
    This tool adds a new Markdown chapter file to a document collection. The chapter
    will be ordered based on its filename when listed with other chapters. Supports
    optional initial content to bootstrap the chapter with starter text.
    
    Parameters:
        document_name (str): Name of the existing document directory to add chapter to
        chapter_name (str): Filename for the new chapter. Must be:
            - Valid .md filename (e.g., "03-advanced-features.md")
            - ≤100 characters
            - Valid filesystem filename
            - Cannot contain path separators (/ or \)
            - Cannot be reserved name like "_manifest.json" or "_SUMMARY.md"
            - Must not already exist in the document
        initial_content (str, optional): Starting content for the new chapter. Can be:
            - Empty string (default): Creates empty chapter file
            - Any valid UTF-8 text content
            - Must be ≤1MB in size
    
    Returns:
        OperationStatus: Structured result object containing:
            - success (bool): True if chapter was created successfully, False otherwise
            - message (str): Human-readable description of the operation result
            - details (Dict[str, Any], optional): Additional context including:
                - document_name (str): Name of the parent document (on success)
                - chapter_name (str): Name of the created chapter (on success)
    
    Example Usage:
        ```json
        {
            "name": "create_chapter",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "03-advanced-features.md",
                "initial_content": "# Advanced Features\n\nThis chapter covers advanced functionality."
            }
        }
        ```
    
    Example Success Response:
        ```json
        {
            "success": true,
            "message": "Chapter '03-advanced-features.md' created successfully in document 'user_guide'.",
            "details": {
                "document_name": "user_guide",
                "chapter_name": "03-advanced-features.md"
            }
        }
        ```
    
    Example Error Response:
        ```json
        {
            "success": false,
            "message": "Chapter '03-advanced-features.md' already exists in document 'user_guide'.",
            "details": null
        }
        ```
    """
    # Validate inputs
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        return OperationStatus(success=False, message=doc_error)

    is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
    if not is_valid_chapter:
        return OperationStatus(success=False, message=chapter_error)

    is_valid_content, content_error = _validate_content(initial_content)
    if not is_valid_content:
        return OperationStatus(success=False, message=content_error)

    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        return OperationStatus(
            success=False, message=f"Document '{document_name}' not found."
        )

    if not _is_valid_chapter_filename(chapter_name):
        return OperationStatus(
            success=False,
            message=f"Invalid chapter name '{chapter_name}'. Must be a .md file and not a reserved name like '{CHAPTER_MANIFEST_FILE}'.",
        )

    chapter_path = _get_chapter_path(document_name, chapter_name)
    if chapter_path.exists():
        return OperationStatus(
            success=False,
            message=f"Chapter '{chapter_name}' already exists in document '{document_name}'.",
        )

    try:
        chapter_path.write_text(initial_content, encoding="utf-8")
        return OperationStatus(
            success=True,
            message=f"Chapter '{chapter_name}' created successfully in document '{document_name}'.",
            details={"document_name": document_name, "chapter_name": chapter_name},
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error creating chapter '{chapter_name}' in document '{document_name}': {e}",
        )


@mcp_server.tool()
@log_mcp_call
def delete_chapter(document_name: str, chapter_name: str) -> OperationStatus:
    """
    Delete a chapter file from a document directory.
    
    This tool permanently removes a chapter file from a document collection.
    The operation is irreversible and will delete the file from the filesystem.
    Use with caution as the chapter content will be lost.
    
    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to delete. Must be:
            - Valid .md filename (e.g., "02-old-chapter.md")
            - Existing file within the specified document directory
            - Not a reserved filename like "_manifest.json"
    
    Returns:
        OperationStatus: Structured result object containing:
            - success (bool): True if chapter was deleted successfully, False otherwise
            - message (str): Human-readable description of the operation result
            - details (Dict[str, Any], optional): Additional context (currently None)
    
    Example Usage:
        ```json
        {
            "name": "delete_chapter",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "02-outdated-section.md"
            }
        }
        ```
    
    Example Success Response:
        ```json
        {
            "success": true,
            "message": "Chapter '02-outdated-section.md' deleted successfully from document 'user_guide'.",
            "details": null
        }
        ```
    
    Example Error Response:
        ```json
        {
            "success": false,
            "message": "Chapter '02-outdated-section.md' not found in document 'user_guide'.",
            "details": null
        }
        ```
    """
    if not _is_valid_chapter_filename(
        chapter_name
    ):  # Check early to avoid issues with non-MD files
        return OperationStatus(
            success=False,
            message=f"Invalid target '{chapter_name}'. Not a valid chapter Markdown file name.",
        )

    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.is_file():
        return OperationStatus(
            success=False,
            message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
        )

    try:
        chapter_path.unlink()
        return OperationStatus(
            success=True,
            message=f"Chapter '{chapter_name}' deleted successfully from document '{document_name}'.",
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error deleting chapter '{chapter_name}' from document '{document_name}': {e}",
        )


@mcp_server.tool()
@log_mcp_call
def write_chapter_content(
    document_name: str, chapter_name: str, new_content: str
) -> OperationStatus:
    r"""
    Overwrite the entire content of a chapter file with new content.

    .. deprecated:: 0.18.0
       This tool's behavior of creating a chapter if it doesn't exist is deprecated and will be removed in a future version.
       In the future, this tool will only write to existing chapters. Please use `create_chapter` for new chapters.
    
    This tool completely replaces the content of an existing chapter file or creates
    a new chapter if it doesn't exist. The operation provides diff information showing
    exactly what changed between the original and new content.
    
    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to write. Must be:
            - Valid .md filename (e.g., "01-introduction.md")  
            - Valid filesystem filename
            - Not a reserved filename like "_manifest.json"
        new_content (str): Complete new content for the chapter file. Can be:
            - Any valid UTF-8 text content
            - Empty string (creates empty chapter)
            - Must be ≤1MB in size
    
    Returns:
        OperationStatus: Structured result object containing:
            - success (bool): True if content was written successfully, False otherwise
            - message (str): Human-readable description of the operation result  
            - details (Dict[str, Any], optional): Includes diff information:
                - changed (bool): Whether content was actually modified
                - diff (str): Unified diff showing changes made
                - summary (str): Brief description of changes
                - lines_added (int): Number of lines added
                - lines_removed (int): Number of lines removed
    
    Example Usage:
        ```json
        {
            "name": "write_chapter_content",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-introduction.md",
                "new_content": "# Introduction\n\nWelcome to our comprehensive user guide.\n\nThis guide will help you get started."
            }
        }
        ```
    
    Example Success Response:
        ```json
        {
            "success": true,
            "message": "Content of chapter '01-introduction.md' in document 'user_guide' updated successfully.",
            "details": {
                "changed": true,
                "diff": "--- 01-introduction.md (before)\n+++ 01-introduction.md (after)\n@@ -1,3 +1,4 @@\n # Introduction\n \n-Welcome to our guide.\n+Welcome to our comprehensive user guide.\n+\n+This guide will help you get started.",
                "summary": "Modified content: +3 lines, -1 lines",
                "lines_added": 3,
                "lines_removed": 1
            }
        }
        ```
    """
    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        # Option: create document if not exists? For now, require existing document.
        return OperationStatus(
            success=False, message=f"Document '{document_name}' not found."
        )

    if not _is_valid_chapter_filename(chapter_name):
        return OperationStatus(
            success=False, message=f"Invalid chapter name '{chapter_name}'."
        )

    chapter_path = _get_chapter_path(document_name, chapter_name)
    try:
        # Read original content before overwriting
        original_content = ""
        if chapter_path.exists():
            original_content = chapter_path.read_text(encoding="utf-8")
        
        # Write new content
        chapter_path.write_text(new_content, encoding="utf-8")
        
        # Generate diff for details
        diff_info = _generate_content_diff(original_content, new_content, chapter_name)
        
        return OperationStatus(
            success=True,
            message=f"Content of chapter '{chapter_name}' in document '{document_name}' updated successfully.",
            details=diff_info,
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error writing to chapter '{chapter_name}' in document '{document_name}': {e}",
        )


# ============================================================================
# PARAGRAPH MANAGEMENT TOOLS (ATOMIC)
# ============================================================================
# Atomic tools for precise paragraph-level operations within chapters

@mcp_server.tool()
@log_mcp_call
def replace_paragraph(
    document_name: str, chapter_name: str, paragraph_index: int, new_content: str
) -> OperationStatus:
    """
    Replace the content of a specific paragraph within a chapter.
    
    This atomic tool replaces an existing paragraph at the specified index with
    new content. The paragraph index is zero-based, and the operation will fail
    if the index is out of bounds.
    
    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to modify (must end with .md)
        paragraph_index (int): Zero-indexed position of the paragraph to replace (≥0)
        new_content (str): New content to replace the existing paragraph with
    
    Returns:
        OperationStatus: Result object with success status, message, and diff details
    
    Example Usage:
        ```json
        {
            "name": "replace_paragraph",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-intro.md",
                "paragraph_index": 2,
                "new_content": "This is the updated paragraph content."
            }
        }
        ```
    """
    # Validate inputs
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        return OperationStatus(success=False, message=doc_error)

    is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
    if not is_valid_chapter:
        return OperationStatus(success=False, message=chapter_error)

    is_valid_index, index_error = _validate_paragraph_index(paragraph_index)
    if not is_valid_index:
        return OperationStatus(success=False, message=index_error)

    is_valid_content, content_error = _validate_content(new_content)
    if not is_valid_content:
        return OperationStatus(success=False, message=content_error)

    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
        return OperationStatus(
            success=False,
            message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
        )

    try:
        original_full_content = chapter_path.read_text(encoding="utf-8")
        paragraphs = _split_into_paragraphs(original_full_content)
        total_paragraphs = len(paragraphs)

        if not (0 <= paragraph_index < total_paragraphs):
            return OperationStatus(
                success=False,
                message=f"Paragraph index {paragraph_index} is out of bounds (0-{total_paragraphs-1}).",
            )

        paragraphs[paragraph_index] = new_content
        final_content = "\n\n".join(paragraphs)
        chapter_path.write_text(final_content, encoding="utf-8")

        # Generate diff for details
        diff_info = _generate_content_diff(original_full_content, final_content, chapter_name)
        
        return OperationStatus(
            success=True,
            message=f"Paragraph {paragraph_index} in '{chapter_name}' ({document_name}) successfully replaced.",
            details=diff_info,
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error replacing paragraph in '{chapter_name}' ({document_name}): {str(e)}",
        )


@mcp_server.tool()
@log_mcp_call
def insert_paragraph_before(
    document_name: str, chapter_name: str, paragraph_index: int, new_content: str
) -> OperationStatus:
    """
    Insert a new paragraph before the specified index within a chapter.
    
    This atomic tool inserts new content as a paragraph before the existing
    paragraph at the specified index. All subsequent paragraphs are shifted down.
    
    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to modify (must end with .md)
        paragraph_index (int): Zero-indexed position to insert before (≥0)
        new_content (str): Content for the new paragraph to insert
    
    Returns:
        OperationStatus: Result object with success status, message, and diff details
    
    Example Usage:
        ```json
        {
            "name": "insert_paragraph_before",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-intro.md",
                "paragraph_index": 1,
                "new_content": "This paragraph will be inserted before paragraph 1."
            }
        }
        ```
    """
    # Validate inputs
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        return OperationStatus(success=False, message=doc_error)

    is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
    if not is_valid_chapter:
        return OperationStatus(success=False, message=chapter_error)

    is_valid_index, index_error = _validate_paragraph_index(paragraph_index)
    if not is_valid_index:
        return OperationStatus(success=False, message=index_error)

    is_valid_content, content_error = _validate_content(new_content)
    if not is_valid_content:
        return OperationStatus(success=False, message=content_error)

    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
        return OperationStatus(
            success=False,
            message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
        )

    try:
        original_full_content = chapter_path.read_text(encoding="utf-8")
        paragraphs = _split_into_paragraphs(original_full_content)
        total_paragraphs = len(paragraphs)

        if not (0 <= paragraph_index <= total_paragraphs):
            return OperationStatus(
                success=False,
                message=f"Paragraph index {paragraph_index} is out of bounds (0-{total_paragraphs}).",
            )

        paragraphs.insert(paragraph_index, new_content)
        final_content = "\n\n".join(paragraphs)
        chapter_path.write_text(final_content, encoding="utf-8")

        # Generate diff for details
        diff_info = _generate_content_diff(original_full_content, final_content, chapter_name)
        
        return OperationStatus(
            success=True,
            message=f"Paragraph inserted before index {paragraph_index} in '{chapter_name}' ({document_name}).",
            details=diff_info,
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error inserting paragraph in '{chapter_name}' ({document_name}): {str(e)}",
        )


@mcp_server.tool()
@log_mcp_call
def insert_paragraph_after(
    document_name: str, chapter_name: str, paragraph_index: int, new_content: str
) -> OperationStatus:
    """
    Insert a new paragraph after the specified index within a chapter.
    
    This atomic tool inserts new content as a paragraph after the existing
    paragraph at the specified index. All subsequent paragraphs are shifted down.
    
    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to modify (must end with .md)
        paragraph_index (int): Zero-indexed position to insert after (≥0)
        new_content (str): Content for the new paragraph to insert
    
    Returns:
        OperationStatus: Result object with success status, message, and diff details
    
    Example Usage:
        ```json
        {
            "name": "insert_paragraph_after",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-intro.md",
                "paragraph_index": 1,
                "new_content": "This paragraph will be inserted after paragraph 1."
            }
        }
        ```
    """
    # Validate inputs
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        return OperationStatus(success=False, message=doc_error)

    is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
    if not is_valid_chapter:
        return OperationStatus(success=False, message=chapter_error)

    is_valid_index, index_error = _validate_paragraph_index(paragraph_index)
    if not is_valid_index:
        return OperationStatus(success=False, message=index_error)

    is_valid_content, content_error = _validate_content(new_content)
    if not is_valid_content:
        return OperationStatus(success=False, message=content_error)

    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
        return OperationStatus(
            success=False,
            message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
        )

    try:
        original_full_content = chapter_path.read_text(encoding="utf-8")
        paragraphs = _split_into_paragraphs(original_full_content)
        total_paragraphs = len(paragraphs)

        if not (0 <= paragraph_index < total_paragraphs) and not (
            total_paragraphs == 0 and paragraph_index == 0
        ):
            return OperationStatus(
                success=False,
                message=f"Paragraph index {paragraph_index} is out of bounds (0-{total_paragraphs-1}).",
            )

        if total_paragraphs == 0 and paragraph_index == 0:  # Insert into empty doc
            paragraphs.append(new_content)
        else:
            paragraphs.insert(paragraph_index + 1, new_content)
        
        final_content = "\n\n".join(paragraphs)
        chapter_path.write_text(final_content, encoding="utf-8")

        # Generate diff for details
        diff_info = _generate_content_diff(original_full_content, final_content, chapter_name)
        
        return OperationStatus(
            success=True,
            message=f"Paragraph inserted after index {paragraph_index} in '{chapter_name}' ({document_name}).",
            details=diff_info,
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error inserting paragraph in '{chapter_name}' ({document_name}): {str(e)}",
        )


@mcp_server.tool()
@log_mcp_call
def delete_paragraph(
    document_name: str, chapter_name: str, paragraph_index: int
) -> OperationStatus:
    """
    Delete a specific paragraph from a chapter.
    
    This atomic tool removes the paragraph at the specified index from the chapter.
    All subsequent paragraphs are shifted up to fill the gap.
    
    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to modify (must end with .md)
        paragraph_index (int): Zero-indexed position of the paragraph to delete (≥0)
    
    Returns:
        OperationStatus: Result object with success status, message, and diff details
    
    Example Usage:
        ```json
        {
            "name": "delete_paragraph",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-intro.md",
                "paragraph_index": 2
            }
        }
        ```
    """
    # Validate inputs
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        return OperationStatus(success=False, message=doc_error)

    is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
    if not is_valid_chapter:
        return OperationStatus(success=False, message=chapter_error)

    is_valid_index, index_error = _validate_paragraph_index(paragraph_index)
    if not is_valid_index:
        return OperationStatus(success=False, message=index_error)

    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
        return OperationStatus(
            success=False,
            message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
        )

    try:
        original_full_content = chapter_path.read_text(encoding="utf-8")
        paragraphs = _split_into_paragraphs(original_full_content)
        total_paragraphs = len(paragraphs)

        if total_paragraphs == 0:
            return OperationStatus(
                success=False,
                message="Cannot delete paragraph from an empty chapter.",
            )

        if not (0 <= paragraph_index < total_paragraphs):
            return OperationStatus(
                success=False,
                message=f"Paragraph index {paragraph_index} is out of bounds for chapter with {total_paragraphs} paragraphs (valid range 0-{total_paragraphs-1}).",
            )

        del paragraphs[paragraph_index]
        final_content = "\n\n".join(paragraphs)
        chapter_path.write_text(final_content, encoding="utf-8")

        # Generate diff for details
        diff_info = _generate_content_diff(original_full_content, final_content, chapter_name)
        
        return OperationStatus(
            success=True,
            message=f"Paragraph {paragraph_index} deleted from '{chapter_name}' ({document_name}).",
            details=diff_info,
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error deleting paragraph from '{chapter_name}' ({document_name}): {str(e)}",
        )



@mcp_server.tool()
@log_mcp_call
def append_paragraph_to_chapter(
    document_name: str, chapter_name: str, paragraph_content: str
) -> OperationStatus:
    r"""
    Append a new paragraph to the end of a specific chapter.
    
    This tool adds new content as a paragraph at the end of an existing chapter file.
    The new paragraph will be separated from existing content by proper paragraph
    spacing (blank lines). Provides diff information showing the changes made.
    
    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to append to. Must be:
            - Valid .md filename (e.g., "01-introduction.md")
            - Existing file within the specified document directory
            - Not a reserved filename like "_manifest.json"
        paragraph_content (str): Content to append as a new paragraph. Must be:
            - Non-empty string
            - Valid UTF-8 text content
            - Must be ≤1MB in size
    
    Returns:
        OperationStatus: Structured result object containing:
            - success (bool): True if paragraph was appended successfully, False otherwise
            - message (str): Human-readable description of the operation result
            - details (Dict[str, Any], optional): Includes diff information:
                - changed (bool): Whether content was actually modified
                - diff (str): Unified diff showing changes made
                - summary (str): Brief description of changes
                - lines_added (int): Number of lines added
                - lines_removed (int): Number of lines removed
    
    Example Usage:
        ```json
        {
            "name": "append_paragraph_to_chapter",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-introduction.md",
                "paragraph_content": "This additional paragraph provides more context about the features covered in this guide."
            }
        }
        ```
    
    Example Success Response:
        ```json
        {
            "success": true,
            "message": "Paragraph appended to chapter '01-introduction.md' in document 'user_guide'.",
            "details": {
                "changed": true,
                "diff": "--- 01-introduction.md (before)\n+++ 01-introduction.md (after)\n@@ -10,0 +10,2 @@\n+\n+This additional paragraph provides more context about the features covered in this guide.",
                "summary": "Modified content: +2 lines, -0 lines", 
                "lines_added": 2,
                "lines_removed": 0
            }
        }
        ```
    """
    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
        return OperationStatus(
            success=False,
            message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
        )

    try:
        original_full_content = chapter_path.read_text(encoding="utf-8")
        paragraphs = _split_into_paragraphs(original_full_content)
        paragraphs.append(paragraph_content)
        # Filter out any potential empty strings that might arise if original_full_content was just newlines
        # or if _split_into_paragraphs somehow yields them, though it's designed not to.
        final_paragraphs = [p for p in paragraphs if p]
        final_content = "\n\n".join(final_paragraphs)
        # If the only content is the new paragraph and it's not empty, ensure no leading newlines.
        if (
            len(final_paragraphs) == 1
            and final_paragraphs[0] == paragraph_content
            and paragraph_content
        ):
            final_content = paragraph_content
        chapter_path.write_text(final_content, encoding="utf-8")
        
        # Generate diff for details
        diff_info = _generate_content_diff(original_full_content, final_content, chapter_name)
        
        return OperationStatus(
            success=True,
            message=f"Paragraph appended to chapter '{chapter_name}' in document '{document_name}'.",
            details=diff_info,
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error appending paragraph to '{chapter_name}': {str(e)}",
        )


# ============================================================================
# SEARCH AND REPLACE TOOLS
# ============================================================================
# Tools for finding and replacing text content within documents and chapters

@mcp_server.tool()
@log_mcp_call
def replace_text_in_chapter(
    document_name: str, chapter_name: str, text_to_find: str, replacement_text: str
) -> OperationStatus:
    r"""
    Replace all occurrences of a text string with another string in a specific chapter.
    
    This tool performs case-sensitive text replacement throughout a chapter file.
    All instances of the search text will be replaced with the replacement text.
    Provides detailed information about the number of replacements made and diff output.
    
    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to modify. Must be:
            - Valid .md filename (e.g., "01-introduction.md")
            - Existing file within the specified document directory
            - Not a reserved filename like "_manifest.json"
        text_to_find (str): Text string to search for and replace. Must be:
            - Non-empty string
            - Exact text match (case-sensitive)
        replacement_text (str): Text to replace found occurrences with. Can be:
            - Any valid UTF-8 text content
            - Empty string (effectively deletes the found text)
            - Must be ≤1MB in size
    
    Returns:
        OperationStatus: Structured result object containing:
            - success (bool): True if operation completed successfully, False otherwise
            - message (str): Human-readable description of the operation result
            - details (Dict[str, Any], optional): Includes replacement information:
                - changed (bool): Whether content was actually modified
                - diff (str): Unified diff showing changes made (if any)
                - summary (str): Brief description of changes
                - lines_added (int): Number of lines added
                - lines_removed (int): Number of lines removed
                - occurrences_replaced (int): Number of text occurrences replaced
    
    Example Usage:
        ```json
        {
            "name": "replace_text_in_chapter",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-introduction.md",
                "text_to_find": "old version",
                "replacement_text": "new version"
            }
        }
        ```
    
    Example Success Response:
        ```json
        {
            "success": true,
            "message": "All 2 occurrences of 'old version' replaced with 'new version' in chapter '01-introduction.md'.",
            "details": {
                "changed": true,
                "diff": "--- 01-introduction.md (before)\n+++ 01-introduction.md (after)\n@@ -5,1 +5,1 @@\n-This old version is deprecated.\n+This new version is current.",
                "summary": "Modified content: +1 lines, -1 lines",
                "lines_added": 1,
                "lines_removed": 1,
                "occurrences_replaced": 2
            }
        }
        ```
    
    Example No Matches Response:
        ```json
        {
            "success": true,
            "message": "Text 'nonexistent' not found in chapter '01-introduction.md'. No changes made.",
            "details": {
                "occurrences_found": 0
            }
        }
        ```
    """
    # Validate inputs
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        return OperationStatus(success=False, message=doc_error)

    is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
    if not is_valid_chapter:
        return OperationStatus(success=False, message=chapter_error)

    if not text_to_find:
        return OperationStatus(success=False, message="Text to find cannot be empty.")

    is_valid_replacement, replacement_error = _validate_content(replacement_text)
    if not is_valid_replacement:
        return OperationStatus(success=False, message=replacement_error)

    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
        return OperationStatus(
            success=False,
            message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
        )

    try:
        original_content = chapter_path.read_text(encoding="utf-8")
        if text_to_find not in original_content:
            return OperationStatus(
                success=True,  # Success, but no change
                message=f"Text '{text_to_find}' not found in chapter '{chapter_name}'. No changes made.",
                details={"occurrences_found": 0},
            )

        modified_content = original_content.replace(text_to_find, replacement_text)
        occurrences = original_content.count(text_to_find)
        chapter_path.write_text(modified_content, encoding="utf-8")
        
        # Generate diff for details
        diff_info = _generate_content_diff(original_content, modified_content, chapter_name)
        diff_info["occurrences_replaced"] = occurrences
        
        return OperationStatus(
            success=True,
            message=f"All {occurrences} occurrences of '{text_to_find}' replaced with '{replacement_text}' in chapter '{chapter_name}'.",
            details=diff_info,
        )
    except Exception as e:
        return OperationStatus(
            success=False, message=f"Error replacing text in '{chapter_name}': {str(e)}"
        )


@mcp_server.tool()
@log_mcp_call
def replace_text_in_document(
    document_name: str, text_to_find: str, replacement_text: str
) -> OperationStatus:
    """
    Replace all occurrences of a text string with another string throughout all chapters of a document.
    
    This tool performs case-sensitive text replacement across all chapter files within
    a document directory. It processes each chapter sequentially and provides comprehensive
    reporting on which chapters were modified and how many replacements were made.
    
    Parameters:
        document_name (str): Name of the document directory to process
        text_to_find (str): Text string to search for and replace. Must be:
            - Non-empty string
            - Exact text match (case-sensitive)
        replacement_text (str): Text to replace found occurrences with. Can be:
            - Any valid UTF-8 text content
            - Empty string (effectively deletes the found text)
            - Must be ≤1MB in size
    
    Returns:
        OperationStatus: Structured result object containing:
            - success (bool): True if operation completed successfully, False otherwise
            - message (str): Human-readable description of the operation result
            - details (Dict[str, Any], optional): Includes comprehensive replacement information:
                - chapters_modified_count (int): Number of chapters that had replacements
                - total_occurrences_replaced (int): Total number of text occurrences replaced
                - modified_chapters (List[Dict]): Details for each modified chapter including:
                    - chapter_name (str): Name of the modified chapter file
                    - occurrences_replaced (int): Number of replacements in this chapter
    
    Example Usage:
        ```json
        {
            "name": "replace_text_in_document",
            "arguments": {
                "document_name": "user_guide",
                "text_to_find": "version 1.0",
                "replacement_text": "version 2.0"
            }
        }
        ```
    
    Example Success Response:
        ```json
        {
            "success": true,
            "message": "Text replacement completed in document 'user_guide'. 5 occurrences replaced across 3 chapter(s).",
            "details": {
                "chapters_modified_count": 3,
                "total_occurrences_replaced": 5,
                "modified_chapters": [
                    {
                        "chapter_name": "01-introduction.md",
                        "occurrences_replaced": 2
                    },
                    {
                        "chapter_name": "02-setup.md", 
                        "occurrences_replaced": 1
                    },
                    {
                        "chapter_name": "03-features.md",
                        "occurrences_replaced": 2
                    }
                ]
            }
        }
        ```
    
    Example No Matches Response:
        ```json
        {
            "success": true,
            "message": "Text 'nonexistent' not found in any chapters of document 'user_guide'. No changes made.",
            "details": {
                "chapters_modified_count": 0,
                "total_occurrences_replaced": 0
            }
        }
            """
    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        return OperationStatus(
            success=False, message=f"Document '{document_name}' not found."
        )

    ordered_chapter_files = _get_ordered_chapter_files(document_name)
    if not ordered_chapter_files:
        return OperationStatus(
            success=True,
            message=f"Document '{document_name}' contains no chapters. No changes made.",
            details={"chapters_modified_count": 0},
        )

    chapters_modified_count = 0
    total_occurrences_replaced = 0
    modified_chapters_details = []

    for chapter_file_path in ordered_chapter_files:
        try:
            original_content = chapter_file_path.read_text(encoding="utf-8")
            if text_to_find in original_content:
                occurrences_in_chapter = original_content.count(text_to_find)
                modified_content = original_content.replace(
                    text_to_find, replacement_text
                )
                chapter_file_path.write_text(modified_content, encoding="utf-8")
                chapters_modified_count += 1
                total_occurrences_replaced += occurrences_in_chapter
                modified_chapters_details.append(
                    {
                        "chapter_name": chapter_file_path.name,
                        "occurrences_replaced": occurrences_in_chapter,
                    }
                )
        except Exception as e:
            # Log or collect errors per chapter? For now, fail fast on first error.
            return OperationStatus(
                success=False,
                message=f"Error replacing text in chapter '{chapter_file_path.name}' of document '{document_name}': {e}",
                details={"chapters_processed_before_error": chapters_modified_count},
            )

    if chapters_modified_count == 0:
        return OperationStatus(
            success=True,
            message=f"Text '{text_to_find}' not found in any chapters of document '{document_name}'. No changes made.",
            details={"chapters_modified_count": 0, "total_occurrences_replaced": 0},
        )

    return OperationStatus(
        success=True,
        message=f"Text replacement completed in document '{document_name}'. {total_occurrences_replaced} occurrences replaced across {chapters_modified_count} chapter(s).",
        details={
            "chapters_modified_count": chapters_modified_count,
            "total_occurrences_replaced": total_occurrences_replaced,
            "modified_chapters": modified_chapters_details,
        },
    )


# ============================================================================
# STATISTICS AND ANALYTICS TOOLS
# ============================================================================
# Tools for analyzing document and chapter metrics, word counts, etc.

@mcp_server.tool()
@log_mcp_call
def get_chapter_statistics(
    document_name: str, chapter_name: str
) -> Optional[StatisticsReport]:
    """
    Retrieve statistical information for a specific chapter.
    
    This tool analyzes a chapter file and returns comprehensive statistics including
    word count and paragraph count. Useful for content analysis, progress tracking,
    and document planning.
    
    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to analyze. Must be:
            - Valid .md filename (e.g., "01-introduction.md")
            - Existing file within the specified document directory
            - Not a reserved filename like "_manifest.json"
    
    Returns:
        Optional[StatisticsReport]: Statistics object if chapter found, None if not found.
        StatisticsReport contains:
            - scope (str): Description of what was analyzed (e.g., "chapter: user_guide/01-intro.md")
            - word_count (int): Total number of words in the chapter
            - paragraph_count (int): Total number of paragraphs in the chapter
            - chapter_count (None): Not applicable for chapter-level statistics
        
        Returns None if document doesn't exist or chapter doesn't exist.
    
    Example Usage:
        ```json
        {
            "name": "get_chapter_statistics",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-introduction.md"
            }
        }
        ```
    
    Example Success Response:
        ```json
        {
            "scope": "chapter: user_guide/01-introduction.md",
            "word_count": 342,
            "paragraph_count": 8,
            "chapter_count": null
        }
        ```
    
    Example Not Found Response:
        ```json
        null
        ```
    """
    chapter_details = read_chapter_content(
        document_name, chapter_name
    )  # Leverages existing tool
    if not chapter_details:
        print(
            f"Could not retrieve chapter '{chapter_name}' in document '{document_name}' for statistics."
        )
        return None

    return StatisticsReport(
        scope=f"chapter: {document_name}/{chapter_name}",
        word_count=chapter_details.word_count,
        paragraph_count=chapter_details.paragraph_count,
    )


@mcp_server.tool()
@log_mcp_call
def get_document_statistics(document_name: str) -> Optional[StatisticsReport]:
    """
    Retrieve aggregated statistical information for an entire document.
    
    This tool analyzes all chapters within a document directory and returns
    comprehensive statistics including total word count, paragraph count, and
    chapter count. Useful for document-level analysis and progress tracking.
    
    Parameters:
        document_name (str): Name of the document directory to analyze
    
    Returns:
        Optional[StatisticsReport]: Statistics object if document found, None if not found.
        StatisticsReport contains:
            - scope (str): Description of what was analyzed (e.g., "document: user_guide")
            - word_count (int): Total number of words across all chapters
            - paragraph_count (int): Total number of paragraphs across all chapters
            - chapter_count (int): Total number of chapter files in the document
        
        Returns None if document directory doesn't exist.
        Returns zero counts if document exists but has no chapter files.
    
    Example Usage:
        ```json
        {
            "name": "get_document_statistics",
            "arguments": {
                "document_name": "user_guide"
            }
        }
        ```
    
    Example Success Response:
        ```json
        {
            "scope": "document: user_guide",
            "word_count": 2847,
            "paragraph_count": 156,
            "chapter_count": 8
        }
        ```
    
    Example Not Found Response:
        ```json
        null
        ```
    """
    # Option 1: Re-use list_documents and find the specific document.
    # This is good for consistency if list_documents is already computed/cached by agent.
    # However, it might be slightly less direct if we only need one document.
    all_docs_info = list_documents()  # This computes stats for all docs
    target_doc_info = next(
        (doc for doc in all_docs_info if doc.document_name == document_name), None
    )

    if not target_doc_info:
        # Check if the document directory exists even if it has no chapters or failed to be processed by list_documents
        doc_path = _get_document_path(document_name)
        if not doc_path.is_dir():
            print(f"Document '{document_name}' not found for statistics.")
            return None
        # If dir exists but not in target_doc_info (e.g. no chapters or processing error in list_documents)
        # We might want to recalculate directly
        ordered_chapter_files = _get_ordered_chapter_files(document_name)
        if not ordered_chapter_files:  # Empty document
            return StatisticsReport(
                scope=f"document: {document_name}",
                word_count=0,
                paragraph_count=0,
                chapter_count=0,
            )
        # If it has chapters but wasn't in list_documents output, it implies an issue. Recalculate:
        # This part is a bit redundant with read_full_document logic but more direct for stats
        total_word_count = 0
        total_paragraph_count = 0
        chapter_count = 0
        for chapter_file_path in ordered_chapter_files:
            details = _read_chapter_content_details(document_name, chapter_file_path)
            if details:
                total_word_count += details.word_count
                total_paragraph_count += details.paragraph_count
                chapter_count += 1
        return StatisticsReport(
            scope=f"document: {document_name}",
            word_count=total_word_count,
            paragraph_count=total_paragraph_count,
            chapter_count=chapter_count,
        )

    return StatisticsReport(
        scope=f"document: {document_name}",
        word_count=target_doc_info.total_word_count,
        paragraph_count=target_doc_info.total_paragraph_count,
        chapter_count=target_doc_info.total_chapters,
    )


@mcp_server.tool()
@log_mcp_call
def find_text_in_chapter(
    document_name: str, chapter_name: str, query: str, case_sensitive: bool = False
) -> List[ParagraphDetail]:
    """
    Search for paragraphs containing a specific text string within a single chapter.
    
    This tool performs exact text matching within paragraph content, returning all
    paragraphs that contain the search query. Supports both case-sensitive and
    case-insensitive searching. Useful for locating specific content for review or editing.
    
    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to search (must end with .md)
        query (str): Text string to search for. Must be:
            - Non-empty string
            - Not only whitespace
            - Exact text match (not regex or wildcard)
        case_sensitive (bool, optional): Whether to perform case-sensitive matching.
            - False (default): Case-insensitive search
            - True: Exact case matching required
    
    Returns:
        List[ParagraphDetail]: List of paragraph objects containing the search query.
        Each ParagraphDetail contains:
            - document_name (str): Name of the parent document
            - chapter_name (str): Name of the chapter file  
            - paragraph_index_in_chapter (int): Zero-indexed position within the chapter
            - content (str): Full text content of the matching paragraph
            - word_count (int): Number of words in the paragraph
        
        Returns empty list [] if no matches found or if chapter/document doesn't exist.
    
    Example Usage:
        ```json
        {
            "name": "find_text_in_chapter",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-introduction.md", 
                "query": "installation",
                "case_sensitive": false
            }
        }
        ```
    
    Example Success Response:
        ```json
        [
            {
                "document_name": "user_guide",
                "chapter_name": "01-introduction.md",
                "paragraph_index_in_chapter": 3,
                "content": "The installation process is straightforward and should take only a few minutes.",
                "word_count": 13
            },
            {
                "document_name": "user_guide", 
                "chapter_name": "01-introduction.md",
                "paragraph_index_in_chapter": 7,
                "content": "For advanced installation options, see the next chapter.",
                "word_count": 9
            }
        ]
        ```
    
    Example No Matches Response:
        ```json
        []
        ```
    """
    # Validate inputs
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        print(f"Invalid document name: {doc_error}")
        return []

    is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
    if not is_valid_chapter:
        print(f"Invalid chapter name: {chapter_error}")
        return []

    is_valid_query, query_error = _validate_search_query(query)
    if not is_valid_query:
        print(f"Invalid search query: {query_error}")
        return []

    results = []
    chapter_content_obj = read_chapter_content(document_name, chapter_name)
    if not chapter_content_obj:
        return results  # Empty list if chapter not found

    paragraphs_text = _split_into_paragraphs(chapter_content_obj.content)
    search_query = query if case_sensitive else query.lower()

    for i, para_text in enumerate(paragraphs_text):
        current_para_content = para_text if case_sensitive else para_text.lower()
        if search_query in current_para_content:
            results.append(
                ParagraphDetail(
                    document_name=document_name,
                    chapter_name=chapter_name,
                    paragraph_index_in_chapter=i,
                    content=para_text,  # Return original case paragraph
                    word_count=_count_words(para_text),
                )
            )

    return results


@mcp_server.tool()
@log_mcp_call
def find_text_in_document(
    document_name: str, query: str, case_sensitive: bool = False
) -> List[ParagraphDetail]:
    """
    Search for paragraphs containing a specific text string across all chapters in a document.
    
    This tool performs text matching across all chapter files within a document directory,
    returning all paragraphs that contain the search query. Supports both case-sensitive
    and case-insensitive searching. Useful for document-wide content discovery and analysis.
    
    Parameters:
        document_name (str): Name of the document directory to search in
        query (str): Text string to search for within paragraph content. Must be:
            - Non-empty string
            - Can include spaces and special characters
            - Matched as exact substring (not regex)
        case_sensitive (bool, optional): Whether to perform case-sensitive matching.
            - True: Exact case matching (e.g., "API" ≠ "api")
            - False (default): Case-insensitive matching (e.g., "API" = "api")
    
    Returns:
        List[ParagraphDetail]: List of matching paragraph objects. Each ParagraphDetail contains:
            - document_name (str): Name of the document containing the paragraph
            - chapter_name (str): Name of the chapter file containing the paragraph
            - paragraph_index_in_chapter (int): Zero-indexed position within the chapter
            - content (str): Full text content of the matching paragraph
            - word_count (int): Number of words in the paragraph
        
        Returns empty list if no matches found or if document doesn't exist.
        Results are ordered by chapter filename, then by paragraph position within chapter.
    
    Example Usage:
        ```json
        {
            "name": "find_text_in_document",
            "arguments": {
                "document_name": "user_guide",
                "query": "authentication",
                "case_sensitive": false
            }
        }
        ```
    
    Example Success Response:
        ```json
        [
            {
                "document_name": "user_guide",
                "chapter_name": "02-setup.md",
                "paragraph_index_in_chapter": 3,
                "content": "Configure authentication settings in your application to ensure secure access.",
                "word_count": 12
            },
            {
                "document_name": "user_guide", 
                "chapter_name": "05-security.md",
                "paragraph_index_in_chapter": 1,
                "content": "Authentication is a critical component of any secure application.",
                "word_count": 10
            }
        ]
        ```
    
    Example No Matches Response:
        ```json
        []
        ```
    """
    all_results = []
    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        print(f"Document '{document_name}' not found for text search.")
        return all_results

    ordered_chapter_files = _get_ordered_chapter_files(document_name)
    if not ordered_chapter_files:
        return all_results  # Empty list if no chapters

    for chapter_file_path in ordered_chapter_files:
        chapter_name = chapter_file_path.name
        # Delegate to find_text_in_chapter for each chapter
        chapter_results = find_text_in_chapter(
            document_name, chapter_name, query, case_sensitive
        )
        all_results.extend(chapter_results)

    return all_results


@mcp_server.tool()
@log_mcp_call
def append_to_chapter_content(
    document_name: str, chapter_name: str, content_to_append: str
) -> OperationStatus:
    """
    Append content to the end of a chapter without treating it as a separate paragraph.
    
    This atomic tool adds content directly to the end of the existing chapter content,
    maintaining the current paragraph structure. Unlike append_paragraph, this does not
    create a new paragraph but extends the existing content.
    
    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to append to (must end with .md)
        content_to_append (str): Content to append to the end of the chapter
    
    Returns:
        OperationStatus: Result object with success status, message, and diff details
    
    Example Usage:
        ```json
        {
            "name": "append_to_chapter_content",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-intro.md",
                "content_to_append": "\n\nAdditional note: This content is appended directly."
            }
        }
        ```
    """
    # Validate inputs
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        return OperationStatus(success=False, message=doc_error)

    is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
    if not is_valid_chapter:
        return OperationStatus(success=False, message=chapter_error)

    is_valid_content, content_error = _validate_content(content_to_append)
    if not is_valid_content:
        return OperationStatus(success=False, message=content_error)

    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
        return OperationStatus(
            success=False,
            message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
        )

    try:
        original_content = chapter_path.read_text(encoding="utf-8")
        new_content = original_content + content_to_append
        chapter_path.write_text(new_content, encoding="utf-8")

        # Generate diff for details
        diff_info = _generate_content_diff(original_content, new_content, chapter_name)
        
        return OperationStatus(
            success=True,
            message=f"Content appended to chapter '{chapter_name}' in document '{document_name}'.",
            details=diff_info,
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error appending content to '{chapter_name}': {str(e)}",
        )


@mcp_server.tool()
@log_mcp_call
def move_paragraph_before(
    document_name: str, chapter_name: str, paragraph_to_move_index: int, target_paragraph_index: int
) -> OperationStatus:
    """
    Move a paragraph to appear before another paragraph within the same chapter.
    
    This atomic tool reorders paragraphs within a chapter by moving the paragraph
    at the source index to appear before the paragraph at the target index.
    
    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to modify (must end with .md)
        paragraph_to_move_index (int): Zero-indexed position of the paragraph to move (≥0)
        target_paragraph_index (int): Zero-indexed position to move before (≥0)
    
    Returns:
        OperationStatus: Result object with success status, message, and diff details
    
    Example Usage:
        ```json
        {
            "name": "move_paragraph_before",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-intro.md",
                "paragraph_to_move_index": 3,
                "target_paragraph_index": 1
            }
        }
        ```
    """
    # Validate inputs
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        return OperationStatus(success=False, message=doc_error)

    is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
    if not is_valid_chapter:
        return OperationStatus(success=False, message=chapter_error)

    is_valid_move_index, move_index_error = _validate_paragraph_index(paragraph_to_move_index)
    if not is_valid_move_index:
        return OperationStatus(success=False, message=f"Move index: {move_index_error}")

    is_valid_target_index, target_index_error = _validate_paragraph_index(target_paragraph_index)
    if not is_valid_target_index:
        return OperationStatus(success=False, message=f"Target index: {target_index_error}")

    if paragraph_to_move_index == target_paragraph_index:
        return OperationStatus(
            success=False,
            message="Cannot move a paragraph before itself.",
        )

    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
        return OperationStatus(
            success=False,
            message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
        )

    try:
        original_full_content = chapter_path.read_text(encoding="utf-8")
        paragraphs = _split_into_paragraphs(original_full_content)
        total_paragraphs = len(paragraphs)

        if not (0 <= paragraph_to_move_index < total_paragraphs):
            return OperationStatus(
                success=False,
                message=f"Paragraph to move index {paragraph_to_move_index} is out of bounds (0-{total_paragraphs-1}).",
            )

        if not (0 <= target_paragraph_index < total_paragraphs):
            return OperationStatus(
                success=False,
                message=f"Target paragraph index {target_paragraph_index} is out of bounds (0-{total_paragraphs-1}).",
            )

        # Move the paragraph
        paragraph_to_move = paragraphs.pop(paragraph_to_move_index)
        
        # Adjust target index if necessary (if we moved from before the target)
        if paragraph_to_move_index < target_paragraph_index:
            target_paragraph_index -= 1
            
        paragraphs.insert(target_paragraph_index, paragraph_to_move)
        
        final_content = "\n\n".join(paragraphs)
        chapter_path.write_text(final_content, encoding="utf-8")

        # Generate diff for details
        diff_info = _generate_content_diff(original_full_content, final_content, chapter_name)
        
        return OperationStatus(
            success=True,
            message=f"Paragraph {paragraph_to_move_index} moved before paragraph {target_paragraph_index} in '{chapter_name}' ({document_name}).",
            details=diff_info,
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error moving paragraph in '{chapter_name}' ({document_name}): {str(e)}",
        )


@mcp_server.tool()
@log_mcp_call
def move_paragraph_to_end(
    document_name: str, chapter_name: str, paragraph_to_move_index: int
) -> OperationStatus:
    """
    Move a paragraph to the end of a chapter.
    
    This atomic tool moves the paragraph at the specified index to the end of the
    chapter, after all other paragraphs.
    
    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to modify (must end with .md)
        paragraph_to_move_index (int): Zero-indexed position of the paragraph to move (≥0)
    
    Returns:
        OperationStatus: Result object with success status, message, and diff details
    
    Example Usage:
        ```json
        {
            "name": "move_paragraph_to_end",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-intro.md",
                "paragraph_to_move_index": 1
            }
        }
        ```
    """
    # Validate inputs
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        return OperationStatus(success=False, message=doc_error)

    is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
    if not is_valid_chapter:
        return OperationStatus(success=False, message=chapter_error)

    is_valid_index, index_error = _validate_paragraph_index(paragraph_to_move_index)
    if not is_valid_index:
        return OperationStatus(success=False, message=index_error)

    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
        return OperationStatus(
            success=False,
            message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
        )

    try:
        original_full_content = chapter_path.read_text(encoding="utf-8")
        paragraphs = _split_into_paragraphs(original_full_content)
        total_paragraphs = len(paragraphs)

        if not (0 <= paragraph_to_move_index < total_paragraphs):
            return OperationStatus(
                success=False,
                message=f"Paragraph index {paragraph_to_move_index} is out of bounds (0-{total_paragraphs-1}).",
            )

        # If already at the end, no need to move
        if paragraph_to_move_index == total_paragraphs - 1:
            return OperationStatus(
                success=True,
                message=f"Paragraph {paragraph_to_move_index} is already at the end of '{chapter_name}' ({document_name}).",
                details={"changed": False, "summary": "No changes made - paragraph already at end"}
            )

        # Move the paragraph to the end
        if paragraph_to_move_index == len(paragraphs) - 1:
            return OperationStatus(
                success=True,
                message=f"Paragraph {paragraph_to_move_index} is already at the end of '{chapter_name}' ({document_name}).",
                details={"changed": False},
            )
    
        paragraph_to_move = paragraphs.pop(paragraph_to_move_index)
        paragraphs.append(paragraph_to_move)
        
        final_content = "\n\n".join(paragraphs)
        chapter_path.write_text(final_content, encoding="utf-8")

        # Generate diff for details
        diff_info = _generate_content_diff(original_full_content, final_content, chapter_name)
        
        return OperationStatus(
            success=True,
            message=f"Paragraph {paragraph_to_move_index} moved to end of '{chapter_name}' ({document_name}).",
            details=diff_info,
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error moving paragraph to end in '{chapter_name}' ({document_name}): {str(e)}",
        )


# ============================================================================
# MCP RESOURCE HANDLERS
# ============================================================================
# Handlers for exposing document and chapter data as standardized resources

def _create_resource_error(message: str, code: int = -1) -> McpError:
    """Helper function to create properly formatted McpError for resource operations."""
    return McpError(ErrorData(code=code, message=message))

async def _list_resources_handler():
    """
    List all available document chapters as MCP resources.
    
    Scans the DOCS_ROOT_PATH for all document directories and their .md chapter files,
    generating a TextResource for each chapter that can be accessed by LLM clients.
    
    Returns:
        List[Resource]: List of available chapter resources. Each Resource contains:
            - uri (str): Unique resource identifier (e.g., "file://document_name/chapter_name")
            - name (str): Human-readable name (e.g., "document_name/chapter_name")
            - mimeType (str): Set to "text/markdown" for all chapters
    """
    resources = []
    
    # Ensure DOCS_ROOT_PATH is a Path object (defensive programming)
    root_path = Path(DOCS_ROOT_PATH) if not isinstance(DOCS_ROOT_PATH, Path) else DOCS_ROOT_PATH
    
    if not root_path.exists() or not root_path.is_dir():
        return resources

    # Scan all document directories
    for doc_dir in root_path.iterdir():
        if doc_dir.is_dir():
            document_name = doc_dir.name
            chapter_files = _get_ordered_chapter_files(document_name)
            
            # Create a resource for each chapter file
            for chapter_file_path in chapter_files:
                chapter_name = chapter_file_path.name
                uri = f"file://{document_name}/{chapter_name}"
                name = f"{document_name}/{chapter_name}"
                
                resources.append(Resource(
                    uri=uri,
                    name=name,
                    mimeType="text/markdown"
                ))
    
    return resources


async def _read_resource_handler(uri: str):
    """
    Read the content of a specific document chapter resource.
    
    Parses the provided URI to extract document and chapter names, then securely
    reads the corresponding chapter file content. Validates that the requested
    file is within the DOCS_ROOT_PATH to prevent directory traversal attacks.
    
    Parameters:
        uri (str): Resource URI in format "file://document_name/chapter_name"
    
    Returns:
        TextResourceContents: Resource content object containing:
            - uri (str): Original resource URI
            - mime_type (str): Set to "text/markdown"
            - text (str): Full content of the chapter file
    
    Raises:
        McpError: If URI is invalid, file doesn't exist, or security violation
    """
    # Parse and validate the URI
    try:
        parsed_uri = urllib.parse.urlparse(uri)
        if parsed_uri.scheme != "file":
            raise _create_resource_error(f"Invalid URI scheme: {parsed_uri.scheme}. Expected 'file'")
        
        # Extract document_name and chapter_name from netloc and path
        if parsed_uri.netloc:
            # Format: file://document_name/chapter_name
            document_name = parsed_uri.netloc
            chapter_name = parsed_uri.path.strip('/')
            if not chapter_name:
                raise _create_resource_error(f"Invalid URI: missing chapter name. Expected 'file://document_name/chapter_name'")
        else:
            # Format: file:///document_name/chapter_name
            path_parts = parsed_uri.path.strip('/').split('/')
            if len(path_parts) != 2:
                raise _create_resource_error(f"Invalid URI path format: {parsed_uri.path}. Expected '/document_name/chapter_name'")
            document_name, chapter_name = path_parts
        
        # Validate the names using existing validation functions
        is_valid_doc, doc_error = _validate_document_name(document_name)
        if not is_valid_doc:
            raise _create_resource_error(f"Invalid document name: {doc_error}")
        
        is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
        if not is_valid_chapter:
            raise _create_resource_error(f"Invalid chapter name: {chapter_error}")
        
    except ValueError as e:
        raise _create_resource_error(f"Failed to parse URI: {e}")
    
    # Securely resolve the file path and verify it's within DOCS_ROOT_PATH
    try:
        chapter_path = _get_chapter_path(document_name, chapter_name)
        
        # Ensure the resolved path is within DOCS_ROOT_PATH (prevent directory traversal)
        root_path = Path(DOCS_ROOT_PATH) if not isinstance(DOCS_ROOT_PATH, Path) else DOCS_ROOT_PATH
        resolved_chapter_path = chapter_path.resolve()
        resolved_root_path = root_path.resolve()
        
        if not str(resolved_chapter_path).startswith(str(resolved_root_path)):
            raise _create_resource_error(f"Access denied: Path outside document root")
        
        # Verify the file exists and is a valid chapter file
        if not resolved_chapter_path.is_file():
            raise _create_resource_error(f"Chapter file not found: {document_name}/{chapter_name}")
        
        if not _is_valid_chapter_filename(chapter_name):
            raise _create_resource_error(f"Invalid chapter file: {chapter_name}")
        
        # Read and return the content
        content = resolved_chapter_path.read_text(encoding="utf-8")
        
        return [TextResourceContents(
            uri=uri,
            mimeType="text/markdown",
            text=content
        )]
        
    except FileNotFoundError:
        raise _create_resource_error(f"Chapter file not found: {document_name}/{chapter_name}")
    except PermissionError:
        raise _create_resource_error(f"Permission denied accessing: {document_name}/{chapter_name}")
    except UnicodeDecodeError as e:
        raise _create_resource_error(f"Failed to decode file content: {e}")
    except Exception as e:
        raise _create_resource_error(f"Error reading resource: {e}")

# Override the FastMCP resource methods
mcp_server.list_resources = _list_resources_handler
mcp_server.read_resource = _read_resource_handler

# Expose sync wrappers for testing
@log_mcp_call
def list_resources():
    """Sync wrapper for list_resources for testing purposes."""
    import asyncio
    return asyncio.run(_list_resources_handler())

@log_mcp_call
def read_resource(uri: str):
    """Sync wrapper for read_resource for testing purposes."""
    import asyncio
    return asyncio.run(_read_resource_handler(uri))


@mcp_server.custom_route("/health", methods=["GET"], name="health")
async def health_check(request: Request) -> Response:
    """Health check endpoint to verify server readiness."""
    return Response(status_code=200)


@mcp_server.custom_route("/metrics", methods=["GET"], name="metrics")
async def metrics_endpoint(request: Request) -> Response:
    """Prometheus metrics endpoint for monitoring MCP tool usage."""
    if not METRICS_AVAILABLE:
        return Response(
            content="# Metrics not available - OpenTelemetry not installed\n",
            status_code=503,
            media_type="text/plain"
        )
    
    try:
        metrics_data, content_type = get_metrics_export()
        return Response(
            content=metrics_data,
            status_code=200,
            media_type=content_type
        )
    except Exception as e:
        error_msg = f"# Error generating metrics: {e}\n"
        return Response(
            content=error_msg,
            status_code=500,
            media_type="text/plain"
        )


@mcp_server.custom_route("/metrics/summary", methods=["GET"], name="metrics_summary")
async def metrics_summary_endpoint(request: Request) -> Response:
    """JSON endpoint providing a summary of current metrics configuration and status."""
    try:
        from .metrics_config import get_metrics_summary
        summary = get_metrics_summary()
        
        import json
        return Response(
            content=json.dumps(summary, indent=2),
            status_code=200,
            media_type="application/json"
        )
    except Exception as e:
        error_response = {
            "error": f"Failed to get metrics summary: {e}",
            "status": "error"
        }
        import json
        return Response(
            content=json.dumps(error_response),
            status_code=500,
            media_type="application/json"
        )


# --- Main Server Execution ---
def main():
    """Run the main entry point for the server with argument parsing."""
    parser = argparse.ArgumentParser(description="Document MCP Server")
    parser.add_argument(
        "transport",
        choices=["sse", "stdio"],
        default="stdio",
        nargs="?",
        help="Transport type: 'sse' for HTTP Server-Sent Events or 'stdio' for standard I/O (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Host to bind to for SSE transport (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to bind to for SSE transport (default: {DEFAULT_PORT})",
    )

    args = parser.parse_args()

    # This print will show the path used by the subprocess
    print(
        f"doc_tool_server.py: Initializing with DOCS_ROOT_PATH = {DOCS_ROOT_PATH.resolve()}"
    )
    print(f"Document tool server starting. Tools exposed by '{mcp_server.name}':")
    print(f"Serving tools for root directory: {DOCS_ROOT_PATH.resolve()}")
    
    # Show metrics status
    if METRICS_AVAILABLE:
        try:
            from .metrics_config import METRICS_ENABLED
            status = "enabled" if METRICS_ENABLED else "disabled"
            print(f"Metrics: {status}")
        except ImportError:
            print("Metrics: available but not configured")
    else:
        print("Metrics: not available (install prometheus-client)")

    if args.transport == "stdio":
        print(
            "MCP server running with stdio transport. Waiting for client connection..."
        )
        mcp_server.run(transport="stdio")
    else:
        print(f"MCP server running with HTTP SSE transport on {args.host}:{args.port}")
        print(f"SSE endpoint: http://{args.host}:{args.port}/sse")
        print(f"Health endpoint: http://{args.host}:{args.port}/health")
        if METRICS_AVAILABLE:
            print(f"Metrics endpoint: http://{args.host}:{args.port}/metrics")
        # Update server settings before running
        mcp_server.settings.host = args.host
        mcp_server.settings.port = args.port
        mcp_server.run(transport="sse")


if __name__ == "__main__":
    main()
