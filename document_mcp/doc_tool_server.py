import argparse
import datetime
import os
import re  # Added for robust paragraph splitting
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp.server import FastMCP
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import Response

# import logging # No longer needed here
# import functools # No longer needed here
from .logger_config import (  # Import the configured logger and the decorator
    log_mcp_call,
)

# --- Configuration ---
# Each "document" will be a subdirectory within DOCS_ROOT_DIR.
# Chapters will be .md files within their respective document subdirectory.
load_dotenv()
DOCS_ROOT_DIR_NAME = os.environ.get("DOCUMENT_ROOT_DIR", ".documents_storage")
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

mcp_server = FastMCP("DocumentManagementTools")

# --- Input Validation Helpers ---


def _validate_document_name(document_name: str) -> tuple[bool, str]:
    """Validate document name input."""
    if not document_name:
        return False, "Document name cannot be empty"
    if not isinstance(document_name, str):
        return False, "Document name must be a string"
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
    if not chapter_name:
        return False, "Chapter name cannot be empty"
    if not isinstance(chapter_name, str):
        return False, "Chapter name must be a string"
    if len(chapter_name) > MAX_CHAPTER_NAME_LENGTH:
        return (
            False,
            f"Chapter name too long (max {MAX_CHAPTER_NAME_LENGTH} characters)",
        )
    if not chapter_name.lower().endswith(".md"):
        return False, "Chapter name must end with .md"
    if "/" in chapter_name or "\\" in chapter_name:
        return False, "Chapter name cannot contain path separators"
    if chapter_name == CHAPTER_MANIFEST_FILE:
        return False, f"Chapter name cannot be reserved name '{CHAPTER_MANIFEST_FILE}'"
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


class StatisticsReport(BaseModel):
    """Report for analytical queries."""

    scope: str  # e.g., "document: my_doc", "chapter: my_doc/ch1.md"
    word_count: int
    paragraph_count: int
    chapter_count: Optional[int] = None  # Only for document-level stats


# --- Helper Functions ---


def _get_document_path(document_name: str) -> Path:
    """Returns the Path object for a given document directory."""
    return DOCS_ROOT_PATH / document_name


def _get_chapter_path(document_name: str, chapter_filename: str) -> Path:
    """Returns the Path object for a given chapter file."""
    doc_path = _get_document_path(document_name)
    return doc_path / chapter_filename

def _is_valid_chapter_filename(filename: str) -> bool:
    """
    Checks if a filename is a valid Markdown file for a chapter.
    Excludes manifest and summary files.
    """
    if not filename.lower().endswith(".md"):
        return False
    if filename == CHAPTER_MANIFEST_FILE:
        return False
    if filename == DOCUMENT_SUMMARY_FILE: # Exclude document summary file
        return False
    return True


def _split_into_paragraphs(text: str) -> List[str]:
    """Splits text into paragraphs. Handles multiple blank lines.
    A paragraph is a block of text separated by one or more completely blank lines.
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
    """Counts words in a given text string."""
    return len(text.split())


def _get_ordered_chapter_files(document_name: str) -> List[Path]:
    """
    Retrieves chapter files for a document, ordered by filename.
    Excludes any manifest file.
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
    """Reads a chapter file and returns its content and metadata."""
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
        print(
            f"Error reading chapter file {chapter_file_path.name} in document {document_name}: {e}"
        )
        return None


def _get_chapter_metadata(
    document_name: str, chapter_file_path: Path
) -> Optional[ChapterMetadata]:
    """Helper to get metadata for a single chapter."""
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
        print(
            f"Error getting metadata for chapter {chapter_file_path.name} in {document_name}: {e}"
        )
        return None


# --- Example of a refactored tool (list_documents) ---
@mcp_server.tool()
@log_mcp_call
def list_documents() -> List[DocumentInfo]:
    """
    Lists all available documents.
    A document is a directory containing chapter files (.md).
    Returns a list of objects, each detailing a document's name, chapter count, word counts, and last modified time.
    """
    docs_info = []
    if not DOCS_ROOT_PATH.exists() or not DOCS_ROOT_PATH.is_dir():
        return []

    for doc_dir in DOCS_ROOT_PATH.iterdir():
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


# --- Implement Read Tools ---


@mcp_server.tool()
@log_mcp_call
def list_chapters(document_name: str) -> Optional[List[ChapterMetadata]]:
    """
    Lists all chapters for a given document, ordered by filename.
    Requires `document_name`.
    Returns a list of chapter metadata objects or None if the document doesn't exist.
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
    """
    Reads the content of a specific chapter within a document.
    Requires `document_name` and `chapter_name` (e.g., "01-intro.md").
    Returns an object containing the chapter's content and metadata, or None if not found.
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
def read_document_summary(document_name: str) -> Optional[str]:
    """
    Reads the content of the _SUMMARY.md file for a given document.
    Requires `document_name`.
    Returns the summary content as a string, or None if the summary file doesn't exist or cannot be read.
    """
    # Validate document name (optional, but good practice if it can be invalid)
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        print(f"Invalid document name provided: {doc_error}")
        # Depending on desired strictness, could return None or raise error
        return None # For now, let's be lenient if the path check below handles it

    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        print(f"Document '{document_name}' not found at {doc_path}")
        return None

    summary_file_path = doc_path / DOCUMENT_SUMMARY_FILE
    if not summary_file_path.is_file():
        print(f"Summary file '{DOCUMENT_SUMMARY_FILE}' not found in document '{document_name}'.")
        return None

    try:
        summary_content = summary_file_path.read_text(encoding="utf-8")
        return summary_content
    except Exception as e:
        print(f"Error reading summary file for document '{document_name}': {e}")
        return None


@mcp_server.tool()
@log_mcp_call
def read_paragraph_content(
    document_name: str, chapter_name: str, paragraph_index_in_chapter: int
) -> Optional[ParagraphDetail]:
    """
    Reads a specific paragraph from a chapter in a document.
    Requires `document_name`, `chapter_name`, and a 0-indexed `paragraph_index_in_chapter`.
    Returns an object with paragraph details or None if not found or index is out of bounds.
    """
    chapter_content_obj = read_chapter_content(document_name, chapter_name)
    if not chapter_content_obj:
        return None

    paragraphs = _split_into_paragraphs(chapter_content_obj.content)
    total_paragraphs = len(paragraphs)

    if not (0 <= paragraph_index_in_chapter < total_paragraphs):
        print(
            f"Error: Paragraph index {paragraph_index_in_chapter} is out of bounds (0-{total_paragraphs-1}) for chapter '{chapter_name}' in document '{document_name}'."
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
    """
    Reads the entire content of a document, including all its chapters in order.
    Requires `document_name`.
    Returns an object containing the document name and a list of all chapter contents, or None if the document is not found.
    """
    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        print(f"Document '{document_name}' not found at {doc_path}")
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
            print(
                f"Warning: Could not read chapter '{chapter_file_path.name}' in document '{document_name}'. Skipping."
            )

    return FullDocumentContent(
        document_name=document_name,
        chapters=all_chapter_contents,
        total_word_count=doc_total_word_count,
        total_paragraph_count=doc_total_paragraph_count,
    )


# --- Implement Write Tools ---


@mcp_server.tool()
@log_mcp_call
def create_document(document_name: str) -> OperationStatus:
    """
    Creates a new document (as a directory).
    Requires `document_name`.
    The document name should be suitable for a directory name.
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
    Deletes a document and all its chapters.
    Requires `document_name`.
    This operation is irreversible.
    """
    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        return OperationStatus(
            success=False, message=f"Document '{document_name}' not found."
        )
    try:
        # Delete all files within the directory first
        for item in doc_path.iterdir():
            if item.is_file():
                item.unlink()
            elif (
                item.is_dir()
            ):  # Should not happen based on current structure, but good practice
                # Recursively delete subdirectories if any (e.g. import shutil; shutil.rmtree(item))
                # For now, assume no subdirs other than files.
                print(
                    f"Warning: Subdirectory {item} found in document {document_name}. Manual cleanup might be needed if it wasn't deleted."
                )
        doc_path.rmdir()  # Remove the now-empty directory
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
    """
    Creates a new chapter file within a document.
    Requires `document_name` and `chapter_name` (e.g., "02-next-steps.md").
    `chapter_name` must end with .md and be a valid filename.
    Optionally accepts `initial_content` for the chapter.
    Chapter order is determined by alphanumeric sorting of chapter filenames.
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
    Deletes a chapter file from a document.
    Requires `document_name` and `chapter_name`.
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
    """
    Overwrites the entire content of a specific chapter.
    Requires `document_name`, `chapter_name`, and the `new_content` for the chapter.
    If the chapter doesn't exist, it will be created.
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
    # No need to read original content for OperationStatus in this case, as we overwrite.
    try:
        chapter_path.write_text(new_content, encoding="utf-8")
        updated_content_details = _read_chapter_content_details(
            document_name, chapter_path
        )
        return OperationStatus(
            success=True,
            message=f"Content of chapter '{chapter_name}' in document '{document_name}' updated successfully.",
            details=(
                updated_content_details.model_dump()
                if updated_content_details
                else None
            ),
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error writing to chapter '{chapter_name}' in document '{document_name}': {e}",
        )


@mcp_server.tool()
@log_mcp_call
def modify_paragraph_content(
    document_name: str,
    chapter_name: str,
    paragraph_index: int,
    new_paragraph_content: str,
    mode: str,
) -> OperationStatus:
    """
    Modifies a paragraph in a chapter. Modes: "replace", "insert_before", "insert_after", "delete".
    Requires `document_name`, `chapter_name`, `paragraph_index` (0-indexed), `new_paragraph_content` (ignored for "delete"), and `mode`.
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

    if mode != "delete":  # Content validation not needed for delete mode
        is_valid_content, content_error = _validate_content(new_paragraph_content)
        if not is_valid_content:
            return OperationStatus(success=False, message=content_error)

    allowed_modes = ["replace", "insert_before", "insert_after", "delete"]
    if mode not in allowed_modes:
        return OperationStatus(
            success=False,
            message=f"Invalid mode '{mode}'. Allowed modes: {allowed_modes}",
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

        if mode == "replace":
            if not (0 <= paragraph_index < total_paragraphs):
                return OperationStatus(
                    success=False,
                    message=f"Paragraph index {paragraph_index} for replacement is out of bounds (0-{total_paragraphs-1}).",
                )
            paragraphs[paragraph_index] = new_paragraph_content
        elif mode == "insert_before":
            if not (
                0 <= paragraph_index <= total_paragraphs
            ):  # Can insert at the very beginning (idx=0) or before any existing para up to total_paragraphs
                return OperationStatus(
                    success=False,
                    message=f"Paragraph index {paragraph_index} for insert_before is out of bounds (0-{total_paragraphs}).",
                )
            paragraphs.insert(paragraph_index, new_paragraph_content)
        elif mode == "insert_after":
            if not (0 <= paragraph_index < total_paragraphs) and not (
                total_paragraphs == 0 and paragraph_index == 0
            ):
                return OperationStatus(
                    success=False,
                    message=f"Paragraph index {paragraph_index} for insert_after is out of bounds (0-{total_paragraphs-1}).",
                )
            if total_paragraphs == 0 and paragraph_index == 0:  # Insert into empty doc
                paragraphs.append(new_paragraph_content)
            else:
                paragraphs.insert(paragraph_index + 1, new_paragraph_content)
        elif mode == "delete":
            if not (0 <= paragraph_index < total_paragraphs):
                return OperationStatus(
                    success=False,
                    message=f"Paragraph index {paragraph_index} for deletion is out of bounds (0-{total_paragraphs-1}).",
                )
            if not paragraphs:  # Should be caught by above, but defensive
                return OperationStatus(
                    success=False,
                    message="Cannot delete paragraph from an empty chapter.",
                )
            del paragraphs[paragraph_index]

        final_content = "\n\n".join(
            paragraphs
        )  # Ensure consistent paragraph separation
        chapter_path.write_text(final_content, encoding="utf-8")

        updated_content_details = _read_chapter_content_details(
            document_name, chapter_path
        )
        return OperationStatus(
            success=True,
            message=f"Paragraph {paragraph_index} in '{chapter_name}' ({document_name}) successfully modified with mode '{mode}'.",
            details=(
                updated_content_details.model_dump()
                if updated_content_details
                else None
            ),
        )
    except Exception as e:
        # Attempt to return original content if read, otherwise none
        error_details = None
        if "original_full_content" in locals():
            error_details = {
                "content_before_error": original_full_content
            }  # Could be large
        return OperationStatus(
            success=False,
            message=f"Error modifying paragraph in '{chapter_name}' ({document_name}): {str(e)}",
            details=error_details,
        )


@mcp_server.tool()
@log_mcp_call
def append_paragraph_to_chapter(
    document_name: str, chapter_name: str, paragraph_content: str
) -> OperationStatus:
    """
    Appends a new paragraph to the end of a specific chapter.
    Requires `document_name`, `chapter_name`, and `paragraph_content`.
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
        updated_content_details = _read_chapter_content_details(
            document_name, chapter_path
        )
        return OperationStatus(
            success=True,
            message=f"Paragraph appended to chapter '{chapter_name}' in document '{document_name}'.",
            details=(
                updated_content_details.model_dump()
                if updated_content_details
                else None
            ),
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error appending paragraph to '{chapter_name}': {str(e)}",
        )


@mcp_server.tool()
@log_mcp_call
def replace_text_in_chapter(
    document_name: str, chapter_name: str, text_to_find: str, replacement_text: str
) -> OperationStatus:
    """
    Replaces all occurrences of a string with another string in a specific chapter.
    Requires `document_name`, `chapter_name`, `text_to_find`, and `replacement_text`.
    This is case-sensitive.
    """
    # Validate inputs
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        return OperationStatus(success=False, message=doc_error)

    is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
    if not is_valid_chapter:
        return OperationStatus(success=False, message=chapter_error)

    if not text_to_find:
        return OperationStatus(success=False, message="Text to find cannot be empty")

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
        updated_content_details = _read_chapter_content_details(
            document_name, chapter_path
        )
        return OperationStatus(
            success=True,
            message=f"All {occurrences} occurrences of '{text_to_find}' replaced with '{replacement_text}' in chapter '{chapter_name}'.",
            details=(
                updated_content_details.model_dump()
                if updated_content_details
                else None
            ),  # Consider adding occurrences_found to details
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
    Replaces all occurrences of a string with another string throughout all chapters of a document.
    Requires `document_name`, `text_to_find`, and `replacement_text`.
    This is case-sensitive. Reports on chapters modified.
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


# --- Implement Analyze Tools ---


@mcp_server.tool()
@log_mcp_call
def get_chapter_statistics(
    document_name: str, chapter_name: str
) -> Optional[StatisticsReport]:
    """
    Retrieves statistics for a specific chapter (word count, paragraph count).
    Requires `document_name` and `chapter_name`.
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
    Retrieves aggregated statistics for an entire document (total word/paragraph/chapter count).
    Requires `document_name`.
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


# --- Implement Retrieval Tools (Exact Text Match) ---


@mcp_server.tool()
@log_mcp_call
def find_text_in_chapter(
    document_name: str, chapter_name: str, query: str, case_sensitive: bool = False
) -> List[ParagraphDetail]:
    """
    Finds paragraphs containing the exact query string within a specific chapter.
    Requires `document_name`, `chapter_name`, and `query` string.
    `case_sensitive` defaults to False (case-insensitive search).
    Returns a list of ParagraphDetail objects for matching paragraphs.
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
    Finds paragraphs containing the exact query string across all chapters of a document.
    Requires `document_name` and `query` string.
    `case_sensitive` defaults to False (case-insensitive search).
    Returns a list of ParagraphDetail objects for matching paragraphs from any chapter.
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


@mcp_server.custom_route("/health", methods=["GET"], name="health")
async def health_check(request: Request) -> Response:
    """Health check endpoint to verify server readiness."""
    return Response(status_code=200)


# --- Main Server Execution ---
def main():
    """Main entry point for the server with argument parsing."""
    parser = argparse.ArgumentParser(description="Document MCP Server")
    parser.add_argument(
        "transport",
        choices=["sse", "stdio"],
        default="sse",
        nargs="?",
        help="Transport type: 'sse' for HTTP Server-Sent Events or 'stdio' for standard I/O (default: sse)",
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
    print(
        f"doc_tool_server.py: Environment DOCUMENT_ROOT_DIR = {os.environ.get('DOCUMENT_ROOT_DIR')}"
    )
    print(f"Document tool server starting. Tools exposed by '{mcp_server.name}':")
    print(f"Serving tools for root directory: {DOCS_ROOT_PATH.resolve()}")

    if args.transport == "sse":
        print(f"MCP server running with HTTP SSE transport on {args.host}:{args.port}")
        print(f"SSE endpoint: http://{args.host}:{args.port}/sse")
        # Update server settings before running
        mcp_server.settings.host = args.host
        mcp_server.settings.port = args.port
        mcp_server.run(transport="sse")
    else:
        print(
            "MCP server running with stdio transport. Waiting for client connection..."
        )
        mcp_server.run(transport="stdio")


if __name__ == "__main__":
    main()
