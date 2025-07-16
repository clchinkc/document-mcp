"""Tool import compatibility layer for tests.

This module provides a centralized way to import tools during the server refactoring process.
Tests can use this module to get the tools they need without worrying about the underlying
implementation changes during the transition from monolithic to modular structure.
"""

# ============================================================================
# DOCUMENT TOOLS
# ============================================================================
# Imports from main server which acts as the central registry for all tools
# ============================================================================
# HELPER FUNCTIONS AND CONSTANTS
# ============================================================================
from document_mcp.doc_tool_server import CHAPTER_MANIFEST_FILE
from document_mcp.doc_tool_server import DOCS_ROOT_PATH
from document_mcp.doc_tool_server import DOCUMENT_SUMMARY_FILE
from document_mcp.doc_tool_server import _check_file_freshness
from document_mcp.doc_tool_server import _count_words
from document_mcp.doc_tool_server import _execute_batch_operation
from document_mcp.doc_tool_server import _get_modification_history_path
from document_mcp.doc_tool_server import _get_snapshots_path
from document_mcp.doc_tool_server import _is_valid_chapter_filename
from document_mcp.doc_tool_server import _resolve_operation_dependencies
from document_mcp.doc_tool_server import _split_into_paragraphs
from document_mcp.doc_tool_server import _validate_chapter_name
from document_mcp.doc_tool_server import _validate_content
from document_mcp.doc_tool_server import _validate_document_name
from document_mcp.doc_tool_server import _validate_paragraph_index
from document_mcp.doc_tool_server import _validate_search_query

# ============================================================================
# PARAGRAPH TOOLS
# ============================================================================
from document_mcp.doc_tool_server import append_paragraph_to_chapter

# ============================================================================
# BATCH TOOLS
# ============================================================================
from document_mcp.doc_tool_server import batch_apply_operations
from document_mcp.doc_tool_server import check_content_freshness

# ============================================================================
# SAFETY TOOLS (Version Control)
# ============================================================================
from document_mcp.doc_tool_server import check_content_status

# ============================================================================
# CHAPTER TOOLS
# ============================================================================
from document_mcp.doc_tool_server import create_chapter
from document_mcp.doc_tool_server import create_document
from document_mcp.doc_tool_server import delete_chapter
from document_mcp.doc_tool_server import delete_document
from document_mcp.doc_tool_server import delete_paragraph
from document_mcp.doc_tool_server import diff_content
from document_mcp.doc_tool_server import diff_snapshots

# ============================================================================
# CONTENT TOOLS (Unified/Scope-based)
# ============================================================================
from document_mcp.doc_tool_server import find_text
from document_mcp.doc_tool_server import get_modification_history
from document_mcp.doc_tool_server import get_statistics
from document_mcp.doc_tool_server import insert_paragraph_after
from document_mcp.doc_tool_server import insert_paragraph_before
from document_mcp.doc_tool_server import list_chapters
from document_mcp.doc_tool_server import list_documents
from document_mcp.doc_tool_server import list_snapshots
from document_mcp.doc_tool_server import manage_snapshots
from document_mcp.doc_tool_server import move_paragraph_before
from document_mcp.doc_tool_server import move_paragraph_to_end
from document_mcp.doc_tool_server import read_chapter_content
from document_mcp.doc_tool_server import read_content
from document_mcp.doc_tool_server import read_document_summary
from document_mcp.doc_tool_server import read_full_document
from document_mcp.doc_tool_server import read_paragraph_content
from document_mcp.doc_tool_server import replace_paragraph
from document_mcp.doc_tool_server import replace_text
from document_mcp.doc_tool_server import restore_snapshot
from document_mcp.doc_tool_server import snapshot_document
from document_mcp.doc_tool_server import write_chapter_content

# ============================================================================
# MODELS AND TYPES
# ============================================================================
from document_mcp.models import BatchApplyRequest
from document_mcp.models import BatchApplyResult
from document_mcp.models import BatchOperation
from document_mcp.models import ChapterContent
from document_mcp.models import ChapterMetadata

# ContentFreshnessStatus is now properly imported from models.py
from document_mcp.models import ContentFreshnessStatus
from document_mcp.models import DocumentInfo
from document_mcp.models import DocumentSummary
from document_mcp.models import FullDocumentContent
from document_mcp.models import ModificationHistory
from document_mcp.models import ModificationHistoryEntry
from document_mcp.models import OperationResult
from document_mcp.models import OperationStatus
from document_mcp.models import ParagraphDetail
from document_mcp.models import SnapshotInfo
from document_mcp.models import SnapshotsList
from document_mcp.models import StatisticsReport

__all__ = [
    # Document tools
    "create_document",
    "delete_document",
    "list_documents",
    "read_document_summary",
    # Chapter tools
    "create_chapter",
    "delete_chapter",
    "list_chapters",
    "write_chapter_content",
    "read_chapter_content",
    # Paragraph tools
    "append_paragraph_to_chapter",
    "delete_paragraph",
    "insert_paragraph_after",
    "insert_paragraph_before",
    "move_paragraph_before",
    "move_paragraph_to_end",
    "read_paragraph_content",
    "replace_paragraph",
    # Content tools
    "find_text",
    "get_statistics",
    "read_content",
    "read_full_document",
    "replace_text",
    # Safety tools
    "check_content_status",
    "diff_content",
    "manage_snapshots",
    "snapshot_document",
    "restore_snapshot",
    "list_snapshots",
    "diff_snapshots",
    "check_content_freshness",
    "get_modification_history",
    # Batch tools
    "batch_apply_operations",
    # Models and types
    "BatchApplyRequest",
    "BatchApplyResult",
    "BatchOperation",
    "ChapterContent",
    "ChapterMetadata",
    "ContentFreshnessStatus",
    "DocumentInfo",
    "DocumentSummary",
    "FullDocumentContent",
    "ModificationHistory",
    "ModificationHistoryEntry",
    "OperationResult",
    "OperationStatus",
    "ParagraphDetail",
    "SnapshotInfo",
    "SnapshotsList",
    "StatisticsReport",
    # Helper functions and constants
    "CHAPTER_MANIFEST_FILE",
    "DOCUMENT_SUMMARY_FILE",
    "DOCS_ROOT_PATH",
    "_count_words",
    "_is_valid_chapter_filename",
    "_split_into_paragraphs",
    "_validate_content",
    "_validate_document_name",
    "_validate_chapter_name",
    "_validate_paragraph_index",
    "_validate_search_query",
    "_get_snapshots_path",
    "_get_modification_history_path",
    "_check_file_freshness",
    "_execute_batch_operation",
    "_resolve_operation_dependencies",
]
