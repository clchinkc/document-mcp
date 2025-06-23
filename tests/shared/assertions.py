"""
Assertion helpers for Document MCP testing.

This module provides consistent assertion patterns used across
all test types to ensure robust and readable test validation.
"""

from typing import Any, List, Optional

# Import models for type checking
from document_mcp.doc_tool_server import (
    ChapterMetadata,
    DocumentInfo,
    OperationStatus,
    StatisticsReport,
)


def assert_operation_success(
    status: OperationStatus, expected_message_part: Optional[str] = None
) -> None:
    """
    Assert that an operation was successful.

    Args:
        status: The OperationStatus object to check
        expected_message_part: Optional substring that should be in the success message

    Raises:
        AssertionError: If operation failed or message doesn't match
    """
    assert status is not None, "Operation status should not be None"
    assert hasattr(status, "success"), "Status missing 'success' field"
    assert hasattr(status, "message"), "Status missing 'message' field"

    assert (
        status.success is True
    ), f"Operation should succeed but got failure: {status.message}"

    assert (
        isinstance(status.message, str) and len(status.message) > 0
    ), "Success status should have a meaningful message"

    if expected_message_part:
        assert (
            expected_message_part.lower() in status.message.lower()
        ), f"Expected '{expected_message_part}' in success message: '{status.message}'"


def assert_operation_failure(
    status: OperationStatus, expected_message_part: Optional[str] = None
) -> None:
    """
    Assert that an operation failed.

    Args:
        status: The OperationStatus object to check
        expected_message_part: Optional substring that should be in the error message

    Raises:
        AssertionError: If operation succeeded or message doesn't match
    """
    assert status is not None, "Operation status should not be None"
    assert hasattr(status, "success"), "Status missing 'success' field"
    assert hasattr(status, "message"), "Status missing 'message' field"

    assert (
        status.success is False
    ), f"Operation should fail but got success: {status.message}"

    assert (
        isinstance(status.message, str) and len(status.message) > 0
    ), "Failure status should have a meaningful error message"

    if expected_message_part:
        assert (
            expected_message_part.lower() in status.message.lower()
        ), f"Expected '{expected_message_part}' to be in '{status.message}'"


def assert_agent_response_valid(
    response: Any, response_type: str = "agent", require_details: bool = False
) -> None:
    """
    Assert that an agent response is valid and properly structured.

    Args:
        response: The response object to validate
        response_type: Type of response for better error messages
        require_details: Whether details field is required

    Raises:
        AssertionError: If response is invalid
    """
    assert response is not None, f"{response_type} should provide a response"

    # Check required fields
    assert hasattr(
        response, "summary"
    ), f"{response_type} response missing 'summary' field"
    assert isinstance(
        response.summary, str
    ), f"{response_type} summary must be a string"
    assert len(response.summary) > 0, f"{response_type} summary should not be empty"

    # Check optional fields exist
    assert hasattr(
        response, "details"
    ), f"{response_type} response missing 'details' field"
    assert hasattr(
        response, "error_message"
    ), f"{response_type} response missing 'error_message' field"

    # Validate details if required
    if require_details:
        assert (
            response.details is not None
        ), f"{response_type} details should not be None when required"

    # If error_message is present, it should be a string
    if response.error_message is not None:
        assert isinstance(
            response.error_message, str
        ), f"{response_type} error_message must be a string"


def assert_document_info_valid(
    doc_info: DocumentInfo, expected_doc_name: Optional[str] = None
) -> None:
    """
    Validate that document info has the expected structure and values.

    Args:
        doc_info: DocumentInfo object to validate
        expected_doc_name: Expected document name

    Raises:
        AssertionError: If document info is invalid
    """
    assert doc_info is not None, "Document info should not be None"

    # Check required fields
    required_fields = [
        "document_name",
        "total_chapters",
        "total_word_count",
        "total_paragraph_count",
        "last_modified",
        "chapters",
    ]
    for field in required_fields:
        assert hasattr(doc_info, field), f"Document info missing {field}"

    # Validate field types and values
    if expected_doc_name:
        assert doc_info.document_name == expected_doc_name

    assert isinstance(doc_info.total_chapters, int), "Total chapters should be integer"
    assert doc_info.total_chapters >= 0, "Total chapters should be non-negative"

    assert isinstance(
        doc_info.total_word_count, int
    ), "Total word count should be integer"
    assert doc_info.total_word_count >= 0, "Total word count should be non-negative"

    assert isinstance(
        doc_info.total_paragraph_count, int
    ), "Total paragraph count should be integer"
    assert (
        doc_info.total_paragraph_count >= 0
    ), "Total paragraph count should be non-negative"

    assert isinstance(doc_info.chapters, list), "Chapters should be a list"

    # Validate chapter count consistency
    assert (
        len(doc_info.chapters) == doc_info.total_chapters
    ), f"Chapter list length ({len(doc_info.chapters)}) should match total_chapters ({doc_info.total_chapters})"


def assert_chapter_metadata_valid(
    metadata: ChapterMetadata, expected_chapter_name: Optional[str] = None
) -> None:
    """
    Validate that chapter metadata has the expected structure and values.

    Args:
        metadata: ChapterMetadata object to validate
        expected_chapter_name: Expected chapter name

    Raises:
        AssertionError: If chapter metadata is invalid
    """
    assert metadata is not None, "Chapter metadata should not be None"

    # Check required fields
    required_fields = ["chapter_name", "word_count", "paragraph_count", "last_modified"]
    for field in required_fields:
        assert hasattr(metadata, field), f"Metadata missing {field}"

    # Validate field values
    if expected_chapter_name:
        assert metadata.chapter_name == expected_chapter_name

    assert isinstance(metadata.word_count, int), "Word count should be integer"
    assert metadata.word_count >= 0, "Word count should be non-negative"

    assert isinstance(
        metadata.paragraph_count, int
    ), "Paragraph count should be integer"
    assert metadata.paragraph_count >= 0, "Paragraph count should be non-negative"


def assert_statistics_report_valid(
    stats: StatisticsReport, expected_scope: Optional[str] = None
) -> None:
    """
    Validate that statistics report has the expected structure and values.

    Args:
        stats: StatisticsReport object to validate
        expected_scope: Expected scope string

    Raises:
        AssertionError: If statistics report is invalid
    """
    assert stats is not None, "Statistics report should not be None"

    # Check required fields
    required_fields = ["scope", "word_count", "paragraph_count"]
    for field in required_fields:
        assert hasattr(stats, field), f"Statistics missing {field}"

    # Validate field values
    if expected_scope:
        assert expected_scope.lower() in stats.scope.lower()

    assert isinstance(stats.word_count, int), "Word count should be integer"
    assert stats.word_count >= 0, "Word count should be non-negative"

    assert isinstance(stats.paragraph_count, int), "Paragraph count should be integer"
    assert stats.paragraph_count >= 0, "Paragraph count should be non-negative"

    # Chapter count is optional for chapter-level stats
    if hasattr(stats, "chapter_count") and stats.chapter_count is not None:
        assert isinstance(stats.chapter_count, int), "Chapter count should be integer"
        assert stats.chapter_count >= 0, "Chapter count should be non-negative"


def assert_list_response_valid(
    response_list: List[Any],
    expected_length: Optional[int] = None,
    allow_empty: bool = True,
) -> None:
    """
    Validate that a list response is properly structured.

    Args:
        response_list: List to validate
        expected_length: Expected length (if known)
        allow_empty: Whether empty lists are allowed

    Raises:
        AssertionError: If list response is invalid
    """
    assert response_list is not None, "Response list should not be None"
    assert isinstance(response_list, list), "Response should be a list"

    if not allow_empty:
        assert len(response_list) > 0, "Response list should not be empty"

    if expected_length is not None:
        assert (
            len(response_list) == expected_length
        ), f"Expected list length {expected_length}, got {len(response_list)}"


def assert_no_error_in_response(response: Any) -> None:
    """
    Assert that a response contains no error information.

    Args:
        response: Response object to check

    Raises:
        AssertionError: If response contains error information
    """
    if hasattr(response, "error_message") and response.error_message is not None:
        assert False, f"Response should not contain error: {response.error_message}"

    if hasattr(response, "success") and response.success is False:
        message = getattr(response, "message", "Unknown error")
        assert False, f"Response should be successful: {message}"


def assert_contains_text(
    text: str, expected_substring: str, case_sensitive: bool = False
) -> None:
    """
    Assert that text contains expected substring.

    Args:
        text: Text to search in
        expected_substring: Substring to find
        case_sensitive: Whether search should be case sensitive

    Raises:
        AssertionError: If substring not found
    """
    if case_sensitive:
        assert (
            expected_substring in text
        ), f"Expected '{expected_substring}' in text: '{text}'"
    else:
        assert (
            expected_substring.lower() in text.lower()
        ), f"Expected '{expected_substring}' (case-insensitive) in text: '{text}'"
