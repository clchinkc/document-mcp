"""Custom exception hierarchy for the Document MCP system.

This module provides a unified exception framework with consistent error handling
and user-friendly error messages across all components.
"""

from typing import Any


class DocumentMCPError(Exception):
    """Base exception for all Document MCP system errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        details: dict[str, Any] | None = None,
        user_message: str | None = None,
    ):
        """Initialize DocumentMCPError.

        Args:
            message: Technical error message for logging
            error_code: Standardized error code for programmatic handling
            details: Additional error context and debugging information
            user_message: User-friendly error message (defaults to message)
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.user_message = user_message or message

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for structured logging/responses."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "user_message": self.user_message,
            "details": self.details,
        }


class ValidationError(DocumentMCPError):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: str | None = None, value: Any = None, **kwargs):
        """Initialize ValidationError.

        Args:
            message: Error message
            field: Name of the field that failed validation
            value: Invalid value that caused the error
            **kwargs: Additional arguments passed to parent
        """
        details = kwargs.get("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["invalid_value"] = str(value)

        kwargs["details"] = details
        kwargs.setdefault("error_code", "VALIDATION_ERROR")
        super().__init__(message, **kwargs)


class DocumentNotFoundError(DocumentMCPError):
    """Raised when a requested document does not exist."""

    def __init__(self, document_name: str, **kwargs):
        message = f"Document '{document_name}' not found"
        user_message = f"The document '{document_name}' does not exist"

        details = kwargs.get("details", {})
        details["document_name"] = document_name

        kwargs.update(
            {
                "error_code": "DOCUMENT_NOT_FOUND",
                "details": details,
                "user_message": user_message,
            }
        )
        super().__init__(message, **kwargs)


class ChapterNotFoundError(DocumentMCPError):
    """Raised when a requested chapter does not exist."""

    def __init__(self, document_name: str, chapter_name: str, **kwargs):
        message = f"Chapter '{chapter_name}' not found in document '{document_name}'"
        user_message = f"The chapter '{chapter_name}' does not exist in document '{document_name}'"

        details = kwargs.get("details", {})
        details.update(
            {
                "document_name": document_name,
                "chapter_name": chapter_name,
            }
        )

        kwargs.update(
            {
                "error_code": "CHAPTER_NOT_FOUND",
                "details": details,
                "user_message": user_message,
            }
        )
        super().__init__(message, **kwargs)


class ParagraphNotFoundError(DocumentMCPError):
    """Raised when a requested paragraph does not exist."""

    def __init__(self, document_name: str, chapter_name: str, paragraph_index: int, **kwargs):
        message = (
            f"Paragraph {paragraph_index} not found in chapter '{chapter_name}' of document '{document_name}'"
        )
        user_message = f"Paragraph {paragraph_index} does not exist in the specified chapter"

        details = kwargs.get("details", {})
        details.update(
            {
                "document_name": document_name,
                "chapter_name": chapter_name,
                "paragraph_index": paragraph_index,
            }
        )

        kwargs.update(
            {
                "error_code": "PARAGRAPH_NOT_FOUND",
                "details": details,
                "user_message": user_message,
            }
        )
        super().__init__(message, **kwargs)


class ContentFreshnessError(DocumentMCPError):
    """Raised when content has been modified externally and is no longer fresh."""

    def __init__(self, file_path: str, operation: str = "modify", **kwargs):
        message = f"Content freshness check failed for {file_path} during {operation} operation"
        user_message = (
            "The file has been modified by another process. Please refresh and try again to avoid conflicts."
        )

        details = kwargs.get("details", {})
        details.update(
            {
                "file_path": file_path,
                "operation": operation,
            }
        )

        kwargs.update(
            {
                "error_code": "CONTENT_FRESHNESS_ERROR",
                "details": details,
                "user_message": user_message,
            }
        )
        super().__init__(message, **kwargs)


class OperationError(DocumentMCPError):
    """Raised when an MCP tool operation fails."""

    def __init__(self, operation: str, reason: str, **kwargs):
        message = f"Operation '{operation}' failed: {reason}"

        details = kwargs.get("details", {})
        details.update(
            {
                "operation": operation,
                "failure_reason": reason,
            }
        )

        kwargs.update(
            {
                "error_code": "OPERATION_ERROR",
                "details": details,
            }
        )
        super().__init__(message, **kwargs)

        # Add operation property for backward compatibility with tests
        self.operation = operation


class AgentError(DocumentMCPError):
    """Base class for agent-related errors."""

    def __init__(self, agent_type: str, message: str, **kwargs):
        details = kwargs.get("details", {})
        details["agent_type"] = agent_type

        kwargs.update(
            {
                "error_code": "AGENT_ERROR",
                "details": details,
            }
        )
        super().__init__(message, **kwargs)


class AgentConfigurationError(AgentError):
    """Raised when agent configuration is invalid."""

    def __init__(self, agent_type: str, config_issue: str, **kwargs):
        message = f"Agent configuration error in {agent_type}: {config_issue}"
        user_message = f"Configuration error: {config_issue}"

        kwargs.update(
            {
                "error_code": "AGENT_CONFIGURATION_ERROR",
                "user_message": user_message,
            }
        )
        super().__init__(agent_type, message, **kwargs)

        # Add agent_type property for backward compatibility with tests
        self.agent_type = agent_type


class LLMError(AgentError):
    """Raised when LLM operations fail."""

    def __init__(self, provider: str, model: str, reason: str, **kwargs):
        message = f"LLM operation failed with {provider} {model}: {reason}"
        user_message = f"AI model error: {reason}"

        details = kwargs.get("details", {})
        details.update(
            {
                "provider": provider,
                "model": model,
                "failure_reason": reason,
            }
        )

        kwargs.update(
            {
                "error_code": "LLM_ERROR",
                "details": details,
                "user_message": user_message,
            }
        )
        super().__init__("LLM", message, **kwargs)


class MCPToolError(DocumentMCPError):
    """Raised when MCP tool execution fails."""

    def __init__(self, tool_name: str, reason: str, **kwargs):
        message = f"MCP tool '{tool_name}' failed: {reason}"

        details = kwargs.get("details", {})
        details.update(
            {
                "tool_name": tool_name,
                "failure_reason": reason,
            }
        )

        kwargs.update(
            {
                "error_code": "MCP_TOOL_ERROR",
                "details": details,
            }
        )
        super().__init__(message, **kwargs)


class SemanticSearchError(DocumentMCPError):
    """Raised when semantic search operations fail."""

    def __init__(self, reason: str, **kwargs):
        message = f"Semantic search failed: {reason}"
        user_message = f"Search failed: {reason}"

        kwargs.update(
            {
                "error_code": "SEMANTIC_SEARCH_ERROR",
                "user_message": user_message,
            }
        )
        super().__init__(message, **kwargs)


class FileSystemError(DocumentMCPError):
    """Raised when file system operations fail."""

    def __init__(self, operation: str, path: str, reason: str, **kwargs):
        message = f"File system {operation} failed for {path}: {reason}"
        user_message = f"File operation failed: {reason}"

        details = kwargs.get("details", {})
        details.update(
            {
                "operation": operation,
                "file_path": path,
                "failure_reason": reason,
            }
        )

        kwargs.update(
            {
                "error_code": "FILE_SYSTEM_ERROR",
                "details": details,
                "user_message": user_message,
            }
        )
        super().__init__(message, **kwargs)
