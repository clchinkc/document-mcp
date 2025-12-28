"""Unit tests for the custom exception hierarchy."""

from __future__ import annotations

import pytest

from document_mcp.exceptions import AgentConfigurationError
from document_mcp.exceptions import AgentError
from document_mcp.exceptions import ChapterNotFoundError
from document_mcp.exceptions import ContentFreshnessError
from document_mcp.exceptions import DocumentMCPError
from document_mcp.exceptions import DocumentNotFoundError
from document_mcp.exceptions import FileSystemError
from document_mcp.exceptions import LLMError
from document_mcp.exceptions import MCPToolError
from document_mcp.exceptions import OperationError
from document_mcp.exceptions import ParagraphNotFoundError
from document_mcp.exceptions import SemanticSearchError
from document_mcp.exceptions import ValidationError


class TestDocumentMCPError:
    """Tests for the base DocumentMCPError class."""

    def test_basic_initialization(self):
        """Test basic exception creation."""
        error = DocumentMCPError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.error_code == "UNKNOWN_ERROR"
        assert error.details == {}
        assert error.user_message == "Something went wrong"

    def test_with_all_parameters(self):
        """Test exception with all parameters."""
        error = DocumentMCPError(
            message="Technical error",
            error_code="CUSTOM_ERROR",
            details={"key": "value"},
            user_message="User-friendly message",
        )

        assert error.message == "Technical error"
        assert error.error_code == "CUSTOM_ERROR"
        assert error.details == {"key": "value"}
        assert error.user_message == "User-friendly message"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        error = DocumentMCPError(
            message="Test error",
            error_code="TEST_ERROR",
            details={"info": "data"},
            user_message="Test message",
        )

        result = error.to_dict()

        assert result["error_type"] == "DocumentMCPError"
        assert result["error_code"] == "TEST_ERROR"
        assert result["message"] == "Test error"
        assert result["user_message"] == "Test message"
        assert result["details"] == {"info": "data"}

    def test_to_dict_preserves_subclass_name(self):
        """Test that to_dict preserves subclass name."""
        error = ValidationError("Invalid input")
        result = error.to_dict()

        assert result["error_type"] == "ValidationError"


class TestValidationError:
    """Tests for ValidationError."""

    def test_basic_validation_error(self):
        """Test basic validation error."""
        error = ValidationError("Invalid input")

        assert "Invalid input" in str(error)
        assert error.error_code == "VALIDATION_ERROR"

    def test_with_field_and_value(self):
        """Test validation error with field and value."""
        error = ValidationError("Invalid field", field="username", value="ab")

        assert error.details["field"] == "username"
        assert error.details["invalid_value"] == "ab"

    def test_with_only_field(self):
        """Test validation error with only field."""
        error = ValidationError("Required field", field="email")

        assert error.details["field"] == "email"
        assert "invalid_value" not in error.details

    def test_value_none_not_added_to_details(self):
        """Test that None value is not added to details."""
        error = ValidationError("Error", field="test", value=None)

        assert "invalid_value" not in error.details


class TestDocumentNotFoundError:
    """Tests for DocumentNotFoundError."""

    def test_initialization(self):
        """Test document not found error creation."""
        error = DocumentNotFoundError("my_document")

        assert "my_document" in str(error)
        assert error.error_code == "DOCUMENT_NOT_FOUND"
        assert error.details["document_name"] == "my_document"
        assert "does not exist" in error.user_message

    def test_with_additional_details(self):
        """Test with additional details passed."""
        error = DocumentNotFoundError("doc", details={"extra": "info"})

        assert error.details["document_name"] == "doc"
        assert error.details["extra"] == "info"


class TestChapterNotFoundError:
    """Tests for ChapterNotFoundError."""

    def test_initialization(self):
        """Test chapter not found error creation."""
        error = ChapterNotFoundError("my_doc", "01-intro.md")

        assert "01-intro.md" in str(error)
        assert "my_doc" in str(error)
        assert error.error_code == "CHAPTER_NOT_FOUND"
        assert error.details["document_name"] == "my_doc"
        assert error.details["chapter_name"] == "01-intro.md"

    def test_user_message_format(self):
        """Test user message format."""
        error = ChapterNotFoundError("doc", "chapter.md")

        assert "chapter.md" in error.user_message
        assert "doc" in error.user_message
        assert "does not exist" in error.user_message


class TestParagraphNotFoundError:
    """Tests for ParagraphNotFoundError."""

    def test_initialization(self):
        """Test paragraph not found error creation."""
        error = ParagraphNotFoundError("doc", "chapter.md", 5)

        assert "5" in str(error)
        assert error.error_code == "PARAGRAPH_NOT_FOUND"
        assert error.details["document_name"] == "doc"
        assert error.details["chapter_name"] == "chapter.md"
        assert error.details["paragraph_index"] == 5

    def test_user_message_format(self):
        """Test user message format."""
        error = ParagraphNotFoundError("doc", "chapter.md", 10)

        assert "10" in error.user_message
        assert "does not exist" in error.user_message


class TestContentFreshnessError:
    """Tests for ContentFreshnessError."""

    def test_initialization(self):
        """Test content freshness error creation."""
        error = ContentFreshnessError("/path/to/file.md")

        assert "/path/to/file.md" in str(error)
        assert error.error_code == "CONTENT_FRESHNESS_ERROR"
        assert error.details["file_path"] == "/path/to/file.md"
        assert error.details["operation"] == "modify"

    def test_with_custom_operation(self):
        """Test with custom operation."""
        error = ContentFreshnessError("/path/file", operation="delete")

        assert error.details["operation"] == "delete"
        assert "delete" in str(error)

    def test_user_message(self):
        """Test user-friendly message."""
        error = ContentFreshnessError("/path")

        assert "modified by another process" in error.user_message


class TestOperationError:
    """Tests for OperationError."""

    def test_initialization(self):
        """Test operation error creation."""
        error = OperationError("create_document", "Permission denied")

        assert "create_document" in str(error)
        assert "Permission denied" in str(error)
        assert error.error_code == "OPERATION_ERROR"
        assert error.details["operation"] == "create_document"
        assert error.details["failure_reason"] == "Permission denied"

    def test_operation_property(self):
        """Test backward compatibility operation property."""
        error = OperationError("test_op", "Failed")

        assert error.operation == "test_op"


class TestAgentError:
    """Tests for AgentError."""

    def test_initialization(self):
        """Test agent error creation."""
        error = AgentError("simple", "Agent failed")

        assert "Agent failed" in str(error)
        assert error.error_code == "AGENT_ERROR"
        assert error.details["agent_type"] == "simple"


class TestAgentConfigurationError:
    """Tests for AgentConfigurationError."""

    def test_initialization(self):
        """Test agent configuration error creation."""
        error = AgentConfigurationError("react", "Missing API key")

        assert "react" in str(error)
        assert "Missing API key" in str(error)
        # Note: Parent AgentError sets error_code to AGENT_ERROR
        assert error.error_code == "AGENT_ERROR"
        assert error.agent_type == "react"

    def test_user_message(self):
        """Test user-friendly message."""
        error = AgentConfigurationError("simple", "Invalid model")

        assert "Configuration error" in error.user_message
        assert "Invalid model" in error.user_message


class TestLLMError:
    """Tests for LLMError."""

    def test_initialization(self):
        """Test LLM error creation."""
        error = LLMError("openai", "gpt-4", "Rate limit exceeded")

        assert "openai" in str(error)
        assert "gpt-4" in str(error)
        assert "Rate limit exceeded" in str(error)
        # Note: Parent AgentError sets error_code to AGENT_ERROR
        assert error.error_code == "AGENT_ERROR"
        assert error.details["provider"] == "openai"
        assert error.details["model"] == "gpt-4"
        assert error.details["failure_reason"] == "Rate limit exceeded"

    def test_user_message(self):
        """Test user-friendly message."""
        error = LLMError("gemini", "gemini-pro", "Timeout")

        assert "AI model error" in error.user_message
        assert "Timeout" in error.user_message


class TestMCPToolError:
    """Tests for MCPToolError."""

    def test_initialization(self):
        """Test MCP tool error creation."""
        error = MCPToolError("list_documents", "Connection failed")

        assert "list_documents" in str(error)
        assert "Connection failed" in str(error)
        assert error.error_code == "MCP_TOOL_ERROR"
        assert error.details["tool_name"] == "list_documents"
        assert error.details["failure_reason"] == "Connection failed"


class TestSemanticSearchError:
    """Tests for SemanticSearchError."""

    def test_initialization(self):
        """Test semantic search error creation."""
        error = SemanticSearchError("Embedding API unavailable")

        assert "Embedding API unavailable" in str(error)
        assert error.error_code == "SEMANTIC_SEARCH_ERROR"

    def test_user_message(self):
        """Test user-friendly message."""
        error = SemanticSearchError("No embeddings found")

        assert "Search failed" in error.user_message
        assert "No embeddings found" in error.user_message


class TestFileSystemError:
    """Tests for FileSystemError."""

    def test_initialization(self):
        """Test file system error creation."""
        error = FileSystemError("write", "/path/to/file", "Disk full")

        assert "write" in str(error)
        assert "/path/to/file" in str(error)
        assert "Disk full" in str(error)
        assert error.error_code == "FILE_SYSTEM_ERROR"
        assert error.details["operation"] == "write"
        assert error.details["file_path"] == "/path/to/file"
        assert error.details["failure_reason"] == "Disk full"

    def test_user_message(self):
        """Test user-friendly message."""
        error = FileSystemError("read", "/path", "Permission denied")

        assert "File operation failed" in error.user_message
        assert "Permission denied" in error.user_message


class TestExceptionInheritance:
    """Tests for exception inheritance hierarchy."""

    def test_all_inherit_from_base(self):
        """Test that all custom exceptions inherit from DocumentMCPError."""
        exceptions = [
            ValidationError("test"),
            DocumentNotFoundError("doc"),
            ChapterNotFoundError("doc", "chapter"),
            ParagraphNotFoundError("doc", "chapter", 0),
            ContentFreshnessError("/path"),
            OperationError("op", "reason"),
            AgentError("type", "msg"),
            AgentConfigurationError("type", "issue"),
            LLMError("provider", "model", "reason"),
            MCPToolError("tool", "reason"),
            SemanticSearchError("reason"),
            FileSystemError("op", "/path", "reason"),
        ]

        for exc in exceptions:
            assert isinstance(exc, DocumentMCPError)
            assert isinstance(exc, Exception)

    def test_agent_errors_inherit_from_agent_error(self):
        """Test that agent-related errors inherit from AgentError."""
        config_error = AgentConfigurationError("type", "issue")
        llm_error = LLMError("provider", "model", "reason")

        assert isinstance(config_error, AgentError)
        assert isinstance(llm_error, AgentError)

    def test_can_be_caught_as_base_exception(self):
        """Test that all can be caught as DocumentMCPError."""
        with pytest.raises(DocumentMCPError):
            raise ValidationError("test")

        with pytest.raises(DocumentMCPError):
            raise DocumentNotFoundError("doc")

        with pytest.raises(DocumentMCPError):
            raise LLMError("p", "m", "r")
