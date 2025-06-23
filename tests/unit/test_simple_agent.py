"""
Unit tests for Simple Agent functions.

This module tests individual functions in the Simple agent in isolation with focus on:
- Configuration loading and validation
- Response parsing and model validation
- Error handling logic
- Function behavior without external dependencies
"""

import asyncio
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import ValidationError

# Import the data models to test DetailsType compatibility
from document_mcp.doc_tool_server import (
    ChapterContent,
    ChapterMetadata,
    DocumentInfo,
    OperationStatus,
    ParagraphDetail,
    StatisticsReport,
)
from src.agents.simple_agent import (
    SYSTEM_PROMPT,
    FinalAgentResponse,
    check_api_keys_config,
    initialize_agent_and_mcp_server,
    load_llm_config,
    process_single_user_query,
)

# Import shared test utilities
from tests.shared import assert_agent_response_valid as assert_response_valid
from tests.shared import (
    create_mock_agent,
    create_mock_mcp_server,
)


class TestConfigurationFunctions:
    """Test suite for configuration-related functions."""

    def test_load_llm_config_with_openai_key(self):
        """Test loading LLM config with OpenAI API key."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test_openai_key", "OPENAI_MODEL_NAME": "gpt-4o"},
            clear=True,
        ):
            with patch("src.agents.simple_agent.load_dotenv"):
                model = load_llm_config()
                assert model is not None
                # The actual model type checking would require importing OpenAIModel
                # but we're testing the function logic, not the model instantiation

    def test_load_llm_config_with_gemini_key(self):
        """Test loading LLM config with Gemini API key."""
        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "test_gemini_key",
                "GEMINI_MODEL_NAME": "gemini-2.5-flash",
            },
            clear=True,
        ):
            with patch("src.agents.simple_agent.load_dotenv"):
                model = load_llm_config()
                assert model is not None

    def test_load_llm_config_openai_priority_over_gemini(self):
        """Test that OpenAI has priority over Gemini when both keys are present."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test_openai_key", "GEMINI_API_KEY": "test_gemini_key"},
            clear=True,
        ):
            with patch("src.agents.simple_agent.load_dotenv"):
                with patch("src.agents.simple_agent.OpenAIModel") as mock_openai:
                    with patch("src.agents.simple_agent.GeminiModel") as mock_gemini:
                        load_llm_config()
                        mock_openai.assert_called_once()
                        mock_gemini.assert_not_called()

    def test_load_llm_config_no_api_keys(self):
        """Test loading LLM config with no API keys raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.agents.simple_agent.load_dotenv"):
                with pytest.raises(ValueError) as exc_info:
                    load_llm_config()
                assert "No valid API key found" in str(exc_info.value)

    def test_load_llm_config_empty_api_keys(self):
        """Test loading LLM config with empty API keys raises ValueError."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "", "GEMINI_API_KEY": "   "},  # Whitespace only
            clear=True,
        ):
            with patch("src.agents.simple_agent.load_dotenv"):
                with pytest.raises(ValueError) as exc_info:
                    load_llm_config()
                assert "No valid API key found" in str(exc_info.value)

    def test_check_api_keys_config_no_keys(self):
        """Test checking API keys config with no keys configured."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.agents.simple_agent.load_dotenv"):
                config = check_api_keys_config()
                assert config["openai_configured"] is False
                assert config["gemini_configured"] is False
                assert config["active_provider"] is None
                assert config["active_model"] is None

    def test_check_api_keys_config_openai_only(self):
        """Test checking API keys config with only OpenAI configured."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test_openai_key", "OPENAI_MODEL_NAME": "gpt-4o"},
            clear=True,
        ):
            with patch("src.agents.simple_agent.load_dotenv"):
                config = check_api_keys_config()
                assert config["openai_configured"] is True
                assert config["gemini_configured"] is False
                assert config["active_provider"] == "openai"
                assert config["active_model"] == "gpt-4o"
                assert config["openai_model"] == "gpt-4o"

    def test_check_api_keys_config_gemini_only(self):
        """Test checking API keys config with only Gemini configured."""
        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "test_gemini_key",
                "GEMINI_MODEL_NAME": "gemini-2.5-flash",
            },
            clear=True,
        ):
            with patch("src.agents.simple_agent.load_dotenv"):
                config = check_api_keys_config()
                assert config["openai_configured"] is False
                assert config["gemini_configured"] is True
                assert config["active_provider"] == "gemini"
                assert config["active_model"] == "gemini-2.5-flash"
                assert config["gemini_model"] == "gemini-2.5-flash"

    def test_check_api_keys_config_both_keys_openai_priority(self):
        """Test checking API keys config with both keys configured - OpenAI has priority."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test_openai_key",
                "GEMINI_API_KEY": "test_gemini_key",
                "OPENAI_MODEL_NAME": "gpt-4o",
                "GEMINI_MODEL_NAME": "gemini-2.5-flash",
            },
            clear=True,
        ):
            with patch("src.agents.simple_agent.load_dotenv"):
                config = check_api_keys_config()
                assert config["openai_configured"] is True
                assert config["gemini_configured"] is True
                assert config["active_provider"] == "openai"  # OpenAI has priority
                assert config["active_model"] == "gpt-4o"
                assert config["openai_model"] == "gpt-4o"
                assert config["gemini_model"] == "gemini-2.5-flash"

    def test_check_api_keys_config_default_model_names(self):
        """Test checking API keys config uses default model names when not specified."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test_openai_key"
                # No OPENAI_MODEL_NAME specified
            },
            clear=True,
        ):
            with patch("src.agents.simple_agent.load_dotenv"):
                config = check_api_keys_config()
                assert config["openai_configured"] is True
                assert config["active_model"] == "gpt-4.1-mini"  # Default
                assert config["openai_model"] == "gpt-4.1-mini"

    def test_load_llm_config_with_model_instantiation(self):
        """Test that load_llm_config actually instantiates model objects."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test_openai_key", "OPENAI_MODEL_NAME": "gpt-4o"},
            clear=True,
        ):
            with patch("src.agents.simple_agent.load_dotenv"):
                with patch("src.agents.simple_agent.OpenAIModel") as mock_openai_model:
                    mock_model_instance = Mock()
                    mock_openai_model.return_value = mock_model_instance

                    result = load_llm_config()

                    mock_openai_model.assert_called_once_with(model_name="gpt-4o")
                    assert result == mock_model_instance

    def test_load_llm_config_prints_model_selection(self):
        """Test that load_llm_config prints the selected model."""
        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "test_gemini_key",
                "GEMINI_MODEL_NAME": "gemini-2.5-flash",
            },
            clear=True,
        ):
            with patch("src.agents.simple_agent.load_dotenv"):
                with patch("src.agents.simple_agent.GeminiModel"):
                    with patch("builtins.print") as mock_print:
                        load_llm_config()
                        mock_print.assert_called_with(
                            "Using Gemini model: gemini-2.5-flash"
                        )

    def test_load_llm_config_with_whitespace_only_key(self):
        """Test load_llm_config handles whitespace-only API keys correctly."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "   \t\n   ",  # Only whitespace
                "GEMINI_API_KEY": "valid_key",
            },
            clear=True,
        ):
            with patch("src.agents.simple_agent.load_dotenv"):
                with patch("src.agents.simple_agent.GeminiModel") as mock_gemini:
                    load_llm_config()
                    mock_gemini.assert_called_once()  # Should fall back to Gemini

    def test_check_api_keys_config_with_whitespace_keys(self):
        """Test check_api_keys_config properly handles whitespace in API keys."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "  \t  ",  # Whitespace only
                "GEMINI_API_KEY": "valid_key",
            },
            clear=True,
        ):
            with patch("src.agents.simple_agent.load_dotenv"):
                config = check_api_keys_config()
                assert (
                    config["openai_configured"] is False
                )  # Whitespace should not count
                assert config["gemini_configured"] is True
                assert config["active_provider"] == "gemini"


class TestResponseModel:
    """Test suite for FinalAgentResponse model validation."""

    def test_final_agent_response_valid_minimal(self):
        """Test FinalAgentResponse with minimal valid data."""
        response = FinalAgentResponse(summary="Test summary")
        assert response.summary == "Test summary"
        assert response.details is None
        assert response.error_message is None

    def test_final_agent_response_valid_complete(self):
        """Test FinalAgentResponse with all fields populated."""
        test_details = {"key": "value"}
        response = FinalAgentResponse(
            summary="Test summary", details=test_details, error_message="Test error"
        )
        assert response.summary == "Test summary"
        assert response.details == test_details
        assert response.error_message == "Test error"

    def test_final_agent_response_empty_summary_allowed(self):
        """Test FinalAgentResponse allows empty summary (no validation constraint)."""
        response = FinalAgentResponse(summary="")
        assert response.summary == ""
        assert response.details is None
        assert response.error_message is None

    def test_final_agent_response_none_summary_invalid(self):
        """Test FinalAgentResponse with None summary."""
        with pytest.raises(ValidationError):
            FinalAgentResponse(summary=None)

    def test_final_agent_response_various_details_types(self):
        """Test FinalAgentResponse accepts various details types."""
        # Test with list
        response = FinalAgentResponse(summary="Test", details=[])
        assert response.details == []

        # Test with dict
        response = FinalAgentResponse(summary="Test", details={"test": "value"})
        assert response.details == {"test": "value"}

        # Test with None
        response = FinalAgentResponse(summary="Test", details=None)
        assert response.details is None

    def test_final_agent_response_with_document_info_list(self):
        """Test FinalAgentResponse with List[DocumentInfo] details."""
        doc_info = DocumentInfo(
            document_name="test_doc",
            total_chapters=2,
            total_word_count=100,
            total_paragraph_count=5,
            last_modified="2023-01-01T00:00:00Z",
            chapters=[],
        )
        response = FinalAgentResponse(summary="Document list", details=[doc_info])
        assert response.details == [doc_info]
        assert len(response.details) == 1
        assert response.details[0].document_name == "test_doc"

    def test_final_agent_response_with_operation_status(self):
        """Test FinalAgentResponse with OperationStatus details."""
        status = OperationStatus(success=True, message="Operation completed")
        response = FinalAgentResponse(summary="Operation result", details=status)
        assert response.details == status
        assert response.details.success is True
        assert response.details.message == "Operation completed"

    def test_final_agent_response_with_statistics_report(self):
        """Test FinalAgentResponse with StatisticsReport details."""
        stats = StatisticsReport(
            scope="document: test_doc",
            word_count=500,
            paragraph_count=20,
            chapter_count=3,
        )
        response = FinalAgentResponse(summary="Statistics", details=stats)
        assert response.details == stats
        assert response.details.word_count == 500

    def test_final_agent_response_with_chapter_content(self):
        """Test FinalAgentResponse with ChapterContent details."""
        from datetime import datetime, timezone

        chapter = ChapterContent(
            document_name="test_doc",
            chapter_name="01-intro.md",
            content="# Introduction\n\nThis is the intro.",
            word_count=20,
            paragraph_count=2,
            last_modified=datetime.now(timezone.utc),
        )
        response = FinalAgentResponse(summary="Chapter content", details=chapter)
        assert response.details == chapter
        assert response.details.chapter_name == "01-intro.md"

    def test_final_agent_response_with_paragraph_detail_list(self):
        """Test FinalAgentResponse with List[ParagraphDetail] details."""
        paragraph = ParagraphDetail(
            document_name="test_doc",
            chapter_name="01-intro.md",
            paragraph_index_in_chapter=0,
            content="This is a test paragraph.",
            word_count=5,
        )
        response = FinalAgentResponse(summary="Search results", details=[paragraph])
        assert response.details == [paragraph]
        assert len(response.details) == 1
        assert response.details[0].content == "This is a test paragraph."

    def test_final_agent_response_model_dump(self):
        """Test FinalAgentResponse model serialization."""
        response = FinalAgentResponse(
            summary="Test summary", details={"key": "value"}, error_message="Test error"
        )
        dumped = response.model_dump()
        assert dumped["summary"] == "Test summary"
        assert dumped["details"] == {"key": "value"}
        assert dumped["error_message"] == "Test error"

    def test_final_agent_response_model_dump_exclude_none(self):
        """Test FinalAgentResponse model serialization excluding None values."""
        response = FinalAgentResponse(summary="Test summary")
        dumped = response.model_dump(exclude_none=True)
        assert dumped["summary"] == "Test summary"
        assert "details" not in dumped
        assert "error_message" not in dumped


class TestErrorHandling:
    """Test suite for error handling in agent functions."""

    @pytest.mark.asyncio
    async def test_process_single_user_query_timeout_error(self):
        """Test process_single_user_query handles timeout errors."""
        mock_agent = AsyncMock()
        mock_agent.run.side_effect = asyncio.TimeoutError()

        result = await process_single_user_query(mock_agent, "test query")

        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        assert "timed out" in result.summary.lower()
        assert result.error_message == "Timeout error"

    @pytest.mark.asyncio
    async def test_process_single_user_query_cancelled_error(self):
        """Test process_single_user_query handles cancelled errors."""
        mock_agent = AsyncMock()
        mock_agent.run.side_effect = asyncio.CancelledError()

        result = await process_single_user_query(mock_agent, "test query")

        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        assert "cancelled" in result.summary.lower()
        assert result.error_message == "Cancelled error"

    @pytest.mark.asyncio
    async def test_process_single_user_query_runtime_error_event_loop_closed(self):
        """Test process_single_user_query handles event loop closed errors."""
        mock_agent = AsyncMock()
        mock_agent.run.side_effect = RuntimeError("Event loop is closed")

        result = await process_single_user_query(mock_agent, "test query")

        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        assert ("event loop closed" in result.summary.lower() or "event loop was closed" in result.summary.lower())
        assert (result.error_message == "Event loop closed" or result.error_message is None)

    @pytest.mark.asyncio
    async def test_process_single_user_query_runtime_error_generator(self):
        """Test process_single_user_query handles generator cleanup errors."""
        mock_agent = AsyncMock()
        mock_agent.run.side_effect = RuntimeError("generator didn't stop after athrow")

        result = await process_single_user_query(mock_agent, "test query")

        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        assert "generator cleanup error" in result.summary.lower()
        assert result.error_message == "Generator cleanup error"

    @pytest.mark.asyncio
    async def test_process_single_user_query_generic_runtime_error(self):
        """Test process_single_user_query handles generic runtime errors."""
        mock_agent = AsyncMock()
        mock_agent.run.side_effect = RuntimeError("Some other runtime error")

        result = await process_single_user_query(mock_agent, "test query")

        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        assert "runtime error" in result.summary.lower()
        assert "Some other runtime error" in result.error_message

    @pytest.mark.asyncio
    async def test_process_single_user_query_read_timeout_error(self):
        """Test process_single_user_query handles ReadTimeout errors gracefully."""
        mock_agent = AsyncMock()
        mock_agent.run.side_effect = Exception("ReadTimeout occurred")

        result = await process_single_user_query(mock_agent, "test query")

        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        assert "api connection timeout" in result.summary.lower()
        assert "ReadTimeout occurred" in result.error_message

    @pytest.mark.asyncio
    async def test_process_single_user_query_api_key_error(self):
        """Test process_single_user_query handles API key placeholder errors."""
        mock_agent = AsyncMock()
        mock_agent.run.side_effect = Exception("API error")

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_placeholder"}):
            result = await process_single_user_query(mock_agent, "test query")

        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        assert "api authentication error" in result.summary.lower()
        assert "API error" in result.error_message

    @pytest.mark.asyncio
    async def test_process_single_user_query_generic_exception(self):
        """Test process_single_user_query handles generic exceptions."""
        mock_agent = AsyncMock()
        mock_agent.run.side_effect = Exception("Generic error")

        result = await process_single_user_query(mock_agent, "test query")

        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        assert "exception during query processing" in result.summary.lower()
        assert "Generic error" in result.error_message

    @pytest.mark.asyncio
    async def test_process_single_user_query_successful_run_result(self):
        """Test process_single_user_query with successful run result."""
        mock_agent = AsyncMock()
        mock_run_result = Mock()
        mock_run_result.output = FinalAgentResponse(summary="Success")
        mock_run_result.error_message = None
        mock_agent.run.return_value = mock_run_result

        result = await process_single_user_query(mock_agent, "test query")

        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        assert result.summary == "Success"

    @pytest.mark.asyncio
    async def test_process_single_user_query_run_result_with_error(self):
        """Test process_single_user_query with run result containing error."""
        mock_agent = AsyncMock()
        mock_run_result = Mock()
        mock_run_result.output = None
        mock_run_result.error_message = "Agent internal error"
        mock_agent.run.return_value = mock_run_result

        result = await process_single_user_query(mock_agent, "test query")

        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        assert "agent error" in result.summary.lower()
        assert result.error_message == "Agent internal error"

    @pytest.mark.asyncio
    async def test_process_single_user_query_empty_run_result(self):
        """Test process_single_user_query with empty run result."""
        mock_agent = AsyncMock()
        mock_run_result = Mock()
        mock_run_result.output = None
        mock_run_result.error_message = None
        mock_agent.run.return_value = mock_run_result

        result = await process_single_user_query(mock_agent, "test query")

        assert result is None

    @pytest.mark.asyncio
    async def test_process_single_user_query_with_wait_for_timeout(self):
        """Test that process_single_user_query uses asyncio.wait_for with correct timeout."""
        mock_agent = AsyncMock()
        mock_agent.run.return_value = Mock(output=FinalAgentResponse(summary="Success"))

        with patch("src.agents.simple_agent.asyncio.wait_for") as mock_wait_for:
            mock_wait_for.return_value = Mock(
                output=FinalAgentResponse(summary="Success")
            )

            await process_single_user_query(mock_agent, "test query")

            mock_wait_for.assert_called_once()
            args, kwargs = mock_wait_for.call_args
            assert kwargs["timeout"] == 45.0

    @pytest.mark.asyncio
    async def test_process_single_user_query_prints_errors_to_stderr(self):
        """Test that process_single_user_query prints errors to stderr."""
        mock_agent = AsyncMock()
        mock_agent.run.side_effect = RuntimeError("Test error")

        with patch("builtins.print") as mock_print:
            await process_single_user_query(mock_agent, "test query")

            # Check that print was called with stderr argument
            stderr_calls = [
                call
                for call in mock_print.call_args_list
                if len(call[1]) > 0 and "file" in call[1]
            ]
            assert len(stderr_calls) > 0

    @pytest.mark.asyncio
    async def test_process_single_user_query_handles_none_agent(self):
        """Test process_single_user_query handles None agent gracefully."""
        # The function catches all exceptions and returns a FinalAgentResponse
        result = await process_single_user_query(None, "test query")

        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        assert "exception during query processing" in result.summary.lower()
        assert "'NoneType' object has no attribute 'run'" in result.error_message

    @pytest.mark.asyncio
    async def test_process_single_user_query_handles_empty_query(self):
        """Test process_single_user_query handles empty query string."""
        mock_agent = AsyncMock()
        mock_agent.run.return_value = Mock(
            output=FinalAgentResponse(summary="Empty query handled")
        )

        result = await process_single_user_query(mock_agent, "")

        mock_agent.run.assert_called_once_with("")
        assert result.summary == "Empty query handled"

    @pytest.mark.asyncio
    async def test_process_single_user_query_with_complex_error_message(self):
        """Test process_single_user_query handles complex error messages correctly."""
        mock_agent = AsyncMock()
        complex_error = Exception(
            "ReadTimeout: Connection timeout after 30s\nEvent loop is closed"
        )
        mock_agent.run.side_effect = complex_error

        result = await process_single_user_query(mock_agent, "test query")

        assert result is not None
        assert "api connection timeout" in result.summary.lower()
        assert "ReadTimeout" in result.error_message


class TestAgentInitialization:
    """Test suite for agent initialization functions."""

    @pytest.mark.asyncio
    async def test_initialize_agent_and_mcp_server_success(self):
        """Test successful agent and MCP server initialization."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test_key",
                "MCP_SERVER_HOST": "localhost",
                "MCP_SERVER_PORT": "3001",
            },
        ):
            with patch("src.agents.simple_agent.load_llm_config") as mock_load_llm:
                with patch("src.agents.simple_agent.MCPServerSSE") as mock_mcp_server:
                    with patch("src.agents.simple_agent.Agent") as mock_agent:
                        mock_llm = Mock()
                        mock_load_llm.return_value = mock_llm

                        agent, mcp_server = await initialize_agent_and_mcp_server()

                        # Verify LLM config was loaded
                        mock_load_llm.assert_called_once()

                        # Verify MCP server was created with correct URL
                        mock_mcp_server.assert_called_once_with(
                            "http://localhost:3001/sse"
                        )

                        # Verify agent was created with correct parameters
                        mock_agent.assert_called_once()
                        call_args = mock_agent.call_args
                        assert (
                            call_args[0][0] == mock_llm
                        )  # First positional arg is LLM
                        assert "mcp_servers" in call_args[1]
                        assert "system_prompt" in call_args[1]
                        assert "output_type" in call_args[1]
                        assert call_args[1]["output_type"] == FinalAgentResponse

    @pytest.mark.asyncio
    async def test_initialize_agent_and_mcp_server_llm_config_error(self):
        """Test agent initialization handles LLM config errors."""
        with patch("src.agents.simple_agent.load_llm_config") as mock_load_llm:
            mock_load_llm.side_effect = ValueError("No API key found")

            with pytest.raises(ValueError) as exc_info:
                await initialize_agent_and_mcp_server()

            assert "No API key found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_initialize_agent_and_mcp_server_custom_host_port(self):
        """Test agent initialization with custom host and port."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test_key",
                "MCP_SERVER_HOST": "custom.host",
                "MCP_SERVER_PORT": "8080",
            },
        ):
            with patch("src.agents.simple_agent.load_llm_config") as mock_load_llm:
                with patch("src.agents.simple_agent.MCPServerSSE") as mock_mcp_server:
                    with patch("src.agents.simple_agent.Agent") as mock_agent:
                        mock_llm = Mock()
                        mock_load_llm.return_value = mock_llm

                        await initialize_agent_and_mcp_server()

                        # Verify MCP server was created with custom URL
                        mock_mcp_server.assert_called_once_with(
                            "http://custom.host:8080/sse"
                        )

    @pytest.mark.asyncio
    async def test_initialize_agent_and_mcp_server_default_host_port(self):
        """Test agent initialization with default host and port."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test_key"
                # No MCP_SERVER_HOST or MCP_SERVER_PORT specified
            },
        ):
            with patch("src.agents.simple_agent.load_llm_config") as mock_load_llm:
                with patch("src.agents.simple_agent.MCPServerSSE") as mock_mcp_server:
                    with patch("src.agents.simple_agent.Agent") as mock_agent:
                        mock_llm = Mock()
                        mock_load_llm.return_value = mock_llm

                        await initialize_agent_and_mcp_server()

                        # Verify MCP server was created with default URL
                        mock_mcp_server.assert_called_once_with(
                            "http://localhost:3001/sse"
                        )

    @pytest.mark.asyncio
    async def test_initialize_agent_and_mcp_server_with_port_conversion(self):
        """Test agent initialization converts port string to integer."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test_key", "MCP_SERVER_PORT": "9999"},  # String port
        ):
            with patch("src.agents.simple_agent.load_llm_config") as mock_load_llm:
                with patch("src.agents.simple_agent.MCPServerSSE") as mock_mcp_server:
                    with patch("src.agents.simple_agent.Agent") as mock_agent:
                        mock_llm = Mock()
                        mock_load_llm.return_value = mock_llm

                        await initialize_agent_and_mcp_server()

                        # Verify port was converted to int and used in URL
                        mock_mcp_server.assert_called_once_with(
                            "http://localhost:9999/sse"
                        )

    @pytest.mark.asyncio
    async def test_initialize_agent_and_mcp_server_error_propagation(self):
        """Test that initialization errors are properly propagated."""
        with patch("src.agents.simple_agent.load_llm_config") as mock_load_llm:
            mock_load_llm.side_effect = ValueError("Test config error")

            with patch("builtins.print") as mock_print:
                with pytest.raises(ValueError) as exc_info:
                    await initialize_agent_and_mcp_server()

                assert "Test config error" in str(exc_info.value)
                # Verify error was printed to stderr
                stderr_calls = [
                    call
                    for call in mock_print.call_args_list
                    if len(call[1]) > 0 and "file" in call[1]
                ]
                assert len(stderr_calls) > 0


class TestSystemPrompt:
    """Test suite for system prompt validation and content."""

    def test_system_prompt_contains_key_elements(self):
        """Test that the system prompt contains essential elements."""
        # Check for key sections that should be in the system prompt
        assert "MCP tools" in SYSTEM_PROMPT
        assert "document" in SYSTEM_PROMPT.lower()
        assert "FinalAgentResponse" in SYSTEM_PROMPT
        assert "tool" in SYSTEM_PROMPT.lower()

        # Check for specific tool mentions
        assert "list_documents" in SYSTEM_PROMPT
        assert "read_full_document" in SYSTEM_PROMPT
        assert "create_document" in SYSTEM_PROMPT

    def test_system_prompt_tool_selection_rules(self):
        """Test that system prompt contains critical tool selection rules."""
        assert "CRITICAL TOOL SELECTION RULES" in SYSTEM_PROMPT
        assert "list_documents" in SYSTEM_PROMPT
        assert "read_full_document" in SYSTEM_PROMPT
        assert "get_document_statistics" in SYSTEM_PROMPT
        assert "find_text_in_document" in SYSTEM_PROMPT

    def test_system_prompt_error_handling_section(self):
        """Test that system prompt contains error handling guidance."""
        assert "ERROR HANDLING" in SYSTEM_PROMPT
        assert "error_message" in SYSTEM_PROMPT
        assert "OperationStatus" in SYSTEM_PROMPT

    def test_system_prompt_document_vs_listing_distinction(self):
        """Test that system prompt clearly distinguishes listing vs reading documents."""
        assert "CRITICAL DISTINCTION - LISTING vs READING DOCUMENTS" in SYSTEM_PROMPT
        assert "LISTING DOCUMENTS" in SYSTEM_PROMPT
        assert "READING DOCUMENT CONTENT" in SYSTEM_PROMPT

    def test_system_prompt_statistics_tool_usage(self):
        """Test that system prompt contains specific statistics tool guidance."""
        assert "STATISTICS TOOL USAGE" in SYSTEM_PROMPT
        assert "StatisticsReport" in SYSTEM_PROMPT
        assert "get_document_statistics" in SYSTEM_PROMPT
        assert "get_chapter_statistics" in SYSTEM_PROMPT

    def test_system_prompt_constraint_mention(self):
        """Test that system prompt mentions the core constraint."""
        assert "CORE CONSTRAINT" in SYSTEM_PROMPT
        assert "at most one MCP tool per user query" in SYSTEM_PROMPT

    def test_system_prompt_length_reasonable(self):
        """Test that system prompt is substantial but not excessively long."""
        # Should be comprehensive but not overwhelming
        assert (
            1000 < len(SYSTEM_PROMPT) < 10000
        ), f"System prompt length: {len(SYSTEM_PROMPT)}"

    def test_system_prompt_contains_examples(self):
        """Test that system prompt contains practical examples."""
        assert "KEY TOOLS EXAMPLES" in SYSTEM_PROMPT
        assert "document_name=" in SYSTEM_PROMPT  # Parameter examples
        assert "chapter_name=" in SYSTEM_PROMPT


class TestUtilityIntegration:
    """Test suite for integration with utility functions and edge cases."""

    def test_final_agent_response_with_conftest_validation(self):
        """Test FinalAgentResponse validation using conftest utilities."""
        response = FinalAgentResponse(summary="Test response")

        # Use the conftest utility
        assert_response_valid(response, "simple_agent")

        # Test edge case
        response_with_details = FinalAgentResponse(
            summary="Response with details", details={"key": "value"}
        )
        assert_response_valid(response_with_details, "simple_agent")

    @pytest.mark.asyncio
    async def test_process_single_user_query_with_conftest_mock(self):
        """Test process_single_user_query using conftest mock utilities."""
        mock_agent = create_mock_agent()

        # Set up the mock to return a proper response
        mock_response = FinalAgentResponse(summary="Mock response")
        mock_run_result = Mock()
        mock_run_result.output = mock_response
        mock_run_result.error_message = None
        mock_agent.run.return_value = mock_run_result

        result = await process_single_user_query(mock_agent, "test query")

        assert result == mock_response
        mock_agent.run.assert_called_once_with("test query")
        assert_response_valid(result, "process_single_user_query")

    @pytest.mark.asyncio
    async def test_initialize_agent_with_conftest_mock_server(self):
        """Test agent initialization using conftest MCP server mock."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            with patch("src.agents.simple_agent.load_llm_config") as mock_load_llm:
                with patch(
                    "src.agents.simple_agent.MCPServerSSE"
                ) as mock_mcp_server_class:
                    with patch("src.agents.simple_agent.Agent") as mock_agent_class:
                        mock_llm = Mock()
                        mock_load_llm.return_value = mock_llm

                        # Use conftest utility
                        mock_mcp_server = create_mock_mcp_server()
                        mock_mcp_server_class.return_value = mock_mcp_server

                        mock_agent = create_mock_agent()
                        mock_agent_class.return_value = mock_agent

                        agent, mcp_server = await initialize_agent_and_mcp_server()

                        assert agent == mock_agent
                        assert mcp_server == mock_mcp_server
                        mock_agent_class.assert_called_once()

    def test_details_type_union_comprehensive(self):
        """Test that DetailsType union accepts all expected types comprehensively."""
        from datetime import datetime, timezone

        # Test all the union types
        test_cases = [
            # List[DocumentInfo]
            [
                DocumentInfo(
                    document_name="test",
                    total_chapters=1,
                    total_word_count=10,
                    total_paragraph_count=2,
                    last_modified=datetime.now(timezone.utc),
                    chapters=[],
                )
            ],
            # Optional[List[ChapterMetadata]]
            [
                ChapterMetadata(
                    chapter_name="test.md",
                    word_count=10,
                    paragraph_count=2,
                    last_modified=datetime.now(timezone.utc),
                )
            ],
            None,  # None case
            # Optional[ChapterContent]
            ChapterContent(
                document_name="test",
                chapter_name="test.md",
                content="content",
                word_count=10,
                paragraph_count=2,
                last_modified=datetime.now(timezone.utc),
            ),
            # OperationStatus
            OperationStatus(success=True, message="Success"),
            # Optional[StatisticsReport]
            StatisticsReport(scope="test", word_count=10, paragraph_count=2),
            # List[ParagraphDetail]
            [
                ParagraphDetail(
                    document_name="test",
                    chapter_name="test.md",
                    paragraph_index_in_chapter=0,
                    content="content",
                    word_count=5,
                )
            ],
            # Dict[str, Any]
            {"key": "value", "number": 42},
        ]

        for details in test_cases:
            response = FinalAgentResponse(summary="Test", details=details)
            assert response.details == details

    def test_error_message_formatting_edge_cases(self):
        """Test FinalAgentResponse with various error message formats."""
        # Test with multiline error
        multiline_error = "Error line 1\nError line 2\nError line 3"
        response = FinalAgentResponse(
            summary="Multi-line error", error_message=multiline_error
        )
        assert response.error_message == multiline_error

        # Test with very long error message
        long_error = "Error: " + "x" * 1000
        response = FinalAgentResponse(summary="Long error", error_message=long_error)
        assert len(response.error_message) == len(long_error)

        # Test with special characters
        special_error = "Error: 特殊字符 🚨 <script>alert('xss')</script>"
        response = FinalAgentResponse(
            summary="Special chars error", error_message=special_error
        )
        assert response.error_message == special_error
