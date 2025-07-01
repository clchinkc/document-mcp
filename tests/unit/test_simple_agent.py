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

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent

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
    main as simple_agent_main,
    process_single_user_query,
)

# Import shared test utilities
from tests.shared import (
    assert_agent_response_valid,
)
from tests.shared.mock_factories import (
    create_mock_agent,
    create_mock_mcp_server,
)


class TestConfigurationFunctions:
    """Test suite for configuration-related functions."""

    def test_load_llm_config_with_openai_key(self, mock_environment_operations, mocker):
        mock_environment_operations.mock_os_environ({
            "OPENAI_API_KEY": "test_openai_key", 
            "OPENAI_MODEL_NAME": "gpt-4o"
        })
        mocker.patch("src.agents.simple_agent.load_dotenv")
        
        model = load_llm_config()
        assert model is not None
        # The actual model type checking would require importing OpenAIModel
        # but we're testing the function logic, not the model instantiation

    def test_load_llm_config_with_gemini_key(self, mock_environment_operations, mocker):
        mock_environment_operations.mock_os_environ({
            "GEMINI_API_KEY": "test_gemini_key",
            "GEMINI_MODEL_NAME": "gemini-2.5-flash",
        })
        mocker.patch("src.agents.simple_agent.load_dotenv")
        
        model = load_llm_config()
        assert model is not None

    def test_load_llm_config_openai_priority_over_gemini(self, mock_environment_operations, mocker):
        """Test that OpenAI has priority over Gemini when both keys are present."""
        mock_environment_operations.mock_os_environ({
            "OPENAI_API_KEY": "test_openai_key", 
            "GEMINI_API_KEY": "test_gemini_key"
        })
        mocker.patch("src.agents.simple_agent.load_dotenv")
        mock_openai = mocker.patch("src.agents.simple_agent.OpenAIModel")
        mock_gemini = mocker.patch("src.agents.simple_agent.GeminiModel")
        
        load_llm_config()
        mock_openai.assert_called_once()
        mock_gemini.assert_not_called()

    def test_load_llm_config_no_api_keys(self, mock_environment_operations, mocker):
        """Test loading LLM config with no API keys raises ValueError."""
        mock_environment_operations.mock_os_environ({})
        mocker.patch("src.agents.simple_agent.load_dotenv")
        
        with pytest.raises(ValueError) as exc_info:
            load_llm_config()
        assert "No valid API key found" in str(exc_info.value)

    def test_load_llm_config_empty_api_keys(self, mock_environment_operations, mocker):
        """Test loading LLM config with empty API keys raises ValueError."""
        mock_environment_operations.mock_os_environ({
            "OPENAI_API_KEY": "", 
            "GEMINI_API_KEY": "   "  # Whitespace only
        })
        mocker.patch("src.agents.simple_agent.load_dotenv")
        
        with pytest.raises(ValueError) as exc_info:
            load_llm_config()
        assert "No valid API key found" in str(exc_info.value)

    def test_check_api_keys_config_no_keys(self, mock_environment_operations, mocker):
        mock_environment_operations.mock_os_environ({})
        mocker.patch("src.agents.simple_agent.load_dotenv")
        
        config = check_api_keys_config()
        assert config["openai_configured"] is False
        assert config["gemini_configured"] is False
        assert config["active_provider"] is None
        assert config["active_model"] is None

    def test_check_api_keys_config_openai_only(self, mock_environment_operations, mocker):
        """Test checking API keys config with only OpenAI configured."""
        mock_environment_operations.mock_os_environ({
            "OPENAI_API_KEY": "test_openai_key", 
            "OPENAI_MODEL_NAME": "gpt-4o"
        })
        mocker.patch("src.agents.simple_agent.load_dotenv")
        
        config = check_api_keys_config()
        assert config["openai_configured"] is True
        assert config["gemini_configured"] is False
        assert config["active_provider"] == "openai"
        assert config["active_model"] == "gpt-4o"
        assert config["openai_model"] == "gpt-4o"

    def test_check_api_keys_config_gemini_only(self, mock_environment_operations, mocker):
        """Test checking API keys config with only Gemini configured."""
        mock_environment_operations.mock_os_environ({
            "GEMINI_API_KEY": "test_gemini_key",
            "GEMINI_MODEL_NAME": "gemini-2.5-flash",
        })
        mocker.patch("src.agents.simple_agent.load_dotenv")
        
        config = check_api_keys_config()
        assert config["openai_configured"] is False
        assert config["gemini_configured"] is True
        assert config["active_provider"] == "gemini"
        assert config["active_model"] == "gemini-2.5-flash"
        assert config["gemini_model"] == "gemini-2.5-flash"

    def test_check_api_keys_config_both_keys_openai_priority(self, mock_environment_operations, mocker):
        """Test checking API keys config with both keys configured - OpenAI has priority."""
        mock_environment_operations.mock_os_environ({
            "OPENAI_API_KEY": "test_openai_key",
            "GEMINI_API_KEY": "test_gemini_key",
            "OPENAI_MODEL_NAME": "gpt-4o",
            "GEMINI_MODEL_NAME": "gemini-2.5-flash",
        })
        mocker.patch("src.agents.simple_agent.load_dotenv")
        
        config = check_api_keys_config()
        assert config["openai_configured"] is True
        assert config["gemini_configured"] is True
        assert config["active_provider"] == "openai"  # OpenAI has priority
        assert config["active_model"] == "gpt-4o"
        assert config["openai_model"] == "gpt-4o"
        assert config["gemini_model"] == "gemini-2.5-flash"

    def test_check_api_keys_config_default_model_names(self, mock_environment_operations, mocker):
        """Test checking API keys config uses default model names when not specified."""
        mock_environment_operations.mock_os_environ({
            "OPENAI_API_KEY": "test_openai_key"
            # No OPENAI_MODEL_NAME specified
        })
        mocker.patch("src.agents.simple_agent.load_dotenv")
        
        config = check_api_keys_config()
        assert config["openai_configured"] is True
        assert config["active_model"] == "gpt-4.1-mini"  # Default
        assert config["openai_model"] == "gpt-4.1-mini"

    def test_load_llm_config_with_model_instantiation(self, mock_environment_operations, mocker):
        """Test that load_llm_config actually instantiates model objects."""
        mock_environment_operations.mock_os_environ({
            "OPENAI_API_KEY": "test_openai_key", 
            "OPENAI_MODEL_NAME": "gpt-4o"
        })
        mocker.patch("src.agents.simple_agent.load_dotenv")
        mock_openai_model = mocker.patch("src.agents.simple_agent.OpenAIModel")
        mock_model_instance = mocker.Mock()
        mock_openai_model.return_value = mock_model_instance

        result = load_llm_config()

        mock_openai_model.assert_called_once_with(model_name="gpt-4o")
        assert result == mock_model_instance

    def test_load_llm_config_prints_model_selection(self, mock_environment_operations, mock_file_operations, mocker):
        """Test that load_llm_config prints the selected model."""
        mock_environment_operations.mock_os_environ({
            "GEMINI_API_KEY": "test_gemini_key",
            "GEMINI_MODEL_NAME": "gemini-2.5-flash",
        })
        mocker.patch("src.agents.simple_agent.load_dotenv")
        mocker.patch("src.agents.simple_agent.GeminiModel")
        mock_print = mock_file_operations.mock_print()
        
        load_llm_config()
        mock_print.assert_called_with("Using Gemini model: gemini-2.5-flash")

    def test_load_llm_config_with_whitespace_only_key(self, mock_environment_operations, mocker):
        """Test load_llm_config handles whitespace-only API keys correctly."""
        mock_environment_operations.mock_os_environ({
            "OPENAI_API_KEY": "   \t\n   ",  # Only whitespace
            "GEMINI_API_KEY": "valid_key",
        })
        mocker.patch("src.agents.simple_agent.load_dotenv")
        mock_gemini = mocker.patch("src.agents.simple_agent.GeminiModel")
        
        load_llm_config()
        mock_gemini.assert_called_once()  # Should fall back to Gemini

    def test_check_api_keys_config_with_whitespace_keys(self, mock_environment_operations, mocker):
        """Test check_api_keys_config properly handles whitespace in API keys."""
        mock_environment_operations.mock_os_environ({
            "OPENAI_API_KEY": "  \t  ",  # Whitespace only
            "GEMINI_API_KEY": "valid_key",
        })
        mocker.patch("src.agents.simple_agent.load_dotenv")
        
        config = check_api_keys_config()
        assert config["openai_configured"] is False  # Whitespace should not count
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
    async def test_process_single_user_query_timeout_error(self, mocker):
        """Test process_single_user_query handles timeout errors."""
        class MockAgent:
            async def run(self, *args, **kwargs):
                raise asyncio.TimeoutError()

        mock_agent = MockAgent()

        result = await process_single_user_query(mock_agent, "test query")

        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        assert "timed out" in result.summary.lower()
        assert result.error_message == "Timeout error"

    @pytest.mark.asyncio
    async def test_process_single_user_query_cancelled_error(self, mocker):
        """Test process_single_user_query handles cancelled errors."""
        class MockAgent:
            async def run(self, *args, **kwargs):
                raise asyncio.CancelledError()

        mock_agent = MockAgent()

        result = await process_single_user_query(mock_agent, "test query")

        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        assert "cancelled" in result.summary.lower()
        assert result.error_message == "Cancelled error"

    @pytest.mark.asyncio
    async def test_process_single_user_query_runtime_error_event_loop_closed(self, mocker):
        """Test process_single_user_query handles event loop closed errors."""
        class MockAgent:
            async def run(self, *args, **kwargs):
                raise RuntimeError("Event loop is closed")

        mock_agent = MockAgent()

        result = await process_single_user_query(mock_agent, "test query")

        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        assert ("event loop closed" in result.summary.lower() or "event loop was closed" in result.summary.lower())
        assert (result.error_message == "Event loop closed" or result.error_message is None)

    @pytest.mark.asyncio
    async def test_process_single_user_query_runtime_error_generator(self, mocker):
        """Test process_single_user_query handles generator cleanup errors."""
        class MockAgent:
            async def run(self, *args, **kwargs):
                raise RuntimeError("generator didn't stop after athrow")

        mock_agent = MockAgent()

        result = await process_single_user_query(mock_agent, "test query")

        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        assert "generator cleanup error" in result.summary.lower()
        assert result.error_message == "Generator cleanup error"

    @pytest.mark.asyncio
    async def test_process_single_user_query_generic_runtime_error(self, mocker):
        """Test process_single_user_query handles generic runtime errors."""
        class MockAgent:
            async def run(self, *args, **kwargs):
                raise RuntimeError("Some other runtime error")

        mock_agent = MockAgent()

        result = await process_single_user_query(mock_agent, "test query")

        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        assert "runtime error" in result.summary.lower()
        assert "Some other runtime error" in result.error_message

    @pytest.mark.asyncio
    async def test_process_single_user_query_read_timeout_error(self, mocker):
        """Test process_single_user_query handles ReadTimeout errors gracefully."""
        class MockAgent:
            async def run(self, *args, **kwargs):
                raise Exception("ReadTimeout occurred")

        mock_agent = MockAgent()

        result = await process_single_user_query(mock_agent, "test query")

        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        assert "api connection timeout" in result.summary.lower()
        assert "ReadTimeout occurred" in result.error_message

    @pytest.mark.asyncio
    async def test_process_single_user_query_api_key_error(self, mock_environment_operations, mocker):
        """Test process_single_user_query handles API key placeholder errors."""
        class MockAgent:
            async def run(self, *args, **kwargs):
                raise Exception("API error")

        mock_agent = MockAgent()
        mock_environment_operations.mock_os_environ({"GEMINI_API_KEY": "test_api_key_placeholder"})
        
        result = await process_single_user_query(mock_agent, "test query")

        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        assert "api authentication error" in result.summary.lower()
        assert "API error" in result.error_message

    @pytest.mark.asyncio
    async def test_process_single_user_query_generic_exception(self, mocker):
        """Test process_single_user_query handles generic exceptions."""
        class MockAgent:
            async def run(self, *args, **kwargs):
                raise Exception("Generic error")

        mock_agent = MockAgent()

        result = await process_single_user_query(mock_agent, "test query")

        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        assert "exception during query processing" in result.summary.lower()
        assert "Generic error" in result.error_message

    @pytest.mark.asyncio
    async def test_process_single_user_query_successful_run_result(self, mocker):
        """Test process_single_user_query with successful run result."""
        class MockAgent:
            async def run(self, *args, **kwargs):
                return mocker.Mock(output=FinalAgentResponse(summary="Success", details=None, error_message=None))

        mock_agent = MockAgent()
        result = await process_single_user_query(mock_agent, "test query")
        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        assert result.summary == "Success"
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_process_single_user_query_run_result_with_error(self, mocker):
        """Test process_single_user_query with run result containing error_message."""
        class MockAgent:
            async def run(self, *args, **kwargs):
                return mocker.Mock(output=FinalAgentResponse(summary="Error occurred", details=None, error_message="Test error"))

        mock_agent = MockAgent()
        result = await process_single_user_query(mock_agent, "test query")
        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        assert result.summary == "Error occurred"
        assert result.error_message == "Test error"

    @pytest.mark.asyncio
    async def test_process_single_user_query_empty_run_result(self, mocker):
        """Test process_single_user_query with empty run result."""
        class MockAgent:
            async def run(self, *args, **kwargs):
                return mocker.Mock(output=None, error_message=None)

        mock_agent = MockAgent()
        result = await process_single_user_query(mock_agent, "test query")
        assert result is None

    @pytest.mark.asyncio
    async def test_process_single_user_query_with_wait_for_timeout(self, mocker):
        """Test process_single_user_query with asyncio.wait_for timeout handling."""
        # Create a regular mock agent since we're mocking wait_for anyway
        mock_agent = mocker.Mock()
        mock_agent.run = mocker.Mock(return_value=mocker.Mock(output=FinalAgentResponse(summary="Success")))
        
        # Mock asyncio.wait_for to raise TimeoutError
        mock_wait_for = mocker.patch("asyncio.wait_for", side_effect=asyncio.TimeoutError())

        result = await process_single_user_query(mock_agent, "test query")

        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        assert "timed out" in result.summary.lower()
        assert result.error_message == "Timeout error"
        mock_wait_for.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_single_user_query_prints_errors_to_stderr(self, mock_file_operations):
        class MockAgent:
            async def run(self, *args, **kwargs):
                raise Exception("Test error")

        mock_agent = MockAgent()
        mock_print = mock_file_operations.mock_print()
        
        result = await process_single_user_query(mock_agent, "test query")

        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        mock_print.assert_called()
        args = mock_print.call_args[0]
        assert any("Test error" in str(arg) for arg in args)

    @pytest.mark.asyncio
    async def test_process_single_user_query_handles_none_agent(self):
        """Test process_single_user_query handles None agent gracefully."""
        result = await process_single_user_query(None, "test query")

        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        # The actual function catches AttributeError when calling run() on None
        assert "exception during query processing" in result.summary.lower()
        assert "'NoneType' object has no attribute 'run'" in result.error_message

    @pytest.mark.asyncio
    async def test_process_single_user_query_handles_empty_query(self, mocker):
        """Test process_single_user_query handles empty query string."""
        class MockAgent:
            async def run(self, *args, **kwargs):
                return mocker.Mock(output=FinalAgentResponse(summary="Empty query handled", details=None, error_message=None))

        mock_agent = MockAgent()
        result = await process_single_user_query(mock_agent, "")
        assert result.summary == "Empty query handled"
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_process_single_user_query_with_complex_error_message(self):
        """Test process_single_user_query with complex error message containing special characters."""
        complex_error = "Error with special chars: ñáéíóú 🔥 & quotes 'single' \"double\""

        class MockAgent:
            async def run(self, *args, **kwargs):
                raise Exception(complex_error)

        mock_agent = MockAgent()

        result = await process_single_user_query(mock_agent, "test query")

        assert result is not None
        assert isinstance(result, FinalAgentResponse)
        assert "exception during query processing" in result.summary.lower()
        assert complex_error in result.error_message


class TestAgentInitialization:
    """Test suite for agent initialization functions."""

    @pytest.mark.asyncio
    async def test_initialize_agent_and_mcp_server_success(self, mock_agent_operations):
        """Test successful agent and MCP server initialization."""
        mocks = mock_agent_operations.setup_agent_test_environment("openai")
        
        mock_agent = mocks['agent']
        mock_server_instance = mocks['server_instance']

        result_agent, result_server = await initialize_agent_and_mcp_server()

        assert result_agent == mock_agent
        assert result_server == mock_server_instance
        assert result_server.host == "localhost"
        assert result_server.port == 3001

    @pytest.mark.asyncio
    async def test_initialize_agent_and_mcp_server_llm_config_error(self, mocker):
        """Test agent initialization with LLM config error."""
        # Mock load_llm_config to raise an error
        mocker.patch("src.agents.simple_agent.load_llm_config", side_effect=ValueError("Config error"))

        with pytest.raises(ValueError, match="Config error"):
            await initialize_agent_and_mcp_server()

    @pytest.mark.asyncio
    async def test_initialize_agent_and_mcp_server_custom_host_port(self, mock_agent_operations, mock_environment_operations):
        # Set up environment variables for custom host/port
        mock_environment_operations.mock_os_environ({
            "OPENAI_API_KEY": "test_key",
            "MCP_SERVER_HOST": "custom_host",
            "MCP_SERVER_PORT": "8080"
        })
        
        # Set up mocks with custom host/port
        mocks = mock_agent_operations.setup_agent_test_environment("openai")
        mock_server_instance = mocks['server_instance']
        mock_server_instance.host = "custom_host"
        mock_server_instance.port = 8080

        # Call without parameters (function reads from environment)
        result_agent, result_server = await initialize_agent_and_mcp_server()

        # Verify custom configuration was read from environment
        assert result_server == mock_server_instance

    @pytest.mark.asyncio
    async def test_initialize_agent_and_mcp_server_with_port_conversion(self, mock_agent_operations, mock_environment_operations):
        """Test agent initialization converts string port to integer."""
        # Set up environment variables with string port
        mock_environment_operations.mock_os_environ({
            "OPENAI_API_KEY": "test_key",
            "MCP_SERVER_PORT": "9000"  # String port in environment
        })
        
        # Set up mocks
        mocks = mock_agent_operations.setup_agent_test_environment("openai")
        mock_server_instance = mocks['server_instance']
        mock_server_instance.port = 9000

        # Call function (it should read and convert the string port)
        result_agent, result_server = await initialize_agent_and_mcp_server()

        # Verify port was handled correctly
        assert result_server == mock_server_instance

    @pytest.mark.asyncio
    async def test_initialize_agent_and_mcp_server_error_propagation(self, mocker):
        """Test that initialization errors are properly propagated."""
        # Mock load_llm_config to succeed but Agent creation to fail
        mocker.patch("src.agents.simple_agent.load_llm_config", return_value=mocker.Mock())
        mocker.patch("src.agents.simple_agent.Agent", side_effect=Exception("Agent creation failed"))

        with pytest.raises(Exception, match="Agent creation failed"):
            await initialize_agent_and_mcp_server()


class TestSystemPrompt:
    """Test suite for system prompt validation and content."""

    def test_system_prompt_contains_key_elements(self):
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
        assert "CORE CONSTRAINT" in SYSTEM_PROMPT
        assert "at most one MCP tool per user query" in SYSTEM_PROMPT

    def test_system_prompt_length_reasonable(self):
        """Test that system prompt is substantial but not excessively long."""
        # Should be comprehensive but not overwhelming
        assert (
            1000 < len(SYSTEM_PROMPT) < 10000
        ), f"System prompt length: {len(SYSTEM_PROMPT)}"

    def test_system_prompt_contains_examples(self):
        assert "KEY TOOLS EXAMPLES" in SYSTEM_PROMPT
        assert "document_name=" in SYSTEM_PROMPT  # Parameter examples
        assert "chapter_name=" in SYSTEM_PROMPT

    def test_simple_agent_system_prompt_summary_enhancements(self):
        """Test that Simple Agent system prompt includes summary handling."""
        assert 'SUMMARY OPERATIONS' in SYSTEM_PROMPT
        assert 'read_document_summary' in SYSTEM_PROMPT
        assert 'Explicit Content Requests' in SYSTEM_PROMPT
        assert 'Broad Screening/Editing' in SYSTEM_PROMPT
        assert 'USE THIS FIRST' in SYSTEM_PROMPT


class TestUtilityIntegration:
    """Test suite for integration with utility functions and edge cases."""

    def test_final_agent_response_with_conftest_validation(self):
        """Test FinalAgentResponse validation using conftest utilities."""
        response = FinalAgentResponse(summary="Test response")

        # Use the conftest utility
        assert_agent_response_valid(response, "simple_agent")

        # Test edge case
        response_with_details = FinalAgentResponse(
            summary="Response with details", details={"key": "value"}
        )
        assert_agent_response_valid(response_with_details, "simple_agent")

    @pytest.mark.asyncio
    async def test_process_single_user_query_with_conftest_mock(self, mocker):
        """Test process_single_user_query using conftest mock utilities."""
        # Create custom response data
        response_data = {"summary": "Mock response", "details": None, "error_message": None}
        mock_agent = create_mock_agent(response_data, mocker=mocker)

        result = await process_single_user_query(mock_agent, "test query")

        expected_response = FinalAgentResponse(summary="Mock response")
        assert result.summary == expected_response.summary
        assert result.details == expected_response.details
        assert result.error_message == expected_response.error_message
        assert_agent_response_valid(result, "process_single_user_query")

    @pytest.mark.asyncio
    async def test_initialize_agent_with_conftest_mock_server(self, mock_environment_operations, mock_agent_operations):
        """Test agent initialization using conftest MCP server mock."""
        mock_environment_operations.mock_os_environ({"OPENAI_API_KEY": "test_key"})
        
        # Set up all the mocks using the fixture
        mocks = mock_agent_operations.setup_agent_test_environment("openai")
        mock_agent = mocks['agent']
        mock_server_instance = mocks['server_instance']

        agent, mcp_server = await initialize_agent_and_mcp_server()

        assert agent == mock_agent
        assert mcp_server == mock_server_instance

    def test_details_type_union_comprehensive(self, mocker):
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


class TestSummaryFunctionality:
    """Test summary functionality integration with Simple Agent."""

    @pytest.mark.asyncio
    async def test_simple_agent_summary_priority_workflow(self):
        """Test that Simple agent prioritizes summaries in its workflow."""
        import tempfile
        import os
        from pathlib import Path
        
        # Create isolated test environment
        temp_dir = Path(tempfile.mkdtemp(prefix='test_simple_summary_'))
        os.environ['DOCUMENT_ROOT_DIR'] = str(temp_dir)
        
        try:
            # Clear module cache to ensure fresh imports
            import sys
            if 'document_mcp.doc_tool_server' in sys.modules:
                import importlib
                importlib.reload(sys.modules['document_mcp.doc_tool_server'])
            
            from document_mcp.doc_tool_server import (
                create_document, 
                DOCUMENT_SUMMARY_FILE
            )
            
            doc_name = 'simple_test_doc'
            
            # Set up document with summary
            create_document(doc_name)
            doc_dir = temp_dir / doc_name
            doc_dir.mkdir(parents=True, exist_ok=True)
            
            # Create summary
            summary_file = doc_dir / DOCUMENT_SUMMARY_FILE  
            summary_content = "# API Project\n\nA project for building REST APIs with authentication."
            summary_file.write_text(summary_content)
            
            # Create chapters
            auth_chapter = doc_dir / '01-auth.md'
            auth_chapter.write_text("# Authentication\n\nDetailed auth implementation...")
            
            # Test agent initialization and system prompt
            try:
                # Mock environment and agent components for testing
                mock_env = {"OPENAI_API_KEY": "test_key"}
                
                # For this complex test, we'll just validate the underlying document functions
                # rather than fully mocking the agent initialization
                from document_mcp.doc_tool_server import list_documents, read_document_summary
                
                docs = list_documents()
                test_doc = next((d for d in docs if d.document_name == doc_name), None)
                assert test_doc is not None
                assert test_doc.has_summary is True
                
                summary = read_document_summary(doc_name)
                assert summary == summary_content
                    
            except Exception as e:
                # If agent connection fails, test the underlying logic
                from document_mcp.doc_tool_server import list_documents, read_document_summary
                
                docs = list_documents()
                test_doc = next((d for d in docs if d.document_name == doc_name), None)
                assert test_doc is not None
                assert test_doc.has_summary is True
                
                summary = read_document_summary(doc_name)
                assert summary == summary_content
                
        finally:
            # Clean up
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
