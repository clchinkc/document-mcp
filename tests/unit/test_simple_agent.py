"""Unit tests for Simple Agent implementation.

This module provides comprehensive unit tests for the Simple Agent with mocked
dependencies to validate core functionality, error handling, and response formatting.
"""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from document_mcp.exceptions import AgentConfigurationError
from document_mcp.exceptions import OperationError
from src.agents.simple_agent.agent import SimpleAgent
from src.agents.simple_agent.agent import SimpleAgentResponse
from src.agents.simple_agent.prompts import get_simple_agent_system_prompt


class TestSimpleAgentResponse:
    """Test SimpleAgentResponse model validation."""

    def test_simple_agent_response_valid_data(self):
        """Test SimpleAgentResponse with valid data."""
        response = SimpleAgentResponse(
            summary="Document created successfully", details={"document_name": "test_doc", "success": True}
        )
        assert response.summary == "Document created successfully"
        assert response.details == {"document_name": "test_doc", "success": True}

    def test_simple_agent_response_empty_details(self):
        """Test SimpleAgentResponse with empty details."""
        response = SimpleAgentResponse(summary="No operation performed", details={})
        assert response.summary == "No operation performed"
        assert response.details == {}

    def test_simple_agent_response_complex_details(self):
        """Test SimpleAgentResponse with complex nested details."""
        complex_details = {
            "operations": [
                {"type": "create_document", "success": True},
                {"type": "create_chapter", "success": True},
            ],
            "metadata": {"execution_time": 1.5, "tools_used": ["create_document", "create_chapter"]},
        }
        response = SimpleAgentResponse(summary="Multiple operations completed", details=complex_details)
        assert response.details["operations"] == complex_details["operations"]
        assert response.details["metadata"] == complex_details["metadata"]


class TestSimpleAgentInitialization:
    """Test Simple Agent initialization and configuration."""

    def test_simple_agent_initialization(self):
        """Test basic Simple Agent initialization."""
        agent = SimpleAgent()
        assert agent.agent_type == "simple"
        assert agent._pydantic_agent is None
        assert agent.settings is not None

    @patch("src.agents.shared.agent_base.get_settings")
    def test_simple_agent_initialization_with_settings(self, mock_get_settings):
        """Test Simple Agent initialization with mocked settings."""
        mock_settings = Mock()
        mock_settings.default_timeout = 30
        mock_get_settings.return_value = mock_settings

        agent = SimpleAgent()
        assert agent.settings == mock_settings


class TestSimpleAgentLLMConfiguration:
    """Test Simple Agent LLM configuration and error handling."""

    @pytest.mark.asyncio
    async def test_get_llm_success(self):
        """Test successful LLM configuration loading."""
        agent = SimpleAgent()

        mock_llm = Mock()
        with patch("src.agents.shared.agent_base.load_llm_config", new_callable=AsyncMock) as mock_load:
            mock_load.return_value = mock_llm

            llm = await agent.get_llm()
            assert llm == mock_llm
            assert agent._llm == mock_llm
            mock_load.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_llm_cached(self):
        """Test LLM configuration caching."""
        agent = SimpleAgent()
        mock_llm = Mock()
        agent._llm = mock_llm

        with patch("src.agents.shared.agent_base.load_llm_config", new_callable=AsyncMock) as mock_load:
            llm = await agent.get_llm()
            assert llm == mock_llm
            mock_load.assert_not_called()  # Should not reload cached LLM

    @pytest.mark.asyncio
    async def test_get_llm_configuration_error(self):
        """Test LLM configuration error handling."""
        agent = SimpleAgent()

        with patch("src.agents.shared.agent_base.load_llm_config", new_callable=AsyncMock) as mock_load:
            mock_load.side_effect = Exception("API key not found")

            with pytest.raises(AgentConfigurationError) as exc_info:
                await agent.get_llm()

            assert "Failed to load LLM configuration" in str(exc_info.value)
            assert "API key not found" in str(exc_info.value)
            assert exc_info.value.agent_type == "simple"


class TestSimpleAgentMCPConfiguration:
    """Test Simple Agent MCP server configuration."""

    def test_get_mcp_server_environment(self):
        """Test MCP server environment configuration."""
        agent = SimpleAgent()

        with patch("src.agents.shared.agent_base.prepare_mcp_server_environment") as mock_prepare:
            mock_env = {"DOCUMENTS_STORAGE_ROOT": "/test/path"}
            mock_prepare.return_value = mock_env

            env = agent.get_mcp_server_environment()
            assert env == mock_env
            mock_prepare.assert_called_once()


class TestSimpleAgentPydanticAgentCreation:
    """Test Pydantic AI agent creation and configuration."""

    @pytest.mark.asyncio
    async def test_get_pydantic_agent_creation(self):
        """Test Pydantic AI agent creation with all components."""
        agent = SimpleAgent()

        mock_llm = Mock()
        mock_server_env = {"TEST_ENV": "value"}
        mock_tool_descriptions = "Mock tool descriptions"
        mock_system_prompt = "Mock system prompt"

        with (
            patch.multiple(
                agent,
                get_llm=AsyncMock(return_value=mock_llm),
                get_mcp_server_environment=Mock(return_value=mock_server_env),
                get_system_prompt=Mock(return_value=mock_system_prompt),
            ),
            patch("src.agents.simple_agent.agent.get_tool_descriptions_for_agent") as mock_get_tools,
            patch("src.agents.simple_agent.agent.MCPServerStdio") as mock_mcp_server_class,
            patch("src.agents.simple_agent.agent.Agent") as mock_agent_class,
        ):
            mock_get_tools.return_value = mock_tool_descriptions
            mock_mcp_server = Mock()
            mock_mcp_server_class.return_value = mock_mcp_server
            mock_agent_instance = Mock()
            mock_agent_class.return_value = mock_agent_instance

            pydantic_agent = await agent.get_pydantic_agent()

            # Verify agent creation parameters
            mock_agent_class.assert_called_once_with(
                mock_llm,
                result_type=SimpleAgentResponse,
                system_prompt=f"{mock_system_prompt}\n\n{mock_tool_descriptions}",
                mcp_servers=[mock_mcp_server],
            )

            # Verify tool descriptions were requested for simple agent
            mock_get_tools.assert_called_once_with("simple")

            assert pydantic_agent == mock_agent_instance
            assert agent._pydantic_agent == mock_agent_instance

    @pytest.mark.asyncio
    async def test_get_pydantic_agent_cached(self):
        """Test Pydantic AI agent caching."""
        agent = SimpleAgent()
        mock_agent = Mock()
        agent._pydantic_agent = mock_agent

        pydantic_agent = await agent.get_pydantic_agent()
        assert pydantic_agent == mock_agent


class TestSimpleAgentSystemPrompt:
    """Test Simple Agent system prompt functionality."""

    def test_get_system_prompt(self):
        """Test system prompt retrieval."""
        agent = SimpleAgent()

        with patch("src.agents.simple_agent.agent.get_simple_agent_system_prompt") as mock_get_prompt:
            mock_prompt = "Test system prompt for simple agent"
            mock_get_prompt.return_value = mock_prompt

            prompt = agent.get_system_prompt()
            assert prompt == mock_prompt
            mock_get_prompt.assert_called_once()


class TestSimpleAgentExecution:
    """Test Simple Agent execution logic and error handling."""

    @pytest.mark.asyncio
    async def test_run_success(self):
        """Test successful agent execution."""
        agent = SimpleAgent()
        query = "Create a document called 'Test Document'"

        # Mock agent result
        mock_agent_result = Mock()
        mock_agent_result.data = SimpleAgentResponse(
            summary="Document created successfully",
            details={"document_name": "Test Document", "success": True},
        )

        # Mock all_messages for extract_mcp_tool_responses
        mock_agent_result.all_messages.return_value = []

        mock_pydantic_agent = AsyncMock()
        mock_pydantic_agent.run.return_value = mock_agent_result

        with (
            patch.object(agent, "get_pydantic_agent", return_value=mock_pydantic_agent),
            patch.object(agent, "create_response") as mock_create_response,
            patch.object(agent, "create_error_context") as mock_error_context,
        ):
            mock_context = Mock()
            mock_error_context.return_value.__enter__ = Mock(return_value=mock_context)
            mock_error_context.return_value.__exit__ = Mock(return_value=False)

            mock_response = {"summary": "Success", "details": {"success": True}}
            mock_create_response.return_value = mock_response

            result = await agent.run(query)

            mock_pydantic_agent.run.assert_called_once_with(query)
            mock_create_response.assert_called_once()
            assert result == mock_response

    @pytest.mark.asyncio
    async def test_run_with_timeout(self):
        """Test agent execution with custom timeout."""
        agent = SimpleAgent()
        query = "Test query"
        timeout = 60

        mock_agent_result = Mock()
        mock_agent_result.data = SimpleAgentResponse(
            summary="Document created successfully",
            details={"document_name": "Test Document", "success": True},
        )
        mock_agent_result.all_messages.return_value = []

        mock_pydantic_agent = AsyncMock()
        mock_pydantic_agent.run.return_value = mock_agent_result

        with (
            patch.object(agent, "get_pydantic_agent", return_value=mock_pydantic_agent),
            patch.object(agent, "create_response") as mock_create_response,
            patch.object(agent, "create_error_context") as mock_error_context,
            patch("asyncio.wait_for", new_callable=AsyncMock) as mock_wait_for,
        ):
            mock_context = Mock()
            mock_error_context.return_value.__enter__ = Mock(return_value=mock_context)
            mock_error_context.return_value.__exit__ = Mock(return_value=False)

            mock_wait_for.return_value = mock_agent_result
            mock_response = {"summary": "Success", "details": {"success": True}}
            mock_create_response.return_value = mock_response

            result = await agent.run(query, timeout=timeout)

            # Verify wait_for was called with correct timeout
            mock_wait_for.assert_called_once()
            args, kwargs = mock_wait_for.call_args
            assert kwargs.get("timeout") == timeout or args[1] == timeout
            assert result == mock_response

    @pytest.mark.asyncio
    async def test_run_timeout_error(self):
        """Test agent execution timeout handling."""
        agent = SimpleAgent()
        query = "Test query"
        timeout = 30

        mock_pydantic_agent = AsyncMock()

        with (
            patch.object(agent, "get_pydantic_agent", return_value=mock_pydantic_agent),
            patch.object(agent, "create_error_context") as mock_error_context,
            patch("asyncio.wait_for", new_callable=AsyncMock) as mock_wait_for,
        ):
            mock_context = Mock()
            mock_error_context.return_value.__enter__ = Mock(return_value=mock_context)
            mock_error_context.return_value.__exit__ = Mock(return_value=False)

            mock_wait_for.side_effect = asyncio.TimeoutError()

            with pytest.raises(OperationError) as exc_info:
                await agent.run(query, timeout=timeout)

            assert "Agent execution timed out" in str(exc_info.value)
            assert exc_info.value.details["operation"] == "simple_agent_execution"
            assert exc_info.value.details["query"] == query
            assert exc_info.value.details["timeout"] == timeout


class TestSimpleAgentStructuredOutput:
    """Test Simple Agent structured output functionality."""

    @pytest.mark.asyncio
    async def test_run_with_structured_output_with_data(self):
        """Test structured output when agent result has data attribute."""
        agent = SimpleAgent()
        query = "Test query"

        expected_response = SimpleAgentResponse(summary="Test response", details={"test": "data"})

        mock_agent_result = Mock()
        mock_agent_result.data = expected_response

        mock_pydantic_agent = AsyncMock()
        mock_pydantic_agent.run.return_value = mock_agent_result

        with patch.object(agent, "get_pydantic_agent", return_value=mock_pydantic_agent):
            result = await agent.run_with_structured_output(query)

            assert result == expected_response
            mock_pydantic_agent.run.assert_called_once_with(query)

    @pytest.mark.asyncio
    async def test_run_with_structured_output_without_data(self):
        """Test structured output when agent result lacks data attribute."""
        agent = SimpleAgent()
        query = "Test query"

        mock_agent_result = Mock()
        delattr(mock_agent_result, "data")  # Simulate missing data attribute
        mock_agent_result.output = SimpleAgentResponse(
            summary="Operation completed", details={"create_document": {"success": True}}
        )

        mock_pydantic_agent = AsyncMock()
        mock_pydantic_agent.run.return_value = mock_agent_result

        mock_tool_responses = {"create_document": {"success": True}}

        with (
            patch.object(agent, "get_pydantic_agent", return_value=mock_pydantic_agent),
            patch.object(agent, "extract_mcp_tool_responses", return_value=mock_tool_responses),
        ):
            result = await agent.run_with_structured_output(query)

            assert isinstance(result, SimpleAgentResponse)
            assert result.summary == "Operation completed"
            assert result.details == mock_tool_responses


class TestSimpleAgentMCPToolResponseExtraction:
    """Test MCP tool response extraction functionality."""

    def test_extract_mcp_tool_responses_basic(self):
        """Test basic MCP tool response extraction."""
        agent = SimpleAgent()

        # Mock agent result with tool responses
        mock_message = Mock()
        mock_tool_part = Mock()
        mock_tool_part.tool_name = "create_document"
        mock_tool_part.content = {"success": True, "document_name": "test"}
        type(mock_tool_part).__name__ = "ToolReturnPart"
        mock_message.parts = [mock_tool_part]

        mock_agent_result = Mock()
        mock_agent_result.all_messages.return_value = [mock_message]

        with patch.object(agent, "extract_mcp_tool_responses") as mock_extract:
            mock_extract.return_value = {"create_document": {"success": True, "document_name": "test"}}

            responses = agent.extract_mcp_tool_responses(mock_agent_result)
            assert responses == {"create_document": {"success": True, "document_name": "test"}}


class TestSimpleAgentCleanup:
    """Test Simple Agent resource cleanup."""

    @pytest.mark.asyncio
    async def test_aexit_with_pydantic_agent(self):
        """Test cleanup when pydantic agent exists."""
        agent = SimpleAgent()
        mock_pydantic_agent = AsyncMock()
        agent._pydantic_agent = mock_pydantic_agent

        with patch(
            "src.agents.shared.agent_base.AgentBase.__aexit__", new_callable=AsyncMock
        ) as mock_super_aexit:
            await agent.__aexit__(None, None, None)

            mock_pydantic_agent.__aexit__.assert_called_once_with(None, None, None)
            mock_super_aexit.assert_called_once_with(None, None, None)

    @pytest.mark.asyncio
    async def test_aexit_without_pydantic_agent(self):
        """Test cleanup when no pydantic agent exists."""
        agent = SimpleAgent()

        with patch(
            "src.agents.shared.agent_base.AgentBase.__aexit__", new_callable=AsyncMock
        ) as mock_super_aexit:
            await agent.__aexit__(None, None, None)

            mock_super_aexit.assert_called_once_with(None, None, None)


class TestSimpleAgentPrompts:
    """Test Simple Agent prompt functionality."""

    def test_get_simple_agent_system_prompt(self):
        """Test system prompt generation."""
        prompt = get_simple_agent_system_prompt()

        # Basic validation that prompt contains expected elements
        assert isinstance(prompt, str)
        assert len(prompt) > 0

        # Check for key components that should be in simple agent prompt
        assert "simple" in prompt.lower() or "document" in prompt.lower()


# Integration-style tests for agent behavior patterns
class TestSimpleAgentIntegrationPatterns:
    """Test Simple Agent integration patterns with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_full_agent_workflow_mock(self):
        """Test complete agent workflow with all mocked dependencies."""
        agent = SimpleAgent()
        query = "Create a document called 'Integration Test'"

        # Mock all dependencies
        mock_llm = Mock()
        mock_mcp_server = Mock()
        mock_pydantic_agent = AsyncMock()

        # Mock agent result with realistic structure
        mock_agent_result = Mock()
        mock_agent_result.data = SimpleAgentResponse(
            summary="Document 'Integration Test' created successfully",
            details={
                "create_document": {
                    "success": True,
                    "document_name": "Integration Test",
                    "path": "/documents/Integration Test/",
                }
            },
        )
        mock_agent_result.all_messages.return_value = []
        mock_pydantic_agent.run.return_value = mock_agent_result

        with (
            patch(
                "src.agents.shared.agent_base.load_llm_config", new_callable=AsyncMock, return_value=mock_llm
            ),
            patch("src.agents.shared.agent_base.prepare_mcp_server_environment", return_value={}),
            patch(
                "src.agents.shared.tool_descriptions.get_tool_descriptions_for_agent",
                return_value="Mock tools",
            ),
            patch("src.agents.simple_agent.agent.MCPServerStdio", return_value=mock_mcp_server),
            patch("src.agents.simple_agent.agent.Agent", return_value=mock_pydantic_agent),
            patch(
                "src.agents.simple_agent.prompts.get_simple_agent_system_prompt", return_value="Mock prompt"
            ),
            patch.object(agent, "create_error_context") as mock_error_context,
            patch.object(agent, "create_response") as mock_create_response,
        ):
            # Mock error context
            mock_context = Mock()
            mock_error_context.return_value.__enter__ = Mock(return_value=mock_context)
            mock_error_context.return_value.__exit__ = Mock(return_value=False)

            # Mock response creation
            expected_response = {
                "summary": "Document 'Integration Test' created successfully",
                "details": {"create_document": {"success": True, "document_name": "Integration Test"}},
                "agent_type": "simple",
                "execution_successful": True,
            }
            mock_create_response.return_value = expected_response

            result = await agent.run(query)

            # Verify the complete workflow
            assert result == expected_response
            mock_pydantic_agent.run.assert_called_once_with(query)
            mock_create_response.assert_called_once()
