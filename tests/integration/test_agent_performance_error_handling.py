"""Integration tests for agent error handling.

This module tests core agent error handling patterns to ensure
robust operation in production environments.
"""

import asyncio
import sys
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from pydantic_ai.mcp import MCPServerStdio

from document_mcp.exceptions import AgentConfigurationError
from document_mcp.exceptions import OperationError
from src.agents.simple_agent.agent import SimpleAgent
from src.agents.simple_agent.agent import SimpleAgentResponse


@pytest.fixture
async def mcp_server():
    """Provide a real MCP server for integration testing."""
    server = MCPServerStdio(
        command=sys.executable, args=["-m", "document_mcp.doc_tool_server", "stdio"], timeout=60.0
    )
    yield server


class TestAgentErrorHandling:
    """Test agent error handling under various failure conditions."""

    @pytest.mark.asyncio
    async def test_simple_agent_llm_configuration_error(self):
        """Test Simple Agent LLM configuration error handling."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "", "GEMINI_API_KEY": ""}, clear=False):
            # Reset settings to pick up cleared API keys
            from document_mcp.config.settings import reset_settings

            reset_settings()

            agent = SimpleAgent()

            # Should raise AgentConfigurationError when no API keys available
            with pytest.raises(AgentConfigurationError) as exc_info:
                await agent.get_llm()

            assert exc_info.value.agent_type == "simple"
            assert "Failed to load LLM configuration" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_simple_agent_mcp_server_failure(self, temp_docs_root):
        """Test Simple Agent behavior when MCP server fails."""
        # Provide basic environment to allow LLM initialization
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            agent = SimpleAgent()

            with patch("src.agents.shared.config.load_llm_config") as mock_load_llm:
                mock_llm = AsyncMock()
                mock_load_llm.return_value = mock_llm

                # Mock MCP server that fails to start
                with patch("src.agents.simple_agent.agent.MCPServerStdio") as mock_mcp_server_class:
                    mock_mcp_server = Mock()
                    mock_mcp_server_class.return_value = mock_mcp_server

                    # Mock Agent class to simulate MCP connection failure
                    with patch("src.agents.simple_agent.agent.Agent") as mock_agent_class:
                        mock_agent_class.side_effect = Exception("Failed to connect to MCP server")

                        # Should handle MCP server connection errors
                        with pytest.raises(Exception) as exc_info:
                            await agent.get_pydantic_agent()

                        assert "Failed to connect to MCP server" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_simple_agent_timeout_handling(self, mcp_server, temp_docs_root):
        """Test Simple Agent timeout handling."""
        query = "Create a test document"
        short_timeout = 0.001  # Very short timeout

        # Provide basic environment to allow LLM initialization
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch("src.agents.shared.config.load_llm_config") as mock_load_llm:
                mock_llm = AsyncMock()
                mock_load_llm.return_value = mock_llm

                agent = SimpleAgent()

                async with mcp_server:
                    with patch("src.agents.simple_agent.agent.Agent") as mock_agent_class:
                        mock_pydantic_agent = AsyncMock()

                        # Make agent run take longer than timeout
                        async def slow_run(query):
                            await asyncio.sleep(0.1)  # Longer than timeout
                            return Mock()

                        mock_pydantic_agent.run = slow_run
                        mock_agent_class.return_value = mock_pydantic_agent

                        # Should raise OperationError due to timeout
                        with pytest.raises(OperationError) as exc_info:
                            await agent.run(query, timeout=short_timeout)

                        assert "Agent execution timed out" in str(exc_info.value)
                        assert exc_info.value.operation == "simple_agent_execution"
                        assert exc_info.value.details["timeout"] == short_timeout

    @pytest.mark.asyncio
    async def test_agent_malformed_response_handling(self, mcp_server, temp_docs_root):
        """Test agent handling of malformed LLM responses."""
        # Provide basic environment to allow LLM initialization
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch("src.agents.shared.config.load_llm_config") as mock_load_llm:
                mock_llm = AsyncMock()
                mock_load_llm.return_value = mock_llm

                agent = SimpleAgent()

                async with mcp_server:
                    with patch("src.agents.simple_agent.agent.Agent") as mock_agent_class:
                        mock_pydantic_agent = AsyncMock()
                        mock_agent_result = Mock()

                        # Simulate malformed response (missing required fields)
                        # Create a custom mock that returns False for hasattr checks on data/output
                        class MalformedResult:
                            def __getattr__(self, name):
                                if name in ["data", "output"]:
                                    raise AttributeError(
                                        f"'{self.__class__.__name__}' object has no attribute '{name}'"
                                    )
                                return Mock()

                        mock_agent_result = MalformedResult()
                        mock_pydantic_agent.run.return_value = mock_agent_result
                        mock_agent_class.return_value = mock_pydantic_agent

                        # Mock extract_mcp_tool_responses to return empty dict
                        with patch.object(agent, "extract_mcp_tool_responses", return_value={}):
                            result = await agent.run_with_structured_output("Test query")

                            # Should create default response structure
                            assert isinstance(result, SimpleAgentResponse)
                            assert result.summary == "Operation completed"
                            assert result.details == {}

    @pytest.mark.asyncio
    async def test_agent_resource_cleanup_on_error(self, mcp_server, temp_docs_root):
        """Test agent resource cleanup when errors occur."""
        # Provide basic environment to allow LLM initialization
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch("src.agents.shared.config.load_llm_config") as mock_load_llm:
                mock_llm = AsyncMock()
                mock_load_llm.return_value = mock_llm

                agent = SimpleAgent()
                mock_pydantic_agent = AsyncMock()
                agent._pydantic_agent = mock_pydantic_agent

                # Test cleanup on error
                test_exception = Exception("Test error")

                with patch(
                    "src.agents.shared.agent_base.AgentBase.__aexit__", new_callable=AsyncMock
                ) as mock_super_aexit:
                    await agent.__aexit__(Exception, test_exception, None)

                    # Should call cleanup on pydantic agent
                    mock_pydantic_agent.__aexit__.assert_called_once_with(Exception, test_exception, None)

                    # Should call parent cleanup
                    mock_super_aexit.assert_called_once_with(Exception, test_exception, None)


class TestAgentPerformanceConstraints:
    """Test agent performance under various constraints."""


class TestAgentRetryPatterns:
    """Test agent retry and error recovery patterns."""

    @pytest.mark.asyncio
    async def test_agent_retry_pattern(self, mcp_server, temp_docs_root):
        """Test agent retry pattern for transient failures."""
        # Provide basic environment to allow LLM initialization
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch("src.agents.shared.config.load_llm_config") as mock_load_llm:
                mock_llm = AsyncMock()
                mock_load_llm.return_value = mock_llm

                agent = SimpleAgent()

                async with mcp_server:
                    with patch("src.agents.simple_agent.agent.Agent") as mock_agent_class:
                        mock_pydantic_agent = AsyncMock()

                        # Simulate transient failure then success
                        call_count = 0

                        async def failing_run(query):
                            nonlocal call_count
                            call_count += 1
                            if call_count < 2:  # Fail first attempt
                                raise Exception("Transient failure")
                            else:  # Succeed on 2nd attempt
                                mock_result = Mock()
                                mock_result.data = SimpleAgentResponse(
                                    summary="Operation succeeded after retry",
                                    details={"operation": {"success": True}},
                                )
                                # Mock all_messages() method to return empty list for extract_mcp_tool_responses
                                mock_result.all_messages.return_value = []
                                return mock_result

                        mock_pydantic_agent.run = failing_run
                        mock_agent_class.return_value = mock_pydantic_agent

                        # Test retry logic
                        try:
                            await agent.run("Test operation")
                        except Exception:
                            # Expected first failure
                            result = await agent.run("Test operation")  # Should succeed
                            assert result is not None


class TestAgentEdgeCases:
    """Test agent behavior in edge cases."""

    @pytest.mark.asyncio
    async def test_agent_empty_query_handling(self, mcp_server, temp_docs_root):
        """Test agent handling of empty queries."""
        # Provide basic environment to allow LLM initialization
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch("src.agents.shared.config.load_llm_config") as mock_load_llm:
                mock_llm = AsyncMock()
                mock_load_llm.return_value = mock_llm

                agent = SimpleAgent()

                async with mcp_server:
                    with patch("src.agents.simple_agent.agent.Agent") as mock_agent_class:
                        mock_pydantic_agent = AsyncMock()
                        mock_result = Mock()
                        mock_result.data = SimpleAgentResponse(
                            summary="No operation performed - empty query", details={}
                        )
                        # Mock all_messages() method to return empty list for extract_mcp_tool_responses
                        mock_result.all_messages.return_value = []
                        mock_pydantic_agent.run.return_value = mock_result
                        mock_agent_class.return_value = mock_pydantic_agent

                        result = await agent.run("")
                        assert result is not None
                        assert "summary" in result

    @pytest.mark.asyncio
    async def test_agent_unicode_handling(self, mcp_server, temp_docs_root):
        """Test agent handling of Unicode characters."""
        # Provide basic environment to allow LLM initialization
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch("src.agents.shared.config.load_llm_config") as mock_load_llm:
                mock_llm = AsyncMock()
                mock_load_llm.return_value = mock_llm

                agent = SimpleAgent()
                unicode_query = "Create document 'ñoñó' with émojis 🚀"

                async with mcp_server:
                    with patch("src.agents.simple_agent.agent.Agent") as mock_agent_class:
                        mock_pydantic_agent = AsyncMock()
                        mock_result = Mock()
                        mock_result.data = SimpleAgentResponse(
                            summary="Processed Unicode query successfully",
                            details={"operation": {"success": True}},
                        )
                        # Mock all_messages() method to return empty list for extract_mcp_tool_responses
                        mock_result.all_messages.return_value = []
                        mock_pydantic_agent.run.return_value = mock_result
                        mock_agent_class.return_value = mock_pydantic_agent

                        result = await agent.run(unicode_query)
                        assert result is not None
                        assert "summary" in result
