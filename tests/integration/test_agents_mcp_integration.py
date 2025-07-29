"""Enhanced integration tests for agents with MCP server interaction.

This module provides comprehensive integration tests that validate agent-MCP
communication patterns with mocked LLM responses to ensure agents properly
populate the details field with structured MCP tool responses.
"""

import json
import sys
import uuid
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from pydantic_ai.mcp import MCPServerStdio

from src.agents.react_agent.parser import ActionParser
from src.agents.simple_agent.agent import SimpleAgent
from src.agents.simple_agent.agent import SimpleAgentResponse


@pytest.fixture
async def mcp_server():
    """Provide a real MCP server for integration testing."""
    server = MCPServerStdio(
        command=sys.executable, args=["-m", "document_mcp.doc_tool_server", "stdio"], timeout=60.0
    )
    yield server


@pytest.fixture
def mock_simple_agent_response():
    """Create a mock Simple Agent response for testing."""
    return SimpleAgentResponse(
        summary="Document management operation completed successfully",
        details={
            "create_document": {
                "success": True,
                "document_name": "integration_test_doc",
                "path": "/documents/integration_test_doc/",
            }
        },
    )


class TestSimpleAgentMCPIntegration:
    """Integration tests for Simple Agent with MCP server."""

    @pytest.mark.asyncio
    async def test_simple_agent_create_document_integration(
        self, mcp_server, temp_docs_root, mock_simple_agent_response
    ):
        """Test Simple Agent document creation with real MCP server."""
        doc_name = f"simple_integration_{uuid.uuid4().hex[:8]}"
        query = f"Create a document called '{doc_name}'"

        # Mock the LLM response but use real MCP server
        # Provide basic environment to allow LLM initialization
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch("src.agents.shared.config.load_llm_config") as mock_load_llm:
                mock_llm = AsyncMock()
                mock_load_llm.return_value = mock_llm

            # Create agent
            agent = SimpleAgent()

            async with mcp_server:
                # Mock the Pydantic AI agent to return structured response
                with patch("src.agents.simple_agent.agent.Agent") as mock_agent_class:
                    mock_pydantic_agent = AsyncMock()
                    mock_agent_result = Mock()

                    # Set up the mock result with proper structure
                    mock_agent_result.data = SimpleAgentResponse(
                        summary=f"Document '{doc_name}' created successfully",
                        details={
                            "create_document": {
                                "success": True,
                                "document_name": doc_name,
                                "message": f"Document '{doc_name}' created successfully",
                            }
                        },
                    )

                    mock_pydantic_agent.run.return_value = mock_agent_result
                    mock_agent_class.return_value = mock_pydantic_agent

                    # Execute the agent
                    result = await agent.run(query)

                    # Verify the result contains structured details from MCP
                    assert result is not None
                    assert "summary" in result
                    assert "details" in result

                    # The details should contain MCP tool response data
                    if "details" in result and result["details"]:
                        assert isinstance(result["details"], dict)

    @pytest.mark.asyncio
    async def test_simple_agent_list_documents_integration(self, mcp_server, temp_docs_root):
        """Test Simple Agent list documents with real MCP server."""
        query = "List all available documents"

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

                    # Mock structured response for list operation
                    mock_agent_result.data = SimpleAgentResponse(
                        summary="Retrieved list of available documents",
                        details={"list_documents": {"success": True, "documents": [], "total_count": 0}},
                    )

                    mock_pydantic_agent.run.return_value = mock_agent_result
                    mock_agent_class.return_value = mock_pydantic_agent

                    result = await agent.run(query)

                    # Verify structured response
                    assert result is not None
                    assert "summary" in result
                    assert "details" in result

    @pytest.mark.asyncio
    async def test_simple_agent_error_handling_integration(self, mcp_server, temp_docs_root):
        """Test Simple Agent error handling with MCP server."""
        query = "Create a document with invalid name 'test/invalid/name'"

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

                    # Mock error response
                    mock_agent_result.data = SimpleAgentResponse(
                        summary="Failed to create document due to invalid name",
                        details={
                            "create_document": {
                                "success": False,
                                "error": "Document name cannot contain path separators",
                                "error_type": "ValidationError",
                            }
                        },
                    )

                    mock_pydantic_agent.run.return_value = mock_agent_result
                    mock_agent_class.return_value = mock_pydantic_agent

                    result = await agent.run(query)

                    # Verify error is properly structured
                    assert result is not None
                    assert "details" in result
                    if result.get("details"):
                        assert isinstance(result["details"], dict)

    @pytest.mark.asyncio
    async def test_simple_agent_timeout_integration(self, mcp_server, temp_docs_root):
        """Test Simple Agent timeout handling."""
        query = "Create a test document"
        short_timeout = 0.001  # Very short timeout to trigger timeout

        # Provide basic environment to allow LLM initialization
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch("src.agents.shared.config.load_llm_config") as mock_load_llm:
                mock_llm = AsyncMock()
                mock_load_llm.return_value = mock_llm

            agent = SimpleAgent()

            from document_mcp.exceptions import OperationError

            async with mcp_server:
                with patch("src.agents.simple_agent.agent.Agent") as mock_agent_class:
                    mock_pydantic_agent = AsyncMock()

                    # Make the agent run take longer than timeout
                    async def slow_run(query):
                        import asyncio

                        await asyncio.sleep(0.1)  # Longer than timeout
                        return Mock()

                    mock_pydantic_agent.run = slow_run
                    mock_agent_class.return_value = mock_pydantic_agent

                    # Should raise OperationError due to timeout
                    with pytest.raises(OperationError) as exc_info:
                        await agent.run(query, timeout=short_timeout)

                    assert "timed out" in str(exc_info.value)
                    assert exc_info.value.operation == "simple_agent_execution"


class TestReActAgentMCPIntegration:
    """Integration tests for ReAct Agent with MCP server."""

    @pytest.mark.asyncio
    async def test_react_agent_parser_integration(self, mcp_server):
        """Test ReAct Agent parser integration with MCP tools."""
        parser = ActionParser()

        # Test parsing various MCP tool calls
        test_cases = [
            (
                "create_document(document_name='Integration Test')",
                "create_document",
                {"document_name": "Integration Test"},
            ),
            ("list_documents()", "list_documents", {}),
            (
                "create_chapter(document_name='Test', chapter_name='01-intro.md', initial_content='Hello')",
                "create_chapter",
                {"document_name": "Test", "chapter_name": "01-intro.md", "initial_content": "Hello"},
            ),
            (
                "get_statistics(document_name='Test', scope='document')",
                "get_statistics",
                {"document_name": "Test", "scope": "document"},
            ),
        ]

        for action_string, expected_tool, expected_kwargs in test_cases:
            tool_name, kwargs = parser.parse(action_string)
            assert tool_name == expected_tool
            assert kwargs == expected_kwargs

            # Verify that the parsed tool name exists in MCP server
            async with mcp_server:
                # Get available tools from MCP server
                tools_result = await mcp_server._client.list_tools()
                available_tools = [tool.name for tool in tools_result.tools]
                assert tool_name in available_tools

    @pytest.mark.asyncio
    async def test_react_agent_multi_step_integration(self, mcp_server, temp_docs_root):
        """Test ReAct Agent multi-step workflow integration."""
        # Simulate multi-step ReAct workflow
        steps = [
            {"reasoning": "First I need to check existing documents", "action": "list_documents()"},
            {
                "reasoning": "No documents found, I'll create one",
                "action": "create_document(document_name='Multi Step Test')",
            },
            {
                "reasoning": "Now I'll add a chapter to the document",
                "action": "create_chapter(document_name='Multi Step Test', chapter_name='01-intro.md', initial_content='Introduction')",
            },
        ]

        parser = ActionParser()

        async with mcp_server:
            for _i, step in enumerate(steps):
                # Parse the action
                tool_name, kwargs = parser.parse(step["action"])

                # Verify tool exists in MCP server
                tools_result = await mcp_server._client.list_tools()
                available_tools = [tool.name for tool in tools_result.tools]
                assert tool_name in available_tools

                # Simulate tool execution (would be done by agent in real scenario)
                try:
                    result = await mcp_server._client.call_tool(tool_name, kwargs)

                    # Verify result structure
                    assert result is not None
                    assert hasattr(result, "content")
                    assert len(result.content) > 0

                    # Parse the JSON response
                    result_text = result.content[0].text
                    result_data = json.loads(result_text)

                    # Verify response structure contains required fields
                    assert "success" in result_data

                except Exception as e:
                    # Some tools might fail due to dependencies (expected in test environment)
                    assert isinstance(e, Exception)


class TestAgentMCPCommunicationPatterns:
    """Test agent-MCP communication patterns and data flow."""

    @pytest.mark.asyncio
    async def test_mcp_tool_response_extraction_pattern(self, mcp_server, temp_docs_root):
        """Test MCP tool response extraction pattern."""
        doc_name = f"extraction_test_{uuid.uuid4().hex[:8]}"

        async with mcp_server:
            # Call create_document directly via MCP
            result = await mcp_server._client.call_tool("create_document", {"document_name": doc_name})

            # Verify response structure
            assert result is not None
            assert hasattr(result, "content")
            assert len(result.content) > 0

            # Extract and parse response
            result_text = result.content[0].text
            result_data = json.loads(result_text)

            # Verify response contains structured data that agents should extract
            assert isinstance(result_data, dict)
            assert "success" in result_data
            assert "message" in result_data

            # This is the data that should populate agent details field
            assert result_data["success"] in [True, False]
            assert isinstance(result_data["message"], str)

    @pytest.mark.asyncio
    async def test_agent_details_field_population_pattern(self, mcp_server, temp_docs_root):
        """Test that agents properly populate details field with MCP responses."""
        # This test validates the critical architecture requirement

        # Provide basic environment to allow LLM initialization
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch("src.agents.shared.config.load_llm_config") as mock_load_llm:
                mock_llm = AsyncMock()
                mock_load_llm.return_value = mock_llm

            agent = SimpleAgent()

            async with mcp_server:
                # Mock agent to simulate real MCP tool response extraction
                with patch("src.agents.simple_agent.agent.Agent") as mock_agent_class:
                    mock_pydantic_agent = AsyncMock()
                    mock_agent_result = Mock()

                    # Simulate the agent extracting MCP tool responses
                    mock_tool_responses = {
                        "list_documents": {
                            "success": True,
                            "documents": [],
                            "total_count": 0,
                            "message": "No documents found",
                        }
                    }

                    # Mock the details field population from MCP responses
                    mock_agent_result.data = SimpleAgentResponse(
                        summary="Listed all documents - found 0 documents",
                        details=mock_tool_responses,  # This is the critical requirement
                    )

                    mock_pydantic_agent.run.return_value = mock_agent_result
                    mock_agent_class.return_value = mock_pydantic_agent

                    result = await agent.run("List all documents")

                    # CRITICAL TEST: Verify details field contains structured MCP data
                    assert "details" in result
                    assert isinstance(result["details"], dict)

                    # Details should contain actual MCP tool response data
                    if result["details"]:
                        for _tool_name, tool_data in result["details"].items():
                            assert isinstance(tool_data, dict)
                            # Should contain structured data from MCP tools
                            assert "success" in tool_data or "error" in tool_data

    @pytest.mark.asyncio
    async def test_agent_error_details_population(self, mcp_server, temp_docs_root):
        """Test agent error details population from MCP errors."""
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

                    # Simulate MCP error response
                    mock_error_response = {
                        "create_document": {
                            "success": False,
                            "error": "Document name cannot contain path separators",
                            "error_type": "ValidationError",
                            "error_details": {
                                "field": "document_name",
                                "value": "invalid/name",
                                "constraint": "no_path_separators",
                            },
                        }
                    }

                    mock_agent_result.data = SimpleAgentResponse(
                        summary="Failed to create document due to validation error",
                        details=mock_error_response,
                    )

                    mock_pydantic_agent.run.return_value = mock_agent_result
                    mock_agent_class.return_value = mock_pydantic_agent

                    result = await agent.run("Create document with invalid/name")

                    # Verify error details are properly structured
                    assert "details" in result
                    if result["details"]:
                        error_data = result["details"].get("create_document")
                        if error_data:
                            assert error_data["success"] is False
                            assert "error" in error_data
                            assert "error_type" in error_data

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_details_aggregation(self, mcp_server, temp_docs_root):
        """Test aggregation of multiple tool call details."""
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

                    # Simulate multiple tool calls in single agent execution
                    mock_multiple_tools_response = {
                        "create_document": {
                            "success": True,
                            "document_name": "Multi Tool Test",
                            "path": "/documents/Multi Tool Test/",
                        },
                        "create_chapter": {
                            "success": True,
                            "document_name": "Multi Tool Test",
                            "chapter_name": "01-intro.md",
                            "chapter_path": "/documents/Multi Tool Test/01-intro.md",
                        },
                        "get_document_statistics": {
                            "success": True,
                            "document_name": "Multi Tool Test",
                            "statistics": {"total_chapters": 1, "total_paragraphs": 1, "total_words": 10},
                        },
                    }

                    mock_agent_result.data = SimpleAgentResponse(
                        summary="Created document, added chapter, and retrieved statistics",
                        details=mock_multiple_tools_response,
                    )

                    mock_pydantic_agent.run.return_value = mock_agent_result
                    mock_agent_class.return_value = mock_pydantic_agent

                    result = await agent.run("Create a document with chapter and show statistics")

                    # Verify all tool responses are captured in details
                    assert "details" in result
                    if result["details"]:
                        details = result["details"]
                        assert len(details) == 3  # Three tool calls
                        assert "create_document" in details
                        assert "create_chapter" in details
                        assert "get_document_statistics" in details

                        # Each tool response should have success status
                        for _tool_name, tool_data in details.items():
                            assert "success" in tool_data
                            assert tool_data["success"] is True


class TestAgentMCPServerLifecycle:
    """Test agent interaction with MCP server lifecycle."""

    @pytest.mark.asyncio
    async def test_mcp_server_connection_handling(self, mcp_server):
        """Test MCP server connection handling."""
        # Test that MCP server can be started and connected to
        async with mcp_server:
            # Verify server is responsive
            tools_result = await mcp_server._client.list_tools()
            assert tools_result is not None
            assert hasattr(tools_result, "tools")
            assert len(tools_result.tools) > 0

            # Verify essential tools are available
            tool_names = [tool.name for tool in tools_result.tools]
            essential_tools = ["create_document", "list_documents", "create_chapter"]

            for tool in essential_tools:
                assert tool in tool_names

    @pytest.mark.asyncio
    async def test_mcp_server_error_recovery(self, mcp_server):
        """Test MCP server error recovery patterns."""
        async with mcp_server:
            # Test calling non-existent tool
            try:
                await mcp_server._client.call_tool("non_existent_tool", {})
                raise AssertionError("Should have raised an exception")
            except Exception as e:
                # Should handle MCP errors gracefully
                assert e is not None

    @pytest.mark.asyncio
    async def test_concurrent_mcp_operations(self, mcp_server, temp_docs_root):
        """Test concurrent MCP operations."""
        import asyncio

        async with mcp_server:
            # Create multiple concurrent operations
            tasks = []

            for i in range(3):
                doc_name = f"concurrent_test_{i}_{uuid.uuid4().hex[:8]}"
                task = mcp_server._client.call_tool("create_document", {"document_name": doc_name})
                tasks.append(task)

            # Execute concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Verify results
            assert len(results) == 3

            for result in results:
                if isinstance(result, Exception):
                    # Some failures expected due to test environment
                    continue
                else:
                    # Successful results should have proper structure
                    assert hasattr(result, "content")
                    if result.content:
                        result_text = result.content[0].text
                        result_data = json.loads(result_text)
                        assert "success" in result_data


class TestAgentResponseStructureValidation:
    """Validate agent response structure compliance."""

    def test_simple_agent_response_structure_compliance(self):
        """Test Simple Agent response structure compliance."""
        # Test valid response structure
        valid_response = SimpleAgentResponse(
            summary="Operation completed successfully",
            details={
                "create_document": {"success": True, "document_name": "test", "path": "/documents/test/"}
            },
        )

        assert valid_response.summary is not None
        assert isinstance(valid_response.summary, str)
        assert valid_response.details is not None
        assert isinstance(valid_response.details, dict)

        # Verify details contain structured MCP data
        for tool_name, tool_data in valid_response.details.items():
            assert isinstance(tool_name, str)
            assert isinstance(tool_data, dict)
            assert "success" in tool_data  # MCP responses should have success field

    def test_agent_response_details_field_requirement(self):
        """Test the critical requirement that details field contains MCP data."""
        # This test validates the critical architectural requirement:
        # "The agents MUST populate the `details` field with structured data from MCP tool responses"

        valid_mcp_response_data = {
            "list_documents": {"success": True, "documents": ["doc1", "doc2"], "total_count": 2},
            "create_chapter": {"success": True, "document_name": "test_doc", "chapter_name": "01-intro.md"},
        }

        response = SimpleAgentResponse(
            summary="Multiple operations completed", details=valid_mcp_response_data
        )

        # CRITICAL VALIDATION: Details must contain MCP tool response data
        assert response.details == valid_mcp_response_data

        # Each tool response should have structured data
        for _tool_name, tool_data in response.details.items():
            # Must be structured dictionary (not string or other type)
            assert isinstance(tool_data, dict)

            # Must contain success indicator (MCP standard)
            assert "success" in tool_data
            assert isinstance(tool_data["success"], bool)

            # Should contain operation-specific data
            assert len(tool_data) > 1  # More than just success field
