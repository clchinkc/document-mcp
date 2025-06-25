#!/usr/bin/env python3
"""
Test suite for the ReAct execution loop implementation.
Tests the parse_action_string function and validates loop structure.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.agents.react_agent.main import ReActStep, parse_action_string, run_react_loop


class TestParseActionString:
    """Test the action string parsing functionality."""

    def test_simple_function_no_args(self):
        """Test parsing function with no arguments."""
        tool_name, kwargs = parse_action_string("list_documents()")
        assert tool_name == "list_documents"
        assert kwargs == {}

    def test_single_string_argument(self):
        """Test parsing function with single string argument."""
        tool_name, kwargs = parse_action_string(
            'create_document(document_name="My Book")'
        )
        assert tool_name == "create_document"
        assert kwargs == {"document_name": "My Book"}

    def test_multiple_string_arguments(self):
        """Test parsing function with multiple string arguments."""
        tool_name, kwargs = parse_action_string(
            'create_chapter(document_name="My Book", chapter_name="01-intro.md", initial_content="# Introduction")'
        )
        assert tool_name == "create_chapter"
        assert kwargs == {
            "document_name": "My Book",
            "chapter_name": "01-intro.md",
            "initial_content": "# Introduction",
        }

    def test_boolean_arguments(self):
        """Test parsing function with boolean arguments."""
        tool_name, kwargs = parse_action_string(
            'find_text_in_document(document_name="My Book", query="test", case_sensitive=true)'
        )
        assert tool_name == "find_text_in_document"
        assert kwargs == {
            "document_name": "My Book",
            "query": "test",
            "case_sensitive": True,
        }

    def test_integer_arguments(self):
        """Test parsing function with integer arguments."""
        tool_name, kwargs = parse_action_string(
            'read_paragraph_content(document_name="My Book", chapter_name="01-intro.md", paragraph_index_in_chapter=0)'
        )
        assert tool_name == "read_paragraph_content"
        assert kwargs == {
            "document_name": "My Book",
            "chapter_name": "01-intro.md",
            "paragraph_index_in_chapter": 0,
        }

    def test_none_arguments(self):
        """Test parsing function with None arguments."""
        tool_name, kwargs = parse_action_string(
            'test_function(param1="value", param2=none)'
        )
        assert tool_name == "test_function"
        assert kwargs == {"param1": "value", "param2": None}

    def test_single_quotes(self):
        """Test parsing function with single quotes."""
        tool_name, kwargs = parse_action_string(
            "create_document(document_name='My Book')"
        )
        assert tool_name == "create_document"
        assert kwargs == {"document_name": "My Book"}

    def test_quotes_with_commas(self):
        """Test parsing strings that contain commas."""
        tool_name, kwargs = parse_action_string(
            'write_chapter_content(document_name="My Book", chapter_name="01-intro.md", new_content="Hello, world! This is a test.")'
        )
        assert tool_name == "write_chapter_content"
        assert kwargs == {
            "document_name": "My Book",
            "chapter_name": "01-intro.md",
            "new_content": "Hello, world! This is a test.",
        }

    def test_quotes_with_equals(self):
        """Test parsing strings that contain equals signs."""
        tool_name, kwargs = parse_action_string(
            'append_paragraph_to_chapter(document_name="My Book", chapter_name="01-intro.md", paragraph_content="E=mc²")'
        )
        assert tool_name == "append_paragraph_to_chapter"
        assert kwargs == {
            "document_name": "My Book",
            "chapter_name": "01-intro.md",
            "paragraph_content": "E=mc²",
        }

    def test_invalid_format(self):
        """Test parsing invalid action format."""
        with pytest.raises(ValueError, match="Invalid action format"):
            parse_action_string("invalid_format")

    def test_empty_string(self):
        """Test parsing empty string."""
        with pytest.raises(ValueError, match="Invalid action format"):
            parse_action_string("")

    def test_malformed_function(self):
        """Test parsing malformed function call."""
        with pytest.raises(ValueError, match="Invalid action format"):
            parse_action_string("create_document(")

    def test_complex_nested_quotes(self):
        """Test parsing complex nested quotes scenario."""
        tool_name, kwargs = parse_action_string(
            'replace_text_in_chapter(document_name="My Book", chapter_name="01-intro.md", text_to_find="\\"old\\"", replacement_text="\\"new\\"")'
        )
        assert tool_name == "replace_text_in_chapter"
        assert kwargs == {
            "document_name": "My Book",
            "chapter_name": "01-intro.md",
            "text_to_find": '"old"',
            "replacement_text": '"new"',
        }

    def test_action_with_special_characters(self):
        """Test parsing action with special characters."""
        tool_name, kwargs = parse_action_string(
            'create_chapter(document_name="Test & More", content="Line 1\\nLine 2")'
        )
        assert tool_name == "create_chapter"
        assert kwargs["document_name"] == "Test & More"
        assert (
            kwargs["content"] == "Line 1\\nLine 2"
        )  # Parser preserves the literal string

    def test_action_with_boolean_and_integer_params(self):
        """Test parsing action with mixed parameter types."""
        tool_name, kwargs = parse_action_string(
            'search_text(document_name="Doc", query="test", case_sensitive=true, max_results=10)'
        )
        assert tool_name == "search_text"
        assert kwargs["document_name"] == "Doc"
        assert kwargs["query"] == "test"
        assert kwargs["case_sensitive"] is True
        assert kwargs["max_results"] == 10

    def test_malformed_action_comprehensive_errors(self):
        """Test that malformed actions raise appropriate errors."""
        with pytest.raises(ValueError, match="Invalid action format"):
            parse_action_string("malformed_action(")

        with pytest.raises(ValueError, match="Invalid action format"):
            parse_action_string("not_a_function_call")

        with pytest.raises(ValueError, match="Invalid action format"):
            parse_action_string("")


class TestRunReactLoop:
    """Test the main ReAct execution loop."""

    @pytest.mark.asyncio
    async def test_loop_basic_structure(self):
        """Test that the loop returns properly structured history."""

        # Mock the entire run_react_loop function call chain
        with patch(
            "src.agents.react_agent.main.load_llm_config"
        ) as mock_load_config, patch(
            "src.agents.react_agent.main.MCPServerSSE"
        ) as mock_mcp_server_class, patch(
            "src.agents.react_agent.main.get_cached_agent"
        ) as mock_get_cached_agent:

            # Setup mocks - make load_llm_config async
            mock_model = Mock()
            mock_load_config = AsyncMock(return_value=mock_model)

            # Mock the MCP server instance with proper async context manager
            mock_mcp_server_instance = Mock()
            mock_mcp_server_instance.__aenter__ = AsyncMock(
                return_value=mock_mcp_server_instance
            )
            mock_mcp_server_instance.__aexit__ = AsyncMock(return_value=False)
            mock_mcp_server_class.return_value = mock_mcp_server_instance

            # Mock the cached agent
            mock_agent = AsyncMock()
            mock_get_cached_agent.return_value = mock_agent

            # Mock agent run to return completion
            mock_result = Mock()
            mock_result.output = ReActStep(
                thought="Task completed successfully", action=None
            )
            mock_agent.run = AsyncMock(return_value=mock_result)

            # Run the function
            history = await run_react_loop("Test query", max_steps=5)

            # Verify structure
            assert isinstance(history, list)
            assert len(history) == 1
            assert "step" in history[0]
            assert "thought" in history[0]
            assert "action" in history[0]
            assert "observation" in history[0]

    @pytest.mark.asyncio
    async def test_max_steps_limit(self):
        """Test that the loop respects the max steps limit."""

        with patch(
            "src.agents.react_agent.main.load_llm_config"
        ) as mock_load_config, patch(
            "src.agents.react_agent.main.MCPServerSSE"
        ) as mock_mcp_server_class, patch(
            "src.agents.react_agent.main.get_cached_agent"
        ) as mock_get_cached_agent, patch(
            "src.agents.react_agent.main.execute_mcp_tool_directly"
        ) as mock_execute_tool:

            # Setup mocks - make load_llm_config async
            mock_model = Mock()
            mock_load_config = AsyncMock(return_value=mock_model)

            # Mock the MCP server instance with proper async context manager
            mock_mcp_server_instance = Mock()
            mock_mcp_server_instance.__aenter__ = AsyncMock(
                return_value=mock_mcp_server_instance
            )
            mock_mcp_server_instance.__aexit__ = AsyncMock(return_value=False)
            mock_mcp_server_class.return_value = mock_mcp_server_instance

            # Mock the cached agent
            mock_agent = AsyncMock()
            mock_get_cached_agent.return_value = mock_agent

            # Mock agent to never complete (always return action)
            mock_result = Mock()
            mock_result.output = ReActStep(
                thought="Still working on it", action="list_documents()"
            )
            mock_agent.run = AsyncMock(return_value=mock_result)
            mock_execute_tool.return_value = '{"success": true}'

            # Run with max_steps=3
            history = await run_react_loop("Test query", max_steps=3)

            # Should have exactly 4 entries (3 attempts + 1 timeout entry)
            assert len(history) == 4
            assert history[-1]["observation"] == "Task incomplete after 3 steps"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in the loop."""
        with patch("src.agents.react_agent.main.load_llm_config") as mock_load_config:
            # Mock load_llm_config to raise an exception - make it async
            async def mock_load_llm_config_side_effect():
                raise Exception("API key not found")

            mock_load_config.side_effect = mock_load_llm_config_side_effect

            # The exception is raised directly from load_llm_config (before try-catch)
            with pytest.raises(Exception, match="API key not found"):
                await run_react_loop("Test query")

    @pytest.mark.asyncio
    async def test_history_format_consistency(self):
        """Test that history entries have consistent format."""

        with patch(
            "src.agents.react_agent.main.load_llm_config"
        ) as mock_load_config, patch(
            "src.agents.react_agent.main.MCPServerSSE"
        ) as mock_mcp_server_class, patch(
            "src.agents.react_agent.main.get_cached_agent"
        ) as mock_get_cached_agent, patch(
            "src.agents.react_agent.main.execute_mcp_tool_directly"
        ) as mock_execute_tool:

            # Setup mocks - make load_llm_config async
            mock_model = Mock()
            mock_load_config = AsyncMock(return_value=mock_model)

            # Mock the MCP server instance with proper async context manager
            mock_mcp_server_instance = Mock()
            mock_mcp_server_instance.__aenter__ = AsyncMock(
                return_value=mock_mcp_server_instance
            )
            mock_mcp_server_instance.__aexit__ = AsyncMock(return_value=False)
            mock_mcp_server_class.return_value = mock_mcp_server_instance

            # Mock the cached agent
            mock_agent = AsyncMock()
            mock_get_cached_agent.return_value = mock_agent

            # Mock two steps: action then completion
            call_count = 0

            async def mock_run_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    result = Mock()
                    result.output = ReActStep(
                        thought="First step", action="list_documents()"
                    )
                    return result
                else:
                    result = Mock()
                    result.output = ReActStep(thought="Task complete", action=None)
                    return result

            mock_agent.run = AsyncMock(side_effect=mock_run_side_effect)
            mock_execute_tool.return_value = '{"success": true}'

            # Run the function
            history = await run_react_loop("Test query")

            # Verify all entries have required keys
            for entry in history:
                assert "step" in entry
                assert "thought" in entry
                assert "action" in entry
                assert "observation" in entry
                assert isinstance(entry["step"], int)
                assert isinstance(entry["thought"], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
