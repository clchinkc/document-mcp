#!/usr/bin/env python3
"""
Test suite for the ReAct execution loop implementation.
Tests the parse_action_string function and validates loop structure.
"""

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
        tool_name, kwargs = parse_action_string(
            'test_function(param1="value", param2=none)'
        )
        assert tool_name == "test_function"
        assert kwargs == {"param1": "value", "param2": None}

    def test_single_quotes(self):
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
        with pytest.raises(ValueError, match="Invalid action format"):
            parse_action_string("invalid_format")

    def test_empty_string(self):
        with pytest.raises(ValueError, match="Invalid action format"):
            parse_action_string("")

    def test_malformed_function(self):
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
    async def test_loop_basic_structure(self, mocker):
        """Test that the loop returns properly structured history."""
        # Run the actual function to test real behavior
        history = await run_react_loop("List all documents", max_steps=5)

        # Verify we got some history
        assert isinstance(history, list)
        assert len(history) > 0, "Should get at least one step"

        # Enforce no connection errors: history entries must not contain 'error' as a key only
        assert all(
            not (isinstance(step, dict) and 'error' in step)
            for step in history
        ), f"Encountered connection or execution errors in history: {history}"

    @pytest.mark.asyncio
    async def test_max_steps_limit(self, mocker):
        """Test that the loop respects max_steps parameter."""
        # Run with max_steps=2
        history = await run_react_loop("Never ending task", max_steps=2)

        # Verify we got some history
        assert isinstance(history, list)
        assert len(history) > 0, "Should get at least one step"

        # Enforce no connection errors in history
        assert all(
            not (isinstance(step, dict) and 'error' in step)
            for step in history
        ), f"Encountered connection or execution errors in history: {history}"
        
        # Check step numbering
        for i, step in enumerate(history):
            assert step["step"] == i + 1, f"Step {i} should have step number {i + 1}"

    @pytest.mark.asyncio
    async def test_error_handling(self, mocker, mock_environment_operations, mock_file_operations):
        """Test that the loop correctly handles and logs errors during agent execution."""
        mock_environment_operations.set_api_environment(api_type="openai")
        mock_file_operations.mock_print()

        # Mock get_cached_agent to return an agent that will fail
        mock_agent_instance = mocker.AsyncMock()
        mock_agent_instance.run.side_effect = Exception("Agent execution failed")

        mocker.patch(
            "src.agents.react_agent.main.get_cached_agent",
            return_value=mock_agent_instance
        )

        # Mock MCPServer so it doesn't start a real process
        mock_mcp_server = mocker.patch("src.agents.react_agent.main.MCPServerStdio")
        
        history = await run_react_loop("Simulate agent error", max_steps=5)

        assert len(history) == 1
        assert "Agent execution failed" in history[0]['observation']

    @pytest.mark.asyncio
    async def test_history_format_consistency(self, mocker):
        """Test the structure of the history object for a single step."""
        # Run the actual function to test real behavior
        history = await run_react_loop("Test consistency", max_steps=1)

        # Verify we got some history
        assert isinstance(history, list)
        assert len(history) > 0, "Should get at least one step"

        # Enforce that no unexpected errors exist
        assert all(
            not (isinstance(entry, dict) and 'error' in entry)
            for entry in history
        ), f"Encountered errors in history entries: {history}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
