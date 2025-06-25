import pytest
from pydantic import ValidationError

from src.agents.react_agent.main import ReActStep


class TestReActStep:
    """Test suite for the ReActStep data model."""

    def test_react_step_with_thought_and_action(self):
        """Test creating a ReActStep with both thought and action."""
        step = ReActStep(
            thought="I need to create a new document for the user's request.",
            action='create_document(document_name="My Book")',
        )

        assert step.thought == "I need to create a new document for the user's request."
        assert step.action == 'create_document(document_name="My Book")'

    def test_react_step_with_thought_only(self):
        """Test creating a ReActStep with only thought (action is None)."""
        step = ReActStep(thought="The task has been completed successfully.")

        assert step.thought == "The task has been completed successfully."
        assert step.action is None

    def test_react_step_empty_thought_validation(self):
        """Test that empty thought string is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ReActStep(thought="")

        assert "String should have at least 1 character" in str(exc_info.value)

    def test_react_step_missing_thought_validation(self):
        """Test that missing thought field is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ReActStep(action='create_document(document_name="Test")')

        assert "Field required" in str(exc_info.value)

    def test_react_step_complex_action_string(self):
        """Test ReActStep with complex action string containing multiple parameters."""
        complex_action = 'write_chapter_content(document_name="Project Alpha", chapter_name="01-introduction.md", new_content="# Introduction\\nThis is the start of our project.")'

        step = ReActStep(
            thought="I need to write the introduction chapter with proper markdown formatting.",
            action=complex_action,
        )

        assert (
            step.thought
            == "I need to write the introduction chapter with proper markdown formatting."
        )
        assert step.action == complex_action

    def test_react_step_action_with_special_characters(self):
        """Test ReActStep with action containing special characters and quotes."""
        action_with_quotes = 'replace_text_in_chapter(document_name="My Book", chapter_name="chapter1.md", text_to_find="old \\"term\\"", replacement_text="new \\"term\\"")'

        step = ReActStep(
            thought="I need to replace quoted text in the chapter.",
            action=action_with_quotes,
        )

        assert step.action == action_with_quotes

    def test_react_step_long_thought_content(self):
        """Test ReActStep with very long thought content."""
        long_thought = (
            "This is a very detailed reasoning process where I need to carefully consider multiple steps. "
            * 10
        )

        step = ReActStep(
            thought=long_thought,
            action='get_document_statistics(document_name="Large Document")',
        )

        assert len(step.thought) > 500  # Ensure it's actually long
        assert step.thought == long_thought

    def test_react_step_serialization(self):
        """Test that ReActStep can be properly serialized to dict and JSON."""
        step = ReActStep(
            thought="Testing serialization functionality.", action="list_documents()"
        )

        # Test dict conversion
        step_dict = step.model_dump()
        expected_dict = {
            "thought": "Testing serialization functionality.",
            "action": "list_documents()",
        }
        assert step_dict == expected_dict

        # Test JSON serialization
        step_json = step.model_dump_json()
        assert '"thought":"Testing serialization functionality."' in step_json
        assert '"action":"list_documents()"' in step_json

    def test_react_step_deserialization(self):
        """Test that ReActStep can be properly deserialized from dict."""
        step_data = {
            "thought": "Deserializing from dictionary data.",
            "action": 'find_text_in_document(document_name="Search Target", search_text="keyword")',
        }

        step = ReActStep(**step_data)

        assert step.thought == step_data["thought"]
        assert step.action == step_data["action"]

    def test_react_step_termination_with_none_action(self):
        """Test that ReActStep correctly handles action=None for termination."""
        step = ReActStep(thought="Task is complete", action=None)
        assert step.action is None
        assert step.thought == "Task is complete"

    def test_react_step_termination_with_omitted_action(self):
        """Test that ReActStep correctly defaults to None when action is omitted."""
        step = ReActStep(thought="Task is complete")
        assert step.action is None
        assert step.thought == "Task is complete"

    def test_react_step_termination_with_valid_action(self):
        """Test that ReActStep correctly handles valid actions (not termination)."""
        step = ReActStep(
            thought="Creating document", action='create_document(document_name="Test")'
        )
        assert step.action is not None
        assert step.action == 'create_document(document_name="Test")'

    def test_react_step_non_termination_with_various_actions(self):
        """Test that valid actions don't trigger termination."""
        actions = [
            'create_document(document_name="Test")',
            "list_documents()",
            'read_chapter_content(document_name="Book", chapter_name="01-intro.md")',
            'delete_document(document_name="Old")',
        ]

        for action in actions:
            step = ReActStep(thought="Executing action", action=action)
            assert step.action is not None
            assert step.action == action
