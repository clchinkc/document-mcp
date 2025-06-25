from src.agents.react_agent.main import REACT_SYSTEM_PROMPT


class TestReActSystemPrompt:
    """Test suite for the ReAct system prompt."""

    def test_prompt_exists_and_not_empty(self):
        """Test that the system prompt exists and has content."""
        assert REACT_SYSTEM_PROMPT is not None
        assert len(REACT_SYSTEM_PROMPT.strip()) > 0
        assert len(REACT_SYSTEM_PROMPT) > 1000  # Should be substantial

    def test_prompt_contains_essential_structure(self):
        """Test that the prompt contains all essential sections and concepts."""
        # Core sections
        required_sections = [
            "ReAct Process Overview",
            "Available Tools",
            "Example ReAct Sequence",
            "Important Guidelines",
        ]

        for section in required_sections:
            assert section in REACT_SYSTEM_PROMPT, f"Missing section: {section}"

        # Core concepts
        core_concepts = ["Thought", "Action", "Observation"]
        for concept in core_concepts:
            assert concept in REACT_SYSTEM_PROMPT, f"Missing concept: {concept}"

        # JSON structure elements
        json_elements = ['"thought"', '"action"', 'tool_name(param1="value1"']
        for element in json_elements:
            assert element in REACT_SYSTEM_PROMPT, f"Missing JSON element: {element}"

    def test_prompt_includes_complete_example_and_guidelines(self):
        """Test that the prompt includes a complete example sequence and operational guidelines."""
        # Complete example elements
        example_elements = [
            "Project Alpha",
            "Step 1:",
            "Step 2:",
            "Step 3:",
            "Observation:",
            '"action": null',
        ]

        for element in example_elements:
            assert element in REACT_SYSTEM_PROMPT, f"Missing example element: {element}"

        # Operational guidelines
        guidelines = [
            "One Action Per Step",
            "Valid JSON",
            "Parameter Formatting",
            "Chapter Names",
            "Error Handling",
            "Completion",
        ]

        for guideline in guidelines:
            assert guideline in REACT_SYSTEM_PROMPT, f"Missing guideline: {guideline}"

    def test_prompt_mentions_termination_condition(self):
        """Test that the prompt explains how to terminate the ReAct loop."""
        termination_phrases = [
            'action" to null',
            "task is now complete",
            "omit it entirely",
        ]

        # At least one termination phrase should be present
        found_termination = any(
            phrase in REACT_SYSTEM_PROMPT for phrase in termination_phrases
        )
        assert found_termination, "No clear termination condition found in prompt"

    def test_prompt_includes_error_handling_guidance(self):
        """Test that the prompt provides guidance on handling tool errors."""
        error_handling_concepts = ["error", "adjust", "Error Handling"]

        for concept in error_handling_concepts:
            assert (
                concept in REACT_SYSTEM_PROMPT
            ), f"Missing error handling concept: {concept}"

    def test_prompt_emphasizes_sequential_processing(self):
        """Test that the prompt emphasizes step-by-step processing."""
        sequential_concepts = ["sequence", "steps", "one tool", "One Action Per Step"]

        for concept in sequential_concepts:
            assert (
                concept in REACT_SYSTEM_PROMPT
            ), f"Missing sequential concept: {concept}"
