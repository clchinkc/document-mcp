"""Unit tests for agent prompt validation.

This module tests core agent prompt functionality and validation
to ensure prompts are effective across agent types.
"""

from src.agents.react_agent.prompts import get_react_system_prompt
from src.agents.simple_agent.prompts import get_simple_agent_system_prompt


class TestPromptStructureValidation:
    """Test prompt structure validation."""

    def test_simple_agent_prompt_structure(self):
        """Test Simple Agent prompt structure."""
        prompt = get_simple_agent_system_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert len(prompt.strip()) == len(prompt)

        prompt_lower = prompt.lower()
        assert any(keyword in prompt_lower for keyword in ["document", "tool", "response"])

    def test_react_agent_prompt_structure(self):
        """Test ReAct Agent prompt structure."""
        prompt = get_react_system_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert len(prompt.strip()) == len(prompt)

        prompt_lower = prompt.lower()
        react_keywords = ["reason", "think", "action", "step"]
        assert any(keyword in prompt_lower for keyword in react_keywords)

    def test_prompt_consistency_between_agents(self):
        """Test consistency between agent prompts."""
        simple_prompt = get_simple_agent_system_prompt()
        react_prompt = get_react_system_prompt()

        assert len(simple_prompt) > 50
        assert len(react_prompt) > 50
        assert simple_prompt != react_prompt

        # Both should mention common concepts
        for concept in ["document", "tool"]:
            assert concept.lower() in simple_prompt.lower()
            assert concept.lower() in react_prompt.lower()


class TestPromptContentValidation:
    """Test prompt content validation."""

    def test_simple_agent_prompt_completeness(self):
        """Test Simple Agent prompt completeness."""
        prompt = get_simple_agent_system_prompt()

        # Should contain key elements
        assert any(indicator in prompt.lower() for indicator in ["you are", "assistant", "agent"])
        assert any(indicator in prompt.lower() for indicator in ["task", "manage", "help"])
        assert any(indicator in prompt.lower() for indicator in ["format", "response", "output"])

    def test_react_agent_prompt_completeness(self):
        """Test ReAct Agent prompt completeness."""
        prompt = get_react_system_prompt()

        # Should contain ReAct elements
        react_methodology = ["reason", "action", "think", "step"]
        assert sum(keyword in prompt.lower() for keyword in react_methodology) >= 2

        # Should contain workflow elements
        assert any(indicator in prompt.lower() for indicator in ["step", "process", "workflow"])

    def test_prompt_tool_integration_instructions(self):
        """Test prompt tool integration instructions."""
        for prompt in [get_simple_agent_system_prompt(), get_react_system_prompt()]:
            # Should mention tools
            tool_mentions = ["tool", "function", "call", "use"]
            assert sum(mention in prompt.lower() for mention in tool_mentions) >= 2


class TestPromptOptimizationValidation:
    """Test prompt optimization validation."""

    def test_prompt_length_optimization(self):
        """Test prompt length is reasonable."""
        for agent_type, prompt in [
            ("simple", get_simple_agent_system_prompt()),
            ("react", get_react_system_prompt()),
        ]:
            assert len(prompt) > 200, f"{agent_type} prompt too short"
            assert len(prompt) < 20000, f"{agent_type} prompt too long"

            word_count = len(prompt.split())
            assert 50 < word_count < 5000, f"{agent_type} prompt word count: {word_count}"

    def test_prompt_instruction_clarity(self):
        """Test prompt instruction clarity."""
        for prompt in [get_simple_agent_system_prompt(), get_react_system_prompt()]:
            # Should contain clear directives
            directive_words = ["should", "must", "will", "always", "never"]
            directive_count = sum(directive in prompt.lower() for directive in directive_words)
            assert directive_count > 0, "Prompt lacks clear directives"


class TestPromptConsistencyValidation:
    """Test prompt consistency validation."""

    def test_prompt_terminology_consistency(self):
        """Test prompt terminology consistency."""
        simple_prompt = get_simple_agent_system_prompt()
        react_prompt = get_react_system_prompt()

        # Both should use consistent core terminology
        core_terms = {
            "document_management": ["document", "manage"],
            "tool_usage": ["tool", "function"],
            "response_format": ["response", "format"],
        }

        for category, terms in core_terms.items():
            simple_has_terms = any(term in simple_prompt.lower() for term in terms)
            react_has_terms = any(term in react_prompt.lower() for term in terms)

            assert simple_has_terms, f"Simple prompt missing {category} terminology"
            assert react_has_terms, f"ReAct prompt missing {category} terminology"


class TestPromptValidationFramework:
    """Test prompt validation framework."""

    def test_prompt_coverage_validation(self):
        """Test prompt functional area coverage."""
        functional_areas = {
            "task_definition": ["task", "goal"],
            "tool_usage": ["tool", "function"],
            "output_format": ["response", "format"],
            "behavior_guidance": ["should", "must"],
        }

        for agent_type, prompt in [
            ("simple", get_simple_agent_system_prompt()),
            ("react", get_react_system_prompt()),
        ]:
            coverage_count = 0

            for _area, keywords in functional_areas.items():
                if any(keyword in prompt.lower() for keyword in keywords):
                    coverage_count += 1

            coverage_ratio = coverage_count / len(functional_areas)
            assert coverage_ratio >= 0.5, (
                f"{agent_type} prompt covers {coverage_ratio:.2%} of functional areas"
            )

    def test_prompt_validation_framework(self):
        """Test basic prompt validation framework."""

        def validate_prompt(prompt: str) -> bool:
            """Basic prompt validation."""
            return (
                200 < len(prompt) < 20000
                and any(phrase in prompt.lower() for phrase in ["you are", "assistant"])
                and any(phrase in prompt.lower() for phrase in ["tool", "function"])
                and any(phrase in prompt.lower() for phrase in ["response", "format"])
            )

        # Both prompts should pass validation
        assert validate_prompt(get_simple_agent_system_prompt())
        assert validate_prompt(get_react_system_prompt())
