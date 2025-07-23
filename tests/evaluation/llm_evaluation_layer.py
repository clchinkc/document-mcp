"""Simple LLM evaluation layer for tests only.

This module provides LLM evaluation capabilities that work alongside
traditional performance metrics in the test layer, without affecting
the core agent implementations.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass

from src.agents.shared.config import AgentSettings
from src.agents.shared.performance_metrics import AgentPerformanceMetrics


@dataclass
class LLMEvaluation:
    """Simple LLM evaluation result for test purposes."""

    score: float = 0.0  # 0-1 scale
    feedback: str = ""
    success: bool = False
    duration: float = 0.0


class TestLLMEvaluator:
    """Simple LLM evaluator for test environments only."""

    def __init__(self):
        """Initialize the LLM evaluation layer."""
        self.client = None
        self.model = "gpt-4o-mini"  # Cost-effective for testing
        self.logger = logging.getLogger(__name__)

        if self.enabled:
            self._try_initialize()

    @property
    def enabled(self) -> bool:
        """Check if LLM evaluation is enabled (dynamically check environment)."""
        return os.environ.get("ENABLE_LLM_EVALUATION", "true").lower() == "true"

    def _try_initialize(self):
        """Try to initialize LLM client, disable if it fails."""
        try:
            config = AgentSettings()

            if config.active_provider == "openai":
                import openai

                self.client = openai.AsyncOpenAI(api_key=config.openai_api_key)
            elif config.active_provider == "gemini":
                import google.generativeai as genai

                genai.configure(api_key=config.gemini_api_key)
                self.client = genai.GenerativeModel("gemini-2.5-flash")
                self.model = "gemini-2.5-flash"
            else:
                # Will be handled by the enabled property
                pass

        except Exception as e:
            self.logger.info(f"LLM evaluation disabled: {e}")
            # Will be handled by the enabled property
            pass

    async def evaluate(self, query: str, response_summary: str) -> LLMEvaluation:
        """Evaluate agent response for test purposes."""
        if not self.enabled:
            return LLMEvaluation()

        start_time = time.time()

        try:
            # Simple evaluation prompt
            prompt = f"""Rate this AI agent response (0.0 to 1.0):

Query: {query}
Response: {response_summary}

Consider accuracy, helpfulness, and task completion.
Respond in JSON: {{"score": 0.0-1.0, "feedback": "brief assessment"}}"""

            # Make LLM call with timeout
            content = await asyncio.wait_for(self._call_llm(prompt), timeout=10.0)

            # Parse response
            try:
                data = json.loads(content.strip())
                score = max(0.0, min(1.0, float(data.get("score", 0.0))))
                feedback = data.get("feedback", "")

                return LLMEvaluation(
                    score=score,
                    feedback=feedback,
                    success=True,
                    duration=time.time() - start_time,
                )

            except (json.JSONDecodeError, ValueError, TypeError):
                # Fallback: extract score from text
                score = self._extract_score_from_text(content)
                return LLMEvaluation(
                    score=score,
                    feedback=content[:100],
                    success=True,
                    duration=time.time() - start_time,
                )

        except Exception as e:
            return LLMEvaluation(
                feedback=f"Evaluation error: {str(e)}",
                duration=time.time() - start_time,
            )

    async def _call_llm(self, prompt: str) -> str:
        """Make LLM API call."""
        if hasattr(self.client, "chat"):  # OpenAI
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150,
            )
            return response.choices[0].message.content
        else:  # Gemini
            response = await self.client.generate_content_async(prompt)
            return response.text

    def _extract_score_from_text(self, text: str) -> float:
        """Extract score from text if JSON parsing fails."""
        import re

        # Look for decimal numbers
        numbers = re.findall(r"\b(0\.\d+|1\.0|0|1)\b", text)
        if numbers:
            try:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
            except ValueError:
                pass

        # Fallback based on keywords
        text_lower = text.lower()
        if any(word in text_lower for word in ["excellent", "perfect", "great"]):
            return 0.9
        elif any(word in text_lower for word in ["good", "correct", "successful"]):
            return 0.7
        elif any(word in text_lower for word in ["okay", "partial", "some"]):
            return 0.5
        elif any(word in text_lower for word in ["poor", "failed", "error"]):
            return 0.2

        return 0.5  # Default neutral score


@dataclass
class EnhancedTestMetrics:
    """Test metrics that combine performance metrics with LLM evaluation."""

    # Core performance metrics (unchanged)
    performance: AgentPerformanceMetrics

    # Test-layer LLM evaluation (optional)
    llm_evaluation: LLMEvaluation | None = None

    @property
    def combined_score(self) -> float:
        """Simple combined score: (speed_factor + quality_score) / 2."""
        if not self.llm_evaluation or not self.llm_evaluation.success:
            return 0.0

        # Normalize execution time to 0-1 scale (faster = higher score)
        time_score = max(0.0, min(1.0, 1.0 / (self.performance.execution_time + 1.0)))

        return (time_score + self.llm_evaluation.score) / 2.0

    def report(self) -> str:
        """Generate simple test report."""
        lines = [
            "📊 Test Results:",
            f"   ⏱️  Time: {self.performance.execution_time:.2f}s",
            f"   🪙 Tokens: {self.performance.token_usage}",
            f"   ✅ Success: {self.performance.success}",
        ]

        if self.llm_evaluation and self.llm_evaluation.success:
            lines.extend(
                [
                    f"   🎯 Quality: {self.llm_evaluation.score:.2f}/1.0",
                    f"   💬 Feedback: {self.llm_evaluation.feedback}",
                ]
            )
        else:
            lines.append("   🎯 Quality: Not evaluated")

        if self.combined_score > 0:
            lines.append(f"   🏆 Combined: {self.combined_score:.2f}/1.0")

        return "\n".join(lines)


# Global test evaluator instance
_test_evaluator = None


def get_test_evaluator() -> TestLLMEvaluator:
    """Get test evaluator instance."""
    global _test_evaluator
    if _test_evaluator is None:
        _test_evaluator = TestLLMEvaluator()
    return _test_evaluator


async def enhance_test_metrics(
    performance_metrics: AgentPerformanceMetrics, query: str, response_summary: str
) -> EnhancedTestMetrics:
    """Enhance performance metrics with LLM evaluation for testing."""
    enhanced = EnhancedTestMetrics(performance=performance_metrics)

    # Only do LLM evaluation in test environment if successful
    if performance_metrics.success and query and response_summary:
        evaluator = get_test_evaluator()
        if evaluator.enabled:
            try:
                enhanced.llm_evaluation = await evaluator.evaluate(query, response_summary)
            except Exception:
                # LLM evaluation failed, continue with just performance metrics
                pass

    return enhanced
