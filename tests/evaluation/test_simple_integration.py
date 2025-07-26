"""Simple integration test showing natural LLM evaluation alongside token metrics.

This demonstrates the correct architecture: agents collect performance metrics,
tests optionally enhance them with LLM evaluation.
"""

import tempfile
import uuid
from pathlib import Path

import pytest

from src.agents.react_agent.main import run_react_agent_with_metrics
from src.agents.simple_agent.main import initialize_agent_and_mcp_server
from src.agents.simple_agent.main import process_single_user_query
from tests.evaluation.llm_evaluation_layer import enhance_test_metrics


@pytest.fixture
def test_docs_root():
    """Provide clean temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.mark.evaluation
class TestNaturalIntegration:
    """Test natural integration of LLM evaluation with existing metrics."""

    @pytest.mark.asyncio
    async def test_simple_agent_with_optional_llm_evaluation(self, test_docs_root):
        """Test simple agent with optional LLM evaluation enhancement."""
        # Set up environment
        import os

        old_doc_root = os.environ.get("DOCUMENT_ROOT_DIR")
        os.environ["DOCUMENT_ROOT_DIR"] = str(test_docs_root)

        try:
            doc_name = f"test_doc_{uuid.uuid4().hex[:8]}"
            query = f"Create a document called '{doc_name}'"

            # Run agent normally - gets standard performance metrics
            agent, mcp_server = await initialize_agent_and_mcp_server()

            async with agent.run_mcp_servers():
                (
                    response,
                    performance_metrics,
                ) = await process_single_user_query(agent, query, collect_metrics=True)

                # Standard performance metrics work as always
                assert performance_metrics.execution_time > 0
                assert performance_metrics.agent_type == "simple"

                # Enhance with LLM evaluation for testing (optional)
                response_summary = response.summary if response else "No response"
                enhanced_metrics = await enhance_test_metrics(performance_metrics, query, response_summary)

                # Show combined results
                print(f"\n{enhanced_metrics.report()}")

                # Assertions work on both layers
                assert enhanced_metrics.performance.execution_time > 0
                if enhanced_metrics.llm_evaluation and enhanced_metrics.llm_evaluation.success:
                    assert 0.0 <= enhanced_metrics.llm_evaluation.score <= 1.0
                    assert enhanced_metrics.combined_score > 0.0
                    print(f"[OK] LLM evaluation successful: {enhanced_metrics.llm_evaluation.score:.2f}")
                else:
                    print("[INFO] LLM evaluation skipped")

        finally:
            
            # Restore environment
            if old_doc_root:
                os.environ["DOCUMENT_ROOT_DIR"] = old_doc_root
            else:
                os.environ.pop("DOCUMENT_ROOT_DIR", None)

    @pytest.mark.asyncio
    async def test_react_agent_with_optional_llm_evaluation(self, test_docs_root):
        """Test react agent with optional LLM evaluation enhancement."""
        import os

        old_doc_root = os.environ.get("DOCUMENT_ROOT_DIR")
        os.environ["DOCUMENT_ROOT_DIR"] = str(test_docs_root)

        try:
            query = "List all documents in the system"

            # Run react agent normally - gets standard performance metrics
            history, performance_metrics = await run_react_agent_with_metrics(query, max_steps=3)

            # Standard performance metrics work as always
            assert performance_metrics.execution_time > 0
            assert performance_metrics.agent_type == "react"

            # Enhance with LLM evaluation for testing (optional)
            response_summary = str(history[-1]) if history else "No response"
            enhanced_metrics = await enhance_test_metrics(performance_metrics, query, response_summary)

            # Show results
            print(f"\n{enhanced_metrics.report()}")

            # Both layers work independently
            assert enhanced_metrics.performance.agent_type == "react"
            if enhanced_metrics.llm_evaluation:
                print(f"[OK] React agent LLM evaluation: {enhanced_metrics.llm_evaluation.score:.2f}")

        finally:
            
            if old_doc_root:
                os.environ["DOCUMENT_ROOT_DIR"] = old_doc_root
            else:
                os.environ.pop("DOCUMENT_ROOT_DIR", None)

    @pytest.mark.asyncio
    async def test_evaluation_is_truly_optional(self, test_docs_root):
        """Test that LLM evaluation is completely optional."""
        import os

        old_doc_root = os.environ.get("DOCUMENT_ROOT_DIR")
        old_llm_eval = os.environ.get("ENABLE_LLM_EVALUATION")

        # Disable LLM evaluation
        os.environ["DOCUMENT_ROOT_DIR"] = str(test_docs_root)
        os.environ["ENABLE_LLM_EVALUATION"] = "false"

        try:
            query = "Create a simple test document"

            # Run agent - still works perfectly
            agent, mcp_server = await initialize_agent_and_mcp_server()

            async with agent.run_mcp_servers():
                (
                    response,
                    performance_metrics,
                ) = await process_single_user_query(agent, query, collect_metrics=True)

                # Performance metrics still work
                assert performance_metrics.execution_time > 0
                assert performance_metrics.token_usage >= 0

                # Try to enhance (will skip LLM evaluation)
                response_summary = response.summary if response else "No response"
                enhanced_metrics = await enhance_test_metrics(performance_metrics, query, response_summary)

                # Show it works without LLM evaluation
                print(f"\n{enhanced_metrics.report()}")

                # Should not have LLM evaluation
                assert enhanced_metrics.llm_evaluation is None or not enhanced_metrics.llm_evaluation.success
                assert enhanced_metrics.combined_score == 0.0

                print("[OK] Everything works perfectly without LLM evaluation")

        finally:
            
            if old_doc_root:
                os.environ["DOCUMENT_ROOT_DIR"] = old_doc_root
            else:
                os.environ.pop("DOCUMENT_ROOT_DIR", None)

            if old_llm_eval:
                os.environ["ENABLE_LLM_EVALUATION"] = old_llm_eval
            else:
                os.environ.pop("ENABLE_LLM_EVALUATION", None)

    @pytest.mark.asyncio
    async def test_simple_comparative_evaluation(self, test_docs_root):
        """Test simple comparative evaluation between agents."""
        import os

        old_doc_root = os.environ.get("DOCUMENT_ROOT_DIR")
        os.environ["DOCUMENT_ROOT_DIR"] = str(test_docs_root)

        try:
            query = "Create a document with a chapter"

            # Test simple agent
            agent, mcp_server = await initialize_agent_and_mcp_server()

            async with agent.run_mcp_servers():
                (
                    simple_response,
                    simple_metrics,
                ) = await process_single_user_query(agent, query, collect_metrics=True)

                simple_enhanced = await enhance_test_metrics(
                    simple_metrics,
                    query,
                    simple_response.summary if simple_response else "",
                )

            # Test react agent
            react_history, react_metrics = await run_react_agent_with_metrics(query, max_steps=3)
            react_enhanced = await enhance_test_metrics(
                react_metrics, query, str(react_history[-1]) if react_history else ""
            )

            # Simple comparison
            print("\n🔀 Agent Comparison:")
            print("Simple Agent:")
            print(f"   Time: {simple_enhanced.performance.execution_time:.2f}s")
            print(f"   Tokens: {simple_enhanced.performance.token_usage}")
            if simple_enhanced.llm_evaluation:
                print(f"   Quality: {simple_enhanced.llm_evaluation.score:.2f}")

            print("React Agent:")
            print(f"   Time: {react_enhanced.performance.execution_time:.2f}s")
            print(f"   Tokens: {react_enhanced.performance.token_usage}")
            if react_enhanced.llm_evaluation:
                print(f"   Quality: {react_enhanced.llm_evaluation.score:.2f}")

            # Both should have valid metrics
            assert simple_enhanced.performance.execution_time > 0
            assert react_enhanced.performance.execution_time > 0

        finally:
            
            if old_doc_root:
                os.environ["DOCUMENT_ROOT_DIR"] = old_doc_root
            else:
                os.environ.pop("DOCUMENT_ROOT_DIR", None)


# Simple demo
async def demo_clean_architecture():
    """Demo the clean separation between agents and test evaluation."""
    print("[ARCH] Clean Architecture Demo")
    print("=" * 40)

    with tempfile.TemporaryDirectory() as tmp_dir:
        import os

        os.environ["DOCUMENT_ROOT_DIR"] = str(tmp_dir)

        query = "Create a document called 'architecture_demo'"

        print(f"[NOTE] Query: {query}")

        # 1. Agent runs normally (no LLM evaluation)
        agent, mcp_server = await initialize_agent_and_mcp_server()

        async with agent.run_mcp_servers():
            response, metrics = await process_single_user_query(agent, query, collect_metrics=True)

        print("\n🤖 Agent Results (Core):")
        print(f"   Success: {metrics.success}")
        print(f"   Time: {metrics.execution_time:.2f}s")
        print(f"   Tokens: {metrics.token_usage}")

        # 2. Test layer optionally enhances with LLM evaluation
        enhanced = await enhance_test_metrics(metrics, query, response.summary if response else "")

        print("\n[TEST] Test Enhancement (Optional):")
        if enhanced.llm_evaluation and enhanced.llm_evaluation.success:
            print(f"   Quality: {enhanced.llm_evaluation.score:.2f}/1.0")
            print(f"   Feedback: {enhanced.llm_evaluation.feedback}")
            print(f"   Combined: {enhanced.combined_score:.2f}/1.0")
        else:
            print("   Quality: LLM evaluation disabled/failed")

        print("\n✨ Key Points:")
        print("   • Agents are unchanged - just collect performance metrics")
        print("   • Tests optionally enhance with LLM evaluation")
        print("   • Everything works with or without LLM evaluation")
        print("   • Clean separation of concerns")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
