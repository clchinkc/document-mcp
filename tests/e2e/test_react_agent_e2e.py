"""
End-to-end tests for the React Agent with real AI models.

These tests require real API keys and make actual calls to AI services.
They test the complete system including AI reasoning and MCP server integration.
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest

from tests.shared.agent_base import E2ETestBase, ReactAgentTestMixin
from tests.conftest import skip_if_no_real_api_key


@pytest.fixture
def test_docs_root():
    """Create a temporary directory for test documents."""
    temp_dir = tempfile.mkdtemp(prefix="react_e2e_test_docs_")
    path = Path(temp_dir)
    
    original_root = os.environ.get("DOCUMENT_ROOT_DIR")
    os.environ["DOCUMENT_ROOT_DIR"] = str(path)
    
    try:
        yield path
    finally:
        if original_root is None:
            if "DOCUMENT_ROOT_DIR" in os.environ:
                del os.environ["DOCUMENT_ROOT_DIR"]
        else:
            os.environ["DOCUMENT_ROOT_DIR"] = original_root
        shutil.rmtree(path, ignore_errors=True)


@pytest.mark.e2e
class TestReactAgentE2E(E2ETestBase, ReactAgentTestMixin):
    """
    End-to-end tests for the React Agent, using real AI models.
    These tests verify the complete workflow from user query to multi-step
    thought process and final MCP execution.
    """

    @pytest.mark.asyncio
    @skip_if_no_real_api_key
    async def test_e2e_react_agent_document_and_chapter_workflow(self, test_docs_root):
        """
        Tests a standard E2E workflow: create a document and add a chapter,
        verifying the agent's multi-step reasoning.
        """
        doc_name = f"e2e_react_doc_{uuid.uuid4().hex[:8]}"
        query = f"Create a new document called '{doc_name}', then add a chapter named 'Introduction' to it."

        agent, mcp_server = await self.initialize_react_agent_and_mcp_server()
        async with mcp_server:
            history = await self.run_react_query_on_agent(agent, query, max_steps=8)

        self.assert_multi_step_workflow(history, min_steps=2)
        assert (
            test_docs_root / doc_name
        ).exists(), "Document should be created on filesystem"
        assert (
            test_docs_root / doc_name / "01-Introduction.md"
        ).exists(), "Chapter should be created within the document"

    @pytest.mark.asyncio
    @skip_if_no_real_api_key
    async def test_e2e_react_agent_read_and_write_workflow(self, sample_documents_fixture):
        """
        Tests a read-then-write workflow to verify the agent can interact
        with existing document content.
        """
        doc_name = sample_documents_fixture["doc_name"]
        chapter_name = "01-chapter1.md"
        new_content = "This paragraph has been programmatically edited by a ReAct agent."
        query = f"In the document '{doc_name}', find the chapter '{chapter_name}' and replace its first paragraph with: '{new_content}'"

        agent, mcp_server = await self.initialize_react_agent_and_mcp_server()
        async with mcp_server:
            history = await self.run_react_query_on_agent(agent, query, max_steps=8)

        self.assert_multi_step_workflow(history)
        final_observation = history[-1].get("observation", "")
        assert (
            "completed" in final_observation.lower()
        ), "Final observation should indicate completion"

        # Verify the file content was actually changed
        file_path = sample_documents_fixture["doc_path"] / chapter_name
        updated_content = file_path.read_text()
        assert new_content in updated_content 