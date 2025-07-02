"""
End-to-end tests for the Simple Agent with real AI models.

These tests require real API keys and make actual calls to AI services.
They test the complete system including AI reasoning and MCP server integration.
"""

import uuid
from pathlib import Path

import pytest

from src.agents.simple_agent import FinalAgentResponse
from tests.shared.agent_base import E2ETestBase, SimpleAgentTestMixin
from tests.conftest import skip_if_no_real_api_key


@pytest.mark.e2e
class TestSimpleAgentE2E(E2ETestBase, SimpleAgentTestMixin):
    """
    End-to-end tests for the Simple Agent, using real AI models.
    These tests verify the complete workflow from user query to MCP execution.
    """

    @pytest.mark.asyncio
    @skip_if_no_real_api_key
    async def test_e2e_simple_agent_document_and_chapter_workflow(self, test_docs_root):
        """
        Tests a standard E2E workflow: create a document, add a chapter,
        and then read the chapter to verify content.
        """
        doc_name = f"e2e_doc_{uuid.uuid4().hex[:8]}"
        chapter_name = "01-intro.md"
        content = "This is a test chapter."

        agent, mcp_server = await self.initialize_simple_agent_and_mcp_server()
        async with mcp_server:
            # 1. Create document
            create_resp = await self.run_simple_query_on_agent(
                agent, f"Create a document called '{doc_name}'"
            )
            assert "created" in create_resp.summary.lower()

            # 2. Add chapter
            add_resp = await self.run_simple_query_on_agent(
                agent,
                f"Add a chapter named '{chapter_name}' to '{doc_name}' with content: {content}",
            )
            assert "added" in add_resp.summary.lower() or "created" in add_resp.summary.lower()

            # 3. Read chapter to verify
            read_resp = await self.run_simple_query_on_agent(
                agent, f"Read chapter '{chapter_name}' from '{doc_name}'"
            )
            assert content in read_resp.details.content

    @pytest.mark.asyncio
    @skip_if_no_real_api_key
    async def test_e2e_simple_agent_error_handling_workflow(self, test_docs_root):
        """
        Tests the agent's ability to handle errors gracefully, such as
        creating a duplicate document, and then proceed with other tasks.
        """
        doc_name = f"e2e_error_doc_{uuid.uuid4().hex[:8]}"

        agent, mcp_server = await self.initialize_simple_agent_and_mcp_server()
        async with mcp_server:
            # 1. Create document
            await self.run_simple_query_on_agent(
                agent, f"Create a document called '{doc_name}'"
            )

            # 2. Attempt to create it again (should fail)
            fail_resp = await self.run_simple_query_on_agent(
                agent, f"Create another document with the same name '{doc_name}'"
            )
            assert "already exists" in fail_resp.summary.lower()

            # 3. List documents to verify state
            list_resp = await self.run_simple_query_on_agent(
                agent, "List all documents"
            )
            doc_names = [d.document_name for d in list_resp.details]
            assert doc_name in doc_names

    @pytest.mark.asyncio
    @skip_if_no_real_api_key
    async def test_simple_agent_e2e_mcp_connection(self):
        """E2E test: Verify agent can run and list documents."""
        agent, mcp_server = await self.initialize_simple_agent_and_mcp_server()
        async with mcp_server:
            response = await self.run_simple_query_on_agent(agent, "List all documents")

        assert response is not None
        assert isinstance(response, FinalAgentResponse)
        assert response.details is not None, "Should get document list details"
        assert isinstance(response.details, list)

    @pytest.mark.asyncio
    @skip_if_no_real_api_key
    async def test_simple_agent_e2e_document_creation(self):
        """E2E test: Simple agent creates a document using real AI."""
        doc_name = f"TestDoc_{uuid.uuid4().hex[:8]}"
        
        agent, mcp_server = await self.initialize_simple_agent_and_mcp_server()
        async with mcp_server:
            response = await self.run_simple_query_on_agent(agent, f"Create a document called '{doc_name}'")

        assert "created" in response.summary.lower()
        global_docs_path = Path(".documents_storage")
        assert any(doc_name in d.name for d in global_docs_path.iterdir())

    @pytest.mark.asyncio
    @skip_if_no_real_api_key
    async def test_simple_agent_e2e_multi_step_workflow(self):
        """E2E test: Simple agent handles multi-step workflow with real AI."""
        doc_name = f"MultiStepDoc_{uuid.uuid4().hex[:8]}"

        agent, mcp_server = await self.initialize_simple_agent_and_mcp_server()
        async with mcp_server:
            # Step 1: Create document
            resp1 = await self.run_simple_query_on_agent(agent, f"Create a document called '{doc_name}'")
            assert "created" in resp1.summary.lower()

            # Step 2: Add chapter
            resp2 = await self.run_simple_query_on_agent(agent, f"Add a chapter called 'Introduction' to '{doc_name}'")
            assert "added" in resp2.summary.lower() or "created" in resp2.summary.lower()

            # Step 3: List documents to verify
            resp3 = await self.run_simple_query_on_agent(agent, "List all documents")
            doc_names = [d.document_name for d in resp3.details if hasattr(d, "document_name")]
            assert doc_name in doc_names
            
        doc_path = Path(".documents_storage") / doc_name
        assert doc_path.exists()
        assert len(list(doc_path.glob("*.md"))) > 0

    @pytest.mark.asyncio
    @skip_if_no_real_api_key
    async def test_simple_agent_e2e_read_write_flow(self, e2e_sample_documents):
        """E2E test: Simple Agent reads and writes documents."""
        doc_name = e2e_sample_documents["doc_name"]
        doc_path = e2e_sample_documents["doc_path"]

        agent, mcp_server = await self.initialize_simple_agent_and_mcp_server()
        async with mcp_server:
            # Read
            resp1 = await self.run_simple_query_on_agent(agent, f"Read paragraph 0 from chapter '01-chapter1.md' of '{doc_name}'")
            assert "first paragraph" in resp1.details.content

            # Write
            await self.run_simple_query_on_agent(
                agent,
                f"Replace paragraph 0 in chapter '01-chapter1.md' of '{doc_name}' with 'Edited first paragraph.'"
            )

            # Re-read
            resp2 = await self.run_simple_query_on_agent(agent, f"Read paragraph 0 from chapter '01-chapter1.md' of '{doc_name}'")
            assert "Edited first paragraph." in resp2.details.content
        
        content = (doc_path / "01-chapter1.md").read_text()
        assert "Edited first paragraph." in content
