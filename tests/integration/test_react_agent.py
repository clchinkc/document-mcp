import uuid
import pytest
import pytest_asyncio
from pathlib import Path

from tests.shared.agent_base import IntegrationTestBase, ReactAgentTestMixin

@pytest.mark.integration
class TestReactAgentIntegration(IntegrationTestBase, ReactAgentTestMixin):
    """
    Integration tests for the React Agent, focusing on its interaction
    with the MCP server and its multi-step reasoning capabilities.
    """

    @pytest.fixture(scope="class", autouse=True)
    async def react_agent_setup(self, request):
        """
        Set up the React Agent and MCP server once for all tests in this class.
        This fixture ensures state is maintained across test methods.
        """
        agent, mcp_server = await self.initialize_react_agent_and_mcp_server()
        request.cls.agent = agent
        request.cls.mcp_server = mcp_server
        yield
        await agent.close()
        await mcp_server.close()

    @pytest.mark.asyncio
    async def test_react_agent_initialization_and_query(self):
        """
        Verify that the React Agent initializes correctly and can execute a basic query.
        """
        assert self.agent is not None, "React Agent should be initialized"

        query = "List all available documents."
        response = await self.run_react_query_on_agent(self.agent, query)

        assert response is not None, "Response should not be None"
        assert isinstance(response, list), "Response should be a list (history)"

    @pytest.mark.asyncio
    async def test_react_agent_multi_step_document_workflow(self, test_docs_root):
        """
        Tests a multi-step workflow: create doc, add chapter, and verify by listing.
        """
        doc_name = f"react_int_multi_step_{uuid.uuid4().hex[:8]}"
        chapter_name = "01-introduction.md"

        # Create document
        h1 = await self.run_react_query_on_agent(
            self.agent, f"Create a document called '{doc_name}'"
        )
        assert any("create_document" in s.get("action", "") for s in h1)

        # Add chapter
        h2 = await self.run_react_query_on_agent(
            self.agent, f"Add a chapter '{chapter_name}' to '{doc_name}'"
        )
        assert any("create_chapter" in s.get("action", "") for s in h2)

        # List documents to verify creation
        h3 = await self.run_react_query_on_agent(self.agent, "List all documents")
        obs = " ".join([s.get("observation", "") for s in h3 if s.get("observation")])
        assert doc_name in obs, "Newly created document should be in the list"

    @pytest.mark.asyncio
    async def test_react_agent_thought_process(self):
        """
        Test that the agent exhibits a multi-step thought process.
        """
        query = "What are the key points in the document?"
        response = await self.run_react_query_on_agent(self.agent, query)
        assert response is not None, "Response should not be None"
        self.assert_multi_step_workflow(response)

    @pytest.mark.asyncio
    async def test_react_agent_error_handling(self):
        """
        Test the agent's behavior with an invalid or empty query.
        """
        query = ""  # Empty query to simulate error
        response = await self.run_react_query_on_agent(self.agent, query)
        assert response is not None, "Response should not be None even on error"
        assert len(response) > 0, "Response history should not be empty"
        final_step = response[-1]
        assert "error" in final_step.get(
            "observation", ""
        ).lower() or "invalid" in final_step.get(
            "observation", ""
        ).lower(), "Response should indicate an error or invalid input"

    @pytest.mark.asyncio
    async def test_react_agent_multiple_queries(self):
        """
        Test that the agent can handle multiple queries sequentially.
        """
        queries = ["Summarize the document.", "What are the key points?"]
        for query in queries:
            response = await self.run_react_query_on_agent(self.agent, query)
            assert response is not None, f"Response for query '{query}' should not be None"
            assert (
                len(response) > 0
            ), f"Response history for query '{query}' should not be empty"

    @pytest.mark.asyncio
    async def test_react_agent_three_round_conversation_with_error_recovery(self, test_docs_root):
        """Tests error recovery: read non-existent, create, then read again."""
        doc_name = f"react_int_error_recovery_{uuid.uuid4().hex[:8]}"
        chapter_name = "01-intro.md"

        agent, mcp_server = await self.initialize_react_agent_and_mcp_server()
        async with mcp_server:
            # Try to read from non-existent doc
            h1 = await self.run_react_query_on_agent(
                agent, f"Read chapter '{chapter_name}' from '{doc_name}'"
            )
            assert any("error" in s.get("observation", "").lower() for s in h1)

            # Create the doc
            await self.run_react_query_on_agent(agent, f"Create document '{doc_name}'")

            # Now add chapter (should succeed)
            h3 = await self.run_react_query_on_agent(
                agent, f"Create chapter '{chapter_name}' in '{doc_name}'"
            )
            assert not any("error" in s.get("observation", "").lower() for s in h3)

    @pytest.mark.asyncio
    async def test_react_agent_comprehensive_workflow(self, test_docs_root):
        """Tests a comprehensive workflow from creation to verification."""
        doc_name = f"react_int_comprehensive_{uuid.uuid4().hex[:8]}"
        chapter_name = "01-intro.md"
        content = "This is a comprehensive test."
        
        agent, mcp_server = await self.initialize_react_agent_and_mcp_server()
        async with mcp_server:
            await self.run_react_query_on_agent(agent, f"Create a document named '{doc_name}'")
            await self.run_react_query_on_agent(agent, f"Add a chapter '{chapter_name}' to '{doc_name}' with content: {content}")
            h3 = await self.run_react_query_on_agent(agent, f"Read chapter '{chapter_name}' from '{doc_name}'")

        obs = " ".join([s.get("observation", "") for s in h3 if s.get("observation")])
        assert content in obs

        doc_path = test_docs_root / doc_name
        assert (doc_path / chapter_name).is_file() 