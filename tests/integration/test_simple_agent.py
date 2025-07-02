import asyncio
import os
import shutil
import uuid
from pathlib import Path

import pytest

from src.agents.simple_agent import FinalAgentResponse
from tests.shared import (
    assert_agent_response_valid,
    generate_unique_name,
    validate_agent_environment,
    validate_package_imports,
    validate_simple_agent_imports,
)
from tests.shared.agent_base import IntegrationTestBase, SimpleAgentTestMixin
from tests.shared.environment import TEST_DOCUMENT_ROOT
from tests.shared.test_data import (
    TestDataRegistry,
    TestDocumentSpec,
    create_test_document_from_spec,
)


def test_agent_environment_setup():
    validate_agent_environment()


def test_agent_package_imports():
    validate_package_imports()
    validate_simple_agent_imports()


@pytest.fixture(autouse=True)
def clean_documents_directory(test_docs_root):
    """
    This fixture relies on test_docs_root from conftest.py to ensure
    a clean state for every test in this file.
    """
    pass


@pytest.mark.integration
class TestSimpleAgentIntegration(IntegrationTestBase, SimpleAgentTestMixin):
    """
    Integration tests for the Simple Agent, using stateful helpers that
    mock LLM responses but use a real MCP server.
    """

    @pytest.mark.asyncio
    async def test_agent_list_documents_empty(self, test_docs_root):
        """Tests that listing documents in an empty directory returns an empty list."""
        agent, mcp_server = await self.initialize_simple_agent_and_mcp_server()
        async with mcp_server:
            response = await self.run_simple_query_on_agent(
                agent, "Show me all available documents"
            )
        assert_agent_response_valid(response, "Simple agent")
        assert isinstance(response.details, list)
        assert len(response.details) == 0

    @pytest.mark.asyncio
    async def test_simple_agent_document_and_chapter_workflow(self, test_docs_root):
        """Tests a standard workflow: create doc, add chapter, and read."""
        doc_name = f"int_doc_{uuid.uuid4().hex[:8]}"
        chapter_name = "01-intro.md"
        content = "This is the first chapter."

        agent, mcp_server = await self.initialize_simple_agent_and_mcp_server()
        async with mcp_server:
            # Create document
            await self.run_simple_query_on_agent(
                agent, f"Create a new document named '{doc_name}'"
            )

            # Add chapter
            await self.run_simple_query_on_agent(
                agent,
                f"Create a chapter named '{chapter_name}' in '{doc_name}' with content: {content}",
            )

            # Read chapter to verify
            response = await self.run_simple_query_on_agent(
                agent, f"Read chapter '{chapter_name}' from '{doc_name}'"
            )

        assert response.details is not None
        assert content in response.details.content

    @pytest.mark.asyncio
    async def test_simple_agent_error_recovery_workflow(self, test_docs_root):
        """Tests agent's ability to recover and proceed after a failed action."""
        doc_name = f"int_error_doc_{uuid.uuid4().hex[:8]}"

        agent, mcp_server = await self.initialize_simple_agent_and_mcp_server()
        async with mcp_server:
            # Create a document
            await self.run_simple_query_on_agent(
                agent, f"Create a document named '{doc_name}'"
            )

            # Attempt to create it again (should fail)
            fail_response = await self.run_simple_query_on_agent(
                agent, f"Create another document with the same name '{doc_name}'"
            )

            # List documents to confirm state
            list_response = await self.run_simple_query_on_agent(
                agent, "List all documents"
            )

        assert "already exists" in fail_response.summary.lower()
        doc_names = [d.document_name for d in list_response.details]
        assert doc_name in doc_names
        assert len(doc_names) >= 1

    @pytest.mark.asyncio
    async def test_simple_agent_state_isolation_across_runs(self, test_docs_root):
        """Ensures that two separate agent runs do not interfere with each other."""
        doc1_name = f"int_iso_doc_1_{uuid.uuid4().hex[:8]}"
        doc2_name = f"int_iso_doc_2_{uuid.uuid4().hex[:8]}"

        # First run
        agent1, mcp_server1 = await self.initialize_simple_agent_and_mcp_server()
        async with mcp_server1:
            await self.run_simple_query_on_agent(
                agent1, f"Create a new document named '{doc1_name}'"
            )

        # Second run
        agent2, mcp_server2 = await self.initialize_simple_agent_and_mcp_server()
        async with mcp_server2:
            await self.run_simple_query_on_agent(
                agent2, f"Create a new document named '{doc2_name}'"
            )
            response = await self.run_simple_query_on_agent(
                agent2, "Show me all available documents"
            )

        doc_names = [d.document_name for d in response.details]
        assert doc1_name in doc_names
        assert doc2_name in doc_names

    @pytest.mark.asyncio
    async def test_simple_agent_three_round_conversation_complex_workflow(
        self, test_docs_root
    ):
        doc_name = generate_unique_name("complex_workflow")
        content = "This document contains searchable content."
        agent, mcp_server = await self.initialize_simple_agent_and_mcp_server()
        async with mcp_server:
            await self.run_simple_query_on_agent(
                agent,
                f"Create a doc named '{doc_name}' with a chapter '01-intro.md' containing: {content}",
            )
            resp_search = await self.run_simple_query_on_agent(
                agent, f"Find 'searchable' in '{doc_name}'"
            )
            resp_stats = await self.run_simple_query_on_agent(
                agent, f"Get stats for '{doc_name}'"
            )

        assert resp_search.details is not None and len(resp_search.details) > 0
        assert resp_stats.details is not None 