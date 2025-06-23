"""
End-to-end tests for the Simple Agent with real AI models.

These tests require real API keys and make actual calls to AI services.
They test the complete system including AI reasoning and MCP server integration.
"""

import asyncio
import os

import pytest

# Import simple agent components
from src.agents.simple_agent import (
    FinalAgentResponse,
    initialize_agent_and_mcp_server,
    process_single_user_query,
)

# Import server management from integration tests
from tests.integration.test_simple_agent import MCPServerManager


def has_real_api_key():
    """Check if a real API key is available (not test/placeholder keys)."""
    api_keys = ["OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"]
    for key in api_keys:
        value = os.environ.get(key, "").strip()
        if value and value != "test_key" and not value.startswith("sk-test"):
            return True
    return False


async def run_simple_agent(query: str) -> FinalAgentResponse:
    """Run simple agent with real AI for E2E testing."""
    try:
        agent, _ = await initialize_agent_and_mcp_server()
        async with agent.run_mcp_servers():
            result = await asyncio.wait_for(
                process_single_user_query(agent, query), timeout=30.0
            )
            return result
    except Exception as e:
        return FinalAgentResponse(
            summary=f"Error during processing: {str(e)}",
            details=None,
            error_message=str(e),
        )


# Skip all tests in this file if no real API key is available
pytestmark = pytest.mark.skipif(
    not has_real_api_key(),
    reason="E2E tests require a real API key (OPENAI_API_KEY, GOOGLE_API_KEY, or GEMINI_API_KEY)",
)


@pytest.fixture
def simple_e2e_server_manager():
    """Create a server manager for E2E tests."""
    import os
    import shutil
    import tempfile
    from pathlib import Path

    # Create a temporary directory for this test session
    test_docs_root = Path(tempfile.mkdtemp(prefix="simple_e2e_session_"))

    # Set environment variables before starting server
    original_port = os.environ.get("MCP_SERVER_PORT")
    original_root = os.environ.get("DOCUMENT_ROOT_DIR")

    manager = MCPServerManager(test_docs_root=test_docs_root)

    # Set environment variables for the agent to use
    os.environ["MCP_SERVER_PORT"] = str(manager.port)
    os.environ["DOCUMENT_ROOT_DIR"] = str(test_docs_root)

    manager.start_server()
    yield manager
    manager.stop_server()

    # Restore environment variables
    if original_port is None:
        if "MCP_SERVER_PORT" in os.environ:
            del os.environ["MCP_SERVER_PORT"]
    else:
        os.environ["MCP_SERVER_PORT"] = original_port

    if original_root is None:
        if "DOCUMENT_ROOT_DIR" in os.environ:
            del os.environ["DOCUMENT_ROOT_DIR"]
    else:
        os.environ["DOCUMENT_ROOT_DIR"] = original_root

    # Cleanup
    if test_docs_root.exists():
        shutil.rmtree(test_docs_root, ignore_errors=True)


@pytest.fixture
def simple_e2e_test_docs_root(simple_e2e_server_manager):
    """Use the server manager's test docs root."""
    return simple_e2e_server_manager.test_docs_root


@pytest.mark.asyncio
async def test_simple_agent_e2e_mcp_connection(simple_e2e_test_docs_root):
    """E2E test: Verify MCP server connection works."""
    query = "List all documents"

    response = await run_simple_agent(query)

    # Verify response
    assert response is not None
    assert isinstance(response, FinalAgentResponse)
    assert len(response.summary) > 0

    # Should get a list response (even if empty)
    assert response.details is not None, "Should get document list details"
    assert isinstance(
        response.details, list
    ), "Details should be a list for document listing"


@pytest.mark.asyncio
async def test_simple_agent_e2e_document_creation(simple_e2e_test_docs_root):
    """E2E test: Simple agent creates a document using real AI."""
    query = "Create a document called 'TestDoc'"

    response = await run_simple_agent(query)

    # Verify response indicates success
    assert response is not None
    assert isinstance(response, FinalAgentResponse)
    assert len(response.summary) > 0

    # Check if response indicates success
    success_indicators = ["created", "success", "added", "made"]
    assert any(
        indicator in response.summary.lower() for indicator in success_indicators
    ), f"Response should indicate success: {response.summary}"

    # Verify document was actually created
    created_dirs = [d for d in simple_e2e_test_docs_root.iterdir() if d.is_dir()]
    assert (
        len(created_dirs) > 0
    ), f"Document should be created. Response: {response.summary}"

    # Check if the created document has the expected name
    doc_names = [d.name.lower() for d in created_dirs]
    assert "testdoc" in doc_names, f"TestDoc should be created. Found: {doc_names}"


@pytest.mark.asyncio
async def test_simple_agent_e2e_multi_step_workflow(simple_e2e_test_docs_root):
    """E2E test: Simple agent handles multi-step workflow with real AI."""
    # Create document
    response1 = await run_simple_agent("Create a document called 'MultiStepDoc'")
    assert (
        "created" in response1.summary.lower() or "success" in response1.summary.lower()
    )

    # Add chapter
    response2 = await run_simple_agent(
        "Add a chapter called 'Introduction' to the document 'MultiStepDoc'"
    )
    assert (
        "added" in response2.summary.lower() or "created" in response2.summary.lower()
    )

    # List documents to verify
    response3 = await run_simple_agent("List all documents")
    # Check if document appears in the details (list of documents)
    doc_names = []
    if response3.details and isinstance(response3.details, list):
        doc_names = [
            doc.document_name
            for doc in response3.details
            if hasattr(doc, "document_name")
        ]
    assert "MultiStepDoc" in doc_names

    # Verify in file system
    created_dirs = [d for d in simple_e2e_test_docs_root.iterdir() if d.is_dir()]
    multi_step_dirs = [d for d in created_dirs if "multistepdoc" in d.name.lower()]
    assert (
        len(multi_step_dirs) > 0
    ), f"MultiStepDoc should exist. Found dirs: {[d.name for d in created_dirs]}"

    # Check for chapters in the multi-step document
    doc_path = multi_step_dirs[0]  # Use the first matching directory
    chapters = list(doc_path.glob("*.md"))
    assert len(chapters) > 0, "At least one chapter should exist"


@pytest.mark.asyncio
async def test_simple_agent_e2e_error_handling(simple_e2e_test_docs_root):
    """E2E test: Simple agent handles errors gracefully with real AI."""
    # Try to add chapter to non-existent document
    response = await run_simple_agent(
        "Add a chapter to a document that doesn't exist called 'NonExistent'"
    )

    # AI should recognize the error and provide helpful response
    assert len(response.summary) > 20, "Should provide meaningful error response"
    assert any(
        word in response.summary.lower()
        for word in ["not found", "doesn't exist", "create", "error"]
    )


@pytest.mark.asyncio
async def test_simple_agent_e2e_content_operations(simple_e2e_test_docs_root):
    """E2E test: Simple agent performs content operations with real AI."""
    # Create document with content
    await run_simple_agent("Create a document called 'ContentTest'")

    response = await run_simple_agent(
        "Add a chapter called 'Chapter1' to 'ContentTest' with content: "
        "'This is a test chapter with some example content.'"
    )
    assert "added" in response.summary.lower() or "created" in response.summary.lower()

    # Read the content back
    response = await run_simple_agent("Read the content of Chapter1 from 'ContentTest'")
    # Check the actual content in details, not just the summary
    content_found = False
    if response.details and hasattr(response.details, "content"):
        content = response.details.content.lower()
        content_found = "test chapter" in content and "example content" in content

    # Fallback to checking summary if content not in details
    if not content_found:
        assert (
            "test chapter" in response.summary.lower()
            or "example content" in response.summary.lower()
        )
    else:
        assert content_found
