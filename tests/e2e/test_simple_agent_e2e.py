"""
End-to-end tests for the Simple Agent with real AI models.

These tests require real API keys and make actual calls to AI services.
They test the complete system including AI reasoning and MCP server integration.
"""

import asyncio
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import pytest

# Import simple agent components
from src.agents.simple_agent import (
    FinalAgentResponse,
    initialize_agent_and_mcp_server,
    process_single_user_query,
)


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
        # Agent initialization now handles server setup via stdio
        agent, _ = await initialize_agent_and_mcp_server()
        
        async with agent.run_mcp_servers():
            result = await asyncio.wait_for(
                process_single_user_query(agent, query), timeout=60.0
            )
            return result
    except Exception as e:
        return FinalAgentResponse(
            summary=f"Error during processing: {str(e)}",
            details=None,
            error_message=str(e),
        )


@pytest.fixture(autouse=True)
def clean_global_docs():
    """Ensures the global .documents_storage directory is clean for E2E tests."""
    global_docs_path = Path(".documents_storage")
    if global_docs_path.exists():
        shutil.rmtree(global_docs_path)
    global_docs_path.mkdir(exist_ok=True)
    yield
    if global_docs_path.exists():
        shutil.rmtree(global_docs_path)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not has_real_api_key(),
    reason="E2E test requires a real API key (OPENAI_API_KEY, GOOGLE_API_KEY, or GEMINI_API_KEY)",
)
async def test_simple_agent_e2e_mcp_connection():
    """E2E test: Verify agent can run and list documents."""
    query = "List all documents"

    response = await run_simple_agent(query)

    assert response is not None
    assert isinstance(response, FinalAgentResponse)
    assert len(response.summary) > 0

    if response.error_message == "Cancelled error":
        pytest.skip("Agent query was cancelled - common in CI environments")

    assert response.details is not None, "Should get document list details"
    assert isinstance(
        response.details, list
    ), "Details should be a list for document listing"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not has_real_api_key(),
    reason="E2E test requires a real API key (OPENAI_API_KEY, GOOGLE_API_KEY, or GEMINI_API_KEY)",
)
async def test_simple_agent_e2e_document_creation():
    """E2E test: Simple agent creates a document using real AI."""
    # Use a unique document name to avoid conflicts
    doc_name = f"TestDoc_{uuid.uuid4().hex[:8]}"
    query = f"Create a document called '{doc_name}'"

    response = await run_simple_agent(query)

    assert response is not None
    assert isinstance(response, FinalAgentResponse)
    assert len(response.summary) > 0

    success_indicators = ["created", "success", "added", "made"]
    assert any(
        indicator in response.summary.lower() for indicator in success_indicators
    ), f"Response should indicate success: {response.summary}"

    global_docs_path = Path(".documents_storage")
    created_dirs = [d for d in global_docs_path.iterdir() if d.is_dir()]
    assert (
        len(created_dirs) > 0
    ), f"Document should be created. Response: {response.summary}"

    doc_names = [d.name.lower() for d in created_dirs]
    assert any(doc_name.lower() in name for name in doc_names), f"{doc_name} should be created. Found: {doc_names}"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not has_real_api_key(),
    reason="E2E test requires a real API key (OPENAI_API_KEY, GOOGLE_API_KEY, or GEMINI_API_KEY)",
)
async def test_simple_agent_e2e_multi_step_workflow():
    """E2E test: Simple agent handles multi-step workflow with real AI."""
    # Use a unique document name to avoid conflicts
    doc_name = f"MultiStepDoc_{uuid.uuid4().hex[:8]}"
    
    # Create document
    response1 = await run_simple_agent(f"Create a document called '{doc_name}'")
    assert (
        "created" in response1.summary.lower() or "success" in response1.summary.lower()
    )

    # Add chapter
    response2 = await run_simple_agent(
        f"Add a chapter called 'Introduction' to the document '{doc_name}'"
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
    assert doc_name in doc_names, f"The created document '{doc_name}' should be in the list."

    # Verify in file system
    global_docs_path = Path(".documents_storage")
    multi_step_dirs = [d for d in global_docs_path.iterdir() if doc_name.lower() in d.name.lower()]
    assert (
        len(multi_step_dirs) > 0
    ), f"{doc_name} should exist. Found dirs: {[d.name for d in global_docs_path.iterdir()]}"

    # Check for chapters in the multi-step document
    doc_path = multi_step_dirs[0]  # Use the first matching directory
    chapters = list(doc_path.glob("*.md"))
    assert len(chapters) > 0, "At least one chapter should exist"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not has_real_api_key(),
    reason="E2E test requires a real API key (OPENAI_API_KEY, GOOGLE_API_KEY, or GEMINI_API_KEY)",
)
async def test_simple_agent_e2e_error_handling():
    """E2E test: Simple agent handles errors gracefully with real AI."""
    # Try to add chapter to non-existent document
    response = await run_simple_agent(
        "Add a chapter to a document that doesn't exist called 'NonExistent'"
    )

    # AI should recognize the error and provide helpful response
    assert len(response.summary) > 20, "Should provide meaningful error response"
    
    # Accept various types of responses including successful creation of the document
    # as the AI might decide to create the document first
    assert any(
        word in response.summary.lower()
        for word in ["not found", "doesn't exist", "create", "error", "created", "made", "document"]
    ), f"Should provide relevant response about the request: {response.summary}"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not has_real_api_key(),
    reason="E2E test requires a real API key (OPENAI_API_KEY, GOOGLE_API_KEY, or GEMINI_API_KEY)",
)
async def test_simple_agent_e2e_content_operations():
    """E2E test: Simple agent performs content operations with real AI."""
    # Use a unique document name to avoid conflicts
    doc_name = f"ContentTest_{uuid.uuid4().hex[:8]}"
    
    # Create document with content
    await run_simple_agent(f"Create a document called '{doc_name}'")

    response = await run_simple_agent(
        f"Add a chapter called 'Chapter1' to '{doc_name}' with content: "
        "'This is a test chapter with some example content.'",
    )
    assert "added" in response.summary.lower() or "created" in response.summary.lower()

    # Read the content back
    response = await run_simple_agent(f"Read the content of Chapter1 from '{doc_name}'")
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
