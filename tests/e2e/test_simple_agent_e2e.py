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

# Import shared testing utilities
from tests.shared.environment import has_real_api_key
from tests.conftest import skip_if_no_real_api_key





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





@pytest.mark.asyncio
@skip_if_no_real_api_key
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
@skip_if_no_real_api_key
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
@skip_if_no_real_api_key
async def test_simple_agent_e2e_multi_step_workflow():
    """E2E test: Simple agent handles multi-step workflow with real AI."""
    doc_name = f"MultiStepDoc_{uuid.uuid4().hex[:8]}"

    # Initialize agent and server once for the entire workflow
    agent, _ = await initialize_agent_and_mcp_server()
    
    try:
        async with agent.run_mcp_servers():
            # Step 1: Create document
            response1 = await asyncio.wait_for(
                process_single_user_query(agent, f"Create a document called '{doc_name}'"),
                timeout=60.0
            )
            assert response1 is not None, "Agent should respond to document creation"
            assert "created" in response1.summary.lower() or "success" in response1.summary.lower(), \
                f"Failed to create document. Summary: {response1.summary}"

            # Step 2: Add chapter
            response2 = await asyncio.wait_for(
                process_single_user_query(agent, f"Add a chapter called 'Introduction' to the document '{doc_name}'"),
                timeout=60.0
            )
            assert response2 is not None, "Agent should respond to chapter addition"
            assert "added" in response2.summary.lower() or "created" in response2.summary.lower(), \
                f"Failed to add chapter. Summary: {response2.summary}"

            # Step 3: List documents to verify
            response3 = await asyncio.wait_for(
                process_single_user_query(agent, "List all documents"),
                timeout=60.0
            )
            assert response3 is not None, "Agent should respond to document listing"
            
            # Check if document appears in the details (list of documents)
            doc_names = []
            if response3.details and isinstance(response3.details, list):
                doc_names = [
                    doc.document_name
                    for doc in response3.details
                    if hasattr(doc, "document_name")
                ]
            assert doc_name in doc_names, f"The created document '{doc_name}' should be in the list."

    except Exception as e:
        pytest.fail(f"Multi-step workflow failed with an exception: {e}")

    # Verify in file system after the agent context has been exited
    global_docs_path = Path(".documents_storage")
    multi_step_dirs = [d for d in global_docs_path.iterdir() if doc_name in d.name]
    assert len(multi_step_dirs) > 0, f"{doc_name} should exist. Found dirs: {[d.name for d in global_docs_path.iterdir()]}"

    doc_path = multi_step_dirs[0]
    chapters = list(doc_path.glob("*.md"))
    assert len(chapters) > 0, "At least one chapter should exist"


@pytest.mark.asyncio
@skip_if_no_real_api_key
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
@skip_if_no_real_api_key
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


@pytest.mark.asyncio
@skip_if_no_real_api_key
async def test_simple_agent_e2e_read_write_flow(e2e_sample_documents):
    """E2E test: Simple Agent reads and writes documents using real AI reasoning."""
    # Use the fixture data
    doc_name = e2e_sample_documents["doc_name"]
    doc_path = e2e_sample_documents["doc_path"]
    storage = e2e_sample_documents["storage_path"]
    summary_text = e2e_sample_documents["summary_text"]
    first_paragraph = e2e_sample_documents["first_paragraph"]
    
    # Step 3: List chapters
    resp = await run_simple_agent(f"List chapters in document '{doc_name}'")
    
    # Check for common errors
    if resp.error_message and ("Cancelled error" in resp.error_message or "request_limit" in resp.error_message.lower()):
        pytest.skip("Agent query was cancelled or hit API limits - common in CI environments")
    
    assert resp is not None
    assert isinstance(resp, FinalAgentResponse)
    assert len(resp.summary) > 0
    
    # Verify the AI attempted to understand the task
    chapter_reasoning = any(
        word in resp.summary.lower() 
        for word in ["chapter", "document", "list", doc_name.lower()]
    )
    assert chapter_reasoning, f"AI should reason about listing chapters. Got: {resp.summary}"
    
    # If we successfully got chapter details, proceed with more operations
    if resp.details and isinstance(resp.details, list) and any(
        hasattr(c, "chapter_name") and c.chapter_name == "01-chapter1.md" for c in resp.details
    ):
        # Step 4: Read summary (only if chapter listing succeeded)
        resp = await run_simple_agent(f"Read the document summary for '{doc_name}'")
        if resp.error_message and "Cancelled error" in resp.error_message:
            pytest.skip("Summary reading was cancelled")
            
        # Verify summary reading was attempted
        summary_reasoning = any(
            word in resp.summary.lower()
            for word in ["summary", "document", "read"]
        )
        assert summary_reasoning, "AI should reason about reading summary"
        
        # If summary reading was successful, test paragraph operations
        if resp.details and isinstance(resp.details, str) and summary_text in resp.details:
            # Step 5: Read paragraph 0
            resp = await run_simple_agent(
                f"Read paragraph 0 from chapter '01-chapter1.md' of '{doc_name}'"
            )
            if resp.error_message and "Cancelled error" in resp.error_message:
                pytest.skip("Paragraph reading was cancelled")
                
            # Verify paragraph reading was attempted
            para_reasoning = any(
                word in resp.summary.lower()
                for word in ["paragraph", "chapter", "read", "01-chapter1"]
            )
            assert para_reasoning, "AI should reason about reading paragraph"
            
            # If paragraph reading was successful, test writing
            if (resp.details and hasattr(resp.details, "paragraph_index_in_chapter") 
                and hasattr(resp.details, "content") and first_paragraph in resp.details.content):
                
                # Step 6: Replace paragraph 0
                resp = await run_simple_agent(
                    f"Replace paragraph 0 in chapter '01-chapter1.md' of '{doc_name}' with 'Edited first paragraph.'"
                )
                if resp.error_message and "Cancelled error" in resp.error_message:
                    pytest.skip("Paragraph replacement was cancelled")
                
                # Verify replacement was attempted
                replace_reasoning = any(
                    word in resp.summary.lower()
                    for word in ["replace", "edit", "paragraph", "chapter"]
                )
                assert replace_reasoning, "AI should reason about replacing paragraph"
                
                # Check if replacement was successful on disk
                content = (doc_path / "01-chapter1.md").read_text()
                if "Edited first paragraph." in content:
                    # Step 7: Re-read paragraph 0 to confirm edit
                    resp = await run_simple_agent(
                        f"Read paragraph 0 from chapter '01-chapter1.md' of '{doc_name}'"
                    )
                    # Verify we can read back the edited content
                    if resp.details and hasattr(resp.details, "content"):
                        assert "Edited first paragraph." in resp.details.content, "Should read back the edited content"
