"""
Integration tests for MCP server over HTTP/SSE transport.

This module tests real MCP server communication using HTTP/SSE transport
instead of in-process function calls.
"""
import pytest
import pytest_asyncio

from tests.shared.mcp_fixtures import (
    mcp_server,
    mcp_client,
    mcp_server_session,
    mcp_client_session,
    mcp_test_data,
    mcp_server_with_data,
)
from tests.shared.mcp_client import (
    MCPConnectionError,
    MCPTimeoutError,
    MCPProtocolError,
)


@pytest.mark.asyncio
async def test_mcp_server_connection(mcp_server, mcp_client):
    """Test basic connection to MCP server."""
    # Client is already connected via fixture
    assert mcp_client._connected
    
    # Test health check
    health = await mcp_client.health_check()
    assert health is True


@pytest.mark.asyncio
async def test_list_tools(mcp_client):
    """Test listing available MCP tools."""
    tools = await mcp_client.list_tools()
    
    # Verify we get a list of tools
    assert isinstance(tools, list)
    assert len(tools) > 0
    
    # Check that essential tools are present
    tool_names = [tool.get("name") for tool in tools]
    assert "list_documents" in tool_names
    assert "create_document" in tool_names
    assert "create_chapter" in tool_names
    
    # Verify tool structure
    for tool in tools:
        assert "name" in tool
        assert "description" in tool
        assert "inputSchema" in tool


@pytest.mark.asyncio
async def test_create_and_list_documents(mcp_client):
    """Test creating a document and listing it."""
    # Create a document
    result = await mcp_client.call_tool(
        "create_document",
        {"document_name": "test_document"}
    )
    
    assert result["success"] is True
    assert "test_document" in result["message"]
    
    # List documents
    documents = await mcp_client.call_tool("list_documents", {})
    
    # Verify the document appears in the list
    assert isinstance(documents, list)
    doc_names = [doc["document_name"] for doc in documents]
    assert "test_document" in doc_names
    
    # Find our document and verify its structure
    test_doc = next(doc for doc in documents if doc["document_name"] == "test_document")
    assert test_doc["total_chapters"] == 0
    assert test_doc["total_word_count"] == 0
    assert test_doc["has_summary"] is False


@pytest.mark.asyncio
async def test_chapter_operations(mcp_client):
    """Test chapter creation and reading."""
    # Create a document first
    await mcp_client.call_tool(
        "create_document",
        {"document_name": "chapter_test_doc"}
    )
    
    # Create a chapter
    chapter_result = await mcp_client.call_tool(
        "create_chapter",
        {
            "document_name": "chapter_test_doc",
            "chapter_name": "01-introduction.md",
            "initial_content": "# Introduction\n\nThis is the introduction chapter."
        }
    )
    
    assert chapter_result["success"] is True
    
    # List chapters
    chapters = await mcp_client.call_tool(
        "list_chapters",
        {"document_name": "chapter_test_doc"}
    )
    
    assert len(chapters) == 1
    assert chapters[0]["chapter_name"] == "01-introduction.md"
    assert chapters[0]["word_count"] > 0
    
    # Read chapter content
    content = await mcp_client.call_tool(
        "read_chapter_content",
        {
            "document_name": "chapter_test_doc",
            "chapter_name": "01-introduction.md"
        }
    )
    
    assert content["content"] == "# Introduction\n\nThis is the introduction chapter."
    assert content["paragraph_count"] == 2  # Title and content


@pytest.mark.asyncio
async def test_error_handling(mcp_client):
    """Test error handling for invalid operations."""
    # Try to read non-existent document
    result = await mcp_client.call_tool(
        "read_chapter_content",
        {
            "document_name": "non_existent_doc",
            "chapter_name": "chapter.md"
        }
    )
    
    # Should return None for non-existent resources
    assert result is None
    
    # Try to create chapter in non-existent document
    result = await mcp_client.call_tool(
        "create_chapter",
        {
            "document_name": "non_existent_doc",
            "chapter_name": "chapter.md",
            "initial_content": "Content"
        }
    )
    
    assert result["success"] is False
    assert "not found" in result["message"]


@pytest.mark.asyncio
async def test_network_failure_simulation(mcp_client):
    """Test handling of network failures."""
    # Create a document first
    await mcp_client.call_tool(
        "create_document",
        {"document_name": "network_test_doc"}
    )
    
    # Simulate network failure
    await mcp_client.simulate_network_failure()
    
    # Try to call a tool after network failure
    with pytest.raises(MCPConnectionError):
        await mcp_client.call_tool("list_documents", {})


@pytest.mark.asyncio
@pytest.mark.mcp_data({
    "documents": [
        {
            "name": "pre_populated_doc",
            "chapters": [
                {"name": "01-intro.md", "content": "# Introduction\n\nWelcome!"},
                {"name": "02-main.md", "content": "# Main Content\n\nThis is the main content."}
            ]
        }
    ]
})
async def test_pre_populated_data(mcp_server_with_data, mcp_client):
    """Test server with pre-populated data."""
    # List documents
    documents = await mcp_client.call_tool("list_documents", {})
    
    # Verify pre-populated document exists
    doc_names = [doc["document_name"] for doc in documents]
    assert "pre_populated_doc" in doc_names
    
    # Check chapters
    chapters = await mcp_client.call_tool(
        "list_chapters",
        {"document_name": "pre_populated_doc"}
    )
    
    assert len(chapters) == 2
    chapter_names = [ch["chapter_name"] for ch in chapters]
    assert "01-intro.md" in chapter_names
    assert "02-main.md" in chapter_names


@pytest.mark.asyncio
async def test_session_persistence(mcp_server_session, mcp_client_session):
    """Test that session-scoped server maintains state."""
    # Create a document in the session
    await mcp_client_session.call_tool(
        "create_document",
        {"document_name": "session_doc"}
    )
    
    # Disconnect and reconnect
    await mcp_client_session.disconnect()
    await mcp_client_session.connect()
    
    # Document should still exist
    documents = await mcp_client_session.call_tool("list_documents", {})
    doc_names = [doc["document_name"] for doc in documents]
    assert "session_doc" in doc_names 