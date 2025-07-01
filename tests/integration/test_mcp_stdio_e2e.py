"""
End-to-end tests for MCP server stdio transport.

These tests validate the complete MCP protocol flow including:
- Server lifecycle management
- Protocol compliance
- Error handling and recovery
- Performance and stress characteristics
"""
import asyncio
import time

import pytest

from tests.integration.mcp_client import (
    MCPError,
    MCPConnectionError,
)

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_mcp_server_lifecycle(mcp_client):
    """Test MCP server startup, initialization, and shutdown."""
    # Connection is implicitly tested by the fixture
    assert mcp_client.is_connected
    
    # Test server info
    server_info = mcp_client.server_info
    assert "name" in server_info
    assert server_info["name"] == "DocumentManagementTools"
    
    # Test tool discovery
    tools = await mcp_client.list_tools()
    assert len(tools) > 0
    
    # Verify essential tools are present
    tool_names = [tool["name"] for tool in tools]
    essential_tools = [
        "create_document", "list_documents", "delete_document",
        "create_chapter", "list_chapters", "read_chapter_content",
        "write_chapter_content", "delete_chapter"
    ]
    for tool in essential_tools:
        assert tool in tool_names
    # Test graceful shutdown
    await mcp_client.disconnect()
    assert not mcp_client.is_connected

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_complete_document_workflow(mcp_client):
    """Test a complete document creation and management workflow."""
    doc_name = "e2e_workflow_document"
    
    # Step 1: Create a document
    create_result = await mcp_client.call_tool("create_document", {"document_name": doc_name})
    assert create_result["success"] is True
    
    # Step 2: List documents to verify creation
    documents = await mcp_client.call_tool("list_documents", {})
    doc_names = [doc["document_name"] for doc in documents]
    assert doc_name in doc_names
    
    # Step 3: Create chapters
    chapters = [
        ("01-introduction.md", "# Introduction"),
        ("02-main-content.md", "# Main Content"),
    ]
    for chapter_name, content in chapters:
        result = await mcp_client.call_tool("create_chapter", {"document_name": doc_name, "chapter_name": chapter_name, "initial_content": content})
        assert result["success"] is True
    
    # Step 4: List chapters
    chapter_list = await mcp_client.call_tool("list_chapters", {"document_name": doc_name})
    assert len(chapter_list) == 2
    
    # Step 5: Read a chapter
    content = await mcp_client.call_tool("read_chapter_content", {"document_name": doc_name, "chapter_name": "02-main-content.md"})
    assert content["content"] == "# Main Content"
    
    # Step 6: Update chapter content
    new_content = "# Updated Content"
    await mcp_client.call_tool("write_chapter_content", {"document_name": doc_name, "chapter_name": "02-main-content.md", "new_content": new_content})
    
    # Step 7: Verify update
    updated_content = await mcp_client.call_tool("read_chapter_content", {"document_name": doc_name, "chapter_name": "02-main-content.md"})
    assert updated_content["content"] == new_content
    
    # Step 8: Clean up
    delete_result = await mcp_client.call_tool("delete_document", {"document_name": doc_name})
    assert delete_result["success"] is True
    # Verify deletion
    remaining = await mcp_client.call_tool("list_documents", {})
    names = [doc["document_name"] for doc in remaining]
    assert doc_name not in names

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_performance_characteristics(mcp_client):
    """Test performance characteristics of MCP operations."""
    doc_name = "perf_test_doc"
    await mcp_client.call_tool("create_document", {"document_name": doc_name})
    
    try:
        # Latency
        start_time = time.time()
        await mcp_client.call_tool("list_documents", {})
        assert time.time() - start_time < 1.0

        # Concurrency: measure time for parallel operations
        start_time = time.time()
        tasks = [mcp_client.call_tool("create_chapter", {"document_name": doc_name, "chapter_name": f"chapter_{i:02d}.md"}) for i in range(10)]
        results = await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        assert all(r["success"] for r in results)
        # Should be faster than sequential 10 * ~0.5s = 5s
        assert concurrent_time < 5.0, f"Concurrent operations took too long: {concurrent_time}s"
        
        # Large content handling: measure SLA
        large_content = "a" * (128 * 1024)  # 128KB
        start_time = time.time()
        result = await mcp_client.call_tool(
            "create_chapter", {"document_name": doc_name, "chapter_name": "large.md", "initial_content": large_content}
        )
        large_content_time = time.time() - start_time
        assert result.get("success") is True
        assert large_content_time < 2.0, f"Large content handling too slow: {large_content_time}s"
        read_content = await mcp_client.call_tool(
            "read_chapter_content", {"document_name": doc_name, "chapter_name": "large.md"}
        )
        assert read_content["content"] == large_content
    finally:
        await mcp_client.call_tool("delete_document", {"document_name": doc_name})

@pytest.mark.stress
@pytest.mark.asyncio
async def test_high_load_scenario(mcp_client):
    """Test server under high concurrent load as per PRD."""
    doc_name = "stress_test_doc"
    await mcp_client.call_tool("create_document", {"document_name": doc_name})
    
    try:
        tasks = []
        for i in range(100):
            task = mcp_client.call_tool("create_chapter", {
                "document_name": doc_name,
                "chapter_name": f"stress_test_{i:02d}.md",
                "initial_content": f"Test content {i}"
            })
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        assert (success_count / len(results)) > 0.95, f"Success rate was only {success_count/len(results):.2%}"
    finally:
        await mcp_client.call_tool("delete_document", {"document_name": doc_name})


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_error_handling_and_recovery(mcp_client, mcp_client_with_error_injection):
    """Test error handling and recovery scenarios."""
    # Test invalid tool calls with standard client
    with pytest.raises(MCPError):
        await mcp_client.call_tool("non_existent_tool", {})
    
    result = await mcp_client.call_tool("create_document", {"invalid_param": "test"})
    # The server may return a dict with success=False or an error string
    if isinstance(result, dict):
        assert result.get("success") is False
    else:
        text = str(result).lower()
        assert "error" in text or "missing" in text

    # Test connection recovery using error-injecting client
    client = mcp_client_with_error_injection
    await client.call_tool("create_document", {"document_name": "recovery_test_doc"})
    
    await client.simulate_network_failure()
    with pytest.raises(MCPConnectionError):
        await client.call_tool("list_documents", {})
    
    # Next request should succeed
    docs = await client.call_tool("list_documents", {})
    assert any(d["document_name"] == "recovery_test_doc" for d in docs)
    await client.call_tool("delete_document", {"document_name": "recovery_test_doc"})


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_mcp_protocol_compliance(mcp_client):
    """Test MCP protocol compliance and message format."""
    tools = await mcp_client.list_tools()
    for tool in tools:
        assert all(k in tool for k in ["name", "description", "inputSchema"])
    
    result = await mcp_client.call_tool("create_document", {"document_name": "protocol_test"})
    assert all(k in result for k in ["success", "message"])
    
    docs = await mcp_client.call_tool("list_documents", {})
    assert isinstance(docs, list)
    
    assert "name" in mcp_client.server_info
    
    await mcp_client.call_tool("delete_document", {"document_name": "protocol_test"}) 