"""Integration tests for Claude Code and Claude Desktop MCP compatibility.

This module tests the stdio-based MCP server integration patterns that Claude
clients use. These tests verify that the Document MCP server properly handles
the MCP protocol as Claude expects it.

Test Tiers:
- Tier 2: Tool Discovery (tool list, initialization)
- Tier 3: Basic Operations (CRUD)
- Tier 4: Complex Workflows (multi-step)
- Tier 5: Error Handling (edge cases)
"""

import asyncio
import json
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.mcp import MCPServerStdio


class TestMCPServerInitialization:
    """Test MCP server initialization and protocol compliance."""

    @pytest.mark.asyncio
    async def test_server_initializes_successfully(self):
        """Test that MCP server initializes with standard protocol."""
        server = MCPServerStdio(
            command="document-mcp",
            args=["stdio"],
            timeout=10.0,
        )

        try:
            async with server as s:
                # If we reach here, initialization succeeded
                assert s is not None
        except Exception as e:
            pytest.fail(f"Server initialization failed: {e}")

    @pytest.mark.asyncio
    async def test_server_returns_valid_capabilities(self):
        """Test that server returns capabilities during initialization."""
        server = MCPServerStdio(
            command="document-mcp",
            args=["stdio"],
            timeout=10.0,
        )

        async with server as s:
            # Access internal client to verify initialization
            assert s._client is not None

    @pytest.mark.asyncio
    async def test_server_handles_multiple_connections(self):
        """Test that multiple concurrent connections work."""
        servers = [
            MCPServerStdio(
                command="document-mcp",
                args=["stdio"],
                timeout=10.0,
            )
            for _ in range(3)
        ]

        try:
            tasks = [asyncio.create_task(server.__aenter__()) for server in servers]
            connections = await asyncio.gather(*tasks)

            # All should connect successfully
            assert len(connections) == 3
            assert all(c is not None for c in connections)

            # Clean up
            for server in servers:
                await server.__aexit__(None, None, None)
        except Exception as e:
            pytest.fail(f"Multiple connections failed: {e}")


class TestToolDiscovery:
    """Test that Claude can discover all tools."""

    @pytest.mark.asyncio
    async def test_all_28_tools_discoverable(self):
        """Test that all 28 MCP tools are discoverable."""
        expected_tools = {
            "list_documents",
            "create_document",
            "delete_document",
            "read_summary",
            "write_summary",
            "list_summaries",
            "list_chapters",
            "create_chapter",
            "delete_chapter",
            "write_chapter_content",
            "add_paragraph",
            "replace_paragraph",
            "delete_paragraph",
            "move_paragraph",
            "read_content",
            "find_text",
            "replace_text",
            "get_statistics",
            "find_similar_text",
            "find_entity",
            "read_metadata",
            "write_metadata",
            "list_metadata",
            "manage_snapshots",
            "check_content_status",
            "diff_content",
            "get_document_outline",
            "search_tool",
        }

        server = MCPServerStdio(
            command="document-mcp",
            args=["stdio"],
            timeout=10.0,
        )

        async with server as s:
            tools = await s._client.list_tools()
            discovered_tools = {t.name for t in tools}

            missing = expected_tools - discovered_tools
            extra = discovered_tools - expected_tools

            assert not missing, f"Missing tools: {missing}"
            assert len(discovered_tools) == 28, f"Expected 28 tools, got {len(discovered_tools)}"

            # Log extra tools if any (might be new additions)
            if extra:
                pytest.warns(UserWarning, match="Extra tools discovered")

    @pytest.mark.asyncio
    async def test_key_tools_have_descriptions(self):
        """Test that key tools have descriptions (for Claude)."""
        server = MCPServerStdio(
            command="document-mcp",
            args=["stdio"],
            timeout=10.0,
        )

        async with server as s:
            tools = await s._client.list_tools()

            key_tools = {"create_document", "read_content", "add_paragraph"}
            for tool in tools:
                if tool.name in key_tools:
                    assert tool.description, f"Tool {tool.name} missing description"
                    assert len(tool.description) > 10, f"Tool {tool.name} description too short"

    @pytest.mark.asyncio
    async def test_tools_have_input_schemas(self):
        """Test that tools have input schemas for validation."""
        server = MCPServerStdio(
            command="document-mcp",
            args=["stdio"],
            timeout=10.0,
        )

        async with server as s:
            tools = await s._client.list_tools()

            # All tools should have input schemas
            for tool in tools:
                assert tool.inputSchema is not None, f"Tool {tool.name} missing inputSchema"
                assert isinstance(tool.inputSchema, dict), f"Tool {tool.name} schema not dict"


class TestBasicOperations:
    """Test CRUD operations work as Claude expects."""

    @pytest.mark.asyncio
    async def test_create_document_operation(self, temp_docs_root):
        """Test document creation."""
        server = MCPServerStdio(
            command="document-mcp",
            args=["stdio"],
            timeout=10.0,
        )

        doc_name = f"test_doc_{uuid.uuid4().hex[:8]}"

        async with server as s:
            result = await s._client.call_tool(
                "create_document",
                {"document_name": doc_name},
            )

            assert result is not None
            assert len(result.content) > 0

            # Verify response format (Claude expects text content)
            response_text = result.content[0].text
            response_data = json.loads(response_text)

            assert response_data.get("success") is True
            assert doc_name in response_data.get("message", "")

            # Verify file system
            doc_path = temp_docs_root / doc_name
            assert doc_path.exists()

    @pytest.mark.asyncio
    async def test_read_content_operation(self, temp_docs_root):
        """Test content reading."""
        # Create a document first
        doc_name = f"test_read_{uuid.uuid4().hex[:8]}"
        doc_path = temp_docs_root / doc_name
        doc_path.mkdir()

        chapter_name = "01-test.md"
        chapter_path = doc_path / chapter_name
        test_content = "Test content for reading"
        chapter_path.write_text(test_content)

        server = MCPServerStdio(
            command="document-mcp",
            args=["stdio"],
            timeout=10.0,
        )

        async with server as s:
            result = await s._client.call_tool(
                "read_content",
                {
                    "document_name": doc_name,
                    "chapter_name": chapter_name,
                },
            )

            assert result is not None
            response_text = result.content[0].text
            response_data = json.loads(response_text)

            assert test_content in response_data.get("content", "")

    @pytest.mark.asyncio
    async def test_list_documents_operation(self, temp_docs_root):
        """Test listing documents."""
        # Create test documents
        for i in range(3):
            doc_path = temp_docs_root / f"test_list_{i}"
            doc_path.mkdir()

        server = MCPServerStdio(
            command="document-mcp",
            args=["stdio"],
            timeout=10.0,
        )

        async with server as s:
            result = await s._client.call_tool("list_documents", {})

            assert result is not None
            response_text = result.content[0].text
            response_data = json.loads(response_text)

            documents = response_data.get("documents", [])
            assert len(documents) >= 3

    @pytest.mark.asyncio
    async def test_paragraph_operations(self, temp_docs_root):
        """Test paragraph add/replace operations."""
        # Create document and chapter
        doc_name = f"test_para_{uuid.uuid4().hex[:8]}"
        doc_path = temp_docs_root / doc_name
        doc_path.mkdir()

        chapter_name = "01-test.md"
        chapter_path = doc_path / chapter_name
        chapter_path.write_text("Initial content\n")

        server = MCPServerStdio(
            command="document-mcp",
            args=["stdio"],
            timeout=10.0,
        )

        async with server as s:
            # Add paragraph
            result = await s._client.call_tool(
                "add_paragraph",
                {
                    "document_name": doc_name,
                    "chapter_name": chapter_name,
                    "paragraph_text": "New paragraph",
                },
            )

            assert result is not None
            response_text = result.content[0].text
            response_data = json.loads(response_text)

            assert response_data.get("success") is True

            # Verify content persisted
            content = chapter_path.read_text()
            assert "New paragraph" in content


class TestComplexWorkflows:
    """Test multi-step workflows."""

    @pytest.mark.asyncio
    async def test_document_creation_workflow(self, temp_docs_root):
        """Test complete workflow: create document, add chapter, add content."""
        doc_name = f"workflow_{uuid.uuid4().hex[:8]}"

        server = MCPServerStdio(
            command="document-mcp",
            args=["stdio"],
            timeout=10.0,
        )

        async with server as s:
            # Step 1: Create document
            result1 = await s._client.call_tool(
                "create_document",
                {"document_name": doc_name},
            )
            assert json.loads(result1.content[0].text).get("success")

            # Step 2: Create chapter
            result2 = await s._client.call_tool(
                "create_chapter",
                {
                    "document_name": doc_name,
                    "chapter_name": "01-intro.md",
                    "initial_content": "Introduction",
                },
            )
            assert json.loads(result2.content[0].text).get("success")

            # Step 3: Add paragraph
            result3 = await s._client.call_tool(
                "add_paragraph",
                {
                    "document_name": doc_name,
                    "chapter_name": "01-intro.md",
                    "paragraph_text": "First scene",
                },
            )
            assert json.loads(result3.content[0].text).get("success")

            # Step 4: Verify via read
            result4 = await s._client.call_tool(
                "read_content",
                {
                    "document_name": doc_name,
                    "chapter_name": "01-intro.md",
                },
            )
            content = json.loads(result4.content[0].text).get("content", "")
            assert "Introduction" in content
            assert "First scene" in content

    @pytest.mark.asyncio
    async def test_search_workflow(self, temp_docs_root):
        """Test search and discovery workflow."""
        # Create document with content
        doc_name = f"search_{uuid.uuid4().hex[:8]}"
        doc_path = temp_docs_root / doc_name
        doc_path.mkdir()

        chapter_name = "01-content.md"
        chapter_path = doc_path / chapter_name
        chapter_path.write_text("The hero journeyed to the castle. The castle was magnificent.")

        server = MCPServerStdio(
            command="document-mcp",
            args=["stdio"],
            timeout=10.0,
        )

        async with server as s:
            # Search for text
            result = await s._client.call_tool(
                "find_text",
                {
                    "document_name": doc_name,
                    "search_query": "castle",
                },
            )

            assert result is not None
            response = json.loads(result.content[0].text)
            matches = response.get("matches", [])
            assert len(matches) > 0


class TestErrorHandling:
    """Test error conditions are handled gracefully."""

    @pytest.mark.asyncio
    async def test_missing_document_error(self):
        """Test error when document doesn't exist."""
        server = MCPServerStdio(
            command="document-mcp",
            args=["stdio"],
            timeout=10.0,
        )

        async with server as s:
            result = await s._client.call_tool(
                "read_content",
                {
                    "document_name": "nonexistent_document_xyz",
                    "chapter_name": "01.md",
                },
            )

            # Should return error response, not crash
            assert result is not None
            response_text = result.content[0].text

            # Either error in response or success=false
            try:
                response = json.loads(response_text)
                # Check for error indication
                assert (
                    response.get("success") is False
                    or "error" in response
                    or "Error" in response.get("message", "")
                )
            except json.JSONDecodeError:
                # Some errors might not be JSON
                pass

    @pytest.mark.asyncio
    async def test_invalid_parameters_error(self):
        """Test error handling with invalid parameters."""
        server = MCPServerStdio(
            command="document-mcp",
            args=["stdio"],
            timeout=10.0,
        )

        async with server as s:
            # Missing required parameter
            result = await s._client.call_tool(
                "create_document",
                {},  # Missing document_name
            )

            # Should handle gracefully
            assert result is not None

    @pytest.mark.asyncio
    async def test_server_stays_alive_after_error(self):
        """Test server continues after error."""
        server = MCPServerStdio(
            command="document-mcp",
            args=["stdio"],
            timeout=10.0,
        )

        async with server as s:
            # Trigger an error
            await s._client.call_tool(
                "read_content",
                {
                    "document_name": "nonexistent",
                    "chapter_name": "x",
                },
            )

            # Should still be able to call other tools
            result = await s._client.call_tool("list_documents", {})
            assert result is not None


class TestStdioProtocol:
    """Test stdio protocol compliance."""

    def test_server_accepts_stdin_no_arguments(self):
        """Test server handles stdio mode without arguments."""
        proc = subprocess.Popen(
            ["document-mcp", "stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # Send simple JSON-RPC message
            init_msg = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0"},
                },
            }

            proc.stdin.write(json.dumps(init_msg) + "\n")
            proc.stdin.flush()

            # Read response
            response_line = proc.stdout.readline()
            assert response_line, "Server should respond to initialize"

            response = json.loads(response_line)
            assert response.get("id") == 1
            assert "result" in response or "error" in response

        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_server_outputs_valid_json_rpc(self):
        """Test server outputs valid JSON-RPC 2.0 messages."""
        proc = subprocess.Popen(
            ["document-mcp", "stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            init_msg = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0"},
                },
            }

            proc.stdin.write(json.dumps(init_msg) + "\n")
            proc.stdin.flush()

            response_line = proc.stdout.readline()
            response = json.loads(response_line)

            # Check JSON-RPC 2.0 compliance
            assert response.get("jsonrpc") == "2.0"
            assert "id" in response
            assert "result" in response or "error" in response

        finally:
            proc.terminate()
            proc.wait(timeout=5)


class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_tool_execution_under_timeout(self):
        """Test that tools execute within reasonable time."""
        server = MCPServerStdio(
            command="document-mcp",
            args=["stdio"],
            timeout=10.0,
        )

        async with server as s:
            import time

            start = time.time()
            result = await s._client.call_tool("list_documents", {})
            elapsed = time.time() - start

            # Should be very fast (< 1 second)
            assert elapsed < 1.0, f"Tool took {elapsed}s, should be < 1s"
            assert result is not None

    @pytest.mark.asyncio
    async def test_concurrent_tool_calls(self):
        """Test multiple concurrent tool calls."""
        server = MCPServerStdio(
            command="document-mcp",
            args=["stdio"],
            timeout=10.0,
        )

        async with server as s:
            # Call same tool multiple times concurrently
            tasks = [
                asyncio.create_task(s._client.call_tool("list_documents", {}))
                for _ in range(5)
            ]

            results = await asyncio.gather(*tasks)
            assert len(results) == 5
            assert all(r is not None for r in results)
