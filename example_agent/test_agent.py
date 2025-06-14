import pytest
import asyncio
import pytest_asyncio
from pathlib import Path
import os
import shutil
from typing import List, Optional, Union, Dict, Any
import uuid
import sys
import tempfile
import subprocess
import time
import requests
import json
from contextlib import asynccontextmanager

# Import from the example agent (local file)
from agent import (
    initialize_agent_and_mcp_server,
    process_single_user_query,
    FinalAgentResponse,
    DocumentInfo, ChapterContent, OperationStatus,
    StatisticsReport, ParagraphDetail, FullDocumentContent
)

# Import from the installed document_mcp package for direct manipulation
try:
    from document_mcp import doc_tool_server
    from document_mcp.doc_tool_server import DOCS_ROOT_PATH as SERVER_DEFAULT_DOCS_ROOT_PATH
except ImportError:
    # Fallback for development/testing without installed package
    import sys
    sys.path.insert(0, '..')
    from document_mcp import doc_tool_server
    from document_mcp.doc_tool_server import DOCS_ROOT_PATH as SERVER_DEFAULT_DOCS_ROOT_PATH

# --- Environment Testing Functions ---

def test_agent_environment_setup():
    """Test agent environment setup and configuration."""
    # Check .env file exists
    env_file = Path(".env")
    assert env_file.exists(), ".env file not found - required for agent API keys"
    
    env_content = env_file.read_text()
    assert "GOOGLE_API_KEY" in env_content or "GEMINI_API_KEY" in env_content, \
        "Google/Gemini API key not found in .env - required for agent"

def test_agent_package_imports():
    """Test if all required packages can be imported for agent functionality."""
    try:
        import pydantic_ai
        assert True
    except ImportError:
        pytest.fail("Failed to import pydantic_ai - required for agent")
    
    try:
        # Test agent imports work
        from agent import FinalAgentResponse, StatisticsReport
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import agent models: {e}")

# --- HTTP SSE Server Management ---

class MCPServerManager:
    """Manages the HTTP SSE MCP server for testing."""
    
    def __init__(self, host="localhost", port=3001, test_docs_root=None):
        self.host = host
        self.port = port
        self.test_docs_root = test_docs_root
        self.process = None
        self.server_url = f"http://{host}:{port}"
        
    async def start_server(self):
        """Start the MCP server process."""
        if self.process is not None:
            return  # Already started
            
        # Set environment for the server process
        env = os.environ.copy()
        if self.test_docs_root:
            env["DOCUMENT_ROOT_DIR"] = str(self.test_docs_root)
            
        # Start the server process using module import to handle relative imports
        cmd = [sys.executable, "-m", "document_mcp.doc_tool_server", "sse", "--host", self.host, "--port", str(self.port)]
        
        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to be ready
        await self._wait_for_server()
        
    async def _wait_for_server(self, timeout=15):
        """Wait for the server to be ready to accept connections."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.server_url}/health", timeout=1)
                if response.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            await asyncio.sleep(0.5)
            
        raise TimeoutError(f"MCP server did not start within {timeout} seconds")
        
    async def stop_server(self):
        """Stop the MCP server process."""
        if self.process is not None:
            try:
                # First, try graceful termination
                self.process.terminate()
                
                try:
                    # Wait for graceful shutdown with timeout
                    await asyncio.wait_for(
                        asyncio.to_thread(self.process.wait),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    # If graceful termination fails, force kill
                    self.process.kill()
                    try:
                        # Wait for force kill to complete
                        await asyncio.wait_for(
                            asyncio.to_thread(self.process.wait),
                            timeout=3.0
                        )
                    except asyncio.TimeoutError:
                        pass  # Process didn't respond to kill signal
                
                # Ensure process cleanup
                try:
                    if self.process.stdout:
                        self.process.stdout.close()
                    if self.process.stderr:
                        self.process.stderr.close()
                    if self.process.stdin:
                        self.process.stdin.close()
                except:
                    pass
                
            except Exception:
                # Force cleanup even if there's an error
                if self.process:
                    try:
                        self.process.kill()
                        await asyncio.wait_for(
                            asyncio.to_thread(self.process.wait),
                            timeout=2.0
                        )
                    except:
                        pass
            finally:
                # Always clear the process reference
                self.process = None
                
                # Wait a bit to ensure all resources are released
                await asyncio.sleep(0.5)

@asynccontextmanager
async def mcp_server_context(test_docs_root=None, host="localhost", port=3001):
    """Context manager for MCP server lifecycle."""
    manager = MCPServerManager(host=host, port=port, test_docs_root=test_docs_root)
    try:
        await manager.start_server()
        yield manager
    finally:
        await manager.stop_server()
        # Additional cleanup time to ensure all resources are released
        await asyncio.sleep(2.0)

# --- Pytest Fixtures ---

@pytest.fixture(scope="module")
def test_docs_root(tmp_path_factory):
    """Create a temporary directory for documents that persists for the module."""
    test_docs_root = tmp_path_factory.mktemp("agent_test_documents_storage")
    
    # Override doc_tool_server paths
    original_server_path = doc_tool_server.DOCS_ROOT_PATH
    doc_tool_server.DOCS_ROOT_PATH = test_docs_root
    
    # Set environment variable for server default path
    original_env_var = os.environ.get("DOCUMENT_ROOT_DIR")
    os.environ["DOCUMENT_ROOT_DIR"] = str(test_docs_root)
    
    yield test_docs_root
    
    # Cleanup: Remove all created documents and restore original state
    try:
        if test_docs_root.exists():
            shutil.rmtree(test_docs_root, ignore_errors=True)
    except Exception as e:
        print(f"Warning: Could not fully clean up test directory: {e}")
    
    # Restore original paths and environment
    doc_tool_server.DOCS_ROOT_PATH = original_server_path
    if original_env_var is None:
        if "DOCUMENT_ROOT_DIR" in os.environ:
            del os.environ["DOCUMENT_ROOT_DIR"]
    else:
        os.environ["DOCUMENT_ROOT_DIR"] = original_env_var

# --- HTTP SSE Test Helper Functions ---

async def run_agent_test(query, test_docs_root=None, server_port=3001):
    """
    Helper function to run a single agent test with HTTP SSE.
    Ensures proper cleanup of async tasks to prevent inter-test conflicts.
    """
    if test_docs_root is None:
        test_docs_root = Path(tempfile.mkdtemp(prefix="agent_test_"))
        cleanup_dir = True
    else:
        cleanup_dir = False

    # Store original environment state
    old_env = os.environ.get("DOCUMENT_ROOT_DIR")
    old_host_env = os.environ.get("MCP_SERVER_HOST")
    old_port_env = os.environ.get("MCP_SERVER_PORT")

    # Set environment for the test
    os.environ["DOCUMENT_ROOT_DIR"] = str(test_docs_root)
    os.environ["MCP_SERVER_HOST"] = "localhost"
    os.environ["MCP_SERVER_PORT"] = str(server_port)

    # Configure server path
    old_path = doc_tool_server.DOCS_ROOT_PATH
    doc_tool_server.DOCS_ROOT_PATH = test_docs_root

    result = None
    initial_tasks = asyncio.all_tasks()

    try:
        async with mcp_server_context(test_docs_root=test_docs_root, port=server_port):
            agent, server_instance = await initialize_agent_and_mcp_server()
            async with agent.run_mcp_servers():
                result = await asyncio.wait_for(
                    process_single_user_query(agent, query),
                    timeout=40.0  # Generous timeout for agent processing
                )
        return result
    finally:
        # --- Comprehensive Task and Environment Cleanup ---

        # 1. Cancel any new tasks created during the test
        current_tasks = asyncio.all_tasks()
        new_tasks = current_tasks - initial_tasks
        for task in new_tasks:
            if not task.done():
                task.cancel()
        
        if new_tasks:
            await asyncio.gather(*new_tasks, return_exceptions=True)

        # 2. Restore environment variables
        if old_env is None:
            os.environ.pop("DOCUMENT_ROOT_DIR", None)
        else:
            os.environ["DOCUMENT_ROOT_DIR"] = old_env

        if old_host_env is None:
            os.environ.pop("MCP_SERVER_HOST", None)
        else:
            os.environ["MCP_SERVER_HOST"] = old_host_env

        if old_port_env is None:
            os.environ.pop("MCP_SERVER_PORT", None)
        else:
            os.environ["MCP_SERVER_PORT"] = old_port_env

        # 3. Restore server path
        doc_tool_server.DOCS_ROOT_PATH = old_path

        # 4. Clean up temporary directory
        if cleanup_dir:
            shutil.rmtree(test_docs_root, ignore_errors=True)

# --- Core Agent Test Cases ---

@pytest.mark.asyncio
async def test_agent_list_documents_empty(test_docs_root):
    """Test that listing documents works when no documents exist."""
    response = await run_agent_test("List all documents", test_docs_root, server_port=3010)
    
    assert response is not None
    assert "document" in response.summary.lower()
    
    # Should indicate empty list or no documents
    if isinstance(response.details, list):
        assert len(response.details) == 0, "Should return empty list when no documents exist"
    else:
        summary_lower = response.summary.lower()
        assert any(phrase in summary_lower for phrase in ["no", "empty", "0", "zero"]), \
            f"Should indicate no documents found: {response.summary}"

@pytest.mark.asyncio
async def test_agent_create_document_and_list(test_docs_root):
    """Test creating a document and then listing it."""
    doc_name = f"test_doc_{uuid.uuid4().hex[:8]}"
    
    # Create document
    create_response = await run_agent_test(
        f"Create a new document named '{doc_name}'", 
        test_docs_root, 
        server_port=3011
    )
    assert create_response is not None
    assert "created" in create_response.summary.lower() or "success" in create_response.summary.lower()
    
    # Verify document directory exists
    doc_dir = test_docs_root / doc_name
    assert doc_dir.exists(), f"Document directory {doc_name} should exist after creation"
    
    # List documents
    list_response = await run_agent_test("List documents", test_docs_root, server_port=3012)
    assert list_response is not None
    
    # Should find our document in the list
    if isinstance(list_response.details, list):
        doc_names = [doc.document_name for doc in list_response.details if hasattr(doc, 'document_name')]
        assert doc_name in doc_names, f"Created document {doc_name} should appear in document list"

@pytest.mark.asyncio
async def test_agent_add_chapter_to_document(test_docs_root):
    """Test adding a chapter to a document that was created by the agent."""
    doc_name = f"doc_chapter_{uuid.uuid4().hex[:8]}"
    chapter_name = "chapter1.md"
    chapter_content = "This is the first chapter."

    # Step 1: Create the document using the agent
    create_doc_response = await run_agent_test(
        f"Create a new document named '{doc_name}'",
        test_docs_root,
        server_port=3013 # Use a unique port for this test step
    )
    assert create_doc_response is not None, "Agent should respond to document creation"
    assert "created" in create_doc_response.summary.lower() or "success" in create_doc_response.summary.lower(), \
        f"Document creation failed, summary: {create_doc_response.summary}"

    # Step 2: Add a chapter to the newly created document
    add_chapter_response = await run_agent_test(
        f"Add a chapter named '{chapter_name}' to document '{doc_name}' with content: {chapter_content}",
        test_docs_root,
        server_port=3014 # Use a different port for the next step
    )
    
    assert add_chapter_response is not None, "Agent should respond to chapter addition"
    assert "added" in add_chapter_response.summary.lower() or "created" in add_chapter_response.summary.lower(), \
        f"Chapter addition failed, summary: {add_chapter_response.summary}"
    
    # Verify chapter file exists with correct content
    chapter_path = test_docs_root / doc_name / chapter_name
    assert chapter_path.exists(), f"Chapter file {chapter_name} should exist after creation"
    
    actual_content = chapter_path.read_text()
    assert chapter_content in actual_content, f"Chapter should contain expected content"

@pytest.mark.asyncio
async def test_agent_read_chapter_content(test_docs_root):
    """Test reading chapter content."""
    doc_name = "readable_doc_test"
    chapter_name = "readable_chapter.md"
    content = "Content to be read."

    # Set up test data
    doc_dir = test_docs_root / doc_name
    doc_dir.mkdir(parents=True, exist_ok=True)
    chapter_path = doc_dir / chapter_name
    chapter_path.write_text(content)
    
    # Read chapter
    response = await run_agent_test(
        f"Read chapter '{chapter_name}' from document '{doc_name}'",
        test_docs_root,
        server_port=3014
    )
    
    assert response is not None
    assert "read" in response.summary.lower() or "content" in response.summary.lower()
    
    # Should return chapter content
    if hasattr(response.details, 'content'):
        assert content in response.details.content
        assert response.details.document_name == doc_name
        assert response.details.chapter_name == chapter_name

@pytest.mark.asyncio
async def test_agent_update_chapter_content(test_docs_root):
    """Test updating chapter content."""
    doc_name = f"doc_for_update_{uuid.uuid4().hex[:8]}"
    chapter_name = "chapter_to_update.md"
    initial_content = "Initial content."
    updated_content = "Updated content here."

    # Set up test data
    doc_dir = test_docs_root / doc_name
    doc_dir.mkdir(parents=True, exist_ok=True)
    chapter_path = doc_dir / chapter_name
    chapter_path.write_text(initial_content)
    
    # Update chapter
    response = await run_agent_test(
        f"Update chapter '{chapter_name}' in document '{doc_name}' with new content: {updated_content}",
        test_docs_root,
        server_port=3015
    )
    
    assert response is not None
    assert "updated" in response.summary.lower() or "modified" in response.summary.lower()
    
    # Verify content was updated
    actual_content = chapter_path.read_text()
    assert updated_content in actual_content, "Chapter content should be updated"

@pytest.mark.asyncio
async def test_agent_get_document_statistics(test_docs_root):
    """Test getting document statistics, accepting dict as a fallback for LLM non-conformance."""
    doc_name = "stats_doc_test"
    
    # Set up test data
    doc_dir = test_docs_root / doc_name
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "chap1.md").write_text("One two three. Four five.")
    (doc_dir / "chap2.md").write_text("Six seven. Eight nine ten.")
    
    # Use an explicit query to guide the LLM
    response = await run_agent_test(
        f"Get document statistics for '{doc_name}'",
        test_docs_root,
        server_port=3016
    )
    
    assert response is not None
    assert "statistics" in response.summary.lower() or "stats" in response.summary.lower()
    
    # --- Validation for LLM non-conformance ---
    # The LLM may fail to return a StatisticsReport object.
    # We will accept a dict as long as it contains the correct data.
    
    details = response.details
    assert details is not None, "Statistics response should contain details"

    if isinstance(details, dict):
        # Handle dict response from a non-conformant LLM
        assert 'word_count' in details, "Response should have 'word_count'"
        assert 'paragraph_count' in details, "Response should have 'paragraph_count'"
        assert details['word_count'] >= 8
        assert details['paragraph_count'] >= 2
    elif hasattr(details, 'word_count') and hasattr(details, 'paragraph_count'):
        # Handle the correct StatisticsReport object
        assert details.word_count >= 8
        assert details.paragraph_count >= 2
    else:
        pytest.fail(f"Response details are not a valid StatisticsReport or a fallback dict. Got {type(details)}")

@pytest.mark.asyncio
async def test_agent_find_text_in_document(test_docs_root):
    """Test finding text in a document."""
    doc_name = "find_text_doc_test"
    text_to_find = "unique_keyword"
    
    # Set up test data
    doc_dir = test_docs_root / doc_name
    doc_dir.mkdir(parents=True, exist_ok=True)
    
    (doc_dir / "c1.md").write_text(f"First chapter with {text_to_find}.")
    (doc_dir / "c2.md").write_text(f"Second chapter, also with {text_to_find}.")
    (doc_dir / "c3.md").write_text("Third chapter, no keyword.")
    
    # Search for text
    response = await run_agent_test(
        f"Find text '{text_to_find}' in document '{doc_name}'",
        test_docs_root,
        server_port=3017
    )
    
    assert response is not None
    assert text_to_find in response.summary.lower()
    assert doc_name in response.summary.lower()
    
    # Should find matches
    if isinstance(response.details, list):
        assert len(response.details) > 0, "Should find matches for the search term"
        # Check that results contain the search term
        for result in response.details:
            if hasattr(result, 'content'):
                assert text_to_find in result.content

# Cleanup function to ensure all test artifacts are removed
def pytest_sessionfinish(session, exitstatus):
    """Clean up any remaining test artifacts after all tests complete."""
    try:
        # Clean up any remaining temporary directories
        import tempfile
        temp_dir = Path(tempfile.gettempdir())
        for item in temp_dir.glob("agent_test_*"):
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
        
        # Ensure doc_tool_server path is restored to default
        if hasattr(doc_tool_server, 'DOCS_ROOT_PATH'):
            default_path = Path.cwd() / ".documents_storage"
            doc_tool_server.DOCS_ROOT_PATH = default_path
        
        # Clean up environment variables
        if "DOCUMENT_ROOT_DIR" in os.environ:
            del os.environ["DOCUMENT_ROOT_DIR"]
            
    except Exception as e:
        print(f"Warning: Could not fully clean up after tests: {e}") 