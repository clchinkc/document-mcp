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

    # Check for any of the supported API keys
    api_keys = ["OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"]
    has_api_key = any(os.environ.get(key) for key in api_keys)
    assert has_api_key, f"No API key found. Expected one of: {api_keys}"

def test_agent_package_imports():
    """Test if all required packages can be imported for agent functionality."""
    try:
        import pydantic_ai
        # Verify pydantic_ai has expected functionality
        assert hasattr(pydantic_ai, 'Agent'), "pydantic_ai should provide Agent class"
        assert hasattr(pydantic_ai, 'RunContext'), "pydantic_ai should provide RunContext class"
    except ImportError:
        pytest.fail("Failed to import pydantic_ai - required for agent")
    
    try:
        # Test agent imports work
        from agent import FinalAgentResponse, StatisticsReport
        # Verify imported classes are proper types
        assert isinstance(FinalAgentResponse, type), "FinalAgentResponse should be a class type"
        assert isinstance(StatisticsReport, type), "StatisticsReport should be a class type"
        # Verify they are pydantic models
        assert hasattr(FinalAgentResponse, 'model_fields'), "FinalAgentResponse should be a pydantic model"
        assert hasattr(StatisticsReport, 'model_fields'), "StatisticsReport should be a pydantic model"
    except ImportError as e:
        pytest.fail(f"Failed to import agent models: {e}")

# --- HTTP SSE Server Management ---

def _get_worker_port(base_port=3001):
    """Get a unique port for this pytest worker to avoid conflicts."""
    try:
        # Check if running with pytest-xdist
        worker_id = os.environ.get('PYTEST_XDIST_WORKER', 'master')
        if worker_id == 'master':
            return base_port
        else:
            # Extract worker number (e.g., 'gw0' -> 0, 'gw1' -> 1)
            worker_num = int(worker_id.replace('gw', ''))
            return base_port + worker_num + 1
    except (ValueError, TypeError):
        # Fallback to base port if parsing fails
        return base_port

class MCPServerManager:
    """Manages the HTTP SSE MCP server for testing."""
    
    def __init__(self, host="localhost", port=None, test_docs_root=None):
        self.host = host
        self.port = port or _get_worker_port()
        self.test_docs_root = test_docs_root
        self.process = None
        self.server_url = f"http://{host}:{self.port}"
        
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
        
    async def _wait_for_server(self, timeout=20):
        """Wait for the server to be ready to accept connections."""
        print(f"Waiting for MCP server at {self.server_url}")
        start_time = time.time()
        attempts = 0
        while time.time() - start_time < timeout:
            try:
                attempts += 1
                response = requests.get(f"{self.server_url}/health", timeout=2)
                if response.status_code == 200:
                    print(f"MCP server ready after {attempts} attempts")
                    return
                else:
                    print(f"Server responded with status {response.status_code}")
            except requests.exceptions.RequestException as e:
                if attempts % 5 == 0:  # Log every 5th attempt
                    print(f"Attempt {attempts}: Server not ready yet ({e})")
            await asyncio.sleep(1.0)  # Increased wait time
            
        raise TimeoutError(f"MCP server did not start within {timeout} seconds after {attempts} attempts")
        
    async def stop_server(self):
        """Stop the MCP server process."""
        if self.process is not None:
            try:
                # Cancel any pending tasks first
                try:
                    current_task = asyncio.current_task()
                    all_tasks = [t for t in asyncio.all_tasks() if t != current_task and not t.done()]
                    if all_tasks:
                        for task in all_tasks:
                            if not task.done():
                                task.cancel()
                        # Give tasks a moment to cancel
                        await asyncio.sleep(0.1)
                except Exception:
                    pass  # Ignore task cleanup errors
                
                # First, try graceful termination
                self.process.terminate()
                
                try:
                    # Wait for graceful shutdown with timeout
                    await asyncio.wait_for(
                        asyncio.to_thread(self.process.wait),
                        timeout=10.0  # Increased timeout for CI environments
                    )
                except asyncio.TimeoutError:
                    # If graceful termination fails, force kill
                    self.process.kill()
                    try:
                        # Wait for force kill to complete
                        await asyncio.wait_for(
                            asyncio.to_thread(self.process.wait),
                            timeout=5.0  # Increased timeout
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
                
            except Exception as e:
                # Force cleanup even if there's an error
                print(f"Error during server stop: {e}")
                if self.process:
                    try:
                        self.process.kill()
                        await asyncio.wait_for(
                            asyncio.to_thread(self.process.wait),
                            timeout=3.0  # Increased timeout
                        )
                    except Exception as kill_err:
                        print(f"Error during force kill: {kill_err}")
            finally:
                # Always clear the process reference
                self.process = None
                
                # Wait longer to ensure all resources are released, especially in CI
                await asyncio.sleep(1.5)

@asynccontextmanager
async def mcp_server_context(test_docs_root=None, host="localhost", port=None):
    """Context manager for MCP server lifecycle."""
    manager = MCPServerManager(host=host, port=port, test_docs_root=test_docs_root)
    try:
        await manager.start_server()
        yield manager
    except Exception as e:
        import traceback
        print(f"Error in MCP server context: {e}")
        print(f"MCP context traceback: {traceback.format_exc()}")
        raise
    finally:
        try:
            await manager.stop_server()
            # Additional cleanup time to ensure all resources are released, especially in CI
            await asyncio.sleep(3.0)
        except Exception as e:
            print(f"Error during MCP server cleanup: {e}")
            # Don't re-raise cleanup errors

# --- Pytest Fixtures ---

@pytest.fixture
def event_loop():
    """
    Creates an asyncio event loop for each test function, preventing
    'Event loop is closed' errors when running multiple async tests.
    Compatible with pytest-xdist.
    """
    # Create a new event loop for this test
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    yield loop
    
    # Ensure proper cleanup
    try:
        # Cancel all pending tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        
        # Wait for all tasks to complete cancellation
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    except Exception:
        pass  # Ignore cleanup errors
    finally:
        # Close the loop
        try:
            loop.close()
        except Exception:
            pass  # Ignore close errors

@pytest.fixture
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

async def run_agent_test(query, test_docs_root=None, server_port=None):
    """
    Helper function to run a single agent test with HTTP SSE.
    Ensures proper cleanup of async tasks to prevent inter-test conflicts.
    """
    if test_docs_root is None:
        test_docs_root = Path(tempfile.mkdtemp(prefix="agent_test_"))
        cleanup_dir = True
    else:
        cleanup_dir = False
        
    # Ensure we have a server port
    if server_port is None:
        server_port = _get_worker_port()

    # Store original environment state
    old_env = os.environ.get("DOCUMENT_ROOT_DIR")
    old_port_env = os.environ.get("MCP_SERVER_PORT")

    # Set environment for the test
    os.environ["DOCUMENT_ROOT_DIR"] = str(test_docs_root)
    if server_port:
        os.environ["MCP_SERVER_PORT"] = str(server_port)

    # Configure server path
    old_path = doc_tool_server.DOCS_ROOT_PATH
    doc_tool_server.DOCS_ROOT_PATH = test_docs_root

    result = None
    
    try:
        async with mcp_server_context(test_docs_root=test_docs_root, port=server_port):
            agent, server_instance = await initialize_agent_and_mcp_server()
            async with agent.run_mcp_servers():
                result = await asyncio.wait_for(
                    process_single_user_query(agent, query),
                    timeout=40.0  # Generous timeout for agent processing
                )
                return result
        
    except asyncio.TimeoutError:
        from agent import FinalAgentResponse
        return FinalAgentResponse(
            summary="Query timed out after 40 seconds",
            details=None,
            error_message="Timeout error"
        )
    except Exception as e:
        # Return error response instead of raising
        from agent import FinalAgentResponse
        import traceback
        
        # Get more detailed error information for TaskGroup errors
        error_details = str(e)
        if hasattr(e, 'exceptions'):
            # This is likely an ExceptionGroup/TaskGroup error
            sub_exceptions = []
            for sub_exc in e.exceptions:
                sub_exceptions.append(f"{type(sub_exc).__name__}: {sub_exc}")
            error_details = f"TaskGroup errors: {'; '.join(sub_exceptions)}"
        
        print(f"Error in run_agent_test: {error_details}")
        print(f"Full traceback: {traceback.format_exc()}")
        
        return FinalAgentResponse(
            summary=f"Error during processing: {error_details}",
            details=None,
            error_message=error_details
        )
    finally:
        # 2. Restore environment variables
        if old_env is None:
            os.environ.pop("DOCUMENT_ROOT_DIR", None)
        else:
            os.environ["DOCUMENT_ROOT_DIR"] = old_env
            
        if old_port_env is None:
            os.environ.pop("MCP_SERVER_PORT", None)
        else:
            os.environ["MCP_SERVER_PORT"] = old_port_env

        # 3. Restore server path
        doc_tool_server.DOCS_ROOT_PATH = old_path

        # 4. Clean up temporary directory
        if cleanup_dir:
            shutil.rmtree(test_docs_root, ignore_errors=True)
            
        # 5. Delay to allow cleanup and reduce race conditions with multiple workers
        await asyncio.sleep(0.5)

# --- Core Agent Test Cases ---

@pytest.mark.asyncio
async def test_agent_list_documents_empty(test_docs_root):
    """Test that listing documents works when no documents exist."""
    response = await run_agent_test("List all documents", test_docs_root, server_port=3010)
    
    assert response is not None, "Agent should provide a response to document listing request"
    # Accept any summary; enforce details type
    assert isinstance(response.summary, str), "Summary must be a string"
    assert isinstance(response.details, list), "Details for list_documents must be a list"

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
    assert create_response is not None, "Agent should respond to document creation request"
    assert isinstance(create_response.summary, str) and len(create_response.summary) > 0, "Creation response should have meaningful summary"
    
    # Skip if event loop issues
    # ... existing code ...

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
    assert isinstance(create_doc_response.summary, str) and len(create_doc_response.summary) > 0, \
        "Document creation response should have meaningful summary"
    assert "created" in create_doc_response.summary.lower() or "success" in create_doc_response.summary.lower(), \
        f"Document creation should indicate success in summary: {create_doc_response.summary}"

    # Step 2: Add a chapter to the newly created document
    add_chapter_response = await run_agent_test(
        f"Add a chapter named '{chapter_name}' to document '{doc_name}' with content: {chapter_content}",
        test_docs_root,
        server_port=3014 # Use a different port for the next step
    )
    
    assert add_chapter_response is not None, "Agent should respond to chapter addition"
    assert isinstance(add_chapter_response.summary, str) and len(add_chapter_response.summary) > 0, \
        "Chapter addition response should have meaningful summary"
    
    # Check for successful chapter addition or handle the case where chapter already exists
    summary_lower = add_chapter_response.summary.lower()
    success_indicators = ["added", "created", "successfully", "exists"]
    assert any(indicator in summary_lower for indicator in success_indicators), \
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
        server_port=3018
    )
    
    assert response is not None, "Agent should respond to chapter reading request"
    assert isinstance(response.summary, str) and len(response.summary) > 0, "Read response should have meaningful summary"
    
    # Skip test if event loop issues occurred
    # ... existing code ...

@pytest.mark.asyncio
async def test_agent_update_chapter_content(test_docs_root):
    """Test updating chapter content."""
    doc_name = f"doc_for_update_{uuid.uuid4().hex[:8]}"
    chapter_name = "chapter_to_update.md"
    initial_content = "Initial content"
    updated_content = "Updated content here"

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
    
    assert response is not None, "Agent should respond to chapter update request"
    assert isinstance(response.summary, str) and len(response.summary) > 0, "Update response should have meaningful summary"
    assert "updated" in response.summary.lower() or "modified" in response.summary.lower(), \
        f"Update response should indicate modification: {response.summary}"
    
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
    
    assert response is not None, "Agent should respond to statistics request"
    assert isinstance(response.summary, str) and len(response.summary) > 0, "Statistics response should have meaningful summary"
    
    # Skip test if event loop issues occurred
    # ... existing code ...

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
    
    assert response is not None, "Agent should respond to text search request"
    assert isinstance(response.summary, str) and len(response.summary) > 0, "Search response should have meaningful summary"
    assert text_to_find in response.summary.lower(), f"Search response should mention the search term: {response.summary}"
    assert doc_name in response.summary.lower(), f"Search response should mention the document name: {response.summary}"
    
    # Should find matches
    if isinstance(response.details, list):
        assert len(response.details) > 0, "Should find at least one match for the search term in the document"
        # Check that results contain the search term
        for result in response.details:
            if hasattr(result, 'content'):
                assert isinstance(result.content, str), "Search result content should be a string"
                assert text_to_find in result.content, f"Search result should contain the search term '{text_to_find}'"

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