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

# --- Simplified Server Management ---

def _get_worker_port():
    """Get a unique port for this pytest worker to avoid conflicts."""
    base_port = 3001
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

class SimpleServerManager:
    """Simplified MCP server manager for testing."""
    
    def __init__(self, test_docs_root=None, port=None):
        self.port = port or _get_worker_port()
        self.test_docs_root = test_docs_root
        self.process = None
        self.server_url = f"http://localhost:{self.port}"
        
    def start_server(self):
        """Start the MCP server process synchronously."""
        if self.process is not None:
            return  # Already started
            
        # Set environment for the server process
        env = os.environ.copy()
        if self.test_docs_root:
            env["DOCUMENT_ROOT_DIR"] = str(self.test_docs_root)
            
        # Start the server process
        cmd = [sys.executable, "-m", "document_mcp.doc_tool_server", "sse", "--host", "localhost", "--port", str(self.port)]
        
        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to be ready
        self._wait_for_server()
        
    def _wait_for_server(self, timeout=30):
        """Wait for the server to be ready to accept connections."""
        print(f"Waiting for MCP server at {self.server_url}")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.server_url}/health", timeout=2)
                if response.status_code == 200:
                    print("MCP server ready")
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(1.0)
            
        raise TimeoutError(f"MCP server did not start within {timeout} seconds")
        
    def stop_server(self):
        """Stop the MCP server process."""
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
            finally:
                self.process = None

# --- Pytest Fixtures ---

@pytest.fixture
def server_manager():
    """Session-scoped server manager to avoid starting/stopping server for each test."""
    # Create a temporary docs root for the entire test session
    temp_root = Path(tempfile.mkdtemp(prefix="agent_test_session_"))
    
    # Use worker-specific port
    port = _get_worker_port()
    manager = SimpleServerManager(test_docs_root=temp_root, port=port)
    manager.start_server()
    
    yield manager
    
    # Cleanup
    manager.stop_server()
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)

@pytest.fixture
def test_docs_root(server_manager):
    """Create a clean subdirectory for each test."""
    test_id = str(uuid.uuid4().hex[:8])
    test_subdir = server_manager.test_docs_root / f"docs_{test_id}"
    test_subdir.mkdir()
    
    # Override doc_tool_server paths for this test
    original_server_path = doc_tool_server.DOCS_ROOT_PATH
    doc_tool_server.DOCS_ROOT_PATH = test_subdir
    
    # Set environment variable with worker-specific port
    original_env_var = os.environ.get("DOCUMENT_ROOT_DIR")
    original_port_var = os.environ.get("MCP_SERVER_PORT")
    os.environ["DOCUMENT_ROOT_DIR"] = str(test_subdir)
    os.environ["MCP_SERVER_PORT"] = str(server_manager.port)
    
    yield test_subdir
    
    # Cleanup test data
    if test_subdir.exists():
        shutil.rmtree(test_subdir, ignore_errors=True)
    
    # Restore original paths and environment
    doc_tool_server.DOCS_ROOT_PATH = original_server_path
    if original_env_var is None:
        if "DOCUMENT_ROOT_DIR" in os.environ:
            del os.environ["DOCUMENT_ROOT_DIR"]
    else:
        os.environ["DOCUMENT_ROOT_DIR"] = original_env_var
        
    if original_port_var is None:
        if "MCP_SERVER_PORT" in os.environ:
            del os.environ["MCP_SERVER_PORT"]
    else:
        os.environ["MCP_SERVER_PORT"] = original_port_var

# --- Simplified Test Helper ---

async def run_simple_agent_test(query: str):
    """Simplified agent test runner."""
    try:
        agent, _ = await initialize_agent_and_mcp_server()
        async with agent.run_mcp_servers():
            result = await asyncio.wait_for(
                process_single_user_query(agent, query),
                timeout=30.0
            )
            return result
    except Exception as e:
        return FinalAgentResponse(
            summary=f"Error during processing: {str(e)}",
            details=None,
            error_message=str(e)
        )

# --- Core Agent Test Cases ---

@pytest.mark.asyncio
async def test_agent_list_documents_empty(test_docs_root):
    """Test that listing documents works when no documents exist."""
    response = await run_simple_agent_test("Show me all available documents")
    
    assert response is not None, "Agent should provide a response to document listing request"
    assert isinstance(response.summary, str), "Summary must be a string"
    assert isinstance(response.details, list), "Details for list_documents must be a list"

@pytest.mark.asyncio
async def test_agent_create_document_and_list(test_docs_root):
    """Test creating a document and then listing it."""
    doc_name = f"test_doc_{uuid.uuid4().hex[:8]}"
    
    # Create document
    create_response = await run_simple_agent_test(f"Create a new document named '{doc_name}'")
    assert create_response is not None, "Agent should respond to document creation request"
    assert isinstance(create_response.summary, str) and len(create_response.summary) > 0, "Creation response should have meaningful summary"

@pytest.mark.asyncio
async def test_agent_add_chapter_to_document(test_docs_root):
    """Test adding a chapter to a document that was created by the agent."""
    doc_name = f"doc_chapter_{uuid.uuid4().hex[:8]}"
    chapter_name = "chapter1.md"
    chapter_content = "This is the first chapter."

    # Step 1: Create the document using the agent
    create_doc_response = await run_simple_agent_test(f"Create a new document named '{doc_name}'")
    assert create_doc_response is not None, "Agent should respond to document creation"
    assert isinstance(create_doc_response.summary, str) and len(create_doc_response.summary) > 0, \
        "Document creation response should have meaningful summary"

    # Step 2: Add a chapter to the newly created document
    add_chapter_response = await run_simple_agent_test(
        f"Create a chapter named '{chapter_name}' in document '{doc_name}' with content: {chapter_content}"
    )
    
    assert add_chapter_response is not None, "Agent should respond to chapter addition"
    assert isinstance(add_chapter_response.summary, str) and len(add_chapter_response.summary) > 0, \
        "Chapter addition response should have meaningful summary"

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
    response = await run_simple_agent_test(f"Read chapter '{chapter_name}' from document '{doc_name}'")
    
    assert response is not None, "Agent should respond to chapter reading request"
    assert isinstance(response.summary, str) and len(response.summary) > 0, "Read response should have meaningful summary"

@pytest.mark.asyncio
async def test_agent_get_document_statistics(test_docs_root):
    """Test getting document statistics."""
    doc_name = "stats_doc_test"
    
    # Set up test data
    doc_dir = test_docs_root / doc_name
    doc_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a few chapters with content
    for i in range(1, 4):
        chapter_path = doc_dir / f"0{i}-chapter.md"
        chapter_path.write_text(f"# Chapter {i}\n\nThis is chapter {i} content.")
    
    # Get statistics
    response = await run_simple_agent_test(f"Get statistics for document '{doc_name}'")
    
    assert response is not None, "Agent should respond to statistics request"
    assert isinstance(response.summary, str) and len(response.summary) > 0, "Statistics response should have meaningful summary"

@pytest.mark.asyncio 
async def test_agent_find_text_in_document(test_docs_root):
    """Test finding text in a document."""
    doc_name = "search_doc_test"
    
    # Set up test data
    doc_dir = test_docs_root / doc_name
    doc_dir.mkdir(parents=True, exist_ok=True)
    
    # Create chapters with searchable content
    chapter1_path = doc_dir / "01-intro.md"
    chapter1_path.write_text("# Introduction\n\nThis chapter contains searchable content.")
    
    chapter2_path = doc_dir / "02-body.md"
    chapter2_path.write_text("# Main Content\n\nMore searchable text here.")
    
    # Search for text
    response = await run_simple_agent_test(f"Find the text 'searchable' in document '{doc_name}'")
    
    assert response is not None, "Agent should respond to search request"
    assert isinstance(response.summary, str) and len(response.summary) > 0, "Search response should have meaningful summary" 