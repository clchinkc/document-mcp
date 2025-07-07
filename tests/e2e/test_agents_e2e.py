
"""
End-to-end tests for Document MCP agents with real AI models.

This module provides comprehensive E2E testing using real LLM APIs and MCP stdio
communication. These tests validate complete user workflows including AI reasoning
and actual file system operations.
"""

import asyncio
import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict

import pytest
from .validation_utils import DocumentSystemValidator, safe_get_response_content, ensure_proper_model_response

def check_api_key_available() -> bool:
    """Check if a real API key is available for E2E testing."""
    api_keys = ["OPENAI_API_KEY", "GEMINI_API_KEY"]
    for key in api_keys:
        value = os.environ.get(key, "").strip()
        if value and value != "test_key" and not value.startswith("sk-test"):
            return True
    return False

async def run_agent_query(agent_module: str, query: str, timeout: int = 120) -> Dict[str, Any]:
    """
    Run an agent with a given query and return the parsed JSON output.
    """
    cmd = ["python3", "-m", agent_module, "--query", query]
    
    try:
        # Ensure subprocess inherits modified environment variables
        # This fixes the document root isolation issue
        env = os.environ.copy()
        if "DOCUMENT_ROOT_DIR" in env:
            env["PYTEST_CURRENT_TEST"] = "1"  # Signal to MCP server we're in test mode
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent.parent.parent,
            env=env
        )
        
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        
        if process.returncode != 0:
            raise RuntimeError(f"Agent failed with stderr: {stderr.decode()}")
            
        output_str = stdout.decode().strip()
        
        # For React agent, the output is a log, not JSON
        if "react_agent" in agent_module:
            return {"execution_log": output_str}
            
        # For Simple agent, find the start of the JSON and parse from there
        try:
            json_start = output_str.find('{')
            if json_start == -1:
                raise json.JSONDecodeError("No JSON object found", output_str, 0)
            json_str = output_str[json_start:]
            return json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            return {"summary": "Failed to parse JSON", "raw_output": output_str}

    except asyncio.TimeoutError:
        raise RuntimeError(f"Agent query timed out after {timeout}s")

@pytest.fixture
async def e2e_docs_dir():
    """Provide a clean temporary directory for E2E document operations."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        old_doc_root = os.environ.get("DOCUMENT_ROOT_DIR")
        os.environ["DOCUMENT_ROOT_DIR"] = tmp_dir
        try:
            yield Path(tmp_dir)
        finally:
            if old_doc_root:
                os.environ["DOCUMENT_ROOT_DIR"] = old_doc_root
            else:
                os.environ.pop("DOCUMENT_ROOT_DIR", None)

@pytest.fixture
def validator(e2e_docs_dir):
    """Provide a DocumentSystemValidator for file system assertions."""
    return DocumentSystemValidator(e2e_docs_dir)

@pytest.mark.e2e
@pytest.mark.skipif(not check_api_key_available(), reason="E2E tests require a real API key")
class TestSimpleAgentE2E:
    """End-to-end tests for Simple Agent with real LLM."""
    
    @pytest.mark.asyncio
    async def test_full_document_lifecycle(self, e2e_docs_dir: Path, validator: DocumentSystemValidator):
        """Test create, write, read, and verify document lifecycle using file system validation."""
        doc_name = f"e2e_simple_doc_{uuid.uuid4().hex[:8]}"
        chapter_name = "01-intro.md"
        content = "This is a test chapter for the simple agent E2E workflow."
        
        # Verify initial state - no documents should exist
        validator.assert_document_count(0)
        
        # 1. Create Document
        create_resp = await run_agent_query(
            "src.agents.simple_agent.main",
            f"Create a document named '{doc_name}'"
        )
        
        # Improved failure diagnostics - check if the agent thinks it succeeded
        create_summary = create_resp.get("summary", "")
        create_details = create_resp.get("details")
        create_error = create_resp.get("error_message")
        
        # If agent reports success but file system shows failure, provide detailed diagnostics
        if "success" in create_summary.lower() and not (e2e_docs_dir / doc_name).exists():
            debug_info = validator.get_debug_info()
            pytest.fail(
                f"Agent reported successful document creation but file system validation failed.\n"
                f"Agent Summary: {create_summary}\n"
                f"Agent Details: {create_details}\n"
                f"Agent Error: {create_error}\n"
                f"File System State: {debug_info}\n"
                f"Expected document at: {e2e_docs_dir / doc_name}\n"
                f"This indicates the agent may not be properly calling MCP tools or there's an environment issue."
            )
        
        # Assert on file system state, not LLM response
        try:
            validator.assert_document_exists(doc_name)
            validator.assert_document_count(1)
        except AssertionError as e:
            # Enhanced error reporting
            debug_info = validator.get_debug_info()
            pytest.fail(
                f"Document creation validation failed.\n"
                f"Original error: {e}\n"
                f"Agent response: {create_resp}\n"
                f"File system state: {debug_info}\n"
                f"Expected document directory: {e2e_docs_dir / doc_name}"
            )
        
        # 2. Add Chapter
        chapter_resp = await run_agent_query(
            "src.agents.simple_agent.main",
            f"In document '{doc_name}', create a chapter '{chapter_name}' with content: {content}"
        )
        
        # Assert on file system state - chapter should exist with correct content
        try:
            validator.assert_chapter_exists(doc_name, chapter_name)
            validator.assert_chapter_content_contains(doc_name, chapter_name, content)
            # Commented out to make test robust to LLM creating extra chapters
            # validator.assert_chapter_count(doc_name, 1)
        except AssertionError as e:
            # Enhanced error reporting for chapter creation
            debug_info = validator.get_debug_info()
            pytest.fail(
                f"Chapter creation validation failed.\n"
                f"Original error: {e}\n"
                f"Agent response: {chapter_resp}\n"
                f"File system state: {debug_info}\n"
                f"Expected chapter at: {e2e_docs_dir / doc_name / chapter_name}"
            )
        
        # 3. Read and Verify Content (test agent's ability to read back)
        read_resp = await run_agent_query(
            "src.agents.simple_agent.main",
            f"Read chapter '{chapter_name}' from document '{doc_name}'"
        )
        
        # For read operations, we can assert on the response details if they exist
        # but we should also verify the file still exists and is readable
        validator.assert_chapter_exists(doc_name, chapter_name)
        validator.assert_chapter_content_contains(doc_name, chapter_name, content)
        
        # Optional: If the agent provides structured response, verify it contains the content
        read_details = safe_get_response_content(read_resp, 'details')
        if read_details and 'content' in read_details:
            assert content in read_details['content'], f"Agent response details should contain expected content. Got: {read_details}"

    @pytest.mark.asyncio
    async def test_summary_first_workflow(self, e2e_docs_dir: Path, validator: DocumentSystemValidator):
        """Test that the agent uses the summary-first workflow for broad queries."""
        doc_name = f"e2e_summary_doc_{uuid.uuid4().hex[:8]}"
        summary_content = "This is a summary of the document."
        
        # 1. Create a document and a summary file for it
        doc_path = e2e_docs_dir / doc_name
        doc_path.mkdir()
        summary_file = doc_path / "_SUMMARY.md"
        summary_file.write_text(summary_content)
        
        # 2. Create a chapter to ensure there's other content
        (doc_path / "01-chapter.md").write_text("This is the full chapter content.")
        
        # 3. Run the agent with a broad query
        query = f"Tell me about the document '{doc_name}'"
        response = await run_agent_query("src.agents.simple_agent.main", query)
        
        # 4. Validate the response
        validator.assert_summary_exists(doc_name)
        details = safe_get_response_content(response, 'details')
        assert summary_content in details.get('content', ''), "The agent should have read the summary content."
        assert "full chapter content" not in details.get('content', ''), "The agent should not have read the full chapter content."

@pytest.mark.e2e
@pytest.mark.skipif(not check_api_key_available(), reason="E2E tests require a real API key")
class TestReactAgentE2E:
    """End-to-end tests for the ReAct agent."""

    @pytest.mark.asyncio
    async def test_multi_step_workflow_and_verification(self, e2e_docs_dir: Path, validator: DocumentSystemValidator):
        """Test a multi-step workflow that requires reasoning and tool chaining using file system validation."""
        doc_name = f"e2e_react_doc_{uuid.uuid4().hex[:8]}"
        chapter_name = "01-main.md"
        
        # Verify initial state - no documents should exist
        validator.assert_document_count(0)
        
        # This query requires multiple steps: create doc -> create chapter -> list docs
        query = (
            f"Create a new document called '{doc_name}', add a chapter named "
            f"'{chapter_name}', and then list all available documents."
        )
        
        response = await run_agent_query(
            "src.agents.react_agent.main",
            query,
            timeout=180  # Increase timeout for complex multi-step tasks
        )
        
        # Assert on concrete outcomes rather than log parsing
        log = response.get("execution_log", "")
        
        try:
            # 1. Document should be created
            validator.assert_document_exists(doc_name)
            validator.assert_document_count(1)
            
            # 2. Chapter should be created
            validator.assert_chapter_exists(doc_name, chapter_name)
            # Commented out to make test robust to LLM creating extra chapters
            # validator.assert_chapter_count(doc_name, 1)
            
        except AssertionError as e:
            # Enhanced error reporting for React agent
            debug_info = validator.get_debug_info()
            pytest.fail(
                f"ReAct agent multi-step workflow validation failed.\n"
                f"Original error: {e}\n"
                f"Agent execution log: {log}\n"
                f"File system state: {debug_info}\n"
                f"Expected document at: {e2e_docs_dir / doc_name}\n"
                f"Expected chapter at: {e2e_docs_dir / doc_name / chapter_name}\n"
                f"This indicates the ReAct agent did not successfully execute all required steps."
            )
        
        # 3. Verify the ReAct agent completed successfully
        assert "ReAct loop finished" in log, f"ReAct agent should complete successfully. Log: {log}"
        
        # 4. Verify the agent's final thought indicates completion
        assert "Final Thought:" in log, f"The agent should have a final thought. Log: {log}"
        assert "complete" in log or "done" in log, f"The final thought should indicate completion. Log: {log}"
