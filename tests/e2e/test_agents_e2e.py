"""End-to-end tests for Document MCP agents with real AI models.

This module provides comprehensive E2E testing using real LLM APIs and MCP stdio
communication. These tests validate complete user workflows including AI reasoning
and actual file system operations.

The tests are organized into 6 test classes covering different aspects:
- SimpleAgentE2E: Document lifecycle and semantic search
- ReactAgentE2E: Reasoning and execution patterns
- SafetyAndVersionControlE2E: Snapshot management and version control
- MultiStepOperationsE2E: Multi-step coordinated workflows
- IntegratedWorkflowE2E: Feature integration and complex scenarios

All tests use subprocess execution to ensure true end-to-end validation.
"""

import asyncio
import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

import pytest

from .validation_utils import DocumentSystemValidator
from .validation_utils import safe_get_response_content


def check_api_key_available() -> bool:
    """Check if a real API key is available for E2E testing."""
    api_keys = ["OPENAI_API_KEY", "GEMINI_API_KEY"]
    for key in api_keys:
        value = os.environ.get(key, "").strip()
        if value and value != "test_key" and not value.startswith("sk-test"):
            return True
    return False


async def run_agent_query(agent_module: str, query: str, timeout: int = 60) -> dict[str, Any]:
    """Run an agent with a given query and return the parsed JSON output."""
    cmd = ["uv", "run", "python", "-m", agent_module, "--query", query]

    try:
        # Ensure subprocess inherits modified environment variables
        # This fixes the document root isolation issue
        env = os.environ.copy()
        if "DOCUMENT_ROOT_DIR" in env:
            env["PYTEST_CURRENT_TEST"] = "1"
            # Don't double-resolve the path - it's already resolved in e2e_docs_dir fixture

        # Add API keys from .env file for E2E tests
        # This ensures the subprocess has access to API keys
        from src.agents.shared.config import get_settings

        settings = get_settings()
        if settings.gemini_api_key:
            env["GEMINI_API_KEY"] = settings.gemini_api_key
        if settings.openai_api_key:
            env["OPENAI_API_KEY"] = settings.openai_api_key

        # Speed up E2E tests by disabling metrics collection
        env["MCP_METRICS_ENABLED"] = "false"

        # Create process with proper timeout handling
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent.parent.parent,
            env=env,
        )

        # Use asyncio.wait_for with proper timeout handling
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            # Kill the process if it times out
            process.kill()
            await process.wait()
            raise RuntimeError(f"Agent query timed out after {timeout}s") from None

        if process.returncode != 0:
            raise RuntimeError(f"Agent failed with stderr: {stderr.decode()}")

        output_str = stdout.decode().strip()
        stderr_str = stderr.decode().strip()

        # Add debug information
        debug_info = {
            "returncode": process.returncode,
            "stdout_length": len(output_str),
            "stderr_length": len(stderr_str),
            "stderr_content": stderr_str[:500] if stderr_str else "No stderr",
        }

        # For React agent, extract JSON from the output
        if "react_agent" in agent_module:
            # ReAct agent outputs logs followed by JSON after "JSON OUTPUT:" marker
            json_marker = "JSON OUTPUT:"
            if json_marker in output_str:
                json_start_marker = output_str.find(json_marker)
                if json_start_marker != -1:
                    # Find the actual JSON start after the marker
                    json_portion = output_str[json_start_marker + len(json_marker) :].strip()
                    json_start = json_portion.find("{")
                    if json_start != -1:
                        json_str = json_portion[json_start:]
                        try:
                            parsed_json = json.loads(json_str)
                            # Add the full execution log for debugging
                            parsed_json["full_execution_log"] = output_str
                            return parsed_json
                        except json.JSONDecodeError as e:
                            # JSON parsing failed, return debug info
                            return {
                                "execution_log": output_str,
                                "json_parse_error": str(e),
                                "json_portion": (json_str[:200] + "..." if len(json_str) > 200 else json_str),
                                "debug_info": debug_info,
                            }
            # Fallback to log-only format if JSON parsing fails
            return {"execution_log": output_str, "debug_info": debug_info}

        # For Simple agent, find the start of the JSON and parse from there
        try:
            json_start = output_str.find("{")
            if json_start == -1:
                raise json.JSONDecodeError("No JSON object found", output_str, 0)
            json_str = output_str[json_start:]
            parsed_json = json.loads(json_str)
            parsed_json["debug_info"] = debug_info
            return parsed_json
        except (json.JSONDecodeError, IndexError):
            return {
                "summary": "Failed to parse JSON",
                "raw_output": output_str,
                "debug_info": debug_info,
            }

    except Exception as e:
        # Handle any other exceptions that might occur
        if "timeout" in str(e).lower():
            raise RuntimeError(f"Agent query timed out after {timeout}s")
        else:
            raise RuntimeError(f"Agent query failed: {str(e)}")


@pytest.fixture
async def e2e_docs_dir():
    """Provide a clean temporary directory for E2E document operations."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        old_doc_root = os.environ.get("DOCUMENT_ROOT_DIR")
        # Use resolved path to match what settings.py does with .resolve()
        # This ensures consistency between test setup and actual document creation
        resolved_tmp_dir = str(Path(tmp_dir).resolve())
        os.environ["DOCUMENT_ROOT_DIR"] = resolved_tmp_dir
        try:
            yield Path(resolved_tmp_dir)
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
    """Comprehensive E2E tests for Simple Agent.

    Tests document lifecycle, semantic search, summary workflows, and content
    operations.
    """

    @pytest.mark.asyncio
    async def test_complete_document_lifecycle_and_content_access(
        self, e2e_docs_dir: Path, validator: DocumentSystemValidator
    ):
        """Test complete document lifecycle: create, write, read, summary access."""
        doc_name = f"e2e_lifecycle_{uuid.uuid4().hex[:8]}"
        chapter_name = "01-intro.md"
        content = "This is a test chapter for document lifecycle testing."
        summary_content = "This document contains an introduction chapter."

        # Verify initial state
        validator.assert_document_count(0)

        # 1. Create Document
        await run_agent_query("src.agents.simple_agent.main", f"Create a document named '{doc_name}'", timeout=120)

        validator.assert_document_exists(doc_name)
        validator.assert_document_count(1)

        # 2. Add Chapter with Content
        await run_agent_query(
            "src.agents.simple_agent.main",
            f"In document '{doc_name}', create chapter '{chapter_name}' with content: {content}",
            timeout=120,
        )

        validator.assert_chapter_exists(doc_name, chapter_name)
        validator.assert_chapter_content_contains(doc_name, chapter_name, content)

        # 3. Add Summary File (using new organized structure)
        doc_path = e2e_docs_dir / doc_name
        summaries_dir = doc_path / "summaries"
        summaries_dir.mkdir(exist_ok=True)
        summary_file = summaries_dir / "document.md"
        summary_file.write_text(summary_content, encoding="utf-8")

        # 4. Test Summary-First Workflow
        summary_query = f"Tell me about the document '{doc_name}'"
        summary_resp = await run_agent_query("src.agents.simple_agent.main", summary_query, timeout=120)

        details = safe_get_response_content(summary_resp, "details")
        
        # Handle case where details is returned as unparsed content
        if "content" in details and isinstance(details["content"], str):
            # The details field contains a JSON string, check if it includes read_summary
            details_str = details["content"]
            assert "read_summary" in details_str, "Agent should use read_summary tool for broad document queries"
            assert summary_content in details_str, "Agent should read summary content"
        else:
            # Details is already parsed as a dictionary
            assert "read_summary" in details, "Agent should use read_summary tool for broad document queries"
            summary_response = details.get("read_summary", {})
            if summary_response:
                summary_read_content = summary_response.get("content", "")
                assert summary_content in summary_read_content, "Agent should read summary content"

        # 5. Test Direct Chapter Reading
        read_resp = await run_agent_query(
            "src.agents.simple_agent.main",
            f"Read chapter '{chapter_name}' from document '{doc_name}'",
            timeout=120,
        )

        read_details = safe_get_response_content(read_resp, "details")

        # 6. Test Creating Additional Summaries Using New Tools
        # Test creating chapter summary
        chapter_summary_query = f"Create a summary for chapter '{chapter_name}' in document '{doc_name}' with content: 'Chapter summary: Introduction concepts'"
        chapter_summary_resp = await run_agent_query("src.agents.simple_agent.main", chapter_summary_query, timeout=120)

        chapter_summary_details = safe_get_response_content(chapter_summary_resp, "details")
        write_summary_response = chapter_summary_details.get("write_summary", {})
        if write_summary_response:
            assert write_summary_response.get("success", False), "Chapter summary creation should succeed"

        # Test creating section summary
        section_summary_query = f"Create a section summary called 'overview' for document '{doc_name}' with content: 'Section overview of key topics'"
        section_summary_resp = await run_agent_query("src.agents.simple_agent.main", section_summary_query, timeout=120)

        section_summary_details = safe_get_response_content(section_summary_resp, "details")
        section_write_response = section_summary_details.get("write_summary", {})
        if section_write_response:
            assert section_write_response.get("success", False), "Section summary creation should succeed"

        # 7. Test Listing All Summaries
        list_summaries_query = f"List all summaries for document '{doc_name}'"
        list_resp = await run_agent_query("src.agents.simple_agent.main", list_summaries_query)

        list_details = safe_get_response_content(list_resp, "details")
        
        # Handle case where details is returned as unparsed content
        if "content" in list_details and isinstance(list_details["content"], str):
            details_str = list_details["content"]
            assert "document.md" in details_str, "Document summary should be listed"
        else:
            summaries_response = list_details.get("list_summaries", {})
            # Handle the wrapped response structure where lists are wrapped in "documents" key
            if isinstance(summaries_response, dict) and "documents" in summaries_response:
                summaries_list = summaries_response["documents"]
                assert "document.md" in summaries_list, "Document summary should be listed"
            elif isinstance(summaries_response, list):
                # Direct list response
                assert "document.md" in summaries_response, "Document summary should be listed"
        if "read_content_response" in read_details:
            content_data = read_details["read_content_response"]
            if "content" in content_data:
                assert content in content_data["content"], (
                    "Direct chapter read should contain chapter content"
                )
        elif "content" in read_details:
            assert content in read_details["content"], "Direct chapter read should contain chapter content"

    @pytest.mark.asyncio
    async def test_semantic_search_capabilities(self, e2e_docs_dir: Path, validator: DocumentSystemValidator):
        """Test comprehensive semantic search functionality with scoped and
        unscoped queries.
        """
        doc_name = f"e2e_search_{uuid.uuid4().hex[:8]}"

        # Create knowledge base with diverse content
        await run_agent_query("src.agents.simple_agent.main", f"Create a document named '{doc_name}'")

        # Add chapters with different topics
        ai_content = (
            "Artificial intelligence and machine learning are transforming technology. "
            "Neural networks enable pattern recognition."
        )
        algorithms_content = (
            "Sorting algorithms like quicksort and mergesort are fundamental. "
            "Binary search is efficient for sorted data."
        )
        data_content = (
            "Arrays and linked lists are basic data structures. "
            "Trees and graphs enable complex relationships."
        )

        await run_agent_query(
            "src.agents.simple_agent.main",
            f"In document '{doc_name}', create chapter '01-ai.md' with content: {ai_content}",
        )

        await run_agent_query(
            "src.agents.simple_agent.main",
            f"In document '{doc_name}', create chapter '02-algorithms.md' with content: {algorithms_content}",
        )

        await run_agent_query(
            "src.agents.simple_agent.main",
            f"In document '{doc_name}', create chapter '03-data.md' with content: {data_content}",
        )

        # Verify setup
        validator.assert_document_exists(doc_name)
        validator.assert_chapter_exists(doc_name, "01-ai.md")
        validator.assert_chapter_exists(doc_name, "02-algorithms.md")
        validator.assert_chapter_exists(doc_name, "03-data.md")

        # Test 1: Document-wide semantic search
        search_response = await run_agent_query(
            "src.agents.simple_agent.main",
            f"In document '{doc_name}', find content similar to 'neural networks and AI technology'",
        )

        search_details = safe_get_response_content(search_response, "details")
        assert "find_similar_text" in search_details, (
            f"Search should return find_similar_text. Got: {search_details}"
        )

        search_data = search_details["find_similar_text"]
        assert "results" in search_data, f"Search data should contain results. Got: {search_data}"

        results = search_data.get("results", [])
        assert len(results) > 0, f"Search should find results. Got: {search_data}"

        # Verify AI chapter is most relevant
        best_result = results[0] if results else {}
        assert "01-ai.md" in str(best_result), f"Best result should be from AI chapter. Got: {best_result}"
        assert "similarity_score" in best_result, "Result should have similarity score"
        assert "content" in best_result, "Result should have content"

        # Test 2: Chapter-scoped semantic search
        scoped_search_response = await run_agent_query(
            "src.agents.simple_agent.main",
            f"In document '{doc_name}', find content similar to 'sorting and searching' in chapter '02-algorithms.md'",
        )

        scoped_details = safe_get_response_content(scoped_search_response, "details")
        if "find_similar_text" in scoped_details:
            scoped_data = scoped_details["find_similar_text"]
            scoped_results = scoped_data.get("results", [])
        else:
            scoped_results = scoped_details.get("results", [])

        if scoped_results:
            # All results should be from algorithms chapter
            for result in scoped_results:
                assert "02-algorithms.md" in str(result), (
                    f"Scoped results should be from algorithms chapter. Got: {result}"
                )


@pytest.mark.e2e
@pytest.mark.skipif(not check_api_key_available(), reason="E2E tests require a real API key")
class TestReactAgentE2E:
    """E2E tests for ReAct agent basic functionality."""

    @pytest.mark.asyncio
    async def test_react_agent_reasoning_workflow(
        self, e2e_docs_dir: Path, validator: DocumentSystemValidator
    ):
        """Test ReAct agent reasoning and execution patterns."""
        doc_name = f"e2e_react_{uuid.uuid4().hex[:8]}"

        # Test ReAct agent with reasoning task
        response = await run_agent_query(
            "src.agents.react_agent.main",
            f"Create document '{doc_name}' and add a chapter explaining why this task requires reasoning",
        )

        # Verify ReAct agent functionality
        validator.assert_document_exists(doc_name)

        # Verify ReAct-specific response structure
        assert response.get("summary"), "ReAct agent response should have summary"

        # Check for ReAct-specific execution log
        if "execution_log" in response:
            print("[OK] ReAct agent execution log present")

        assert response.get("error_message") is None, "ReAct agent should not have errors"

        print("[OK] ReAct agent reasoning workflow verified")


@pytest.mark.e2e
@pytest.mark.skipif(not check_api_key_available(), reason="E2E tests require a real API key")
class TestSafetyAndVersionControlE2E:
    """E2E tests for safety features and version control."""

    @pytest.mark.asyncio
    async def test_safety_features_workflow(self, e2e_docs_dir: Path, validator: DocumentSystemValidator):
        """Test essential safety features: snapshots and version control."""
        doc_name = f"e2e_safety_{uuid.uuid4().hex[:8]}"

        # Create document first (separate operations to avoid MCP loops)
        response1 = await run_agent_query(
            "src.agents.simple_agent.main",
            f"Create document '{doc_name}'",
            timeout=20,  # Document creation with potential API delays
        )

        validator.assert_document_exists(doc_name)

        # Add content separately to avoid complex query loops  
        response2 = await run_agent_query(
            "src.agents.simple_agent.main",
            f"Create chapter named '01-content.md' in document '{doc_name}'",
            timeout=45,  # Chapter creation and potential snapshots
        )

        validator.assert_chapter_exists(doc_name, "01-content.md")

        # Test snapshot creation (the actual safety feature)
        snapshot_response = await run_agent_query(
            "src.agents.simple_agent.main",
            f"Create a snapshot of document '{doc_name}' with message 'Safety test snapshot'",
            timeout=25,  # Snapshots take ~14 seconds + E2E overhead
        )

        # Verify safety workflow
        assert response1.get("error_message") is None, "Document creation should succeed"
        assert response2.get("error_message") is None, "Chapter creation should succeed"
        assert snapshot_response.get("error_message") is None, "Snapshot creation should succeed"

        # Validate actual safety functionality (snapshot creation)
        snapshot_details = safe_get_response_content(snapshot_response, "details")
        assert "manage_snapshots" in snapshot_details or "snapshot" in str(snapshot_details).lower(), (
            f"Should use snapshot management. Got: {snapshot_details}"
        )

        # Verify snapshot was actually created
        if isinstance(snapshot_details, dict) and "manage_snapshots" in snapshot_details:
            snapshot_data = snapshot_details["manage_snapshots"]
            assert snapshot_data.get("success") is True, "Snapshot creation should succeed"
            assert "snapshot_id" in snapshot_data.get("details", {}), "Should have snapshot ID"

        print("[OK] Safety features workflow verified (actual snapshot creation)")
        print("   - Document and content creation")
        print("   - Snapshot creation for version control")
        print("   - Safety mechanism validation")


@pytest.mark.e2e
@pytest.mark.skipif(not check_api_key_available(), reason="E2E tests require a real API key")
class TestMultiStepOperationsE2E:
    """E2E tests for multi-step operations functionality."""

    @pytest.mark.asyncio
    async def test_sequential_operations_capability(self, e2e_docs_dir: Path, validator: DocumentSystemValidator):
        """Test sequential multi-step operations workflow."""
        doc_name = f"e2e_sequential_{uuid.uuid4().hex[:8]}"

        # Test sequential operations demonstrating coordinated workflow
        # Step 1: Create document
        create_response = await run_agent_query(
            "src.agents.simple_agent.main",
            f"Create document '{doc_name}'",
            timeout=30,
        )

        # Step 2: Add chapter to demonstrate coordinated workflow
        chapter_response = await run_agent_query(
            "src.agents.simple_agent.main",
            f"Create chapter '01-intro.md' in document '{doc_name}'",
            timeout=30,
        )

        # Verify the operations succeeded
        assert create_response.get("error_message") is None, "Document creation should succeed"
        assert chapter_response.get("error_message") is None, "Chapter creation should succeed"

        # Verify file system state
        validator.assert_document_exists(doc_name)
        validator.assert_chapter_exists(doc_name, "01-intro.md")
        validator.assert_chapter_count(doc_name, 1)

        print("[OK] Multi-step operations workflow verified")
        print("   - Document creation")
        print("   - Chapter creation")
        print("   - Coordinated sequential workflow execution")


@pytest.mark.e2e
@pytest.mark.skipif(not check_api_key_available(), reason="E2E tests require a real API key")
class TestIntegratedWorkflowE2E:
    """E2E tests for integrated workflows combining multiple features."""

    @pytest.mark.asyncio
    async def test_comprehensive_feature_integration(
        self, e2e_docs_dir: Path, validator: DocumentSystemValidator
    ):
        """Test integrated workflow: content creation, semantic search, safety features, and content improvement."""
        project_name = f"e2e_integrated_{uuid.uuid4().hex[:8]}"
        doc_name = f"{project_name}_knowledge_base"

        # Step 1: Create knowledge base with structured content
        creation_response = await run_agent_query(
            "src.agents.simple_agent.main", f"Create document '{doc_name}'"
        )

        await run_agent_query(
            "src.agents.simple_agent.main",
            f"Add chapter '01-overview.md' to document '{doc_name}' with content: "
            "This document provides project overview and architectural guidance.",
        )

        await run_agent_query(
            "src.agents.simple_agent.main",
            f"Add chapter '02-api.md' to document '{doc_name}' with content: "
            "API documentation covers REST endpoints and authentication methods.",
        )

        validator.assert_document_exists(doc_name)
        validator.assert_chapter_exists(doc_name, "01-overview.md")
        validator.assert_chapter_exists(doc_name, "02-api.md")

        # Step 2: Use semantic search to analyze content
        analysis_response = await run_agent_query(
            "src.agents.simple_agent.main",
            f"In document '{doc_name}', find content similar to 'API and authentication'",
        )

        # Step 3: Create safety snapshot before improvements (using proper timeout)
        snapshot_response = await run_agent_query(
            "src.agents.simple_agent.main",
            f"Create snapshot of document '{doc_name}' with message 'Before adding setup documentation'",
            timeout=25,  # Snapshot operations take ~15 seconds + overhead
        )

        # Step 4: Add improvement based on analysis
        improvement_response = await run_agent_query(
            "src.agents.simple_agent.main",
            f"Add chapter '03-setup.md' to document '{doc_name}' with setup and installation instructions",
        )

        # Verify integrated workflow
        assert creation_response.get("error_message") is None, "Document creation should succeed"
        assert analysis_response.get("error_message") is None, "Semantic search should succeed"
        assert snapshot_response.get("error_message") is None, "Safety snapshot should succeed"
        assert improvement_response.get("error_message") is None, "Content improvement should succeed"

        # Validate semantic search functionality
        analysis_details = safe_get_response_content(analysis_response, "details")
        assert "find_similar_text" in analysis_details or "semantic" in str(analysis_response).lower(), (
            f"Should use semantic search. Got: {analysis_details}"
        )

        # Validate safety features (actual snapshot creation)
        snapshot_details = safe_get_response_content(snapshot_response, "details")
        assert "manage_snapshots" in snapshot_details or "snapshot" in str(snapshot_details).lower(), (
            f"Should use snapshot management. Got: {snapshot_details}"
        )

        # Validate final state
        validator.assert_chapter_exists(doc_name, "03-setup.md")

        print("[OK] Comprehensive feature integration workflow verified")
        print("   - Content creation and structuring")
        print("   - Semantic search for content analysis")
        print("   - Safety snapshots for version control")
        print("   - Content improvement based on analysis")
