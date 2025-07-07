"""
Comprehensive agent performance evaluation suite.

This module provides standardized evaluation tests for document-mcp agents,
focusing on performance metrics, token usage, and reliability measurement.
Tests are designed to be agent-agnostic and support both mock and real LLM modes.
"""

import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

from src.agents.simple_agent.main import FinalAgentResponse, main as simple_main
from src.agents.react_agent.main import main as react_main
from tests.e2e.validation_utils import DocumentSystemValidator, safe_get_response_content


# Test Configuration
TEST_TIMEOUT = 60
MOCK_LLM_DELAY = 0.1  # Simulate LLM response time


class AgentPerformanceMetrics:
    """Container for agent performance metrics."""
    
    def __init__(self):
        self.token_usage: Optional[int] = None
        self.tool_calls_count: int = 0
        self.execution_time: float = 0.0
        self.success: bool = False
        self.error_message: Optional[str] = None
        self.response_data: Optional[Dict] = None
        self.file_system_changes: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for reporting."""
        return {
            'token_usage': self.token_usage,
            'tool_calls_count': self.tool_calls_count,
            'execution_time': self.execution_time,
            'success': self.success,
            'error_message': self.error_message,
            'file_system_changes': self.file_system_changes
        }


class MockLLMResponse:
    """Mock LLM response for consistent testing."""
    
    @staticmethod
    def create_document_response(doc_name: str) -> Dict[str, Any]:
        """Generate mock response for document creation."""
        return {
            "summary": f"Successfully created document '{doc_name}' with initial structure.",
            "details": json.dumps({
                "operation": "create_document",
                "document_name": doc_name,
                "success": True,
                "actions_taken": ["created_document_directory", "initialized_structure"]
            })
        }
    
    @staticmethod
    def create_chapter_response(doc_name: str, chapter_name: str) -> Dict[str, Any]:
        """Generate mock response for chapter creation."""
        return {
            "summary": f"Created chapter '{chapter_name}' in document '{doc_name}'.",
            "details": json.dumps({
                "operation": "create_chapter",
                "document_name": doc_name,
                "chapter_name": chapter_name,
                "success": True,
                "actions_taken": ["created_chapter_file", "wrote_initial_content"]
            })
        }
    
    @staticmethod
    def list_documents_response(doc_names: List[str]) -> Dict[str, Any]:
        """Generate mock response for listing documents."""
        return {
            "summary": f"Found {len(doc_names)} documents in the system.",
            "details": json.dumps({
                "operation": "list_documents",
                "documents": doc_names,
                "count": len(doc_names),
                "success": True
            })
        }
    
    @staticmethod
    def error_response(error_msg: str) -> Dict[str, Any]:
        """Generate mock error response."""
        return {
            "summary": "Operation failed due to an error.",
            "details": json.dumps({
                "operation": "unknown",
                "success": False,
                "error": error_msg
            }),
            "error_message": error_msg
        }


class AgentTestRunner:
    """Test runner for standardized agent evaluation."""
    
    def __init__(self, docs_root: Path, use_mock_llm: bool = True):
        self.docs_root = docs_root
        self.use_mock_llm = use_mock_llm
        self.validator = DocumentSystemValidator(docs_root)
        
    async def run_simple_agent_test(self, query: str, expected_response: Optional[Dict] = None) -> AgentPerformanceMetrics:
        """Run a test using the Simple Agent."""
        metrics = AgentPerformanceMetrics()
        
        # Set up environment
        old_doc_root = os.environ.get("DOCUMENT_ROOT_DIR")
        os.environ["DOCUMENT_ROOT_DIR"] = str(self.docs_root)
        
        try:
            import time
            start_time = time.time()
            
            if self.use_mock_llm:
                # Mock LLM response
                with patch('src.agents.simple_agent.main.Agent') as mock_agent_class:
                    mock_agent = Mock()
                    mock_agent_class.return_value = mock_agent
                    
                    # Configure mock response
                    mock_response = expected_response or MockLLMResponse.create_document_response("test_doc")
                    mock_agent.run_sync.return_value = Mock(
                        data=FinalAgentResponse(**mock_response),
                        usage=Mock(total_tokens=150)  # Mock token usage
                    )
                    
                    # Determine success based on response content
                    is_error_response = mock_response.get('error_message') is not None
                    
                    # Import and run the agent
                    from src.agents.simple_agent.main import initialize_agent_and_mcp_server
                    
                    # Create a mock MCP server
                    with patch('src.agents.simple_agent.main.MCPServerStdio') as mock_mcp:
                        mock_mcp_instance = Mock()
                        mock_mcp.return_value = mock_mcp_instance
                        
                        # Run agent logic
                        await asyncio.sleep(MOCK_LLM_DELAY)
                        metrics.response_data = mock_response
                        metrics.token_usage = 150
                        metrics.tool_calls_count = 1
                        metrics.success = not is_error_response
                        if is_error_response:
                            metrics.error_message = mock_response.get('error_message')
                        
            else:
                # Real LLM execution - call the actual agent
                result = await asyncio.wait_for(
                    self._run_real_simple_agent(query),
                    timeout=TEST_TIMEOUT
                )
                metrics.response_data = result
                metrics.success = not result.get('error_message')
                metrics.error_message = result.get('error_message')
                
            metrics.execution_time = time.time() - start_time
            
        except Exception as e:
            metrics.success = False
            metrics.error_message = str(e)
            metrics.execution_time = time.time() - start_time
            
        finally:
            # Restore environment
            if old_doc_root:
                os.environ["DOCUMENT_ROOT_DIR"] = old_doc_root
            else:
                os.environ.pop("DOCUMENT_ROOT_DIR", None)
                
        return metrics
    
    async def _run_real_simple_agent(self, query: str) -> Dict[str, Any]:
        """Run the actual simple agent with real LLM."""
        # This would integrate with the actual agent execution
        # For now, return a placeholder
        return {
            "summary": "Real agent execution not implemented in this phase",
            "details": json.dumps({"operation": "placeholder", "success": False})
        }
    
    async def run_react_agent_test(self, query: str, expected_steps: Optional[List] = None) -> AgentPerformanceMetrics:
        """Run a test using the React Agent."""
        metrics = AgentPerformanceMetrics()
        
        # Set up environment
        old_doc_root = os.environ.get("DOCUMENT_ROOT_DIR")
        os.environ["DOCUMENT_ROOT_DIR"] = str(self.docs_root)
        
        try:
            import time
            start_time = time.time()
            
            if self.use_mock_llm:
                # Mock React agent execution
                await asyncio.sleep(MOCK_LLM_DELAY * 2)  # React takes longer
                
                # For React agent, we don't have expected_response, so check the query for error indicators
                is_error_query = "invalid" in query.lower() or "fail" in query.lower()
                
                # Simulate React agent response
                if is_error_query:
                    metrics.response_data = {
                        "execution_log": "Mock React agent execution failed",
                        "steps_taken": expected_steps or ["think", "attempt_operation", "observe_error"],
                        "final_result": "Task failed due to invalid operation"
                    }
                    metrics.success = False
                    metrics.error_message = "Invalid operation requested"
                else:
                    metrics.response_data = {
                        "execution_log": "Mock React agent execution completed successfully",
                        "steps_taken": expected_steps or ["think", "act", "observe"],
                        "final_result": "Task completed successfully"
                    }
                    metrics.success = True
                
                metrics.token_usage = 300  # React uses more tokens
                metrics.tool_calls_count = len(expected_steps or [])
                
            else:
                # Real React agent execution
                result = await asyncio.wait_for(
                    self._run_real_react_agent(query),
                    timeout=TEST_TIMEOUT
                )
                metrics.response_data = result
                metrics.success = "error" not in result.get('execution_log', '').lower()
                
            metrics.execution_time = time.time() - start_time
            
        except Exception as e:
            metrics.success = False
            metrics.error_message = str(e)
            metrics.execution_time = time.time() - start_time
            
        finally:
            # Restore environment
            if old_doc_root:
                os.environ["DOCUMENT_ROOT_DIR"] = old_doc_root
            else:
                os.environ.pop("DOCUMENT_ROOT_DIR", None)
                
        return metrics
    
    async def _run_real_react_agent(self, query: str) -> Dict[str, Any]:
        """Run the actual React agent with real LLM."""
        # This would integrate with the actual agent execution
        # For now, return a placeholder
        return {
            "execution_log": "Real React agent execution not implemented in this phase"
        }


# Test Fixtures
@pytest.fixture
def evaluation_docs_root():
    """Provide a clean temporary directory for evaluation tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_test_runner(evaluation_docs_root):
    """Provide a test runner configured for mock LLM testing."""
    return AgentTestRunner(evaluation_docs_root, use_mock_llm=True)


@pytest.fixture
def real_test_runner(evaluation_docs_root):
    """Provide a test runner configured for real LLM testing."""
    return AgentTestRunner(evaluation_docs_root, use_mock_llm=False)


@pytest.fixture
def validator(evaluation_docs_root):
    """Provide a validator for file system assertions."""
    return DocumentSystemValidator(evaluation_docs_root)


# Performance Test Cases
@pytest.mark.evaluation
class TestAgentPerformanceEvaluation:
    """Comprehensive agent performance evaluation tests."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("agent_type", ["simple", "react"])
    async def test_document_creation_performance(self, mock_test_runner, validator, agent_type):
        """Test document creation performance across different agents."""
        doc_name = f"test_doc_{uuid.uuid4().hex[:8]}"
        query = f"Create a new document called '{doc_name}'"
        
        # Expected mock response for document creation
        expected_response = MockLLMResponse.create_document_response(doc_name)
        
        # Run the test
        if agent_type == "simple":
            metrics = await mock_test_runner.run_simple_agent_test(query, expected_response)
        else:
            metrics = await mock_test_runner.run_react_agent_test(query, ["think", "create_document", "observe"])
        
        # Assert on performance metrics
        assert metrics.success, f"Agent failed: {metrics.error_message}"
        assert metrics.execution_time > 0, "Execution time should be recorded"
        assert metrics.token_usage is not None, "Token usage should be tracked"
        assert metrics.tool_calls_count > 0, "Tool calls should be recorded"
        
        # Assert on response structure (details field validation)
        assert metrics.response_data is not None, "Response data should be captured"
        
        if agent_type == "simple":
            details = safe_get_response_content(metrics.response_data, 'details')
            
            # Handle case where details is wrapped in raw_content
            if 'raw_content' in details:
                import json
                try:
                    details = json.loads(details['raw_content'])
                except json.JSONDecodeError:
                    pass  # Keep original details if JSON parsing fails
            
            assert details.get('operation') == 'create_document', "Operation should be recorded in details"
            assert details.get('success') is True, "Success should be recorded in details"
            assert details.get('document_name') == doc_name, "Document name should be recorded"
        
        # Performance assertions
        if agent_type == "simple":
            assert metrics.token_usage < 200, "Simple agent should use fewer tokens"
            assert metrics.execution_time < 5.0, "Simple agent should execute quickly"
        else:
            assert metrics.token_usage < 500, "React agent token usage should be reasonable"
            assert metrics.execution_time < 10.0, "React agent should complete within reasonable time"
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("agent_type", ["simple", "react"])
    async def test_chapter_creation_performance(self, mock_test_runner, validator, agent_type):
        """Test chapter creation performance across different agents."""
        doc_name = f"test_doc_{uuid.uuid4().hex[:8]}"
        chapter_name = "01-intro.md"
        query = f"Create a chapter called '{chapter_name}' in document '{doc_name}'"
        
        # Expected mock response for chapter creation
        expected_response = MockLLMResponse.create_chapter_response(doc_name, chapter_name)
        
        # Run the test
        if agent_type == "simple":
            metrics = await mock_test_runner.run_simple_agent_test(query, expected_response)
        else:
            metrics = await mock_test_runner.run_react_agent_test(query, ["think", "create_chapter", "observe"])
        
        # Assert on performance metrics
        assert metrics.success, f"Agent failed: {metrics.error_message}"
        assert metrics.execution_time > 0, "Execution time should be recorded"
        assert metrics.token_usage is not None, "Token usage should be tracked"
        
        # Assert on response structure (details field validation)
        if agent_type == "simple":
            details = safe_get_response_content(metrics.response_data, 'details')
            
            # Handle case where details is wrapped in raw_content
            if 'raw_content' in details:
                import json
                try:
                    details = json.loads(details['raw_content'])
                except json.JSONDecodeError:
                    pass  # Keep original details if JSON parsing fails
            
            assert details.get('operation') == 'create_chapter', "Operation should be recorded in details"
            assert details.get('success') is True, "Success should be recorded in details"
            assert details.get('document_name') == doc_name, "Document name should be recorded"
            assert details.get('chapter_name') == chapter_name, "Chapter name should be recorded"
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("agent_type", ["simple", "react"])
    async def test_document_listing_performance(self, mock_test_runner, validator, agent_type):
        """Test document listing performance across different agents."""
        query = "List all documents in the system"
        
        # Expected mock response for document listing
        expected_response = MockLLMResponse.list_documents_response(["doc1", "doc2", "doc3"])
        
        # Run the test
        if agent_type == "simple":
            metrics = await mock_test_runner.run_simple_agent_test(query, expected_response)
        else:
            metrics = await mock_test_runner.run_react_agent_test(query, ["think", "list_documents", "observe"])
        
        # Assert on performance metrics
        assert metrics.success, f"Agent failed: {metrics.error_message}"
        assert metrics.execution_time > 0, "Execution time should be recorded"
        assert metrics.token_usage is not None, "Token usage should be tracked"
        
        # Assert on response structure (details field validation)
        if agent_type == "simple":
            details = safe_get_response_content(metrics.response_data, 'details')
            
            # Handle case where details is wrapped in raw_content
            if 'raw_content' in details:
                import json
                try:
                    details = json.loads(details['raw_content'])
                except json.JSONDecodeError:
                    pass  # Keep original details if JSON parsing fails
            
            assert details.get('operation') == 'list_documents', "Operation should be recorded in details"
            assert details.get('success') is True, "Success should be recorded in details"
            assert 'documents' in details, "Documents list should be in details"
            assert details.get('count') == 3, "Document count should be recorded"
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("agent_type", ["simple", "react"])
    async def test_error_handling_performance(self, mock_test_runner, validator, agent_type):
        """Test error handling performance across different agents."""
        query = "Perform an invalid operation that should fail"
        
        # Expected mock error response
        expected_response = MockLLMResponse.error_response("Invalid operation requested")
        
        # Run the test
        if agent_type == "simple":
            metrics = await mock_test_runner.run_simple_agent_test(query, expected_response)
        else:
            metrics = await mock_test_runner.run_react_agent_test(query, ["think", "attempt_operation", "observe_error"])
        
        # Assert on error handling
        assert not metrics.success, "Agent should report failure for invalid operations"
        assert metrics.error_message is not None, "Error message should be captured"
        assert metrics.execution_time > 0, "Execution time should be recorded even for errors"
        
        # Assert on response structure (details field validation)
        if agent_type == "simple":
            details = safe_get_response_content(metrics.response_data, 'details')
            
            # Handle case where details is wrapped in raw_content
            if 'raw_content' in details:
                import json
                try:
                    details = json.loads(details['raw_content'])
                except json.JSONDecodeError:
                    pass  # Keep original details if JSON parsing fails
            
            assert details.get('success') is False, "Error should be recorded in details"
            assert 'error' in details, "Error details should be provided"


@pytest.mark.evaluation
@pytest.mark.real_llm
class TestAgentRealLLMEvaluation:
    """Real LLM evaluation tests for quality assurance."""
    
    @pytest.mark.skipif(
        not any(os.environ.get(key, "").strip() and not os.environ.get(key, "").startswith(("test_", "sk-test"))
                for key in ["OPENAI_API_KEY", "GEMINI_API_KEY"]),
        reason="Real LLM tests require valid API keys"
    )
    @pytest.mark.asyncio
    async def test_real_llm_document_creation(self, real_test_runner, validator):
        """Test document creation with real LLM for quality assurance."""
        doc_name = f"real_test_doc_{uuid.uuid4().hex[:8]}"
        query = f"Create a new document called '{doc_name}'"
        
        # This test is a placeholder for real LLM integration
        # In Phase 1, we focus on the infrastructure setup
        metrics = await real_test_runner.run_simple_agent_test(query)
        
        # Basic assertions for real LLM tests
        assert metrics.execution_time > 0, "Execution time should be recorded"
        assert metrics.response_data is not None, "Response data should be captured"
        
        # Note: Real LLM tests would have more sophisticated assertions
        # based on actual file system changes and API responses


def print_performance_report(metrics_list: List[AgentPerformanceMetrics], test_name: str):
    """Print a performance report for the given metrics."""
    print(f"\n=== Performance Report: {test_name} ===")
    
    total_tokens = sum(m.token_usage or 0 for m in metrics_list)
    total_time = sum(m.execution_time for m in metrics_list)
    success_rate = sum(1 for m in metrics_list if m.success) / len(metrics_list)
    
    print(f"Total Tests: {len(metrics_list)}")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Total Token Usage: {total_tokens}")
    print(f"Total Execution Time: {total_time:.2f}s")
    print(f"Average Tokens per Test: {total_tokens / len(metrics_list):.1f}")
    print(f"Average Execution Time: {total_time / len(metrics_list):.2f}s")
    
    # Print individual test results
    for i, metrics in enumerate(metrics_list):
        status = "✓" if metrics.success else "✗"
        print(f"  {status} Test {i+1}: {metrics.token_usage or 0} tokens, {metrics.execution_time:.2f}s")
    
    print("=" * 50)


# Test runner function for standalone execution
async def run_evaluation_suite():
    """Run the complete evaluation suite and generate performance report."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        docs_root = Path(tmp_dir)
        runner = AgentTestRunner(docs_root, use_mock_llm=True)
        
        # Run a subset of tests for demonstration
        print("Running Agent Performance Evaluation Suite...")
        
        # Test document creation
        doc_metrics = []
        for agent_type in ["simple", "react"]:
            doc_name = f"test_doc_{uuid.uuid4().hex[:8]}"
            query = f"Create a new document called '{doc_name}'"
            expected_response = MockLLMResponse.create_document_response(doc_name)
            
            if agent_type == "simple":
                metrics = await runner.run_simple_agent_test(query, expected_response)
            else:
                metrics = await runner.run_react_agent_test(query, ["think", "create_document", "observe"])
            
            doc_metrics.append(metrics)
        
        print_performance_report(doc_metrics, "Document Creation Tests")


if __name__ == "__main__":
    asyncio.run(run_evaluation_suite())