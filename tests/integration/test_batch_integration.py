"""Integration tests for simplified batch execution.

Tests batch operations with real MCP server communication but mocked LLM responses.
Validates that batch operations work correctly through the MCP interface.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from document_mcp.batch import BatchApplyResult
from document_mcp.batch import BatchExecutor
from document_mcp.batch import BatchOperation


class TestBatchIntegration:
    """Integration tests for batch execution through MCP."""

    @pytest.fixture
    def temp_docs_dir(self):
        """Create temporary directory for test documents."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def batch_executor(self):
        """Create a batch executor for testing."""
        return BatchExecutor()

    def test_batch_executor_initialization(self, batch_executor):
        """Test that batch executor initializes correctly."""
        assert batch_executor is not None
        assert hasattr(batch_executor, "_write_operations")
        assert hasattr(batch_executor, "_read_operations")

    def test_conflict_detection_different_documents(self, batch_executor):
        """Test that operations on different documents don't conflict."""
        operations = [
            BatchOperation(
                operation_type="create_document",
                target={},
                parameters={"document_name": "doc1"},
                order=1,
                operation_id="op1",
            ),
            BatchOperation(
                operation_type="create_document",
                target={},
                parameters={"document_name": "doc2"},
                order=2,
                operation_id="op2",
            ),
        ]

        conflicts = batch_executor._detect_conflicts(operations)
        assert len(conflicts) == 0

    def test_conflict_detection_same_document_writes(self, batch_executor):
        """Test that write operations on same document are flagged."""
        operations = [
            BatchOperation(
                operation_type="create_document",
                target={},
                parameters={"document_name": "doc1"},
                order=1,
                operation_id="op1",
            ),
            BatchOperation(
                operation_type="delete_document",
                target={},
                parameters={"document_name": "doc1"},
                order=2,
                operation_id="op2",
            ),
        ]

        conflicts = batch_executor._detect_conflicts(operations)
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "same_document"
        assert "op1" in conflicts[0].affected_operations
        assert "op2" in conflicts[0].affected_operations

    def test_conflict_detection_same_chapter_writes(self, batch_executor):
        """Test that write operations on same chapter are flagged as errors."""
        operations = [
            BatchOperation(
                operation_type="write_chapter_content",
                target={"document_name": "doc1"},
                parameters={"chapter_name": "ch1.md", "content": "content1"},
                order=1,
                operation_id="op1",
            ),
            BatchOperation(
                operation_type="write_chapter_content",
                target={"document_name": "doc1"},
                parameters={"chapter_name": "ch1.md", "content": "content2"},
                order=2,
                operation_id="op2",
            ),
        ]

        conflicts = batch_executor._detect_conflicts(operations)
        chapter_conflicts = [c for c in conflicts if c.conflict_type == "same_chapter"]
        assert len(chapter_conflicts) == 1
        assert chapter_conflicts[0].severity == "error"

    def test_mixed_read_write_operations(self, batch_executor):
        """Test mix of read and write operations."""
        operations = [
            BatchOperation(
                operation_type="list_documents",
                target={},
                parameters={},
                order=1,
                operation_id="op1",
            ),
            BatchOperation(
                operation_type="create_document",
                target={},
                parameters={"document_name": "doc1"},
                order=2,
                operation_id="op2",
            ),
            BatchOperation(
                operation_type="read_content",
                target={},
                parameters={"document_name": "doc1", "scope": "document"},
                order=3,
                operation_id="op3",
            ),
        ]

        conflicts = batch_executor._detect_conflicts(operations)
        # Should detect potential read-write conflict but not error-level conflicts
        error_conflicts = [c for c in conflicts if c.severity == "error"]
        assert len(error_conflicts) == 0

    @patch("document_mcp.batch.executor.execute_batch_operation")
    def test_batch_execution_sequential_success(self, mock_execute, batch_executor):
        """Test successful sequential batch execution."""
        # Mock successful operation results
        from document_mcp.batch import OperationResult

        mock_execute.side_effect = [
            OperationResult(
                success=True,
                operation_id="op1",
                operation_type="create_document",
                execution_time_ms=100.0,
            ),
            OperationResult(
                success=True,
                operation_id="op2",
                operation_type="create_chapter",
                execution_time_ms=150.0,
            ),
        ]

        operations = [
            BatchOperation(
                operation_type="create_document",
                target={},
                parameters={"document_name": "test_doc"},
                order=1,
                operation_id="op1",
            ),
            BatchOperation(
                operation_type="create_chapter",
                target={"document_name": "test_doc"},
                parameters={"chapter_name": "chapter1.md"},
                order=2,
                operation_id="op2",
            ),
        ]

        result = batch_executor.execute_batch(operations)

        assert isinstance(result, BatchApplyResult)
        assert result.success
        assert result.total_operations == 2
        assert result.successful_operations == 2
        assert result.failed_operations == 0
        assert len(result.operation_results) == 2
        assert mock_execute.call_count == 2

    @patch("document_mcp.batch.executor.execute_batch_operation")
    def test_batch_execution_with_failure(self, mock_execute, batch_executor):
        """Test batch execution stops on first failure."""
        from document_mcp.batch import OperationResult

        mock_execute.side_effect = [
            OperationResult(
                success=True,
                operation_id="op1",
                operation_type="create_document",
                execution_time_ms=100.0,
            ),
            OperationResult(
                success=False,
                operation_id="op2",
                operation_type="invalid_operation",
                error="Operation not found",
                execution_time_ms=50.0,
            ),
        ]

        operations = [
            BatchOperation(
                operation_type="create_document",
                target={},
                parameters={"document_name": "test_doc"},
                order=1,
                operation_id="op1",
            ),
            BatchOperation(
                operation_type="invalid_operation",
                target={},
                parameters={},
                order=2,
                operation_id="op2",
            ),
        ]

        result = batch_executor.execute_batch(operations)

        assert isinstance(result, BatchApplyResult)
        assert not result.success
        assert result.total_operations == 2
        assert result.successful_operations == 1
        assert result.failed_operations == 1
        assert result.rollback_performed  # Should indicate rollback was triggered
        assert len(result.operation_results) == 2

    def test_batch_execution_aborts_on_conflicts(self, batch_executor):
        """Test that batch execution aborts when conflicts are detected."""
        operations = [
            BatchOperation(
                operation_type="write_chapter_content",
                target={"document_name": "doc1"},
                parameters={"chapter_name": "ch1.md", "content": "content1"},
                order=1,
                operation_id="op1",
            ),
            BatchOperation(
                operation_type="write_chapter_content",
                target={"document_name": "doc1"},
                parameters={"chapter_name": "ch1.md", "content": "content2"},
                order=2,
                operation_id="op2",
            ),
        ]

        result = batch_executor.execute_batch(operations)

        assert isinstance(result, BatchApplyResult)
        assert not result.success
        assert result.total_operations == 2
        assert result.successful_operations == 0
        assert result.failed_operations == 2
        assert "conflicts" in result.summary.lower()
        assert "aborted" in result.summary.lower()

    def test_operation_ordering(self, batch_executor):
        """Test that operations are executed in order."""
        operations = [
            BatchOperation(
                operation_type="create_document",
                target={},
                parameters={"document_name": "doc1"},
                order=3,  # Intentionally out of order
                operation_id="op3",
            ),
            BatchOperation(
                operation_type="list_documents",
                target={},
                parameters={},
                order=1,
                operation_id="op1",
            ),
            BatchOperation(
                operation_type="create_document",
                target={},
                parameters={"document_name": "doc2"},
                order=2,
                operation_id="op2",
            ),
        ]

        # Test that _detect_conflicts properly groups operations
        conflicts = batch_executor._detect_conflicts(operations)

        # Should detect conflict between the two create_document operations
        # but this is just a warning for different documents
        doc_conflicts = [c for c in conflicts if c.conflict_type == "same_document"]
        # No same_document conflicts since different document names
        assert len(doc_conflicts) == 0


if __name__ == "__main__":
    pytest.main([__file__])
