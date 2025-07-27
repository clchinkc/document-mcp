"""Unit tests for simplified batch execution features.

Tests the BatchExecutor and related components with mocked dependencies
to ensure proper functionality without external calls.
"""

from unittest.mock import patch

import pytest

from document_mcp.batch import BatchApplyResult

# Import the classes we're testing from the simplified batch module
from document_mcp.batch import BatchExecutor
from document_mcp.batch import BatchOperation
from document_mcp.batch import ConflictInfo
from document_mcp.batch import OperationResult


class TestBatchOperation:
    """Test the BatchOperation model."""

    def test_batch_operation_creation(self):
        """Test that BatchOperation can be created with valid data."""
        operation = BatchOperation(
            operation_type="create_document",
            target={},
            parameters={"document_name": "test_doc"},
            order=1,
            operation_id="op1",
        )
        assert operation.operation_type == "create_document"
        assert operation.parameters["document_name"] == "test_doc"
        assert operation.order == 1
        assert operation.operation_id == "op1"

    def test_batch_operation_defaults(self):
        """Test BatchOperation with default values."""
        operation = BatchOperation(
            operation_type="read_content",
            target={"document_name": "test"},
            parameters={"scope": "document"},
            order=0,
            operation_id="read_op",
        )
        assert operation.target == {"document_name": "test"}
        assert operation.depends_on is None


class TestBatchExecutor:
    """Test the BatchExecutor class."""

    def setup_method(self):
        """Set up test dependencies."""
        self.executor = BatchExecutor()

    def test_initialization(self):
        """Test that batch executor initializes correctly."""
        assert self.executor is not None
        assert hasattr(self.executor, "_write_operations")
        assert hasattr(self.executor, "_read_operations")
        assert "create_document" in self.executor._write_operations
        assert "read_content" in self.executor._read_operations

    def test_detect_conflicts_no_conflicts(self):
        """Test conflict detection with operations that don't conflict."""
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

        conflicts = self.executor._detect_conflicts(operations)
        assert len(conflicts) == 0

    def test_detect_conflicts_same_document(self):
        """Test conflict detection with operations on same document."""
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

        conflicts = self.executor._detect_conflicts(operations)
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "same_document"
        assert conflicts[0].severity == "warning"

    def test_detect_conflicts_same_chapter(self):
        """Test conflict detection with operations on same chapter."""
        operations = [
            BatchOperation(
                operation_type="write_chapter_content",
                target={"document_name": "doc1"},
                parameters={"chapter_name": "ch1.md", "content": "test1"},
                order=1,
                operation_id="op1",
            ),
            BatchOperation(
                operation_type="write_chapter_content",
                target={"document_name": "doc1"},
                parameters={"chapter_name": "ch1.md", "content": "test2"},
                order=2,
                operation_id="op2",
            ),
        ]

        conflicts = self.executor._detect_conflicts(operations)
        assert len(conflicts) >= 1
        chapter_conflicts = [c for c in conflicts if c.conflict_type == "same_chapter"]
        assert len(chapter_conflicts) == 1
        assert chapter_conflicts[0].severity == "error"

    @patch("document_mcp.batch.executor.execute_batch_operation")
    def test_execute_batch_success(self, mock_execute):
        """Test successful batch execution."""
        # Mock successful operation results
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
                parameters={"document_name": "doc1"},
                order=1,
                operation_id="op1",
            ),
            BatchOperation(
                operation_type="create_chapter",
                target={"document_name": "doc1"},
                parameters={"chapter_name": "ch1.md"},
                order=2,
                operation_id="op2",
            ),
        ]

        result = self.executor.execute_batch(operations)

        assert result.success
        assert result.total_operations == 2
        assert result.successful_operations == 2
        assert result.failed_operations == 0
        assert len(result.operation_results) == 2
        assert mock_execute.call_count == 2

    @patch("document_mcp.batch.executor.execute_batch_operation")
    def test_execute_batch_with_failure(self, mock_execute):
        """Test batch execution with operation failure."""
        # Mock one success, one failure
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
                error="Operation failed",
                execution_time_ms=50.0,
            ),
        ]

        operations = [
            BatchOperation(
                operation_type="create_document",
                target={},
                parameters={"document_name": "doc1"},
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

        result = self.executor.execute_batch(operations)

        assert not result.success
        assert result.total_operations == 2
        assert result.successful_operations == 1
        assert result.failed_operations == 1
        assert result.rollback_performed
        assert len(result.operation_results) == 2

    def test_execute_batch_with_conflicts(self):
        """Test batch execution aborts when conflicts detected."""
        operations = [
            BatchOperation(
                operation_type="write_chapter_content",
                target={"document_name": "doc1"},
                parameters={"chapter_name": "ch1.md", "content": "test1"},
                order=1,
                operation_id="op1",
            ),
            BatchOperation(
                operation_type="write_chapter_content",
                target={"document_name": "doc1"},
                parameters={"chapter_name": "ch1.md", "content": "test2"},
                order=2,
                operation_id="op2",
            ),
        ]

        result = self.executor.execute_batch(operations)

        assert not result.success
        assert result.total_operations == 2
        assert result.successful_operations == 0
        assert result.failed_operations == 2
        assert "conflicts" in result.summary.lower()
        assert "aborted" in result.summary.lower()

    def test_get_operation_resource(self):
        """Test resource identification for operations."""
        doc_op = BatchOperation(
            operation_type="create_document",
            target={},
            parameters={"document_name": "doc1"},
            order=1,
            operation_id="op1",
        )

        chapter_op = BatchOperation(
            operation_type="create_chapter",
            target={"document_name": "doc1"},
            parameters={"chapter_name": "ch1.md"},
            order=2,
            operation_id="op2",
        )

        global_op = BatchOperation(
            operation_type="list_documents",
            target={},
            parameters={},
            order=3,
            operation_id="op3",
        )

        assert self.executor._get_operation_resource(doc_op) == "doc1"
        assert self.executor._get_operation_resource(chapter_op) == "doc1::ch1.md"
        assert self.executor._get_operation_resource(global_op) == "global"


class TestBatchModels:
    """Test the batch operation data models."""

    def test_operation_result_creation(self):
        """Test OperationResult model creation."""
        result = OperationResult(
            operation_id="test_op",
            operation_type="create_document",
            success=True,
            execution_time_ms=123.45,
        )

        assert result.operation_id == "test_op"
        assert result.operation_type == "create_document"
        assert result.success
        assert result.execution_time_ms == 123.45
        assert result.result is None
        assert result.error is None

    def test_batch_apply_result_creation(self):
        """Test BatchApplyResult model creation."""
        result = BatchApplyResult(
            success=True,
            total_operations=3,
            successful_operations=2,
            failed_operations=1,
            execution_time_ms=500.0,
            summary="Test batch execution",
        )

        assert result.success
        assert result.total_operations == 3
        assert result.successful_operations == 2
        assert result.failed_operations == 1
        assert result.execution_time_ms == 500.0
        assert result.summary == "Test batch execution"
        assert not result.rollback_performed
        assert len(result.operation_results) == 0

    def test_conflict_info_creation(self):
        """Test ConflictInfo model creation."""
        conflict = ConflictInfo(
            conflict_type="same_document",
            affected_operations=["op1", "op2"],
            severity="warning",
            resolution="Will execute sequentially",
        )

        assert conflict.conflict_type == "same_document"
        assert conflict.affected_operations == ["op1", "op2"]
        assert conflict.severity == "warning"
        assert conflict.resolution == "Will execute sequentially"


if __name__ == "__main__":
    pytest.main([__file__])
