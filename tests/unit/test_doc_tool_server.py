"""Unit tests for document MCP tool server helper functions."""

import datetime

import pytest

from document_mcp.batch import BatchApplyRequest
from document_mcp.batch import BatchApplyResult
from document_mcp.batch import BatchOperation
from document_mcp.batch import OperationResult
from document_mcp.batch.global_registry import get_batch_registry

# Import batch registry directly from batch module
from document_mcp.batch.registry import BatchOperationRegistry

# Import batch functions
from document_mcp.batch.registry import execute_batch_operation as _execute_batch_operation
from document_mcp.helpers import DOCUMENT_SUMMARY_FILE
from document_mcp.helpers import _count_words
from document_mcp.helpers import _get_modification_history_path
from document_mcp.helpers import _get_snapshots_path
from document_mcp.helpers import _get_summaries_path
from document_mcp.helpers import _get_summary_file_path
from document_mcp.helpers import _is_valid_chapter_filename
from document_mcp.helpers import _resolve_operation_dependencies
from document_mcp.helpers import _split_into_paragraphs

# Import constants and models
from document_mcp.utils.validation import CHAPTER_MANIFEST_FILE

# Import helper functions
from document_mcp.utils.validation import check_file_freshness as _check_file_freshness
from document_mcp.utils.validation import validate_content


class TestHelperFunctions:
    """Test suite for helper functions in doc_tool_server."""

    def test_count_words_empty_string(self):
        """Test word counting with empty string."""
        assert _count_words("") == 0

    def test_count_words_single_word(self):
        """Test word counting with single word."""
        assert _count_words("hello") == 1

    def test_count_words_multiple_words(self):
        """Test word counting with multiple words."""
        assert _count_words("hello world test") == 3

    def test_count_words_with_extra_spaces(self):
        """Test word counting with extra spaces."""
        assert _count_words("  hello   world  ") == 2

    def test_count_words_with_newlines(self):
        """Test word counting with newlines."""
        assert _count_words("hello\nworld\ntest") == 3

    def test_split_into_paragraphs_empty_string(self):
        """Test paragraph splitting with empty string."""
        assert _split_into_paragraphs("") == []

    def test_split_into_paragraphs_single_paragraph(self):
        """Test paragraph splitting with single paragraph."""
        text = "This is a single paragraph with multiple sentences."
        result = _split_into_paragraphs(text)
        assert len(result) == 1
        assert result[0] == text

    def test_split_into_paragraphs_multiple_paragraphs(self):
        """Test paragraph splitting with multiple paragraphs."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = _split_into_paragraphs(text)
        assert len(result) == 3
        assert result[0] == "First paragraph."
        assert result[1] == "Second paragraph."
        assert result[2] == "Third paragraph."

    def test_is_valid_chapter_filename_valid_md_files(self):
        """Test filename validation with valid markdown files."""
        assert _is_valid_chapter_filename("chapter1.md") is True
        assert _is_valid_chapter_filename("01-introduction.md") is True

    def test_is_valid_chapter_filename_invalid_extensions(self):
        """Test filename validation rejects non-markdown files."""
        assert _is_valid_chapter_filename("chapter.txt") is False
        assert _is_valid_chapter_filename("chapter") is False

    def test_is_valid_chapter_filename_manifest_file(self):
        """Test filename validation rejects manifest file."""
        assert _is_valid_chapter_filename(CHAPTER_MANIFEST_FILE) is False

    def test_is_valid_chapter_filename_summary_file(self):
        """Test filename validation rejects summary file."""
        assert _is_valid_chapter_filename(DOCUMENT_SUMMARY_FILE) is False


class TestInputValidationHelpers:
    """Test suite for input validation helper functions."""

    @pytest.mark.parametrize(
        "name, expected_valid, expected_error_msg",
        [
            ("doc_name", True, ""),
            ("doc-name-123", True, ""),
            (None, False, "Document name cannot be empty"),
            ("invalid/name", False, "Document name cannot contain path separators"),
        ],
    )
    def testvalidate_document_name(self, name, expected_valid, expected_error_msg):
        from document_mcp.utils.validation import validate_document_name

        is_valid, error = validate_document_name(name)
        assert is_valid is expected_valid
        if not expected_valid:
            assert expected_error_msg == error

    @pytest.mark.parametrize(
        "name, expected_valid, expected_error_msg",
        [
            ("chapter1.md", True, ""),
            ("01-intro.md", True, ""),
            (None, False, "Chapter name cannot be empty"),
            (
                "invalid/chapter.md",
                False,
                "Chapter name cannot contain path separators",
            ),
            ("no_extension", False, "Chapter name must end with .md"),
        ],
    )
    def testvalidate_chapter_name(self, name, expected_valid, expected_error_msg):
        from document_mcp.utils.validation import validate_chapter_name

        is_valid, error = validate_chapter_name(name)
        assert is_valid == expected_valid
        assert error == expected_error_msg

    @pytest.mark.parametrize(
        "content, expected_valid, expected_error_msg",
        [
            ("Some valid content.", True, ""),
            ("", True, ""),
            (None, False, "Content cannot be None"),
            (12345, False, "Content must be a string"),
        ],
    )
    def test_validate_content_general_cases(self, content, expected_valid, expected_error_msg):
        """Test content validation for general cases (None, type, and short strings)."""
        is_valid, error = validate_content(content)
        assert is_valid == expected_valid
        assert error == expected_error_msg

    @pytest.mark.parametrize(
        "index, expected_valid, expected_error_msg",
        [
            (0, True, ""),
            (100, True, ""),
            (-1, False, "Paragraph index cannot be negative"),
        ],
    )
    def testvalidate_paragraph_index(self, index, expected_valid, expected_error_msg):
        from document_mcp.utils.validation import validate_paragraph_index

        is_valid, error = validate_paragraph_index(index)
        assert is_valid is expected_valid
        if not expected_valid:
            assert expected_error_msg == error

    @pytest.mark.parametrize(
        "query, expected_valid, expected_error_msg",
        [
            ("hello", True, ""),
            (None, False, "Search query cannot be None"),
            ("", False, "Search query cannot be empty or whitespace only"),
        ],
    )
    def testvalidate_search_query(self, query, expected_valid, expected_error_msg):
        from document_mcp.utils.validation import validate_search_query

        is_valid, error = validate_search_query(query)
        assert is_valid is expected_valid
        if not expected_valid:
            assert expected_error_msg == error


class TestSafetyHelperFunctions:
    """Test suite for safety feature helper functions."""

    def test_get_snapshots_path_with_default_root(self):
        """Test snapshots path generation with default DOCS_ROOT_PATH."""
        import os

        # Test without DOCUMENT_ROOT_DIR environment variable (production behavior)
        old_doc_root = os.environ.get("DOCUMENT_ROOT_DIR")
        if "DOCUMENT_ROOT_DIR" in os.environ:
            del os.environ["DOCUMENT_ROOT_DIR"]

        try:
            # Reset settings singleton to pick up environment change
            from document_mcp.config.settings import reset_settings

            reset_settings()

            from document_mcp.utils.file_operations import DOCS_ROOT_PATH

            result = _get_snapshots_path("test_doc")
            expected = (DOCS_ROOT_PATH / "test_doc" / ".snapshots").resolve()
            assert result == expected
        finally:
            # Restore environment
            if old_doc_root:
                os.environ["DOCUMENT_ROOT_DIR"] = old_doc_root
            # Reset settings again to pick up the restored environment
            reset_settings()

    def test_get_snapshots_path_with_custom_root(self):
        """Test snapshots path generation with custom DOCUMENT_ROOT_DIR."""
        import os
        import tempfile
        from pathlib import Path

        # Test with DOCUMENT_ROOT_DIR environment variable (test isolation behavior)
        old_doc_root = os.environ.get("DOCUMENT_ROOT_DIR")

        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["DOCUMENT_ROOT_DIR"] = temp_dir

            try:
                result = _get_snapshots_path("test_doc")
                expected = (Path(temp_dir) / "test_doc" / ".snapshots").resolve()
                assert result == expected
            finally:
                # Restore environment
                if old_doc_root:
                    os.environ["DOCUMENT_ROOT_DIR"] = old_doc_root
                else:
                    os.environ.pop("DOCUMENT_ROOT_DIR", None)

    def test_get_modification_history_path_with_default_root(self):
        """Test modification history path generation with default DOCS_ROOT_PATH."""
        import os

        # Test without DOCUMENT_ROOT_DIR environment variable (production behavior)
        old_doc_root = os.environ.get("DOCUMENT_ROOT_DIR")
        if "DOCUMENT_ROOT_DIR" in os.environ:
            del os.environ["DOCUMENT_ROOT_DIR"]

        try:
            # Reset settings singleton to pick up environment change
            from document_mcp.config.settings import reset_settings

            reset_settings()

            from document_mcp.utils.file_operations import DOCS_ROOT_PATH

            result = _get_modification_history_path("test_doc")
            expected = (DOCS_ROOT_PATH / "test_doc" / ".mod_history.json").resolve()
            assert result == expected
        finally:
            # Restore environment
            if old_doc_root:
                os.environ["DOCUMENT_ROOT_DIR"] = old_doc_root
            # Reset settings again to pick up the restored environment
            reset_settings()

    def test_get_modification_history_path_with_custom_root(self):
        """Test modification history path generation with custom DOCUMENT_ROOT_DIR."""
        import os
        import tempfile
        from pathlib import Path

        # Test with DOCUMENT_ROOT_DIR environment variable (test isolation behavior)
        old_doc_root = os.environ.get("DOCUMENT_ROOT_DIR")

        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["DOCUMENT_ROOT_DIR"] = temp_dir

            try:
                result = _get_modification_history_path("test_doc")
                expected = (Path(temp_dir) / "test_doc" / ".mod_history.json").resolve()
                assert result == expected
            finally:
                # Restore environment
                if old_doc_root:
                    os.environ["DOCUMENT_ROOT_DIR"] = old_doc_root
                else:
                    os.environ.pop("DOCUMENT_ROOT_DIR", None)

    def test_check_file_freshness_file_not_exists(self, mocker):
        mock_path = mocker.Mock()
        mock_path.exists.return_value = False

        result = _check_file_freshness(mock_path)

        assert result.__class__.__name__ == "ContentFreshnessStatus"
        assert not result.is_fresh
        assert result.safety_status == "conflict"
        assert "Verify file was not accidentally deleted" in result.recommendations

    def test_check_file_freshness_file_exists_fresh(self, mocker):
        current_time = datetime.datetime.now()
        mock_path = mocker.Mock()
        mock_path.exists.return_value = True
        mock_path.stat.return_value.st_mtime = current_time.timestamp()

        result = _check_file_freshness(mock_path, current_time)

        assert result.__class__.__name__ == "ContentFreshnessStatus"
        assert result.is_fresh
        assert result.safety_status == "safe"

    def test_check_file_freshness_file_exists_stale(self, mocker):
        old_time = datetime.datetime.now() - datetime.timedelta(hours=1)
        new_time = datetime.datetime.now()
        mock_path = mocker.Mock()
        mock_path.exists.return_value = True
        mock_path.stat.return_value.st_mtime = new_time.timestamp()

        result = _check_file_freshness(mock_path, old_time)

        assert result.__class__.__name__ == "ContentFreshnessStatus"
        assert not result.is_fresh
        assert result.safety_status == "warning"
        assert "Content was modified" in result.message


class TestSummaryHelperFunctions:
    """Test suite for summary helper functions."""

    def test_get_summaries_path_with_default_root(self):
        """Test summaries path generation with default DOCS_ROOT_PATH."""
        import os

        # Test without DOCUMENT_ROOT_DIR environment variable (production behavior)
        old_doc_root = os.environ.get("DOCUMENT_ROOT_DIR")
        if "DOCUMENT_ROOT_DIR" in os.environ:
            del os.environ["DOCUMENT_ROOT_DIR"]

        try:
            # Reset settings singleton to pick up environment change
            from document_mcp.config.settings import reset_settings

            reset_settings()

            from document_mcp.utils.file_operations import DOCS_ROOT_PATH

            result = _get_summaries_path("test_doc")
            expected = (DOCS_ROOT_PATH / "test_doc" / "summaries").resolve()
            assert result == expected
        finally:
            # Restore environment
            if old_doc_root:
                os.environ["DOCUMENT_ROOT_DIR"] = old_doc_root
            # Reset settings again to pick up the restored environment
            reset_settings()

    def test_get_summaries_path_with_custom_root(self):
        """Test summaries path generation with custom DOCUMENT_ROOT_DIR."""
        import os
        import tempfile
        from pathlib import Path

        # Test with DOCUMENT_ROOT_DIR environment variable (test isolation behavior)
        old_doc_root = os.environ.get("DOCUMENT_ROOT_DIR")

        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["DOCUMENT_ROOT_DIR"] = temp_dir

            try:
                result = _get_summaries_path("test_doc")
                expected = (Path(temp_dir) / "test_doc" / "summaries").resolve()
                assert result == expected
            finally:
                # Restore environment
                if old_doc_root:
                    os.environ["DOCUMENT_ROOT_DIR"] = old_doc_root
                else:
                    if "DOCUMENT_ROOT_DIR" in os.environ:
                        del os.environ["DOCUMENT_ROOT_DIR"]

    def test_get_summary_file_path_document_scope(self):
        """Test summary file path generation for document scope."""
        result = _get_summary_file_path("test_doc", "document", None)
        assert result.name == "document.md"
        assert result.parent.name == "summaries"

    def test_get_summary_file_path_chapter_scope(self):
        """Test summary file path generation for chapter scope."""
        result = _get_summary_file_path("test_doc", "chapter", "01-intro.md")
        assert result.name == "chapter-01-intro.md"
        assert result.parent.name == "summaries"

    def test_get_summary_file_path_section_scope(self):
        """Test summary file path generation for section scope."""
        result = _get_summary_file_path("test_doc", "section", "introduction")
        assert result.name == "section-introduction.md"
        assert result.parent.name == "summaries"

    def test_get_summary_file_path_chapter_scope_missing_target(self):
        """Test that chapter scope requires target_name."""
        import pytest

        with pytest.raises(ValueError, match="target_name is required for chapter scope"):
            _get_summary_file_path("test_doc", "chapter", None)

    def test_get_summary_file_path_section_scope_missing_target(self):
        """Test that section scope requires target_name."""
        import pytest

        with pytest.raises(ValueError, match="target_name is required for section scope"):
            _get_summary_file_path("test_doc", "section", None)

    def test_get_summary_file_path_invalid_scope(self):
        """Test that invalid scope raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="Invalid scope.*Must be 'document', 'chapter', or 'section'"):
            _get_summary_file_path("test_doc", "invalid", None)


class TestBatchOperations:
    """Test suite for batch operation functionality."""

    def test_batch_operation_model_creation(self):
        """Test creating BatchOperation model with valid data."""
        batch_op = BatchOperation(
            operation_type="create_document",
            target={"document_name": "test_doc"},
            parameters={"document_name": "test_doc"},
            order=1,
            operation_id="test_op_1",
        )

        assert batch_op.operation_type == "create_document"
        assert batch_op.target["document_name"] == "test_doc"
        assert batch_op.parameters["document_name"] == "test_doc"
        assert batch_op.order == 1
        assert batch_op.operation_id == "test_op_1"
        assert batch_op.depends_on is None

    def test_batch_operation_model_with_dependencies(self):
        """Test creating BatchOperation model with dependencies."""
        batch_op = BatchOperation(
            operation_type="create_chapter",
            target={"document_name": "test_doc", "chapter_name": "ch1.md"},
            parameters={"initial_content": "# Chapter 1"},
            order=2,
            operation_id="test_op_2",
            depends_on=["test_op_1"],
        )

        assert batch_op.depends_on == ["test_op_1"]

    def test_operation_result_model_success(self):
        """Test creating OperationResult model for successful operation."""
        result = OperationResult(
            success=True,
            operation_id="test_op_1",
            operation_type="create_document",
            result={"message": "Document created successfully"},
            execution_time_ms=150.0,
        )

        assert result.success is True
        assert result.operation_id == "test_op_1"
        assert result.operation_type == "create_document"
        assert result.result["message"] == "Document created successfully"
        assert result.error is None
        assert result.execution_time_ms == 150.0

    def test_operation_result_model_failure(self):
        """Test creating OperationResult model for failed operation."""
        result = OperationResult(
            success=False,
            operation_id="test_op_2",
            operation_type="create_chapter",
            error="Document not found",
            execution_time_ms=50.0,
        )

        assert result.success is False
        assert result.error == "Document not found"
        assert result.result is None

    def test_batch_apply_result_model(self):
        """Test creating BatchApplyResult model."""
        op_result = OperationResult(
            success=True,
            operation_id="test_op_1",
            operation_type="create_document",
            result_data={"message": "Success"},
            execution_time_ms=100.0,
        )

        batch_result = BatchApplyResult(
            success=True,
            total_operations=1,
            successful_operations=1,
            failed_operations=0,
            execution_time_ms=200.0,
            operation_results=[op_result],
            summary="Batch completed successfully",
        )

        assert batch_result.success is True
        assert batch_result.total_operations == 1
        assert batch_result.successful_operations == 1
        assert batch_result.failed_operations == 0
        assert len(batch_result.operation_results) == 1
        assert batch_result.rollback_performed is False
        assert batch_result.error_summary is None

    def test_batch_operation_registry_creation(self):
        """Test creating a new BatchOperationRegistry."""
        registry = BatchOperationRegistry()

        assert isinstance(registry._operations, dict)
        assert len(registry.get_batchable_operations()) == 0

    def test_batch_operation_registry_register_operation(self):
        """Test registering operations in BatchOperationRegistry."""
        registry = BatchOperationRegistry()

        registry.register_operation("test_operation", "test_function")

        assert registry.is_valid_operation("test_operation") is True
        assert registry.is_valid_operation("nonexistent_operation") is False
        assert registry.get_tool_function_name("test_operation") == "test_function"
        assert "test_operation" in registry.get_batchable_operations()

    def test_global_batch_registry_has_registered_operations(self):
        """Test that the global batch registry can be populated with operations."""
        # In unit tests, the global registry starts empty because tools aren't imported
        # Let's test the registry functionality by manually registering operations
        registry = get_batch_registry()

        # Register test operations
        registry.register_operation("create_document", "create_document")
        registry.register_operation("read_content", "read_content")

        registered_ops = registry.get_batchable_operations()

        # These operations should now be in the registry
        assert "create_document" in registered_ops
        assert "read_content" in registered_ops

        # Test that we can get tool function names
        assert registry.get_tool_function_name("create_document") == "create_document"
        assert registry.get_tool_function_name("read_content") == "read_content"

    def test_execute_batch_operation_unknown_operation_type(self):
        """Test executing batch operation with unknown operation type."""
        batch_op = BatchOperation(
            operation_type="unknown_operation",
            target={},
            parameters={},
            order=1,
            operation_id="test_unknown",
        )

        result = _execute_batch_operation(batch_op)

        assert result.success is False
        assert result.operation_id == "test_unknown"
        assert result.operation_type == "unknown_operation"
        assert "Unknown operation type" in result.error

    def test_execute_batch_operation_missing_tool_function(self, mocker):
        """Test executing batch operation when tool function is missing."""
        # Create a temporary registry with a missing function
        registry = BatchOperationRegistry()
        registry.register_operation("missing_function_op", "nonexistent_function")

        # Mock the global registry function as used within execute_batch_operation
        mock_get_registry = mocker.patch("document_mcp.batch.global_registry.get_batch_registry")
        mock_get_registry.return_value = registry

        # Mock the mcp_client module to simulate missing function
        mock_mcp_client = mocker.MagicMock()
        # Configure the mock so that nonexistent_function attribute doesn't exist
        if hasattr(mock_mcp_client, "nonexistent_function"):
            delattr(mock_mcp_client, "nonexistent_function")
        mocker.patch.dict("sys.modules", {"document_mcp.mcp_client": mock_mcp_client})

        batch_op = BatchOperation(
            operation_type="missing_function_op",
            target={},
            parameters={},
            order=1,
            operation_id="test_missing",
        )

        result = _execute_batch_operation(batch_op)

        assert result.success is False
        assert result.operation_id == "test_missing"
        assert "Tool function not found" in result.error

    def test_batch_apply_request_model_defaults(self):
        """Test BatchApplyRequest model with default values."""
        request = BatchApplyRequest(
            operations=[
                {
                    "operation_type": "create_document",
                    "target": {},
                    "parameters": {"document_name": "test"},
                    "order": 1,
                }
            ]
        )

        assert request.atomic is True
        assert request.validate_only is False
        assert request.snapshot_before is False
        assert request.continue_on_error is False
        assert request.execution_mode == "sequential"

    def test_batch_apply_request_model_custom_values(self):
        """Test BatchApplyRequest model with custom values."""
        request = BatchApplyRequest(
            operations=[],
            atomic=False,
            validate_only=True,
            snapshot_before=True,
            continue_on_error=True,
            execution_mode="parallel_safe",
        )

        assert request.atomic is False
        assert request.validate_only is True
        assert request.snapshot_before is True
        assert request.continue_on_error is True
        assert request.execution_mode == "parallel_safe"


class TestCompositeOperationsValidation:
    """Unit tests for composite operation data structures and validation logic."""


class TestDependencyResolution:
    """Unit tests for batch operation dependency resolution."""

    def test_simple_dependency_chain(self):
        """Test resolving a simple dependency chain."""
        operations = [
            BatchOperation(
                operation_type="create_chapter",
                target={"document_name": "test_doc"},
                parameters={
                    "chapter_name": "chapter1.md",
                    "initial_content": "Chapter 1",
                },
                order=2,
                operation_id="create_ch1",
                depends_on=["create_doc"],
            ),
            BatchOperation(
                operation_type="create_document",
                target={},
                parameters={"document_name": "test_doc"},
                order=1,
                operation_id="create_doc",
                depends_on=[],
            ),
        ]

        resolved_ops = _resolve_operation_dependencies(operations)

        # Document creation should come first, then chapter creation
        assert len(resolved_ops) == 2
        assert resolved_ops[0].operation_id == "create_doc"
        assert resolved_ops[1].operation_id == "create_ch1"

    def test_multiple_dependencies(self):
        """Test operation with multiple dependencies."""
        operations = [
            BatchOperation(
                operation_type="snapshot_document",
                target={"document_name": "test_doc"},
                parameters={"message": "Snapshot after creation"},
                order=4,
                operation_id="snapshot",
                depends_on=["create_ch1", "create_ch2"],
            ),
            BatchOperation(
                operation_type="create_chapter",
                target={"document_name": "test_doc"},
                parameters={"chapter_name": "chapter1.md"},
                order=2,
                operation_id="create_ch1",
                depends_on=["create_doc"],
            ),
            BatchOperation(
                operation_type="create_chapter",
                target={"document_name": "test_doc"},
                parameters={"chapter_name": "chapter2.md"},
                order=3,
                operation_id="create_ch2",
                depends_on=["create_doc"],
            ),
            BatchOperation(
                operation_type="create_document",
                target={},
                parameters={"document_name": "test_doc"},
                order=1,
                operation_id="create_doc",
                depends_on=[],
            ),
        ]

        resolved_ops = _resolve_operation_dependencies(operations)

        # Document should be first, chapters next, snapshot last
        assert len(resolved_ops) == 4
        assert resolved_ops[0].operation_id == "create_doc"
        assert resolved_ops[-1].operation_id == "snapshot"

        # Both chapters should come before snapshot
        ch1_index = next(i for i, op in enumerate(resolved_ops) if op.operation_id == "create_ch1")
        ch2_index = next(i for i, op in enumerate(resolved_ops) if op.operation_id == "create_ch2")
        snapshot_index = next(i for i, op in enumerate(resolved_ops) if op.operation_id == "snapshot")

        assert ch1_index < snapshot_index
        assert ch2_index < snapshot_index

    def test_no_dependencies(self):
        """Test operations with no dependencies (should sort by order)."""
        operations = [
            BatchOperation(
                operation_type="create_document",
                target={},
                parameters={"document_name": "doc2"},
                order=3,
                operation_id="create_doc2",
                depends_on=[],
            ),
            BatchOperation(
                operation_type="create_document",
                target={},
                parameters={"document_name": "doc1"},
                order=1,
                operation_id="create_doc1",
                depends_on=[],
            ),
            BatchOperation(
                operation_type="create_document",
                target={},
                parameters={"document_name": "doc3"},
                order=2,
                operation_id="create_doc3",
                depends_on=[],
            ),
        ]

        resolved_ops = _resolve_operation_dependencies(operations)

        # Should be sorted by order field
        assert len(resolved_ops) == 3
        assert resolved_ops[0].operation_id == "create_doc1"
        assert resolved_ops[1].operation_id == "create_doc3"
        assert resolved_ops[2].operation_id == "create_doc2"

    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        operations = [
            BatchOperation(
                operation_type="operation_a",
                target={},
                parameters={},
                order=1,
                operation_id="op_a",
                depends_on=["op_b"],
            ),
            BatchOperation(
                operation_type="operation_b",
                target={},
                parameters={},
                order=2,
                operation_id="op_b",
                depends_on=["op_a"],
            ),
        ]

        with pytest.raises(ValueError, match="Circular dependency detected"):
            _resolve_operation_dependencies(operations)

    def test_unknown_dependency(self):
        """Test error when depending on unknown operation."""
        operations = [
            BatchOperation(
                operation_type="create_chapter",
                target={"document_name": "test_doc"},
                parameters={"chapter_name": "chapter1.md"},
                order=1,
                operation_id="create_ch1",
                depends_on=["unknown_operation"],
            )
        ]

        with pytest.raises(ValueError, match="depend on unknown operation"):
            _resolve_operation_dependencies(operations)

    def test_self_dependency(self):
        """Test error when operation depends on itself."""
        operations = [
            BatchOperation(
                operation_type="create_document",
                target={},
                parameters={"document_name": "test_doc"},
                order=1,
                operation_id="create_doc",
                depends_on=["create_doc"],
            )
        ]

        with pytest.raises(ValueError, match="Circular dependency detected"):
            _resolve_operation_dependencies(operations)


class TestUnifiedContentTools:
    """Test suite for unified content access tools."""

    def test_read_content_document_scope_validation(self):
        """Test read_content with document scope parameter validation."""
        from document_mcp.mcp_client import read_content

        # Test invalid document name
        result = read_content("", scope="document")
        assert result is None

        # Test invalid scope
        result = read_content("test_doc", scope="invalid")
        assert result is None

        # Test valid document scope for non-existent document
        # Should return None since document doesn't exist
        result = read_content("definitely_nonexistent_doc_12345", scope="document")
        assert result is None

    def test_read_content_chapter_scope_validation(self):
        """Test read_content with chapter scope parameter validation."""
        from document_mcp.mcp_client import read_content

        # Test chapter scope without chapter_name
        result = read_content("test_doc", scope="chapter")
        assert result is None

        # Test chapter scope with invalid chapter_name
        result = read_content("test_doc", scope="chapter", chapter_name="")
        assert result is None

    def test_read_content_paragraph_scope_validation(self):
        """Test read_content with paragraph scope parameter validation."""
        from document_mcp.mcp_client import read_content

        # Test paragraph scope without chapter_name
        result = read_content("test_doc", scope="paragraph")
        assert result is None

        # Test paragraph scope without paragraph_index
        result = read_content("test_doc", scope="paragraph", chapter_name="01-intro.md")
        assert result is None

        # Test paragraph scope with negative paragraph_index
        result = read_content(
            "test_doc",
            scope="paragraph",
            chapter_name="01-intro.md",
            paragraph_index=-1,
        )
        assert result is None

    def test_find_text_document_scope_validation(self):
        """Test find_text with document scope parameter validation."""
        from document_mcp.mcp_client import find_text

        # Test invalid document name
        result = find_text("", "search_term", scope="document")
        assert result is None

        # Test empty search text
        result = find_text("test_doc", "", scope="document")
        assert result is None

        # Test invalid scope
        result = find_text("test_doc", "search_term", scope="invalid")
        assert result is None

    def test_find_text_chapter_scope_validation(self):
        """Test find_text with chapter scope parameter validation."""
        from document_mcp.mcp_client import find_text

        # Test chapter scope without chapter_name
        result = find_text("test_doc", "search_term", scope="chapter")
        assert result is None

        # Test chapter scope with empty chapter_name
        result = find_text("test_doc", "search_term", scope="chapter", chapter_name="")
        assert result is None

    def test_replace_text_document_scope_validation(self):
        """Test replace_text with document scope parameter validation."""
        from document_mcp.mcp_client import replace_text

        # Test invalid document name
        result = replace_text("", "find_text", "replace_text", scope="document")
        assert result is None

        # Test empty find_text
        result = replace_text("test_doc", "", "replace_text", scope="document")
        assert result is None

        # Test invalid scope
        result = replace_text("test_doc", "find_text", "replace_text", scope="invalid")
        assert result is None

    def test_replace_text_chapter_scope_validation(self):
        """Test replace_text with chapter scope parameter validation."""
        from document_mcp.mcp_client import replace_text

        # Test chapter scope without chapter_name
        result = replace_text("test_doc", "find_text", "replace_text", scope="chapter")
        assert result is None

        # Test chapter scope with empty chapter_name
        result = replace_text("test_doc", "find_text", "replace_text", scope="chapter", chapter_name="")
        assert result is None

    def test_get_statistics_document_scope_validation(self):
        """Test get_statistics with document scope parameter validation."""
        from document_mcp.mcp_client import get_statistics

        # Test invalid document name
        result = get_statistics("", scope="document")
        assert result is None

        # Test invalid scope
        result = get_statistics("test_doc", scope="invalid")
        assert result is None

    def test_get_statistics_chapter_scope_validation(self):
        """Test get_statistics with chapter scope parameter validation."""
        from document_mcp.mcp_client import get_statistics

        # Test chapter scope without chapter_name
        result = get_statistics("test_doc", scope="chapter")
        assert result is None

        # Test chapter scope with empty chapter_name
        result = get_statistics("test_doc", scope="chapter", chapter_name="")
        assert result is None

    def test_unified_tools_scope_dispatch(self):
        """Test that unified tools properly dispatch to correct internal functions."""
        from document_mcp.mcp_client import find_text
        from document_mcp.mcp_client import get_statistics
        from document_mcp.mcp_client import read_content
        from document_mcp.mcp_client import replace_text

        # Test that each unified tool properly validates scope parameters
        # This tests the parameter validation and dispatch logic without requiring actual file operations

        # Test all tools with invalid scopes
        invalid_scope = "invalid_scope"

        assert read_content("test_doc", scope=invalid_scope) is None
        assert find_text("test_doc", "search", scope=invalid_scope) is None
        assert replace_text("test_doc", "find", "replace", scope=invalid_scope) is None
        assert get_statistics("test_doc", scope=invalid_scope) is None

    def test_unified_tools_error_handling(self):
        """Test unified tools error handling with various invalid inputs."""
        from document_mcp.mcp_client import find_text
        from document_mcp.mcp_client import get_statistics
        from document_mcp.mcp_client import read_content
        from document_mcp.mcp_client import replace_text

        # Test with None values
        assert read_content(None, scope="document") is None
        assert find_text(None, "search", scope="document") is None
        assert replace_text(None, "find", "replace", scope="document") is None
        assert get_statistics(None, scope="document") is None

        # Test with empty strings
        assert read_content("", scope="document") is None
        assert find_text("", "search", scope="document") is None
        assert replace_text("", "find", "replace", scope="document") is None
        assert get_statistics("", scope="document") is None

    def test_unified_tools_parameter_combinations(self):
        """Test unified tools with various parameter combinations."""
        from document_mcp.mcp_client import find_text
        from document_mcp.mcp_client import read_content

        # Test read_content with all scope variations
        # Document scope (default) for non-existent document
        result = read_content("nonexistent_test_doc_12345")
        assert result is None  # Non-existent document returns None

        # Chapter scope with chapter_name
        result = read_content("test_doc", scope="chapter", chapter_name="01-intro.md")
        assert result is None  # Document doesn't exist, but validation passes

        # Paragraph scope with all required parameters
        result = read_content("test_doc", scope="paragraph", chapter_name="01-intro.md", paragraph_index=0)
        assert result is None  # Document doesn't exist, but validation passes

        # Test find_text with case sensitivity
        result = find_text("test_doc", "search", scope="document", case_sensitive=True)
        assert result is not None  # Returns empty list when document doesn't exist
        assert result == []

        result = find_text(
            "test_doc",
            "search",
            scope="chapter",
            chapter_name="01-intro.md",
            case_sensitive=False,
        )
        assert result is not None  # Returns empty list when chapter doesn't exist
        assert result == []
