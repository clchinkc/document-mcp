"""Integration tests for batch operations in the Document MCP tool server."""

import pytest

from document_mcp.mcp_client import batch_apply_operations
from document_mcp.mcp_client import delete_document
from document_mcp.mcp_client import list_chapters
from document_mcp.mcp_client import list_documents
from document_mcp.mcp_client import read_content


class TestBatchOperationsIntegration:
    """Integration tests for batch operations with real MCP server."""

    @pytest.mark.asyncio
    async def test_batch_apply_operations_create_document_and_chapter(
        self, document_factory
    ):
        """Test batch operation to create document and chapter atomically."""
        doc_name = "test_batch_doc"

        operations = [
            {
                "operation_type": "create_document",
                "target": {},
                "parameters": {"document_name": doc_name},
                "order": 1,
                "operation_id": "create_doc",
            },
            {
                "operation_type": "create_chapter",
                "target": {"document_name": doc_name},
                "parameters": {
                    "chapter_name": "01-intro.md",
                    "initial_content": "# Introduction\n\nThis is a test chapter created via batch operation.",
                },
                "order": 2,
                "operation_id": "create_chapter",
                "depends_on": ["create_doc"],
            },
        ]

        result = batch_apply_operations(
            operations=operations, atomic=True, validate_only=False
        )

        assert result.success is True
        assert result.total_operations == 2
        assert result.successful_operations == 2
        assert result.failed_operations == 0
        assert len(result.operation_results) == 2

        for op_result in result.operation_results:
            assert op_result.success is True

        docs = list_documents()
        doc_names = [doc.document_name for doc in docs]
        assert doc_name in doc_names

        chapters = list_chapters(doc_name)
        assert len(chapters) == 1
        assert chapters[0].chapter_name == "01-intro.md"

        chapter_content = read_content(
            doc_name, scope="chapter", chapter_name="01-intro.md"
        )
        assert (
            "This is a test chapter created via batch operation"
            in chapter_content.content
        )

    @pytest.mark.asyncio
    async def test_batch_apply_operations_validate_only_mode(self, document_factory):
        """Test batch operation in validation-only mode."""
        operations = [
            {
                "operation_type": "create_document",
                "target": {},
                "parameters": {"document_name": "validate_test_doc"},
                "order": 1,
                "operation_id": "validate_doc",
            }
        ]

        result = batch_apply_operations(operations=operations, validate_only=True)

        assert result.success is True
        assert result.total_operations == 1
        assert result.successful_operations == 0
        assert result.failed_operations == 0
        assert "Validation successful" in result.summary

        docs = list_documents()
        doc_names = [doc.document_name for doc in docs]
        assert "validate_test_doc" not in doc_names

    @pytest.mark.asyncio
    async def test_batch_apply_operations_atomic_failure(
        self, temp_docs_root, document_factory
    ):
        """Test batch operation atomic failure with rollback."""
        operations = [
            {
                "operation_type": "create_document",
                "target": {},
                "parameters": {"document_name": "atomic_test_doc"},
                "order": 1,
                "operation_id": "create_valid_doc",
            },
            {
                "operation_type": "unknown_operation",
                "target": {},
                "parameters": {},
                "order": 2,
                "operation_id": "fail_op",
            },
        ]

        result = batch_apply_operations(operations=operations)

        assert result.success is False
        assert result.total_operations == 2
        assert result.successful_operations == 1
        assert result.failed_operations == 1
        assert "Batch failed" in result.error_summary

        failed_op = result.operation_results[1]
        assert failed_op.success is False
        assert "Unknown operation type" in failed_op.error

    @pytest.mark.asyncio
    async def test_batch_apply_operations_continue_on_error_mode(
        self, temp_docs_root, document_factory
    ):
        """Test batch operation with continue_on_error=True."""
        doc_name = "continue_test_doc"

        operations = [
            {
                "operation_type": "create_document",
                "target": {},
                "parameters": {"document_name": doc_name},
                "order": 1,
                "operation_id": "create_doc",
            },
            {
                "operation_type": "unknown_operation",
                "target": {},
                "parameters": {},
                "order": 2,
                "operation_id": "fail_op",
            },
            {
                "operation_type": "create_chapter",
                "target": {"document_name": doc_name},
                "parameters": {"chapter_name": "test.md", "initial_content": "# Test"},
                "order": 3,
                "operation_id": "create_chapter",
            },
        ]

        result = batch_apply_operations(
            operations=operations, atomic=False, continue_on_error=True
        )

        assert result.success is False
        assert result.total_operations == 3
        assert result.successful_operations == 2
        assert result.failed_operations == 1

        docs = list_documents()
        doc_names = [doc.document_name for doc in docs]
        assert doc_name in doc_names

        chapters = list_chapters(doc_name)
        assert len(chapters) == 1
        assert chapters[0].chapter_name == "test.md"

    @pytest.mark.asyncio
    async def test_batch_apply_operations_with_unified_read_content(
        self, temp_docs_root, document_factory
    ):
        """Test batch operation using the unified read_content tool."""
        doc_name = "unified_batch_test"
        chapters = {
            "chapter1.md": "# Chapter 1\n\nFirst chapter content.",
            "chapter2.md": "# Chapter 2\n\nSecond chapter content.",
        }
        document_factory(doc_name, chapters)

        operations = [
            {
                "operation_type": "read_content",
                "target": {"document_name": doc_name},
                "parameters": {"scope": "document"},
                "order": 1,
                "operation_id": "read_full_doc",
            },
            {
                "operation_type": "read_content",
                "target": {"document_name": doc_name},
                "parameters": {"scope": "chapter", "chapter_name": "chapter1.md"},
                "order": 2,
                "operation_id": "read_chapter",
            },
        ]

        result = batch_apply_operations(operations=operations)

        assert result.success is True
        assert result.total_operations == 2
        assert result.successful_operations == 2
        assert result.failed_operations == 0

        doc_read_result = result.operation_results[0]
        assert doc_read_result.success is True
        assert doc_read_result.result["document_name"] == doc_name
        assert len(doc_read_result.result["chapters"]) == 2

        chapter_read_result = result.operation_results[1]
        assert chapter_read_result.success is True
        assert chapter_read_result.result["chapter_name"] == "chapter1.md"
        assert "First chapter content" in chapter_read_result.result["content"]


class TestBatchOperationsForDocumentCreation:
    """Integration tests for batch operations replacing legacy composite operations."""

    def test_batch_create_document_with_chapters_success(self, temp_docs_root):
        """Test successful document creation with multiple chapters using batch operations."""
        doc_name = "test_batch_doc"

        operations = [
            {
                "operation_type": "create_document",
                "target": {},
                "parameters": {"document_name": doc_name},
                "order": 1,
                "operation_id": "create_doc",
            },
            {
                "operation_type": "create_chapter",
                "target": {},
                "parameters": {
                    "document_name": doc_name,
                    "chapter_name": "01-introduction.md",
                    "initial_content": "# Introduction\n\nWelcome to the guide.",
                },
                "order": 2,
                "operation_id": "create_chapter1",
                "depends_on": ["create_doc"],
            },
            {
                "operation_type": "create_chapter",
                "target": {},
                "parameters": {
                    "document_name": doc_name,
                    "chapter_name": "02-setup.md",
                    "initial_content": "# Setup\n\nInstallation instructions.",
                },
                "order": 3,
                "operation_id": "create_chapter2",
                "depends_on": ["create_doc"],
            },
            {
                "operation_type": "create_chapter",
                "target": {},
                "parameters": {
                    "document_name": doc_name,
                    "chapter_name": "03-usage.md",
                    "initial_content": "",
                },
                "order": 4,
                "operation_id": "create_chapter3",
                "depends_on": ["create_doc"],
            },
        ]

        result = batch_apply_operations(operations, atomic=True, snapshot_before=True)

        assert result.success is True
        assert result.total_operations == 4
        assert result.successful_operations == 4

        docs = list_documents()
        doc_names = [doc.document_name for doc in docs]
        assert doc_name in doc_names

        chapters_list = list_chapters(doc_name)
        chapter_names = [ch.chapter_name for ch in chapters_list]
        assert len(chapter_names) == 3
        assert "01-introduction.md" in chapter_names
        assert "02-setup.md" in chapter_names
        assert "03-usage.md" in chapter_names

        intro_content = read_content(
            doc_name, scope="chapter", chapter_name="01-introduction.md"
        )
        assert "Welcome to the guide" in intro_content.content

        setup_content = read_content(
            doc_name, scope="chapter", chapter_name="02-setup.md"
        )
        assert "Installation instructions" in setup_content.content

        delete_document(doc_name)

    def test_batch_create_document_only(self, temp_docs_root):
        """Test batch document creation with no chapters."""
        doc_name = "test_single_doc"

        operations = [
            {
                "operation_type": "create_document",
                "target": {},
                "parameters": {"document_name": doc_name},
                "order": 1,
                "operation_id": "create_doc",
            }
        ]

        result = batch_apply_operations(operations, atomic=True)

        assert result.success is True
        assert result.total_operations == 1
        assert result.successful_operations == 1

        docs = list_documents()
        doc_names = [doc.document_name for doc in docs]
        assert doc_name in doc_names

        delete_document(doc_name)

    def test_batch_rollback_on_invalid_chapter(self, temp_docs_root):
        """Test batch operation rollback when chapter creation fails."""
        doc_name = "test_rollback_batch"

        operations = [
            {
                "operation_type": "create_document",
                "target": {},
                "parameters": {"document_name": doc_name},
                "order": 1,
                "operation_id": "create_doc",
            },
            {
                "operation_type": "create_chapter",
                "target": {},
                "parameters": {
                    "document_name": doc_name,
                    "chapter_name": "01-valid.md",
                    "initial_content": "# Valid Chapter",
                },
                "order": 2,
                "operation_id": "create_chapter1",
                "depends_on": ["create_doc"],
            },
            {
                "operation_type": "create_chapter",
                "target": {},
                "parameters": {
                    "document_name": doc_name,
                    "chapter_name": "invalid_chapter_name_without_md_extension",
                    "initial_content": "This should fail",
                },
                "order": 3,
                "operation_id": "create_chapter2",
                "depends_on": ["create_doc"],
            },
        ]

        result = batch_apply_operations(operations, atomic=True, snapshot_before=True)

        assert result.success is False

        docs = list_documents()
        doc_names = [doc.document_name for doc in docs]
        assert doc_name not in doc_names

    def test_batch_duplicate_document_name(self, document_factory):
        """Test batch operation error handling when document name already exists."""
        doc_name = "existing_batch_doc"

        document_factory(doc_name, {"existing.md": "Content"})

        operations = [
            {
                "operation_type": "create_document",
                "target": {},
                "parameters": {"document_name": doc_name},
                "order": 1,
                "operation_id": "create_doc",
            }
        ]

        result = batch_apply_operations(operations, atomic=True)

        assert result.success is False
        assert "already exists" in str(result).lower()

        original_chapters = list_chapters(doc_name)
        assert len(original_chapters) == 1
        assert original_chapters[0].chapter_name == "existing.md"


class TestBatchOperationsWithDependencies:
    """Integration tests for batch operations with dependency resolution."""

    @pytest.mark.asyncio
    async def test_batch_operations_with_simple_dependencies(
        self, temp_docs_root, document_factory
    ):
        """Test batch operations with simple dependency chain."""
        doc_name = "dependency_test_doc"

        operations = [
            {
                "operation_type": "append_paragraph_to_chapter",
                "target": {"document_name": doc_name, "chapter_name": "intro.md"},
                "parameters": {"new_content": "This is an additional paragraph."},
                "order": 3,
                "operation_id": "append_para",
                "depends_on": ["create_chapter"],
            },
            {
                "operation_type": "create_chapter",
                "target": {"document_name": doc_name},
                "parameters": {
                    "chapter_name": "intro.md",
                    "initial_content": "# Introduction\n\nWelcome to the guide.",
                },
                "order": 2,
                "operation_id": "create_chapter",
                "depends_on": ["create_doc"],
            },
            {
                "operation_type": "create_document",
                "target": {},
                "parameters": {"document_name": doc_name},
                "order": 1,
                "operation_id": "create_doc",
                "depends_on": [],
            },
        ]

        result = batch_apply_operations(operations=operations)

        assert result.success is True
        assert result.total_operations == 3
        assert result.successful_operations == 3
        assert result.failed_operations == 0

        assert len(result.operation_results) == 3
        assert result.operation_results[0].operation_id == "create_doc"
        assert result.operation_results[1].operation_id == "create_chapter"
        assert result.operation_results[2].operation_id == "append_para"

        docs = list_documents()
        doc_names = [doc.document_name for doc in docs]
        assert doc_name in doc_names

        chapters = list_chapters(doc_name)
        assert len(chapters) == 1
        assert chapters[0].chapter_name == "intro.md"

        chapter_content = read_content(
            doc_name, scope="chapter", chapter_name="intro.md"
        )
        assert "Welcome to the guide" in chapter_content.content
        assert "This is an additional paragraph" in chapter_content.content

    @pytest.mark.asyncio
    async def test_batch_operations_with_multiple_dependencies(self, document_factory):
        """Test batch operation where one operation depends on multiple others."""
        doc_name = "multi_dep_test_doc"

        operations = [
            {
                "operation_type": "replace_text",
                "target": {"document_name": doc_name},
                "parameters": {
                    "find_text": "placeholder",
                    "replace_text": "final content",
                    "scope": "document",
                },
                "order": 4,
                "operation_id": "replace_text_op",
                "depends_on": ["create_ch1", "create_ch2"],
            },
            {
                "operation_type": "create_chapter",
                "target": {"document_name": doc_name},
                "parameters": {
                    "chapter_name": "chapter2.md",
                    "initial_content": "# Chapter 2\n\nSecond placeholder content.",
                },
                "order": 3,
                "operation_id": "create_ch2",
                "depends_on": ["create_doc"],
            },
            {
                "operation_type": "create_document",
                "target": {},
                "parameters": {"document_name": doc_name},
                "order": 1,
                "operation_id": "create_doc",
                "depends_on": [],
            },
            {
                "operation_type": "create_chapter",
                "target": {"document_name": doc_name},
                "parameters": {
                    "chapter_name": "chapter1.md",
                    "initial_content": "# Chapter 1\n\nFirst placeholder content.",
                },
                "order": 2,
                "operation_id": "create_ch1",
                "depends_on": ["create_doc"],
            },
        ]

        result = batch_apply_operations(operations=operations)

        assert result.success is True
        assert result.total_operations == 4
        assert result.successful_operations == 4

        op_results = result.operation_results
        assert op_results[0].operation_id == "create_doc"
        assert op_results[-1].operation_id == "replace_text_op"

        ch1_index = next(
            i for i, op in enumerate(op_results) if op.operation_id == "create_ch1"
        )
        ch2_index = next(
            i for i, op in enumerate(op_results) if op.operation_id == "create_ch2"
        )
        replace_index = next(
            i for i, op in enumerate(op_results) if op.operation_id == "replace_text_op"
        )

        assert ch1_index < replace_index
        assert ch2_index < replace_index

        ch1_content = read_content(
            doc_name, scope="chapter", chapter_name="chapter1.md"
        )
        ch2_content = read_content(
            doc_name, scope="chapter", chapter_name="chapter2.md"
        )
        assert "final content" in ch1_content.content
        assert "final content" in ch2_content.content
        assert "placeholder" not in ch1_content.content
        assert "placeholder" not in ch2_content.content

    @pytest.mark.asyncio
    async def test_batch_operations_circular_dependency_failure(self):
        """Test that circular dependencies are properly detected and cause batch failure."""
        operations = [
            {
                "operation_type": "create_document",
                "target": {},
                "parameters": {"document_name": "circular_test"},
                "order": 1,
                "operation_id": "op_a",
                "depends_on": ["op_b"],
            },
            {
                "operation_type": "create_chapter",
                "target": {"document_name": "circular_test"},
                "parameters": {"chapter_name": "test.md", "initial_content": "Test"},
                "order": 2,
                "operation_id": "op_b",
                "depends_on": ["op_a"],
            },
        ]

        result = batch_apply_operations(operations=operations)

        assert result.success is False
        assert result.total_operations == 2
        assert result.successful_operations == 0
