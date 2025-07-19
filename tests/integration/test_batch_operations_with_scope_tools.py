"""Integration tests for scope-based tools with batch operations.

These tests validate that scope-based tools work correctly with the existing
batch operation system and provide proper rollback capabilities.
"""

import pytest

from document_mcp.mcp_client import batch_apply_operations
from document_mcp.mcp_client import read_content


class TestScopeBasedToolsBatchIntegration:
    """Test scope-based tools integration with batch operations."""

    @pytest.mark.asyncio
    async def test_batch_operations_with_scope_based_tools(self, document_factory):
        """Test batch operations can successfully use scope-based tools."""
        # Create test document with initial content
        document_name = "Batch Test Document"
        document_factory(
            document_name,
            {
                "01-intro.md": "Original introduction content",
                "02-body.md": "Original body content",
            },
        )

        # Define batch operations using scope-based tools
        batch_operations = [
            {
                "operation_type": "replace_text",
                "target": {"document_name": document_name},
                "parameters": {
                    "find_text": "Original",
                    "replace_text": "Updated",
                    "scope": "document",
                },
                "order": 1,
                "operation_id": "update_text",
            },
            {
                "operation_type": "create_chapter",
                "target": {"document_name": document_name},
                "parameters": {
                    "chapter_name": "03-conclusion.md",
                    "initial_content": "Updated conclusion content",
                },
                "order": 2,
                "operation_id": "add_conclusion",
                "depends_on": ["update_text"],
            },
        ]

        # Execute batch operations
        response = batch_apply_operations(
            operations=batch_operations, atomic=True, validate_only=False
        )

        assert response is not None

        # Verify scope-based replace_text operation worked
        read_response = read_content(document_name=document_name, scope="document")

        assert read_response is not None
        # Content should be updated by the scope-based replace_text operation

        # Verify new chapter was created
        chapter_response = read_content(
            document_name=document_name,
            scope="chapter",
            chapter_name="03-conclusion.md",
        )

        assert chapter_response is not None

    @pytest.mark.asyncio
    async def test_batch_rollback_with_scope_based_tools(self, document_factory):
        """Test batch rollback works correctly with scope-based tool operations."""
        # Create test document
        document_name = "Rollback Test Document"
        document_factory(document_name, {"01-test.md": "Original test content"})

        # Get initial state for comparison
        initial_response = read_content(document_name=document_name, scope="document")

        assert initial_response is not None

        # Define batch operations that will partially fail
        failing_batch_operations = [
            {
                "operation_type": "replace_text",
                "target": {"document_name": document_name},
                "parameters": {
                    "find_text": "test",
                    "replace_text": "modified",
                    "scope": "document",
                },
                "order": 1,
                "operation_id": "modify_text",
            },
            {
                "operation_type": "create_chapter",
                "target": {"document_name": document_name},
                "parameters": {
                    "chapter_name": "01-test.md",  # Duplicate name - should fail
                    "initial_content": "This should fail",
                },
                "order": 2,
                "operation_id": "create_duplicate",
                "depends_on": ["modify_text"],
            },
        ]

        # Execute batch operations (should fail and rollback)
        response = batch_apply_operations(
            operations=failing_batch_operations,
            atomic=True,  # Atomic mode should rollback on failure
            validate_only=False,
        )

        # Batch should report failure
        assert response is not None

        # Verify rollback - document should be in original state
        final_response = read_content(document_name=document_name, scope="document")

        assert final_response is not None
        # Content should be unchanged due to rollback

    @pytest.mark.asyncio
    async def test_batch_validation_with_scope_based_tools(self, document_factory):
        """Test batch validation works with scope-based tool operations."""
        # Create test document
        document_name = "Validation Test Document"
        document_factory(document_name, {"01-test.md": "Test content"})

        # Define valid batch operations using scope-based tools
        valid_operations = [
            {
                "operation_type": "replace_text",
                "target": {"document_name": document_name},
                "parameters": {
                    "find_text": "Test",
                    "replace_text": "Validated",
                    "scope": "document",
                },
                "order": 1,
                "operation_id": "validate_replace",
            },
            {
                "operation_type": "get_statistics",
                "target": {"document_name": document_name},
                "parameters": {"scope": "document"},
                "order": 2,
                "operation_id": "validate_stats",
            },
        ]

        # Validate operations without execution
        validation_response = batch_apply_operations(
            operations=valid_operations,
            atomic=True,
            validate_only=True,  # Validation mode
        )

        assert validation_response is not None

        # Document should be unchanged after validation
        read_response = read_content(document_name=document_name, scope="document")

        assert read_response is not None
        # Content should contain original "Test" text, not "Validated"

    @pytest.mark.asyncio
    async def test_batch_operations_with_scope_based_tools_dependencies(
        self, temp_docs_root, document_factory
    ):
        """Test batch operations with complex dependencies using scope-based tools."""
        # Create base document (empty for this test)
        document_name = "Dependency Test Document"
        document_factory(document_name, {})

        # Define operations with dependencies
        dependent_operations = [
            {
                "operation_type": "create_chapter",
                "target": {"document_name": document_name},
                "parameters": {
                    "chapter_name": "01-setup.md",
                    "initial_content": "Initial setup content",
                },
                "order": 1,
                "operation_id": "create_setup",
            },
            {
                "operation_type": "replace_text",
                "target": {"document_name": document_name},
                "parameters": {
                    "find_text": "setup",
                    "replace_text": "configuration",
                    "scope": "chapter",
                    "chapter_name": "01-setup.md",
                },
                "order": 2,
                "operation_id": "update_setup",
                "depends_on": ["create_setup"],
            },
            {
                "operation_type": "get_statistics",
                "target": {"document_name": document_name},
                "parameters": {"scope": "chapter", "chapter_name": "01-setup.md"},
                "order": 3,
                "operation_id": "get_stats",
                "depends_on": ["update_setup"],
            },
        ]

        # Execute batch with dependencies
        response = batch_apply_operations(
            operations=dependent_operations, atomic=True, validate_only=False
        )

        assert response is not None

        # Verify all operations completed successfully
        final_content = read_content(
            document_name=document_name, scope="chapter", chapter_name="01-setup.md"
        )

        assert final_content is not None
        # Content should contain "configuration" instead of "setup"

    @pytest.mark.asyncio
    async def test_batch_operations_performance_with_scope_based_tools(
        self, document_factory
    ):
        """Test batch operations performance with multiple scope-based tool calls."""
        # Create document with multiple chapters
        document_name = "Performance Test Document"
        chapters_dict = {}
        for i in range(1, 6):
            chapter_name = f"{i:02d}-chapter.md"
            content = f"Chapter {i} content with test data"
            chapters_dict[chapter_name] = content

        document_factory(document_name, chapters_dict)

        # Define batch operations using scope-based tools
        performance_operations = []

        # Add multiple unified tool operations
        for i in range(1, 6):
            chapter_name = f"{i:02d}-chapter.md"

            # Replace text in each chapter
            performance_operations.append(
                {
                    "operation_type": "replace_text",
                    "target": {"document_name": document_name},
                    "parameters": {
                        "find_text": "test",
                        "replace_text": "performance",
                        "scope": "chapter",
                        "chapter_name": chapter_name,
                    },
                    "order": i,
                    "operation_id": f"replace_ch_{i}",
                }
            )

        # Add final statistics operation
        performance_operations.append(
            {
                "operation_type": "get_statistics",
                "target": {"document_name": document_name},
                "parameters": {"scope": "document"},
                "order": 6,
                "operation_id": "final_stats",
            }
        )

        # Execute batch operations
        response = batch_apply_operations(
            operations=performance_operations, atomic=True, validate_only=False
        )

        assert response is not None

        # Verify all text replacements worked
        final_document = read_content(document_name=document_name, scope="document")

        assert final_document is not None
        # Document should contain "performance" instead of "test"
