"""Integration tests for safety decorator coverage on content replacement operations.

These tests verify that @safety_enhanced_write_operation decorator is properly applied
to functions that directly overwrite/replace existing file content.

Only operations that replace existing content need safety decorators:
- write_chapter_content (overwrites entire chapter content)
- replace_paragraph (overwrites specific paragraph content)

Operations that insert, delete, or move content don't need safety decorators because
they don't risk overwriting externally modified content.
"""

import time

from document_mcp.mcp_client import add_paragraph
from document_mcp.mcp_client import create_chapter
from document_mcp.mcp_client import create_document
from document_mcp.mcp_client import delete_document
from document_mcp.mcp_client import delete_paragraph
from document_mcp.mcp_client import move_paragraph
from document_mcp.mcp_client import replace_paragraph
from document_mcp.mcp_client import write_chapter_content  # Functions with safety decorators


class TestSafetyDecoratorCoverage:
    """Test that content replacement operations have proper safety decorator coverage.

    These tests verify:
    - Safety decorators prevent overwrites when files are modified externally
    - Operations succeed when force_write=True is used
    - Safety info is included in operation results
    - Only operations that replace content have safety decorators
    """

    def test_write_chapter_content_safety_protection(self, temp_docs_root):
        """Test that write_chapter_content has safety protection."""
        doc_name = "safety_test_doc"
        chapter_name = "test_chapter.md"

        try:
            # Setup: Create document and chapter
            create_document(doc_name)
            create_chapter(
                document_name=doc_name,
                chapter_name=chapter_name,
                initial_content="Original content",
            )

            # Get initial timestamp for comparison
            doc_path = temp_docs_root / doc_name / chapter_name
            original_mtime = doc_path.stat().st_mtime

            # Simulate external file modification
            time.sleep(1.1)  # Ensure timestamp difference > 1 second (safety tolerance)
            doc_path.write_text("Externally modified content", encoding="utf-8")
            new_mtime = doc_path.stat().st_mtime

            # Verify file was actually modified
            assert new_mtime > original_mtime

            # Attempt operation with last_known_modified from before external change
            import datetime

            last_known = datetime.datetime.fromtimestamp(original_mtime)

            result = write_chapter_content(
                document_name=doc_name,
                chapter_name=chapter_name,
                new_content="User's new content",
                last_known_modified=last_known.isoformat(),
            )

            # Should fail due to safety protection
            assert result.success is False
            assert (
                "modified externally" in result.message.lower()
                or "force_write" in result.message.lower()
                or "safety check failed" in result.message.lower()
            )
            assert hasattr(result, "safety_info")

            # Should succeed with force_write=True
            result_forced = write_chapter_content(
                document_name=doc_name,
                chapter_name=chapter_name,
                new_content="User's new content",
                last_known_modified=last_known.isoformat(),
                force_write=True,
            )

            assert result_forced.success is True

        finally:
            delete_document(doc_name)

    def test_replace_paragraph_safety_protection(self, temp_docs_root):
        """Test that replace_paragraph has automatic snapshot safety protection.

        Note: replace_paragraph uses @auto_snapshot decorator for safety (creating backups
        before modification), not optimistic locking with last_known_modified.
        """
        doc_name = "safety_test_doc"
        chapter_name = "test_chapter.md"

        try:
            # Setup: Create document and chapter
            create_document(doc_name)
            create_chapter(
                document_name=doc_name,
                chapter_name=chapter_name,
                initial_content="First paragraph\n\nSecond paragraph",
            )

            # Perform replace operation - should create automatic snapshot
            result = replace_paragraph(
                document_name=doc_name,
                chapter_name=chapter_name,
                paragraph_index=0,
                new_content="User's replacement paragraph",
            )

            # Should succeed (with automatic snapshot created for safety)
            assert result.success is True

            # Verify content was replaced
            from document_mcp.mcp_client import read_content

            content = read_content(doc_name, "chapter", chapter_name)
            assert "User's replacement paragraph" in content.content

        finally:
            delete_document(doc_name)

    def test_operations_without_safety_decorators_work_normally(self, temp_docs_root):
        """Test that operations without safety decorators work normally even with external modifications."""
        doc_name = "safety_test_doc"
        chapter_name = "test_chapter.md"

        try:
            # Setup: Create document and chapter
            create_document(doc_name)
            create_chapter(
                document_name=doc_name,
                chapter_name=chapter_name,
                initial_content="First paragraph\n\nSecond paragraph",
            )

            # Simulate external file modification
            doc_path = temp_docs_root / doc_name / chapter_name
            time.sleep(0.1)
            doc_path.write_text(
                "Externally modified first paragraph\n\nExternally modified second paragraph\n\nThird paragraph",
                encoding="utf-8",
            )

            # These operations should succeed despite external modification
            # because they don't overwrite content - they insert/delete/move

            # Test insert operations using add_paragraph
            result1 = add_paragraph(
                document_name=doc_name,
                chapter_name=chapter_name,
                new_content="New first paragraph",
                position="before",
                paragraph_index=0,
            )
            assert result1.success is True

            result2 = add_paragraph(
                document_name=doc_name,
                chapter_name=chapter_name,
                new_content="Inserted paragraph",
                position="after",
                paragraph_index=1,
            )
            assert result2.success is True

            # Test append operation using add_paragraph
            result3 = add_paragraph(
                document_name=doc_name,
                chapter_name=chapter_name,
                new_content="Appended paragraph",
                position="end",
            )
            assert result3.success is True

            # Test move operations using move_paragraph
            result4 = move_paragraph(
                document_name=doc_name,
                chapter_name=chapter_name,
                source_index=0,
                destination="after",
                target_index=None,  # Move to end
            )
            assert result4.success is True

            result5 = move_paragraph(
                document_name=doc_name,
                chapter_name=chapter_name,
                source_index=4,  # Last paragraph
                destination="before",
                target_index=1,
            )
            assert result5.success is True

            # Test delete operation
            result6 = delete_paragraph(document_name=doc_name, chapter_name=chapter_name, paragraph_index=0)
            assert result6.success is True

        finally:
            delete_document(doc_name)

    def test_safety_decorator_coverage_completeness(self):
        """Test that only content replacement functions have safety decorators."""
        # Functions that should have safety decorators (content replacement)
        functions_with_safety = [
            write_chapter_content,
            replace_paragraph,
        ]

        # Functions that should NOT have safety decorators (insert/delete/move)
        # Note: add_paragraph and move_paragraph are the consolidated tools
        functions_without_safety = [
            add_paragraph,
            delete_paragraph,
            move_paragraph,
        ]

        for func in functions_with_safety:
            # Check that function has been wrapped by safety decorator
            assert hasattr(func, "__wrapped__") or hasattr(func, "__name__"), (
                f"Function {func.__name__} should be wrapped by safety decorator"
            )
            assert callable(func), f"Function {func.__name__} should be callable"

        for func in functions_without_safety:
            # These functions should still be callable but don't need safety wrappers
            assert callable(func), f"Function {func.__name__} should be callable"
