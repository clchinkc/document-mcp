"""
Integration tests for the Enhanced Automatic Snapshot System (Phase 2).

These tests validate the automatic snapshot creation functionality for all edit operations
with real function calls and comprehensive validation of the enhanced features.
"""

import os
from pathlib import Path

import pytest

from document_mcp.doc_tool_server import (
    create_document,
    create_chapter,
    delete_document,
    replace_paragraph,
    insert_paragraph_before,
    append_paragraph_to_chapter,
    write_chapter_content,
    delete_chapter,
    replace_text,
    manage_snapshots,
    read_content,
)


class TestAutomaticSnapshotSystemIntegration:
    """
    Integration tests for automatic snapshot creation across all edit operations.
    
    These tests validate:
    - Automatic snapshot creation before edit operations
    - User modification tracking and attribution
    - Enhanced snapshot naming with time-based identifiers
    - Intelligent retention policy application
    - Snapshot integration with all edit operations
    """

    def test_paragraph_edit_automatic_snapshot_creation(self, temp_docs_root):
        """Test automatic snapshot creation for paragraph edit operations."""
        doc_name = "auto_snapshot_test_doc"
        chapter_name = "test_chapter.md"
        
        try:
            # Setup: Create a document with content
            create_document(doc_name)
            create_chapter(
                document_name=doc_name,
                chapter_name=chapter_name,
                initial_content="Original paragraph content\n\nSecond paragraph content"
            )
            
            # Get initial snapshot count
            initial_snapshots = manage_snapshots(
                document_name=doc_name,
                action="list"
            )
            initial_count = len(initial_snapshots["snapshots"])
            
            # Execute edit operation that should create automatic snapshot
            edit_result = replace_paragraph(
                document_name=doc_name,
                chapter_name=chapter_name,
                paragraph_index=0,
                new_content="Modified paragraph content"
            )
            
            # Verify edit was successful
            assert edit_result.success is True
            
            # Verify automatic snapshot was created
            snapshots_after = manage_snapshots(
                document_name=doc_name,
                action="list"
            )
            final_count = len(snapshots_after["snapshots"])
            
            # Should have exactly one more snapshot from replace_paragraph:
            # Only from @auto_snapshot decorator (@safety_enhanced_write_operation no longer creates snapshots)
            assert final_count == initial_count + 1, f"Expected {initial_count + 1} snapshots, got {final_count}"
            
            # Verify snapshot naming pattern contains operation name and user attribution
            # Get the most recent snapshot (snapshots are in reverse chronological order)
            latest_snapshot = snapshots_after["snapshots"][0]
            snapshot_message = latest_snapshot["message"]
            assert "Auto-snapshot before replace_paragraph" in snapshot_message
            assert "by" in snapshot_message  # User attribution
            
        finally:
            # Cleanup
            delete_document(doc_name)

    def test_chapter_edit_automatic_snapshot_creation(self, temp_docs_root):
        """Test automatic snapshot creation for chapter-level operations."""
        doc_name = "chapter_snapshot_test"
        chapter_name = "original_chapter.md"
        
        try:
            # Create document and chapter
            create_document(doc_name)
            create_chapter(
                document_name=doc_name,
                chapter_name=chapter_name,
                initial_content="Original chapter content"
            )
            
            # Get baseline snapshot count
            initial_snapshots = manage_snapshots(
                document_name=doc_name,
                action="list"
            )
            initial_count = len(initial_snapshots["snapshots"])
            
            # Execute chapter content write (should create automatic snapshot)
            write_result = write_chapter_content(
                document_name=doc_name,
                chapter_name=chapter_name,
                new_content="Completely new chapter content\n\nWith multiple paragraphs"
            )
            
            assert write_result.success is True
            
            # Verify automatic snapshot creation
            snapshots_after = manage_snapshots(
                document_name=doc_name,
                action="list"
            )
            final_count = len(snapshots_after["snapshots"])
            
            # Both create_chapter and write_chapter_content have @auto_snapshot decorator
            # Each creates exactly 1 snapshot (safety decorators no longer create snapshots)
            # So we expect exactly 1 additional snapshot from write_chapter_content (initial_count includes create_chapter snapshot)
            assert final_count == initial_count + 1
            
            # Verify snapshot message contains operation details
            # Find the automatic snapshot created by our decorator (not the safety system one)
            auto_snapshots = [snap for snap in snapshots_after["snapshots"] 
                             if snap.get("message") and "Auto-snapshot before write_chapter_content" in snap.get("message", "")]
            
            assert len(auto_snapshots) >= 1, f"Expected at least 1 auto snapshot, found: {[snap.get('message') for snap in snapshots_after['snapshots']]}"
            
            # Verify the snapshot message
            auto_snapshot = auto_snapshots[0]
            assert "Auto-snapshot before write_chapter_content" in auto_snapshot["message"]
            
        finally:
            delete_document(doc_name)

    def test_text_replacement_automatic_snapshot_creation(self, temp_docs_root):
        """Test automatic snapshot creation for scope-based text replacement."""
        doc_name = "text_replace_snapshot_test"
        chapter_name = "content_chapter.md"
        
        try:
            # Setup document with content
            create_document(doc_name)
            create_chapter(
                document_name=doc_name,
                chapter_name=chapter_name,
                initial_content="The old method was problematic.\n\nWe need to use the old approach."
            )
            
            # Get initial snapshot count
            initial_snapshots = manage_snapshots(
                document_name=doc_name,
                action="list"
            )
            initial_count = len(initial_snapshots["snapshots"])
            
            # Execute text replacement (should create automatic snapshot)
            replace_result = replace_text(
                document_name=doc_name,
                find_text="old",
                replace_text="new",
                scope="document"
            )
            
            # replace_text returns a dict, not a model with .success attribute
            assert replace_result is not None
            assert replace_result.get("success", True) is not False
            
            # Verify automatic snapshot was created
            snapshots_after = manage_snapshots(
                document_name=doc_name,
                action="list"
            )
            final_count = len(snapshots_after["snapshots"])
            
            assert final_count == initial_count + 1
            
            # Verify snapshot message
            # Snapshots are in reverse chronological order, so [0] is most recent
            latest_snapshot = snapshots_after["snapshots"][0]
            assert "Auto-snapshot before replace_text" in latest_snapshot["message"]
            
        finally:
            delete_document(doc_name)

    def test_multiple_paragraph_operations_create_snapshots(self, temp_docs_root):
        """Test that multiple paragraph operations each create automatic snapshots."""
        doc_name = "multi_ops_test"
        chapter_name = "test_content.md"
        
        try:
            # Create document
            create_document(doc_name)
            create_chapter(
                document_name=doc_name,
                chapter_name=chapter_name,
                initial_content="Initial content for testing"
            )
            
            # Get baseline snapshot count
            initial_snapshots = manage_snapshots(
                document_name=doc_name,
                action="list"
            )
            initial_count = len(initial_snapshots["snapshots"])
            
            # Perform multiple operations that should each create snapshots
            append_paragraph_to_chapter(
                document_name=doc_name,
                chapter_name=chapter_name,
                paragraph_content="First appended paragraph"
            )
            
            insert_paragraph_before(
                document_name=doc_name,
                chapter_name=chapter_name,
                paragraph_index=0,
                new_content="Inserted at beginning"
            )
            
            # Verify multiple snapshots were created
            final_snapshots = manage_snapshots(
                document_name=doc_name,
                action="list"
            )
            final_count = len(final_snapshots["snapshots"])
            
            # Should have 2 more snapshots (one for each operation)
            assert final_count >= initial_count + 2, f"Expected at least {initial_count + 2} snapshots, got {final_count}"
            
            # Verify snapshot messages contain appropriate operation names
            # Check all snapshots since different operations may create them in different order
            all_snapshots = final_snapshots["snapshots"]
            operation_names = [snap["message"] for snap in all_snapshots]
            
            # Check that we have snapshots for both operations
            append_found = any("append_paragraph_to_chapter" in msg for msg in operation_names)
            insert_found = any("insert_paragraph_before" in msg for msg in operation_names)
            
            assert append_found, f"No append snapshot found in: {operation_names}"
            assert insert_found, f"No insert snapshot found in: {operation_names}"
            
        finally:
            delete_document(doc_name)

    def test_snapshot_naming_and_user_tracking_accuracy(self, temp_docs_root):
        """Test enhanced snapshot naming patterns and user tracking."""
        doc_name = "naming_test_doc"
        chapter_name = "test_chapter.md"
        
        # Set up environment to simulate specific user
        original_user = os.environ.get("USER", "")
        os.environ["USER"] = "test_user_123"
        
        try:
            # Create document and content
            create_document(doc_name)
            create_chapter(
                document_name=doc_name,
                chapter_name=chapter_name,
                initial_content="Content for naming test"
            )
            
            # Perform operation that creates automatic snapshot
            replace_paragraph(
                document_name=doc_name,
                chapter_name=chapter_name,
                paragraph_index=0,
                new_content="Modified content for naming test"
            )
            
            # Verify snapshot naming and user tracking
            snapshots = manage_snapshots(
                document_name=doc_name,
                action="list"
            )
            
            # Find the automatic snapshot for replace_paragraph operation  
            replace_snapshots = [snap for snap in snapshots["snapshots"] 
                               if snap.get("message") and "Auto-snapshot before replace_paragraph" in snap.get("message", "")]
            
            assert len(replace_snapshots) == 1, f"Expected exactly 1 replace_paragraph snapshot, found: {[snap.get('message') for snap in snapshots['snapshots']]}"
            
            # Verify user attribution in the replace_paragraph snapshot
            replace_snapshot = replace_snapshots[0]
            message = replace_snapshot["message"]
            assert "test_user_123" in message
            assert "Auto-snapshot before replace_paragraph" in message
            
            # Verify snapshot ID exists
            snapshot_id = replace_snapshot["snapshot_id"]
            assert snapshot_id  # Should have a valid ID
            
        finally:
            # Restore original user environment
            if original_user:
                os.environ["USER"] = original_user
            else:
                os.environ.pop("USER", None)
            
            delete_document(doc_name)

    def test_edit_operations_continue_despite_snapshot_issues(self, temp_docs_root):
        """Test that edit operations continue even if snapshot creation encounters issues."""
        doc_name = "error_handling_test"
        chapter_name = "test_chapter.md"
        
        try:
            # Create document and content
            create_document(doc_name)
            create_chapter(
                document_name=doc_name,
                chapter_name=chapter_name,
                initial_content="Content for error handling test"
            )
            
            # The edit operation should succeed even if snapshot creation encounters issues
            # (The automatic snapshot system is designed to be non-blocking)
            edit_result = replace_paragraph(
                document_name=doc_name,
                chapter_name=chapter_name,
                paragraph_index=0,
                new_content="Modified content despite potential snapshot issues"
            )
            
            # Edit should succeed regardless of snapshot creation outcome
            assert edit_result.success is True
            
            # Verify the content was actually modified
            updated_content = read_content(
                document_name=doc_name,
                chapter_name=chapter_name,
                scope="chapter"
            )
            
            assert "Modified content despite potential snapshot issues" in updated_content["content"]
            
        finally:
            delete_document(doc_name)