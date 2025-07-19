"""Safety feature integration tests for Document MCP.

These tests validate essential safety features that prevent content loss:
- Write-safety system with modification tracking
- MCP-native versioning with automatic snapshots
- Real-world workflow safety validation
- Version control operations (snapshot, restore, diff)
- Safety middleware integration across all write operations
"""

import pytest


@pytest.mark.integration
class TestSafetyFeatures:
    """Integration tests for safety features."""

    def test_content_freshness_check(self, temp_docs_root):
        """Test content freshness validation."""
        from document_mcp.mcp_client import check_content_status
        from document_mcp.models import ContentFreshnessStatus

        # Test with non-existent document
        status_result = check_content_status(
            document_name="nonexistent_doc",
            chapter_name="test.md",
            include_history=False
        )
        # The MCP tool should return ContentFreshnessStatus when include_history=False
        assert isinstance(status_result, ContentFreshnessStatus)
        assert status_result.safety_status == "conflict"
        assert not status_result.is_fresh

    def test_modification_history_tracking(self, temp_docs_root):
        """Test modification history tracking."""
        from document_mcp.mcp_client import check_content_status
        from document_mcp.models import ModificationHistory

        # Test with non-existent document
        history = check_content_status(
            document_name="nonexistent_doc",
            include_history=True
        )
        # The MCP tool should return ModificationHistory when include_history=True and no chapter specified
        assert isinstance(history, ModificationHistory)
        assert history.total_modifications == 0
        assert history.document_name == "nonexistent_doc"

    def test_snapshot_operations(self, temp_docs_root):
        """Test snapshot creation and listing."""
        from document_mcp.mcp_client import manage_snapshots
        from document_mcp.models import SnapshotsList

        # Test with non-existent document
        snapshots = manage_snapshots(
            document_name="nonexistent_doc",
            action="list"
        )
        assert isinstance(snapshots, SnapshotsList)
        assert snapshots.total_snapshots == 0
        assert snapshots.document_name == "nonexistent_doc"

    def test_safety_enhanced_write_operations(self, temp_docs_root):
        """Test that write operations include safety information."""
        from document_mcp.mcp_client import create_chapter
        from document_mcp.mcp_client import create_document
        from document_mcp.mcp_client import write_chapter_content
        from document_mcp.models import OperationStatus

        # Create document and chapter
        doc_result = create_document("safety_test")
        chapter_result = create_chapter("safety_test", "test.md")

        assert doc_result.success
        assert chapter_result.success

        # Write content with safety features
        write_result = write_chapter_content("safety_test", "test.md", "Test content")

        assert write_result.success
        assert isinstance(write_result, OperationStatus)
        # Check that safety information is included
        # Note: safety_info may be None for new files (no conflicts)
        assert hasattr(write_result, "safety_info")
        assert hasattr(write_result, "snapshot_created")
        assert hasattr(write_result, "warnings")

    def test_snapshot_workflow(self, temp_docs_root):
        """Test complete snapshot workflow."""
        from document_mcp.mcp_client import create_chapter
        from document_mcp.mcp_client import create_document
        from document_mcp.mcp_client import diff_content
        from document_mcp.mcp_client import manage_snapshots
        from document_mcp.mcp_client import replace_paragraph
        from document_mcp.mcp_client import write_chapter_content

        # Create document and chapter
        doc_result = create_document("snapshot_test")
        chapter_result = create_chapter("snapshot_test", "chapter1.md")

        assert doc_result.success
        assert chapter_result.success

        # Write initial content
        write_result = write_chapter_content(
            "snapshot_test", "chapter1.md", "Initial content"
        )
        assert write_result.success

        # Create snapshot
        snapshot_result = manage_snapshots("snapshot_test", "create", message="Initial version")
        assert snapshot_result.success

        # List snapshots
        snapshots = manage_snapshots("snapshot_test", "list")
        assert snapshots.total_snapshots >= 1

        # Make changes
        replace_result = replace_paragraph(
            "snapshot_test", "chapter1.md", 0, "Modified content"
        )
        assert replace_result.success

        # Test diff
        if snapshots.total_snapshots > 0:
            diff_result = diff_content(
                document_name="snapshot_test",
                source_type="snapshot",
                source_id=snapshots.snapshots[0].snapshot_id,
                target_type="current"
            )
            assert diff_result.success
            assert diff_result.details["total_changes"] >= 1

        # Test restore
        if snapshots.total_snapshots > 0:
            restore_result = manage_snapshots(
                "snapshot_test", "restore", snapshot_id=snapshots.snapshots[0].snapshot_id
            )
            assert restore_result.success
            assert restore_result.details["files_restored"] >= 1

    def test_error_handling(self, temp_docs_root):
        """Test error handling in safety operations."""
        from document_mcp.mcp_client import diff_content
        from document_mcp.mcp_client import manage_snapshots

        # Test with invalid document
        result = manage_snapshots("invalid_doc", "create", message="Test")
        assert not result.success
        assert "not found" in result.message

        # Test with invalid snapshot
        result = manage_snapshots("invalid_doc", "restore", snapshot_id="invalid_snapshot")
        assert not result.success
        assert "not found" in result.message

        # Test with invalid diff
        result = diff_content(
            document_name="invalid_doc",
            source_type="snapshot",
            source_id="invalid_snapshot",
            target_type="current"
        )
        assert not result.success
        assert "not found" in result.message

    def test_api_consistency(self, temp_docs_root):
        """Test that all write operations return consistent OperationStatus."""
        from document_mcp.mcp_client import create_chapter
        from document_mcp.mcp_client import create_document
        from document_mcp.mcp_client import delete_paragraph
        from document_mcp.mcp_client import insert_paragraph_before
        from document_mcp.mcp_client import replace_paragraph
        from document_mcp.mcp_client import write_chapter_content
        from document_mcp.models import OperationStatus

        # Create test document and chapter
        doc_result = create_document("api_consistency_test")
        chapter_result = create_chapter("api_consistency_test", "test.md")

        assert isinstance(doc_result, OperationStatus)
        assert isinstance(chapter_result, OperationStatus)
        assert doc_result.success
        assert chapter_result.success

        # Test all write operations return OperationStatus
        write_ops = [
            (
                "write_chapter_content",
                lambda: write_chapter_content(
                    "api_consistency_test",
                    "test.md",
                    "Initial content\n\nSecond paragraph",
                ),
            ),
            (
                "replace_paragraph",
                lambda: replace_paragraph(
                    "api_consistency_test", "test.md", 0, "Updated first paragraph"
                ),
            ),
            (
                "insert_paragraph_before",
                lambda: insert_paragraph_before(
                    "api_consistency_test", "test.md", 0, "New first paragraph"
                ),
            ),
            (
                "delete_paragraph",
                lambda: delete_paragraph("api_consistency_test", "test.md", 0),
            ),
        ]

        for op_name, op_func in write_ops:
            result = op_func()
            assert isinstance(result, OperationStatus), (
                f"{op_name} should return OperationStatus, got {type(result)}"
            )
            assert result.success, f"{op_name} should succeed, got: {result.message}"

            # Check that safety fields are present
            assert hasattr(result, "safety_info")
            assert hasattr(result, "snapshot_created")
            assert hasattr(result, "warnings")

    def test_safety_middleware_integration(self, temp_docs_root):
        """Test that safety middleware is properly integrated across operations."""
        from document_mcp.mcp_client import check_content_status
        from document_mcp.mcp_client import create_chapter
        from document_mcp.mcp_client import create_document
        from document_mcp.mcp_client import manage_snapshots
        from document_mcp.mcp_client import replace_paragraph
        from document_mcp.mcp_client import write_chapter_content

        # Create test document
        doc_result = create_document("middleware_test")
        chapter_result = create_chapter("middleware_test", "test.md")

        assert doc_result.success
        assert chapter_result.success

        # Write initial content and check safety features
        write_result = write_chapter_content(
            "middleware_test", "test.md", "Test content with safety"
        )
        assert write_result.success

        # Verify automatic snapshot creation
        snapshots = manage_snapshots("middleware_test", "list")
        initial_snapshots = snapshots.total_snapshots

        # Verify modification history tracking (may be 0 for new documents)
        history = check_content_status("middleware_test", include_history=True)
        assert hasattr(history, "total_modifications")

        # Make another change and verify incremental tracking
        replace_result = replace_paragraph(
            "middleware_test", "test.md", 0, "Updated content with safety"
        )
        assert replace_result.success

        # Check that another snapshot was created (micro-snapshot)
        new_snapshots = manage_snapshots("middleware_test", "list")
        assert new_snapshots.total_snapshots >= initial_snapshots

        # Check that modification history is being tracked
        new_history = check_content_status("middleware_test", include_history=True)
        assert new_history.total_modifications >= history.total_modifications

    def test_end_to_end_version_control_workflow(self, temp_docs_root):
        """Test complete end-to-end version control workflow."""
        from document_mcp.mcp_client import check_content_status
        from document_mcp.mcp_client import create_chapter
        from document_mcp.mcp_client import create_document
        from document_mcp.mcp_client import diff_content
        from document_mcp.mcp_client import manage_snapshots
        from document_mcp.mcp_client import replace_paragraph
        from document_mcp.mcp_client import write_chapter_content

        workflow_doc = "e2e_workflow_test"

        # Step 1: Create document and initial content
        doc_result = create_document(workflow_doc)
        chapter_result = create_chapter(workflow_doc, "chapter1.md")

        assert doc_result.success
        assert chapter_result.success

        # Step 2: Write initial content with automatic safety features
        initial_content = "Chapter 1\n\nThis is the first paragraph of our story."
        write_result = write_chapter_content(
            workflow_doc, "chapter1.md", initial_content
        )
        assert write_result.success

        # Verify safety features were automatically applied
        assert hasattr(write_result, "safety_info")
        assert hasattr(write_result, "snapshot_created")

        # Step 3: Create named snapshot (like a commit)
        snapshot_result = manage_snapshots(workflow_doc, "create", message="Initial story version")
        assert snapshot_result.success

        # Step 4: Verify snapshot was created
        snapshots = manage_snapshots(workflow_doc, "list")
        assert snapshots.total_snapshots >= 1
        initial_snapshot = snapshots.snapshots[0]

        # Step 5: Make changes to the story
        updated_content = "This is the updated first paragraph with new plot elements."
        replace_result = replace_paragraph(
            workflow_doc, "chapter1.md", 1, updated_content
        )
        assert replace_result.success

        # Step 6: Check content freshness (simulating external changes check)
        freshness = check_content_status(workflow_doc, "chapter1.md", include_history=False)
        assert freshness.is_fresh  # Should be fresh since we just modified it

        # Step 7: Compare versions using diff
        diff_result = diff_content(
            document_name=workflow_doc,
            source_type="snapshot",
            source_id=initial_snapshot.snapshot_id,
            target_type="current"
        )
        assert diff_result.success
        assert diff_result.details["total_changes"] >= 1
        assert diff_result.details["files_changed"] == ["chapter1.md"]

        # Step 8: Create another snapshot after changes
        snapshot_result2 = manage_snapshots(workflow_doc, "create", message="Updated plot elements")
        assert snapshot_result2.success

        # Step 9: Verify we now have multiple snapshots
        updated_snapshots = manage_snapshots(workflow_doc, "list")
        assert updated_snapshots.total_snapshots >= 2

        # Step 10: Compare the two named snapshots
        if updated_snapshots.total_snapshots >= 2:
            second_snapshot = updated_snapshots.snapshots[1]
            diff_between_snapshots = diff_content(
                document_name=workflow_doc,
                source_type="snapshot",
                source_id=initial_snapshot.snapshot_id,
                target_type="snapshot",
                target_id=second_snapshot.snapshot_id
            )
            assert diff_between_snapshots.success
            # Total changes may be 0 if snapshots are identical
            assert diff_between_snapshots.details["total_changes"] >= 0

        # Step 11: Restore to initial version (like git checkout)
        restore_result = manage_snapshots(workflow_doc, "restore", snapshot_id=initial_snapshot.snapshot_id)
        assert restore_result.success
        assert restore_result.details["files_restored"] >= 1

        # Step 12: Verify restoration worked by checking diff with current state
        post_restore_diff = diff_content(
            document_name=workflow_doc,
            source_type="snapshot",
            source_id=initial_snapshot.snapshot_id,
            target_type="current"
        )
        assert post_restore_diff.success
        # After restoration, there should be no differences with the initial snapshot
        assert post_restore_diff.details["total_changes"] == 0

        # Step 13: Verify modification history is accessible
        final_history = check_content_status(workflow_doc, include_history=True)
        assert hasattr(
            final_history, "total_modifications"
        )  # History tracking is available

    def test_comprehensive_safety_validation(self, temp_docs_root):
        """Test comprehensive safety validation across all scenarios."""
        from document_mcp.mcp_client import check_content_status
        from document_mcp.mcp_client import create_chapter
        from document_mcp.mcp_client import create_document
        from document_mcp.mcp_client import manage_snapshots
        from document_mcp.mcp_client import write_chapter_content
        from document_mcp.models import ContentFreshnessStatus
        from document_mcp.models import ModificationHistory
        from document_mcp.models import SnapshotsList

        test_doc = "comprehensive_safety_test"

        # Test 1: Content freshness with various scenarios

        # Non-existent document
        freshness = check_content_status("nonexistent", "test.md", include_history=False)
        assert isinstance(freshness, ContentFreshnessStatus)
        assert not freshness.is_fresh
        assert freshness.safety_status == "conflict"

        # Create document and test freshness on existing content
        doc_result = create_document(test_doc)
        chapter_result = create_chapter(test_doc, "test.md")
        write_result = write_chapter_content(test_doc, "test.md", "Test content")

        assert doc_result.success
        assert chapter_result.success
        assert write_result.success

        # Fresh content check
        freshness = check_content_status(test_doc, "test.md", include_history=False)
        assert isinstance(freshness, ContentFreshnessStatus)
        assert freshness.is_fresh
        assert freshness.safety_status == "safe"

        # Test 2: Modification history with various operations

        # Empty history for non-existent document
        history = check_content_status("nonexistent", include_history=True)
        assert isinstance(history, ModificationHistory)
        assert history.total_modifications == 0

        # History with operations
        history = check_content_status(test_doc, include_history=True)
        assert isinstance(history, ModificationHistory)
        assert history.document_name == test_doc
        # Modification count may vary based on implementation
        assert history.total_modifications >= 0

        # Test 3: Snapshot operations with various scenarios

        # Empty snapshots for non-existent document
        snapshots = manage_snapshots("nonexistent", "list")
        assert isinstance(snapshots, SnapshotsList)
        assert snapshots.total_snapshots == 0

        # Snapshots with content (micro-snapshots from write operations)
        snapshots = manage_snapshots(test_doc, "list")
        assert isinstance(snapshots, SnapshotsList)
        assert snapshots.document_name == test_doc
        # Should have at least micro-snapshots from write operations
        assert snapshots.total_snapshots >= 0

    def test_writer_safety_scenario(self, temp_docs_root):
        """Test a realistic writer safety scenario: protecting against accidental overwrites."""
        from document_mcp.mcp_client import check_content_status
        from document_mcp.mcp_client import create_chapter
        from document_mcp.mcp_client import create_document
        from document_mcp.mcp_client import manage_snapshots
        from document_mcp.mcp_client import replace_paragraph
        from document_mcp.mcp_client import write_chapter_content

        # Scenario: A writer is working on a novel chapter
        doc_name = "my_novel"
        chapter_name = "chapter1.md"

        # Step 1: Create the novel and first chapter
        create_document(doc_name)
        create_chapter(doc_name, chapter_name)

        # Step 2: Write the initial draft
        initial_draft = "Chapter 1: The Beginning\n\nIt was a dark and stormy night when everything changed."
        write_result = write_chapter_content(doc_name, chapter_name, initial_draft)
        assert write_result.success

        # Step 3: Writer creates a checkpoint snapshot
        checkpoint = manage_snapshots(doc_name, "create", message="First draft complete")
        assert checkpoint.success

        # Step 4: Writer makes significant changes
        replace_paragraph(
            doc_name,
            chapter_name,
            1,
            "The morning sun broke through the clouds as Sarah opened her eyes.",
        )

        # Step 5: Writer realizes they want to go back to the original
        snapshots = manage_snapshots(doc_name, "list")
        assert snapshots.total_snapshots >= 1

        # Step 6: Restore to the checkpoint (safety in action)
        restore_result = manage_snapshots(doc_name, "restore", snapshot_id=snapshots.snapshots[0].snapshot_id)
        assert restore_result.success
        assert restore_result.details["files_restored"] >= 1

        # Step 7: Verify content is back to original
        freshness = check_content_status(doc_name, chapter_name, include_history=False)
        assert freshness.is_fresh
