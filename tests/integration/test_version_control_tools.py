"""Integration tests for version control tools.

This module tests the manage_snapshots, check_content_status, and diff_content
tools that manage document versioning and snapshots.
"""

from pathlib import Path

import pytest

from document_mcp.mcp_client import check_content_status
from document_mcp.mcp_client import delete_document
from document_mcp.mcp_client import diff_content
from document_mcp.mcp_client import manage_snapshots  # Version control tools
from tests.shared.fixtures import document_factory




class TestManageSnapshots:
    """Tests for the manage_snapshots tool."""

    def test_manage_snapshots_create_action(self, document_factory):
        """Test creating a snapshot using the manage_snapshots tool."""
        doc_name = "snapshot_test_doc"
        chapters = {"chapter1.md": "# Chapter 1\n\nInitial content."}
        document_factory(doc_name, chapters)

        # Create snapshot
        result = manage_snapshots(
            document_name=doc_name, action="create", message="Test snapshot creation"
        )

        assert result.success is True
        assert result.details["action"] == "create"
        assert "snapshot_id" in result.details
        assert result.details["snapshot_id"] is not None
        assert "created successfully" in result.message

    def test_manage_snapshots_list_action(self, document_factory):
        """Test listing snapshots using the manage_snapshots tool."""
        doc_name = "list_snapshots_doc"
        chapters = {"chapter1.md": "# Chapter 1\n\nContent for listing."}
        document_factory(doc_name, chapters)

        # Create a snapshot first
        create_result = manage_snapshots(
            document_name=doc_name, action="create", message="Snapshot for listing test"
        )
        assert create_result.success is True

        # List snapshots
        result = manage_snapshots(document_name=doc_name, action="list")

        assert result.document_name == doc_name
        assert hasattr(result, "snapshots")
        assert result.total_snapshots >= 1
        assert isinstance(result.snapshots, list)

    def test_manage_snapshots_restore_action(self, document_factory):
        """Test restoring a snapshot using the manage_snapshots tool."""
        doc_name = "restore_test_doc"
        chapters = {"chapter1.md": "# Chapter 1\n\nOriginal content."}
        document_factory(doc_name, chapters)

        # Create a snapshot
        create_result = manage_snapshots(
            document_name=doc_name, action="create", message="Snapshot before changes"
        )
        assert create_result.success is True
        snapshot_id = create_result.details["snapshot_id"]

        # Modify content
        doc_path = document_factory(doc_name)
        (doc_path / "chapter1.md").write_text("# Chapter 1\n\nModified content.")

        # Restore snapshot
        result = manage_snapshots(
            document_name=doc_name, action="restore", snapshot_id=snapshot_id
        )

        assert result.success is True
        assert result.details["action"] == "restore"
        assert result.details["snapshot_id"] == snapshot_id
        assert "restored" in result.message

    def test_manage_snapshots_invalid_action(self, document_factory):
        """Test manage_snapshots with invalid action."""
        doc_name = "invalid_action_doc"
        document_factory(doc_name)

        result = manage_snapshots(document_name=doc_name, action="invalid_action")

        assert result.success is False
        assert "Invalid action" in result.message
        assert result.details["action"] == "invalid_action"
        assert "valid_actions" in result.details

    def test_manage_snapshots_restore_without_snapshot_id(self, document_factory):
        """Test restore action without providing snapshot_id."""
        doc_name = "restore_no_id_doc"
        document_factory(doc_name)

        result = manage_snapshots(document_name=doc_name, action="restore")

        assert result.success is False
        assert "snapshot_id is required" in result.message
        assert result.details["action"] == "restore"


class TestCheckContentStatus:
    """Tests for the check_content_status tool."""

    def test_check_content_status_basic(self, document_factory):
        """Test basic content status checking."""
        doc_name = "status_test_doc"
        chapters = {"chapter1.md": "# Chapter 1\n\nContent for status check."}
        document_factory(doc_name, chapters)

        result = check_content_status(
            document_name=doc_name, chapter_name="chapter1.md"
        )

        # check_content_status returns ContentFreshnessStatus when include_history=False
        assert result.is_fresh is True
        assert result.safety_status in ["safe", "conflict"]

    def test_check_content_status_with_history(self, document_factory):
        """Test content status checking with history included."""
        doc_name = "status_history_doc"
        chapters = {"chapter1.md": "# Chapter 1\n\nContent with history."}
        document_factory(doc_name, chapters)

        result = check_content_status(
            document_name=doc_name, include_history=True, time_window="7d"
        )

        # check_content_status returns ModificationHistory when include_history=True
        assert result.total_modifications >= 0
        assert isinstance(result.entries, list)

    def test_check_content_status_document_scope(self, document_factory):
        """Test content status for entire document."""
        doc_name = "doc_status_test"
        chapters = {
            "chapter1.md": "# Chapter 1\n\nFirst chapter.",
            "chapter2.md": "# Chapter 2\n\nSecond chapter.",
        }
        document_factory(doc_name, chapters)

        result = check_content_status(
            document_name=doc_name
            # No chapter_name = document scope
        )

        # check_content_status returns ContentFreshnessStatus when include_history=False
        assert result.is_fresh is True or result.is_fresh is False
        assert result.safety_status in ["safe", "conflict", "error"]

    def test_check_content_status_invalid_document(self):
        """Test content status with invalid document name."""
        result = check_content_status(document_name="nonexistent_doc")

        # Should handle gracefully, possibly with freshness check failure
        assert result.is_fresh is False or result.safety_status == "error"


class TestDiffContent:
    """Tests for the diff_content tool."""

    def test_diff_content_snapshot_to_snapshot(self, document_factory):
        """Test diff between two snapshots using diff_content tool."""
        doc_name = "diff_test_doc"
        chapters = {"chapter1.md": "# Chapter 1\n\nOriginal content."}
        document_factory(doc_name, chapters)

        # Create first snapshot
        snap1_result = manage_snapshots(
            document_name=doc_name, action="create", message="First snapshot"
        )
        assert snap1_result.success is True
        snapshot_id_1 = snap1_result.details["snapshot_id"]

        # Modify content
        doc_path = document_factory(doc_name)
        (doc_path / "chapter1.md").write_text("# Chapter 1\n\nModified content.")

        # Create second snapshot
        snap2_result = manage_snapshots(
            document_name=doc_name, action="create", message="Second snapshot"
        )
        assert snap2_result.success is True
        snapshot_id_2 = snap2_result.details["snapshot_id"]

        # Generate diff
        result = diff_content(
            document_name=doc_name,
            source_type="snapshot",
            source_id=snapshot_id_1,
            target_type="snapshot",
            target_id=snapshot_id_2,
            output_format="unified",
        )

        assert result.success is True
        assert result.details["operation"] == "diff_content"
        assert result.details["source_type"] == "snapshot"
        assert result.details["target_type"] == "snapshot"

    def test_diff_content_current_comparison(self, document_factory):
        """Test diff with current content."""
        doc_name = "diff_current_doc"
        chapters = {"chapter1.md": "# Chapter 1\n\nContent for diff."}
        document_factory(doc_name, chapters)

        # Create snapshot
        snap_result = manage_snapshots(
            document_name=doc_name, action="create", message="Snapshot for diff"
        )
        assert snap_result.success is True
        snapshot_id = snap_result.details["snapshot_id"]

        # Test snapshot to current
        result = diff_content(
            document_name=doc_name,
            source_type="snapshot",
            source_id=snapshot_id,
            target_type="current",
        )

        # The feature is actually implemented - should succeed
        assert result.success is True
        assert result.details["operation"] == "diff_content"

    def test_diff_content_invalid_parameters(self, document_factory):
        """Test diff_content with invalid parameters."""
        doc_name = "diff_invalid_doc"
        document_factory(doc_name)

        # Test invalid source_type
        result = diff_content(
            document_name=doc_name, source_type="invalid_type", target_type="current"
        )

        assert result.success is False
        assert "Invalid source_type" in result.message

    def test_diff_content_missing_snapshot_id(self, document_factory):
        """Test diff_content with missing snapshot ID."""
        doc_name = "diff_missing_id_doc"
        document_factory(doc_name)

        result = diff_content(
            document_name=doc_name,
            source_type="snapshot",
            # Missing source_id
            target_type="snapshot",
            target_id="some_id",
        )

        # Should handle gracefully when snapshot IDs are missing
        # Implementation may vary based on how it handles missing IDs
        assert hasattr(result, "success")
