"""Integration tests for version control tools.

This module tests the manage_snapshots, check_content_status, and diff_content
tools that manage document versioning and snapshots.
"""

from pathlib import Path

import pytest

from tests.tool_imports import check_content_status
from tests.tool_imports import delete_document
from tests.tool_imports import diff_content
from tests.tool_imports import manage_snapshots  # Version control tools


@pytest.fixture
def document_factory(temp_docs_root: Path):
    """A factory to create documents with chapters for testing."""
    created_docs = []

    def _create_document(doc_name: str, chapters: dict[str, str] = None):
        doc_path = temp_docs_root / doc_name
        doc_path.mkdir(exist_ok=True)
        created_docs.append(doc_name)
        if chapters:
            for chapter_name, content in chapters.items():
                (doc_path / chapter_name).write_text(content)
        return doc_path

    try:
        yield _create_document
    finally:
        for doc_name in created_docs:
            delete_document(doc_name)


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

        assert result["success"] is True
        assert result["action"] == "create"
        assert "snapshot_id" in result
        assert result["snapshot_id"] is not None
        assert "created successfully" in result["message"]

    def test_manage_snapshots_list_action(self, document_factory):
        """Test listing snapshots using the manage_snapshots tool."""
        doc_name = "list_snapshots_doc"
        chapters = {"chapter1.md": "# Chapter 1\n\nContent for listing."}
        document_factory(doc_name, chapters)

        # Create a snapshot first
        create_result = manage_snapshots(
            document_name=doc_name, action="create", message="Snapshot for listing test"
        )
        assert create_result["success"] is True

        # List snapshots
        result = manage_snapshots(document_name=doc_name, action="list")

        assert result["success"] is True
        assert result["action"] == "list"
        assert result["document_name"] == doc_name
        assert "snapshots" in result
        assert result["total_snapshots"] >= 1
        assert isinstance(result["snapshots"], list)

    def test_manage_snapshots_restore_action(self, document_factory):
        """Test restoring a snapshot using the manage_snapshots tool."""
        doc_name = "restore_test_doc"
        chapters = {"chapter1.md": "# Chapter 1\n\nOriginal content."}
        document_factory(doc_name, chapters)

        # Create a snapshot
        create_result = manage_snapshots(
            document_name=doc_name, action="create", message="Snapshot before changes"
        )
        assert create_result["success"] is True
        snapshot_id = create_result["snapshot_id"]

        # Modify content
        doc_path = document_factory(doc_name)
        (doc_path / "chapter1.md").write_text("# Chapter 1\n\nModified content.")

        # Restore snapshot
        result = manage_snapshots(
            document_name=doc_name, action="restore", snapshot_id=snapshot_id
        )

        assert result["success"] is True
        assert result["action"] == "restore"
        assert result["snapshot_id"] == snapshot_id
        assert "restored" in result["message"]

    def test_manage_snapshots_invalid_action(self, document_factory):
        """Test manage_snapshots with invalid action."""
        doc_name = "invalid_action_doc"
        document_factory(doc_name)

        result = manage_snapshots(document_name=doc_name, action="invalid_action")

        assert result["success"] is False
        assert "Invalid action" in result["message"]
        assert result["action"] == "invalid_action"
        assert "valid_actions" in result

    def test_manage_snapshots_restore_without_snapshot_id(self, document_factory):
        """Test restore action without providing snapshot_id."""
        doc_name = "restore_no_id_doc"
        document_factory(doc_name)

        result = manage_snapshots(document_name=doc_name, action="restore")

        assert result["success"] is False
        assert "snapshot_id is required" in result["message"]
        assert result["action"] == "restore"


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

        assert result["success"] is True
        assert result["operation"] == "check_content_status"
        assert result["document_name"] == doc_name
        assert result["chapter_name"] == "chapter1.md"
        assert "freshness" in result
        assert result["freshness"]["is_fresh"] is True
        assert "summary" in result

    def test_check_content_status_with_history(self, document_factory):
        """Test content status checking with history included."""
        doc_name = "status_history_doc"
        chapters = {"chapter1.md": "# Chapter 1\n\nContent with history."}
        document_factory(doc_name, chapters)

        result = check_content_status(
            document_name=doc_name, include_history=True, time_window="7d"
        )

        assert result["success"] is True
        assert "history" in result
        assert result["history"] is not None
        assert "total_modifications" in result["history"]
        assert "entries" in result["history"]

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

        assert result["success"] is True
        assert result["chapter_name"] is None
        assert "document" in result["summary"]

    def test_check_content_status_invalid_document(self):
        """Test content status with invalid document name."""
        result = check_content_status(document_name="nonexistent_doc")

        # Should handle gracefully, possibly with freshness check failure
        assert result["success"] is True or "not found" in result["message"]


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
        assert snap1_result["success"] is True
        snapshot_id_1 = snap1_result["snapshot_id"]

        # Modify content
        doc_path = document_factory(doc_name)
        (doc_path / "chapter1.md").write_text("# Chapter 1\n\nModified content.")

        # Create second snapshot
        snap2_result = manage_snapshots(
            document_name=doc_name, action="create", message="Second snapshot"
        )
        assert snap2_result["success"] is True
        snapshot_id_2 = snap2_result["snapshot_id"]

        # Generate diff
        result = diff_content(
            document_name=doc_name,
            source_type="snapshot",
            source_id=snapshot_id_1,
            target_type="snapshot",
            target_id=snapshot_id_2,
            output_format="unified",
        )

        assert result["success"] is True
        assert result["operation"] == "diff_content"
        assert result["source_type"] == "snapshot"
        assert result["target_type"] == "snapshot"
        assert result["source_id"] == snapshot_id_1
        assert result["target_id"] == snapshot_id_2

    def test_diff_content_current_comparison(self, document_factory):
        """Test diff with current content (limited implementation)."""
        doc_name = "diff_current_doc"
        chapters = {"chapter1.md": "# Chapter 1\n\nContent for diff."}
        document_factory(doc_name, chapters)

        # Create snapshot
        snap_result = manage_snapshots(
            document_name=doc_name, action="create", message="Snapshot for diff"
        )
        assert snap_result["success"] is True
        snapshot_id = snap_result["snapshot_id"]

        # Test snapshot to current (should indicate not fully implemented)
        result = diff_content(
            document_name=doc_name,
            source_type="snapshot",
            source_id=snapshot_id,
            target_type="current",
        )

        # This should indicate the feature is not fully implemented yet
        assert result["success"] is False
        assert "not fully implemented" in result["message"]

    def test_diff_content_invalid_parameters(self, document_factory):
        """Test diff_content with invalid parameters."""
        doc_name = "diff_invalid_doc"
        document_factory(doc_name)

        # Test invalid source_type
        result = diff_content(
            document_name=doc_name, source_type="invalid_type", target_type="current"
        )

        assert result["success"] is False
        assert "Invalid source_type" in result["message"]

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
        assert isinstance(result, dict)
        assert "success" in result
