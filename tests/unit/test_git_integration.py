"""Comprehensive test suite for Git-backed version control integration.

Tests cover:
- Git repository initialization
- Commit creation and history
- Version checkout and restoration
- Diff generation and comparison
- Snapshot migration
- Error handling and edge cases
"""

import datetime
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from story_mcp.utils.git_manager import GitCommit
from story_mcp.utils.git_manager import GitDiff
from story_mcp.utils.git_manager import GitError
from story_mcp.utils.git_manager import GitManager


class TestGitManagerInitialization:
    """Tests for Git repository initialization."""

    def test_init_creates_git_directory(self, tmp_path):
        """Test that __init__ creates .git directory."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()

        manager = GitManager(repo_path)

        assert (repo_path / ".git").exists()
        assert manager.repo_path == repo_path
        assert manager.git_dir == repo_path / ".git"

    def test_init_existing_repo(self, tmp_path):
        """Test initialization with existing Git repository."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()

        # Create repo first
        GitManager(repo_path)

        # Re-initialize should work without error
        manager = GitManager(repo_path)
        assert manager.git_dir.exists()

    def test_init_invalid_path(self):
        """Test initialization with invalid path."""
        invalid_path = Path("/nonexistent/path/that/does/not/exist")

        with pytest.raises((GitError, FileNotFoundError)):
            GitManager(invalid_path)


class TestGitCommit:
    """Tests for commit creation and formatting."""

    def test_commit_basic(self, tmp_path):
        """Test basic commit creation."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()
        (repo_path / "test.md").write_text("content")

        manager = GitManager(repo_path)
        commit = manager.commit(
            operation="edit",
            scope="chapter",
            description="updated introduction",
        )

        assert commit.hash
        assert len(commit.hash) == 40  # SHA1 hash
        assert commit.author
        assert "edit chapter:" in commit.message
        assert "updated introduction" in commit.message
        assert commit.summary == commit.message

    def test_commit_with_author(self, tmp_path):
        """Test commit with specific author."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()
        (repo_path / "test.md").write_text("content")

        manager = GitManager(repo_path)
        commit = manager.commit(
            operation="add",
            scope="paragraph",
            description="new section on performance",
            author="Alice <alice@example.com>",
        )

        assert "alice@example.com" in commit.author

    def test_commit_message_format(self, tmp_path):
        """Test commit message formatting."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()
        (repo_path / "test.md").write_text("content")

        manager = GitManager(repo_path)
        commit = manager.commit(
            operation="replace",
            scope="paragraph",
            description="refined conclusion",
        )

        # Message should match pattern: {operation} {scope}: {description}
        assert commit.message == "replace paragraph: refined conclusion"

    def test_commit_no_changes_fails(self, tmp_path):
        """Test that commit fails when no changes are staged."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()

        manager = GitManager(repo_path)

        with pytest.raises(GitError):
            manager.commit(
                operation="edit",
                scope="chapter",
                description="no actual changes",
            )

    def test_commit_to_dict(self, tmp_path):
        """Test GitCommit.to_dict() conversion."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()
        (repo_path / "test.md").write_text("content")

        manager = GitManager(repo_path)
        commit = manager.commit(
            operation="edit",
            scope="chapter",
            description="test",
        )

        commit_dict = commit.to_dict()
        assert "hash" in commit_dict
        assert "author" in commit_dict
        assert "timestamp" in commit_dict
        assert "message" in commit_dict
        assert "summary" in commit_dict


class TestGitHistory:
    """Tests for version history retrieval."""

    def test_get_version_history_empty(self, tmp_path):
        """Test history retrieval on empty repository."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()

        manager = GitManager(repo_path)
        history = manager.get_version_history(limit=10)

        assert history == []

    def test_get_version_history_multiple_commits(self, tmp_path):
        """Test history retrieval with multiple commits."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()

        manager = GitManager(repo_path)

        # Create 3 commits
        for i in range(3):
            (repo_path / f"file{i}.md").write_text(f"content {i}")
            manager.commit(
                operation="add",
                scope="chapter",
                description=f"commit {i}",
            )

        history = manager.get_version_history(limit=10)

        assert len(history) == 3
        # Should be in reverse order (newest first)
        assert "commit 2" in history[0].message
        assert "commit 0" in history[2].message

    def test_get_version_history_limit(self, tmp_path):
        """Test history limit parameter."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()

        manager = GitManager(repo_path)

        # Create 5 commits
        for i in range(5):
            (repo_path / f"file{i}.md").write_text(f"content {i}")
            manager.commit(
                operation="add",
                scope="chapter",
                description=f"commit {i}",
            )

        # Request only 3
        history = manager.get_version_history(limit=3)

        assert len(history) <= 3

    def test_get_version_history_structure(self, tmp_path):
        """Test returned commit structure."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()
        (repo_path / "test.md").write_text("content")

        manager = GitManager(repo_path)
        manager.commit(
            operation="edit",
            scope="chapter",
            description="test",
        )

        history = manager.get_version_history()

        assert len(history) == 1
        commit = history[0]
        assert isinstance(commit, GitCommit)
        assert commit.hash
        assert commit.author
        assert isinstance(commit.timestamp, datetime.datetime)
        assert commit.message
        assert commit.summary


class TestGitCheckout:
    """Tests for version checkout/restoration."""

    def test_checkout_version_valid(self, tmp_path):
        """Test checking out a valid version."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()

        manager = GitManager(repo_path)

        # Create initial commit
        (repo_path / "test.md").write_text("version 1")
        commit1 = manager.commit(
            operation="add",
            scope="chapter",
            description="version 1",
        )

        # Modify and create new commit
        (repo_path / "test.md").write_text("version 2")
        manager.commit(
            operation="edit",
            scope="chapter",
            description="version 2",
        )

        # Checkout first version
        manager.checkout_version(commit1.hash)

        # Verify content
        content = (repo_path / "test.md").read_text()
        assert "version 1" in content

    def test_checkout_version_by_short_hash(self, tmp_path):
        """Test checking out by short commit hash."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()

        manager = GitManager(repo_path)

        (repo_path / "test.md").write_text("content")
        commit = manager.commit(
            operation="add",
            scope="chapter",
            description="test",
        )

        # Use short hash (7 chars)
        short_hash = commit.hash[:7]
        manager.checkout_version(short_hash)

        # Should not raise error

    def test_checkout_invalid_version_fails(self, tmp_path):
        """Test checkout with invalid hash fails."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()
        (repo_path / "test.md").write_text("content")

        manager = GitManager(repo_path)

        with pytest.raises(GitError):
            manager.checkout_version("invalid_hash_0000000")


class TestGitDiff:
    """Tests for diff generation and comparison."""

    def test_diff_simple(self, tmp_path):
        """Test basic diff between commits."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()

        manager = GitManager(repo_path)

        # Create first version
        (repo_path / "test.md").write_text("line 1\nline 2\n")
        commit1 = manager.commit(
            operation="add",
            scope="chapter",
            description="initial",
        )

        # Create second version
        (repo_path / "test.md").write_text("line 1\nmodified line 2\n")
        manager.commit(
            operation="edit",
            scope="chapter",
            description="modified",
        )

        # Get diff
        diff = manager.compare_versions(commit1.hash)

        assert isinstance(diff, GitDiff)
        assert diff.source_hash == commit1.hash
        assert "modified" in diff.diff_text or len(diff.diff_text) > 0

    def test_diff_stats(self, tmp_path):
        """Test diff statistics parsing."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()

        manager = GitManager(repo_path)

        (repo_path / "test.md").write_text("a\nb\nc\n")
        commit1 = manager.commit(
            operation="add",
            scope="chapter",
            description="initial",
        )

        (repo_path / "test.md").write_text("a\nmodified\nc\nnew line\n")
        manager.commit(
            operation="edit",
            scope="chapter",
            description="modified",
        )

        diff = manager.compare_versions(commit1.hash)

        assert diff.stats["insertions"] >= 0
        assert diff.stats["deletions"] >= 0
        assert diff.stats["files_changed"] >= 0

    def test_diff_no_changes(self, tmp_path):
        """Test diff when comparing identical versions."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()

        manager = GitManager(repo_path)

        (repo_path / "test.md").write_text("content")
        commit1 = manager.commit(
            operation="add",
            scope="chapter",
            description="initial",
        )
        commit2_hash = manager.get_current_hash()

        # Compare same version
        diff = manager.compare_versions(commit1.hash, commit2_hash)

        assert len(diff.diff_text) == 0 or diff.stats["insertions"] == 0

    def test_diff_to_dict(self, tmp_path):
        """Test GitDiff.to_dict() conversion."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()

        manager = GitManager(repo_path)

        (repo_path / "test.md").write_text("content")
        commit1 = manager.commit(
            operation="add",
            scope="chapter",
            description="test",
        )

        diff = manager.compare_versions(commit1.hash)
        diff_dict = diff.to_dict()

        assert "source_hash" in diff_dict
        assert "target_hash" in diff_dict
        assert "diff_text" in diff_dict
        assert "stats" in diff_dict


class TestGitStatus:
    """Tests for repository status."""

    def test_get_status_clean(self, tmp_path):
        """Test status of clean repository."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()
        (repo_path / "test.md").write_text("content")

        manager = GitManager(repo_path)
        manager.commit(
            operation="add",
            scope="chapter",
            description="initial",
        )

        status = manager.get_status()

        assert status["staged"] == []
        assert status["unstaged"] == []
        assert status["untracked"] == []
        assert not status["is_dirty"]

    def test_get_status_dirty(self, tmp_path):
        """Test status with uncommitted changes."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()
        (repo_path / "test.md").write_text("content")

        manager = GitManager(repo_path)
        manager.commit(
            operation="add",
            scope="chapter",
            description="initial",
        )

        # Make changes
        (repo_path / "test.md").write_text("modified")

        status = manager.get_status()

        assert status["is_dirty"]
        assert len(status["unstaged"]) > 0 or len(status["staged"]) > 0

    def test_get_status_untracked(self, tmp_path):
        """Test status with untracked files."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()

        manager = GitManager(repo_path)

        # Create untracked file
        (repo_path / "untracked.md").write_text("content")

        status = manager.get_status()

        assert status["is_dirty"]
        assert "untracked.md" in status["untracked"]


class TestGitTags:
    """Tests for Git tags."""

    def test_create_tag_lightweight(self, tmp_path):
        """Test creating a lightweight tag."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()
        (repo_path / "test.md").write_text("content")

        manager = GitManager(repo_path)
        manager.commit(
            operation="add",
            scope="chapter",
            description="initial",
        )

        tag_name = manager.create_tag("v1.0")

        assert tag_name == "v1.0"

    def test_create_tag_annotated(self, tmp_path):
        """Test creating an annotated tag."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()
        (repo_path / "test.md").write_text("content")

        manager = GitManager(repo_path)
        manager.commit(
            operation="add",
            scope="chapter",
            description="initial",
        )

        tag_name = manager.create_tag("v1.0", message="Version 1.0 release")

        assert tag_name == "v1.0"

    def test_create_tag_duplicate_fails(self, tmp_path):
        """Test that duplicate tag fails."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()
        (repo_path / "test.md").write_text("content")

        manager = GitManager(repo_path)
        manager.commit(
            operation="add",
            scope="chapter",
            description="initial",
        )

        manager.create_tag("v1.0")

        with pytest.raises(GitError):
            manager.create_tag("v1.0")


class TestGitCurrentHash:
    """Tests for retrieving current commit hash."""

    def test_get_current_hash(self, tmp_path):
        """Test getting current HEAD hash."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()
        (repo_path / "test.md").write_text("content")

        manager = GitManager(repo_path)
        manager.commit(
            operation="add",
            scope="chapter",
            description="initial",
        )

        hash_val = manager.get_current_hash()

        assert hash_val
        assert len(hash_val) == 40  # SHA1

    def test_get_current_hash_matches_commit(self, tmp_path):
        """Test that current hash matches last commit."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()
        (repo_path / "test.md").write_text("content")

        manager = GitManager(repo_path)
        commit = manager.commit(
            operation="add",
            scope="chapter",
            description="initial",
        )

        current = manager.get_current_hash()

        assert current == commit.hash


class TestGitMigration:
    """Tests for snapshot migration."""

    def test_migrate_snapshots_no_dir(self, tmp_path):
        """Test migration when snapshots directory doesn't exist."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()

        manager = GitManager(repo_path)
        snapshots_dir = tmp_path / "nonexistent"

        report = manager.migrate_snapshots(snapshots_dir)

        assert report["migrated"] == 0
        assert report["failed"] == 0

    def test_migrate_snapshots_empty_dir(self, tmp_path):
        """Test migration with empty snapshots directory."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()
        snapshots_dir = repo_path / ".snapshots"
        snapshots_dir.mkdir()

        manager = GitManager(repo_path)
        report = manager.migrate_snapshots(snapshots_dir)

        assert report["migrated"] == 0
        assert report["failed"] == 0


class TestGitErrorHandling:
    """Tests for error handling."""

    def test_git_error_message(self):
        """Test GitError exception."""
        error = GitError("Test error")
        assert str(error) == "Test error"

    def test_commit_with_invalid_git_fails(self, tmp_path):
        """Test commit fails gracefully on Git errors."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()

        manager = GitManager(repo_path)

        # Create a file and commit
        (repo_path / "test.md").write_text("content")
        manager.commit(
            operation="add",
            scope="chapter",
            description="initial",
        )

        # Mock git command to fail
        with patch.object(manager, "_run_git", side_effect=subprocess.CalledProcessError(1, "git")):
            with pytest.raises(GitError):
                (repo_path / "test2.md").write_text("more content")
                manager.commit(
                    operation="add",
                    scope="chapter",
                    description="should fail",
                )


class TestGitIntegration:
    """Integration tests for Git operations."""

    def test_full_workflow(self, tmp_path):
        """Test complete Git workflow."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()

        manager = GitManager(repo_path)

        # Create document
        (repo_path / "chapter1.md").write_text("Chapter 1 content")
        commit1 = manager.commit(
            operation="add",
            scope="chapter",
            description="created chapter 1",
        )

        # Add more content
        (repo_path / "chapter2.md").write_text("Chapter 2 content")
        commit2 = manager.commit(
            operation="add",
            scope="chapter",
            description="created chapter 2",
        )

        # Edit chapter 1
        (repo_path / "chapter1.md").write_text("Chapter 1 content - revised")
        commit3 = manager.commit(
            operation="edit",
            scope="chapter",
            description="revised chapter 1",
        )

        # Get history
        history = manager.get_version_history(limit=10)
        assert len(history) == 3

        # Compare versions
        diff = manager.compare_versions(commit1.hash, commit3.hash)
        assert diff.source_hash == commit1.hash

        # Create tag
        tag = manager.create_tag("draft-1", message="First draft")
        assert tag == "draft-1"

        # Checkout earlier version
        manager.checkout_version(commit1.hash)
        content = (repo_path / "chapter1.md").read_text()
        assert "Chapter 1 content" in content
        assert "revised" not in content

    def test_concurrent_edits(self, tmp_path):
        """Test multiple rapid commits."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()

        manager = GitManager(repo_path)

        # Make 5 edits
        for i in range(5):
            (repo_path / f"edit{i}.md").write_text(f"Edit {i}")
            manager.commit(
                operation="edit",
                scope="paragraph",
                description=f"edit {i}",
            )

        history = manager.get_version_history(limit=10)
        assert len(history) == 5

    def test_diff_with_multiple_files(self, tmp_path):
        """Test diff across multiple files."""
        repo_path = tmp_path / "test_doc"
        repo_path.mkdir()

        manager = GitManager(repo_path)

        # Create initial state
        (repo_path / "file1.md").write_text("file1 v1")
        (repo_path / "file2.md").write_text("file2 v1")
        commit1 = manager.commit(
            operation="add",
            scope="chapter",
            description="initial",
        )

        # Modify both files
        (repo_path / "file1.md").write_text("file1 v2")
        (repo_path / "file2.md").write_text("file2 v2")
        manager.commit(
            operation="edit",
            scope="chapter",
            description="modified both",
        )

        diff = manager.compare_versions(commit1.hash)
        assert diff.stats["files_changed"] >= 1
