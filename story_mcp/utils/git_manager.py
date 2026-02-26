"""Git-backed version control manager for document documents.

This module provides Git operations for document version history management,
replacing the snapshot system with proper version control.

Features:
- Automatic Git initialization per document directory
- Commit-based version history with user attribution
- Branch and tag support for complex workflows
- Full diff and version comparison capabilities
- Automatic snapshot migration support
"""

from __future__ import annotations

import datetime
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..logger_config import ErrorCategory
from ..logger_config import log_structured_error

__all__ = [
    "GitManager",
    "GitCommit",
    "GitDiff",
    "GitError",
]


class GitError(Exception):
    """Exception for Git operation failures."""

    pass


class GitCommit:
    """Represents a Git commit in the document history."""

    def __init__(
        self,
        hash: str,
        author: str,
        timestamp: datetime.datetime,
        message: str,
        summary: str | None = None,
    ):
        """Initialize a Git commit.

        Args:
            hash: Commit SHA hash
            author: Author name and email
            timestamp: Commit timestamp
            message: Full commit message
            summary: Optional short summary (extracted from message)
        """
        self.hash = hash
        self.author = author
        self.timestamp = timestamp
        self.message = message
        self.summary = summary or message.split("\n")[0]

    def to_dict(self) -> dict[str, Any]:
        """Convert commit to dictionary representation."""
        return {
            "hash": self.hash,
            "author": self.author,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "summary": self.summary,
        }


class GitDiff:
    """Represents a diff between two Git commits or working tree."""

    def __init__(
        self,
        source_hash: str,
        target_hash: str | None,
        diff_text: str,
        stats: dict[str, int] | None = None,
    ):
        """Initialize a Git diff.

        Args:
            source_hash: Source commit hash or "working"
            target_hash: Target commit hash or "HEAD"
            diff_text: Unified diff format text
            stats: Dictionary with "insertions", "deletions", "files_changed"
        """
        self.source_hash = source_hash
        self.target_hash = target_hash or "HEAD"
        self.diff_text = diff_text
        self.stats = stats or {"insertions": 0, "deletions": 0, "files_changed": 0}

    def to_dict(self) -> dict[str, Any]:
        """Convert diff to dictionary representation."""
        return {
            "source_hash": self.source_hash,
            "target_hash": self.target_hash,
            "diff_text": self.diff_text,
            "stats": self.stats,
        }


class GitManager:
    """Manager for Git-based version control of documents."""

    # Commit message pattern: {operation} {scope}: {description}
    COMMIT_PATTERN = re.compile(r"^(\w+)\s+(\w+):\s(.+)$")

    def __init__(self, repo_path: Path):
        """Initialize Git manager for a document repository.

        Args:
            repo_path: Path to the document directory (becomes Git root)

        Raises:
            GitError: If initialization fails
        """
        self.repo_path = Path(repo_path)
        self.git_dir = self.repo_path / ".git"
        self._ensure_repo_initialized()

    def _ensure_repo_initialized(self) -> None:
        """Ensure the document directory is a Git repository.

        Creates a new repository if one doesn't exist.

        Raises:
            GitError: If initialization fails
        """
        try:
            if not self.git_dir.exists():
                self._run_git(["init"], cwd=self.repo_path)
                # Configure minimal user for commits
                self._run_git(["config", "user.email", "mcp@document.local"], cwd=self.repo_path)
                self._run_git(["config", "user.name", "Document MCP"], cwd=self.repo_path)
        except subprocess.CalledProcessError as e:
            msg = f"Failed to initialize Git repository at {self.repo_path}: {e}"
            log_structured_error(ErrorCategory.ERROR, msg, {"repo_path": str(self.repo_path)})
            raise GitError(msg) from e

    def _run_git(self, args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
        """Execute a Git command.

        Args:
            args: Git command arguments (without 'git' prefix)
            cwd: Working directory for command

        Returns:
            CompletedProcess with stdout/stderr

        Raises:
            subprocess.CalledProcessError: If git command fails
        """
        cmd = ["git"] + args
        result = subprocess.run(
            cmd,
            cwd=cwd or self.repo_path,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                output=result.stdout,
                stderr=result.stderr,
            )

        return result

    def commit(
        self,
        operation: str,
        scope: str,
        description: str,
        author: str | None = None,
        commit_all: bool = True,
    ) -> GitCommit:
        """Create a commit with formatted message.

        Commit format: {operation} {scope}: {description}

        Examples:
            - "edit chapter: updated introduction narrative"
            - "add paragraph: new section on performance"
            - "replace paragraph: refined conclusion"

        Args:
            operation: Operation type (edit, add, replace, delete, move, etc.)
            scope: Scope (chapter, paragraph, document, etc.)
            description: Human-readable description
            author: Author name (defaults to current user)
            commit_all: Whether to stage all changes before committing

        Returns:
            GitCommit object with commit details

        Raises:
            GitError: If commit fails
        """
        try:
            # Get author email
            if not author:
                from ..utils.file_operations import get_current_user

                author = get_current_user()
                # Ensure author is in proper format
                if "@" not in author:
                    author = f"{author} <{author}@local>"

            # Stage changes if requested
            if commit_all:
                self._run_git(["add", "-A"])

            # Format commit message
            message = f"{operation} {scope}: {description}"

            # Check if there are staged changes
            try:
                result = subprocess.run(
                    ["git", "diff", "--cached", "--quiet"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    # No changes to commit
                    raise GitError("No changes staged for commit")
            except Exception as e:
                if "No changes" not in str(e):
                    pass  # Continue even if diff check fails

            # Create commit with properly formatted author
            commit_args = ["commit", "-m", message]
            if author and "@" in author:
                commit_args.append(f"--author={author}")

            self._run_git(commit_args)

            # Get commit hash and details
            result = self._run_git(["rev-parse", "HEAD"])
            commit_hash = result.stdout.strip()

            # Get commit timestamp
            result = self._run_git(["log", "-1", "--format=%aI", "HEAD"])
            timestamp_str = result.stdout.strip()
            timestamp = datetime.datetime.fromisoformat(timestamp_str)

            # Get author from commit
            result = self._run_git(["log", "-1", "--format=%an <%ae>", "HEAD"])
            commit_author = result.stdout.strip()

            return GitCommit(
                hash=commit_hash,
                author=commit_author,
                timestamp=timestamp,
                message=message,
            )

        except subprocess.CalledProcessError as e:
            msg = f"Failed to create commit: {e.stderr}"
            log_structured_error(
                ErrorCategory.ERROR,
                msg,
                {"operation": operation, "scope": scope, "error": e.stderr},
            )
            raise GitError(msg) from e

    def get_version_history(self, limit: int = 10) -> list[GitCommit]:
        """Get commit history for the document.

        Args:
            limit: Maximum number of commits to return

        Returns:
            List of GitCommit objects, newest first

        Raises:
            GitError: If history retrieval fails
        """
        try:
            # Get commit log
            result = subprocess.run(
                [
                    "git",
                    "log",
                    f"--max-count={limit}",
                    "--format=%H%n%an <%ae>%n%aI%n%s%n%b%n---END---",
                ],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False,
            )

            # Handle case where there are no commits yet
            if result.returncode != 0:
                if "no commits yet" in result.stderr or "尚無任何提交" in result.stderr:
                    return []
                raise subprocess.CalledProcessError(
                    result.returncode,
                    ["git", "log"],
                    output=result.stdout,
                    stderr=result.stderr,
                )

            commits = []
            lines = result.stdout.strip().split("\n")
            i = 0

            while i < len(lines):
                if not lines[i]:
                    i += 1
                    continue

                # Parse commit
                commit_hash = lines[i]
                author = lines[i + 1] if i + 1 < len(lines) else "Unknown"
                timestamp_str = lines[i + 2] if i + 2 < len(lines) else datetime.datetime.now().isoformat()
                message = lines[i + 3] if i + 3 < len(lines) else ""

                try:
                    timestamp = datetime.datetime.fromisoformat(timestamp_str)
                except (ValueError, TypeError):
                    timestamp = datetime.datetime.now()

                commits.append(
                    GitCommit(
                        hash=commit_hash,
                        author=author,
                        timestamp=timestamp,
                        message=message,
                    )
                )

                # Skip to next commit marker
                while i < len(lines) and lines[i] != "---END---":
                    i += 1
                i += 1

            return commits

        except subprocess.CalledProcessError as e:
            msg = f"Failed to retrieve version history: {e.stderr}"
            log_structured_error(
                ErrorCategory.ERROR,
                msg,
                {"repo_path": str(self.repo_path), "error": e.stderr},
            )
            raise GitError(msg) from e

    def checkout_version(self, version_hash: str) -> None:
        """Restore document to a specific commit.

        Args:
            version_hash: Commit SHA hash or tag name

        Raises:
            GitError: If checkout fails
        """
        try:
            self._run_git(["checkout", version_hash, "--"])
        except subprocess.CalledProcessError as e:
            msg = f"Failed to checkout version {version_hash}: {e.stderr}"
            log_structured_error(
                ErrorCategory.ERROR,
                msg,
                {"version_hash": version_hash, "error": e.stderr},
            )
            raise GitError(msg) from e

    def compare_versions(
        self,
        version1: str,
        version2: str | None = None,
        stat_only: bool = False,
    ) -> GitDiff:
        """Compare two versions (commits).

        Args:
            version1: Source commit hash or tag
            version2: Target commit hash or tag (defaults to HEAD)
            stat_only: Return only statistics, not full diff

        Returns:
            GitDiff object with comparison results

        Raises:
            GitError: If comparison fails
        """
        try:
            target = version2 or "HEAD"

            # Get diff
            args = ["diff", version1, target]
            if stat_only:
                args.insert(2, "--stat")

            result = self._run_git(args)
            diff_text = result.stdout

            # Get statistics
            stats = self._parse_diff_stats(diff_text)

            return GitDiff(
                source_hash=version1,
                target_hash=target,
                diff_text=diff_text,
                stats=stats,
            )

        except subprocess.CalledProcessError as e:
            msg = f"Failed to compare versions {version1} and {target}: {e.stderr}"
            log_structured_error(
                ErrorCategory.ERROR,
                msg,
                {"version1": version1, "version2": version2, "error": e.stderr},
            )
            raise GitError(msg) from e

    def _parse_diff_stats(self, diff_text: str) -> dict[str, int]:
        """Parse diff statistics from git diff output.

        Args:
            diff_text: Raw diff text

        Returns:
            Dictionary with insertion/deletion/files changed counts
        """
        stats = {"insertions": 0, "deletions": 0, "files_changed": 0}

        for line in diff_text.split("\n"):
            # Count insertions (lines starting with +)
            if line.startswith("+") and not line.startswith("+++"):
                stats["insertions"] += 1
            # Count deletions (lines starting with -)
            elif line.startswith("-") and not line.startswith("---"):
                stats["deletions"] += 1
            # Count files changed (lines with "diff --git")
            elif line.startswith("diff --git"):
                stats["files_changed"] += 1

        return stats

    def create_tag(self, tag_name: str, message: str | None = None) -> str:
        """Create a tag for a version.

        Args:
            tag_name: Name for the tag
            message: Optional annotated tag message

        Returns:
            Tag name (same as input)

        Raises:
            GitError: If tag creation fails
        """
        try:
            args = ["tag", tag_name]
            if message:
                args.extend(["-a", "-m", message])

            self._run_git(args)
            return tag_name

        except subprocess.CalledProcessError as e:
            msg = f"Failed to create tag {tag_name}: {e.stderr}"
            log_structured_error(
                ErrorCategory.ERROR,
                msg,
                {"tag_name": tag_name, "error": e.stderr},
            )
            raise GitError(msg) from e

    def get_current_hash(self) -> str:
        """Get the current HEAD commit hash.

        Returns:
            Current commit SHA hash

        Raises:
            GitError: If retrieval fails
        """
        try:
            result = self._run_git(["rev-parse", "HEAD"])
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            msg = f"Failed to get current commit hash: {e.stderr}"
            log_structured_error(ErrorCategory.ERROR, msg, {"error": e.stderr})
            raise GitError(msg) from e

    def get_status(self) -> dict[str, Any]:
        """Get repository status.

        Returns:
            Dictionary with staged, unstaged, and untracked files

        Raises:
            GitError: If status retrieval fails
        """
        try:
            result = self._run_git(["status", "--porcelain"])
            status_lines = result.stdout.strip().split("\n") if result.stdout.strip() else []

            staged = []
            unstaged = []
            untracked = []

            for line in status_lines:
                if not line:
                    continue

                status_code = line[:2]
                filepath = line[3:]

                if status_code[0] != " ":
                    staged.append(filepath)
                if status_code[1] != " ":
                    unstaged.append(filepath)
                if status_code == "??":
                    untracked.append(filepath)

            return {
                "staged": staged,
                "unstaged": unstaged,
                "untracked": untracked,
                "is_dirty": bool(staged or unstaged or untracked),
            }

        except subprocess.CalledProcessError as e:
            msg = f"Failed to get repository status: {e.stderr}"
            log_structured_error(ErrorCategory.ERROR, msg, {"error": e.stderr})
            raise GitError(msg) from e

    def migrate_snapshots(self, snapshots_dir: Path) -> dict[str, Any]:
        """Migrate snapshots to Git commits.

        Args:
            snapshots_dir: Path to .snapshots directory

        Returns:
            Migration report with success count and any errors

        Raises:
            GitError: If migration fails
        """
        try:
            report = {"migrated": 0, "failed": 0, "errors": []}

            if not snapshots_dir.exists():
                return report

            # Get all snapshot files
            for snapshot_file in sorted(snapshots_dir.glob("*.snapshot")):
                try:
                    # Read snapshot metadata
                    content = snapshot_file.read_text()
                    lines = content.strip().split("\n")

                    timestamp_str = None
                    message = "Manual snapshot"
                    user = "unknown"

                    for line in lines:
                        if line.startswith("Snapshot created at "):
                            timestamp_str = line.split("Snapshot created at ")[1]
                        elif line.startswith("Message: "):
                            message = line.split("Message: ")[1]
                        elif line.startswith("User: "):
                            user = line.split("User: ")[1]

                    # Create commit from snapshot
                    # Note: This is a simplified migration - actual snapshot content
                    # would need to be restored first
                    try:
                        self.commit(
                            operation="snapshot",
                            scope="migrate",
                            description=message,
                            author=user,
                            commit_all=False,
                        )
                        report["migrated"] += 1
                    except GitError:
                        report["failed"] += 1

                except Exception as e:
                    report["failed"] += 1
                    report["errors"].append(str(e))

            return report

        except Exception as e:
            msg = f"Failed to migrate snapshots: {e}"
            log_structured_error(ErrorCategory.ERROR, msg, {"error": str(e)})
            raise GitError(msg) from e
