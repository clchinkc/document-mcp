# Git-Backed Version History: Implementation Guide

## Quick Start

This guide provides step-by-step implementation instructions for converting Document MCP from snapshot-based to Git-backed version history.

---

## Part 1: Core Module Structure

### 1.1 Create Git Backend Module

Create `/Users/clchinkc/Documents/GitHub/document-mcp/document_mcp/utils/git_backend.py`:

```python
"""Git backend for version history management.

This module provides low-level Git operations with proper error handling,
timeouts, and comprehensive logging.
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from ..logger_config import ErrorCategory, log_structured_error


class GitBackendError(Exception):
    """Base exception for Git backend operations."""
    pass


class GitNotAvailableError(GitBackendError):
    """Git executable not found."""
    pass


class GitOperationError(GitBackendError):
    """Git operation failed."""
    pass


def is_git_available() -> bool:
    """Check if Git is installed and available."""
    try:
        subprocess.run(
            ["git", "--version"],
            capture_output=True,
            check=True,
            timeout=5
        )
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
        return False


def git_init(repo_path: Path) -> bool:
    """Initialize a new Git repository.

    Args:
        repo_path: Path to repository directory

    Returns:
        True if successful, False otherwise
    """
    if not is_git_available():
        raise GitNotAvailableError("Git is not installed or not in PATH")

    if (repo_path / ".git").exists():
        return True  # Already initialized

    try:
        subprocess.run(
            ["git", "init"],
            cwd=repo_path,
            capture_output=True,
            check=True,
            timeout=10
        )
        return True
    except subprocess.CalledProcessError as e:
        log_structured_error(
            ErrorCategory.ERROR,
            f"Failed to initialize Git repository",
            {
                "repo_path": str(repo_path),
                "stderr": e.stderr.decode() if e.stderr else ""
            }
        )
        return False


def git_config(repo_path: Path, user_name: str, user_email: str) -> bool:
    """Configure Git user for a repository.

    Args:
        repo_path: Path to repository
        user_name: Git user name
        user_email: Git user email

    Returns:
        True if successful, False otherwise
    """
    try:
        subprocess.run(
            ["git", "config", "user.name", user_name],
            cwd=repo_path,
            capture_output=True,
            check=True,
            timeout=5
        )
        subprocess.run(
            ["git", "config", "user.email", user_email],
            cwd=repo_path,
            capture_output=True,
            check=True,
            timeout=5
        )
        return True
    except subprocess.CalledProcessError as e:
        log_structured_error(
            ErrorCategory.ERROR,
            f"Failed to configure Git",
            {
                "repo_path": str(repo_path),
                "stderr": e.stderr.decode() if e.stderr else ""
            }
        )
        return False


def git_commit(
    repo_path: Path,
    message: str,
    user_name: str | None = None,
    user_email: str | None = None,
    allow_empty: bool = False
) -> tuple[bool, str | None]:
    """Create a Git commit.

    Args:
        repo_path: Path to repository
        message: Commit message
        user_name: Override commit user name
        user_email: Override commit user email
        allow_empty: Allow empty commits (for restoration markers)

    Returns:
        Tuple of (success, commit_hash or error_message)
    """
    try:
        cmd = ["git", "commit", "-m", message]
        if allow_empty:
            cmd.insert(2, "--allow-empty")

        env = os.environ.copy()
        if user_name and user_email:
            env["GIT_AUTHOR_NAME"] = user_name
            env["GIT_AUTHOR_EMAIL"] = user_email
            env["GIT_COMMITTER_NAME"] = user_name
            env["GIT_COMMITTER_EMAIL"] = user_email

        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
            env=env
        )

        # Extract commit hash from output
        # Format: "[branch abc123d] commit message"
        if result.stderr:
            for line in result.stderr.split("\n"):
                if "[" in line and "]" in line:
                    parts = line.split("[")[1].split("]")[0].split()
                    if len(parts) >= 2:
                        return True, parts[1]

        return True, None

    except subprocess.CalledProcessError as e:
        return False, e.stderr.decode() if e.stderr else "Unknown error"
    except subprocess.TimeoutExpired:
        return False, "Git commit timeout (>30s)"


def git_add(repo_path: Path, pattern: str = "-A") -> bool:
    """Stage changes for commit.

    Args:
        repo_path: Path to repository
        pattern: Git add pattern (default: "-A" for all)

    Returns:
        True if successful, False otherwise
    """
    try:
        subprocess.run(
            ["git", "add", pattern],
            cwd=repo_path,
            capture_output=True,
            check=True,
            timeout=30
        )
        return True
    except subprocess.CalledProcessError as e:
        log_structured_error(
            ErrorCategory.ERROR,
            f"Failed to stage changes",
            {
                "repo_path": str(repo_path),
                "pattern": pattern,
                "stderr": e.stderr.decode() if e.stderr else ""
            }
        )
        return False


def has_staged_changes(repo_path: Path) -> bool:
    """Check if repository has staged changes.

    Args:
        repo_path: Path to repository

    Returns:
        True if there are staged changes, False otherwise
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=repo_path,
            capture_output=True
        )
        # Return code 0 = no changes, 1 = has changes
        return result.returncode == 1
    except Exception:
        return False


def git_log(
    repo_path: Path,
    format_string: str = "%h %s",
    max_count: int | None = None
) -> list[str]:
    """Get commit history.

    Args:
        repo_path: Path to repository
        format_string: Git log format (default: short hash + message)
        max_count: Limit number of commits (None = all)

    Returns:
        List of formatted commit strings
    """
    try:
        cmd = ["git", "log", f"--format={format_string}"]
        if max_count:
            cmd.append(f"-{max_count}")

        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )

        return result.stdout.strip().split("\n") if result.stdout.strip() else []

    except Exception as e:
        log_structured_error(
            ErrorCategory.WARNING,
            f"Failed to retrieve Git log",
            {
                "repo_path": str(repo_path),
                "error": str(e)
            }
        )
        return []


def git_diff(
    repo_path: Path,
    revision: str | None = None,
    context_lines: int = 3
) -> str:
    """Get diff output.

    Args:
        repo_path: Path to repository
        revision: Git revision to show (None = unstaged changes)
        context_lines: Lines of context in diff

    Returns:
        Diff output as string
    """
    try:
        if revision:
            cmd = ["git", "show", revision, f"-U{context_lines}"]
        else:
            cmd = ["git", "diff", f"-U{context_lines}"]

        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )

        return result.stdout

    except subprocess.CalledProcessError:
        return ""
    except Exception as e:
        log_structured_error(
            ErrorCategory.WARNING,
            f"Failed to generate diff",
            {"repo_path": str(repo_path), "error": str(e)}
        )
        return ""


def git_reset(repo_path: Path, revision: str) -> bool:
    """Reset repository to specific revision.

    Args:
        repo_path: Path to repository
        revision: Git revision to reset to

    Returns:
        True if successful, False otherwise
    """
    try:
        subprocess.run(
            ["git", "reset", "--hard", revision],
            cwd=repo_path,
            capture_output=True,
            check=True,
            timeout=30
        )
        return True
    except subprocess.CalledProcessError as e:
        log_structured_error(
            ErrorCategory.ERROR,
            f"Failed to reset repository",
            {
                "repo_path": str(repo_path),
                "revision": revision,
                "stderr": e.stderr.decode() if e.stderr else ""
            }
        )
        return False


def git_status(repo_path: Path) -> dict[str, Any]:
    """Get repository status.

    Args:
        repo_path: Path to repository

    Returns:
        Dict with:
        - has_changes: bool
        - modified_files: list[str]
        - staged_files: list[str]
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )

        modified = []
        staged = []

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            status = line[:2]
            filename = line[3:]

            if status[0] in ["M", "A", "D"]:
                staged.append(filename)
            if status[1] in ["M", "D"]:
                modified.append(filename)

        return {
            "has_changes": bool(modified) or bool(staged),
            "modified_files": list(set(modified)),
            "staged_files": list(set(staged))
        }

    except Exception as e:
        log_structured_error(
            ErrorCategory.WARNING,
            f"Failed to get Git status",
            {"repo_path": str(repo_path), "error": str(e)}
        )
        return {
            "has_changes": False,
            "modified_files": [],
            "staged_files": []
        }


def git_gc(repo_path: Path, aggressive: bool = False) -> bool:
    """Run garbage collection on repository.

    Args:
        repo_path: Path to repository
        aggressive: Use aggressive optimization

    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = ["git", "gc"]
        if aggressive:
            cmd.append("--aggressive")

        subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            check=True,
            timeout=120
        )
        return True
    except Exception as e:
        log_structured_error(
            ErrorCategory.WARNING,
            f"Failed to run git gc",
            {"repo_path": str(repo_path), "error": str(e)}
        )
        return False


def get_repo_size(repo_path: Path) -> int:
    """Get total size of .git directory in bytes.

    Args:
        repo_path: Path to repository

    Returns:
        Size in bytes
    """
    git_dir = repo_path / ".git"
    if not git_dir.exists():
        return 0

    total_size = 0
    for dirpath, dirnames, filenames in os.walk(git_dir):
        for filename in filenames:
            filepath = Path(dirpath) / filename
            try:
                total_size += filepath.stat().st_size
            except OSError:
                pass

    return total_size
```

### 1.2 Create Git History Service

Create `/Users/clchinkc/Documents/GitHub/document-mcp/document_mcp/services/git_history_service.py`:

```python
"""High-level Git history service for Document MCP.

Builds on git_backend.py to provide domain-specific Git operations.
"""

import datetime
import json
from pathlib import Path
from typing import Any

from ..helpers import _get_document_path
from ..logger_config import ErrorCategory, log_structured_error
from ..models import OperationStatus
from ..utils.file_operations import get_current_user
from ..utils.git_backend import (
    git_add,
    git_commit,
    git_config,
    git_diff,
    git_init,
    git_log,
    git_reset,
    git_status,
    has_staged_changes,
    is_git_available,
)


class GitHistoryService:
    """Service for managing document version history with Git."""

    @staticmethod
    def ensure_git_initialized(document_name: str) -> bool:
        """Ensure document has Git repository initialized.

        Args:
            document_name: Name of document

        Returns:
            True if Git is available and initialized, False otherwise
        """
        if not is_git_available():
            log_structured_error(
                ErrorCategory.ERROR,
                "Git is not available on this system",
                {"operation": "ensure_git_initialized"}
            )
            return False

        doc_path = _get_document_path(document_name)
        git_dir = doc_path / ".git"

        # Already initialized
        if git_dir.exists():
            return True

        # Initialize new repository
        if not git_init(doc_path):
            return False

        # Configure user
        user = get_current_user() or "document-mcp"
        user_email = f"{user}@document-mcp.local"

        if not git_config(doc_path, user, user_email):
            return False

        # Create initial .gitignore
        GitHistoryService._create_gitignore(doc_path)

        # Create initial commit if there's content
        git_add(doc_path, "-A")
        if has_staged_changes(doc_path):
            git_commit(
                doc_path,
                "Initial document structure",
                user,
                user_email
            )

        return True

    @staticmethod
    def _create_gitignore(repo_path: Path) -> None:
        """Create .gitignore file if it doesn't exist."""
        gitignore_path = repo_path / ".gitignore"
        if gitignore_path.exists():
            return

        gitignore_content = """.embeddings/
summaries/
__pycache__/
*.pyc
*.pyo
.DS_Store
Thumbs.db
*~
*.swp
.vscode/
.idea/
.env
.env.local
"""
        try:
            gitignore_path.write_text(gitignore_content, encoding="utf-8")
        except Exception as e:
            log_structured_error(
                ErrorCategory.WARNING,
                f"Failed to create .gitignore",
                {"repo_path": str(repo_path), "error": str(e)}
            )

    @staticmethod
    def record_change(
        document_name: str,
        operation: str,
        scope: str,
        chapter_name: str | None = None,
        paragraph_index: int | None = None
    ) -> bool:
        """Record a document change as a Git commit.

        Args:
            document_name: Name of document
            operation: Operation type (create, edit, delete, replace, restore)
            scope: Scope of change (document, chapter, paragraph)
            chapter_name: Name of chapter (for chapter/paragraph scope)
            paragraph_index: Index of paragraph (for paragraph scope)

        Returns:
            True if commit was created, False if no changes or error
        """
        if not GitHistoryService.ensure_git_initialized(document_name):
            return False

        doc_path = _get_document_path(document_name)
        user = get_current_user() or "document-mcp"
        user_email = f"{user}@document-mcp.local"

        # Stage all changes
        if not git_add(doc_path, "-A"):
            return False

        # Check if there are staged changes
        if not has_staged_changes(doc_path):
            return False

        # Build commit message
        message = GitHistoryService._build_commit_message(
            operation,
            scope,
            chapter_name,
            paragraph_index,
            user
        )

        # Create commit
        success, _ = git_commit(doc_path, message, user, user_email)
        return success

    @staticmethod
    def _build_commit_message(
        operation: str,
        scope: str,
        chapter_name: str | None,
        paragraph_index: int | None,
        user: str
    ) -> str:
        """Build conventional commit message."""
        # Build scope string
        if scope == "paragraph" and chapter_name and paragraph_index is not None:
            scope_str = f"paragraph({chapter_name}:{paragraph_index})"
        elif scope == "chapter" and chapter_name:
            scope_str = f"chapter({chapter_name})"
        else:
            scope_str = scope

        # Descriptions
        descriptions = {
            "create": f"Add {scope}",
            "edit": f"Modify {scope}",
            "replace": f"Replace {scope}",
            "delete": f"Remove {scope}",
            "move": f"Move {scope}",
            "restore": f"Restore {scope}"
        }

        description = descriptions.get(operation, f"Modify {scope}")

        # Build full message with context
        msg = f"{operation}({scope_str}): {description}\n\n"
        msg += f"User: {user}\n"
        msg += f"Timestamp: {datetime.datetime.now().isoformat()}\n"

        return msg

    @staticmethod
    def get_history(
        document_name: str,
        max_count: int = 50
    ) -> list[dict[str, Any]]:
        """Get commit history for a document.

        Args:
            document_name: Name of document
            max_count: Maximum commits to return

        Returns:
            List of commit dicts with hash, message, author, date
        """
        if not GitHistoryService.ensure_git_initialized(document_name):
            return []

        doc_path = _get_document_path(document_name)

        # Get commits with detailed format
        fmt = "%H%n%an%n%ae%n%ai%n%s%n%b%x00"
        commits = git_log(doc_path, fmt, max_count)

        result = []
        for commit_block in commits:
            if not commit_block.strip():
                continue

            lines = commit_block.split("\n")
            if len(lines) >= 5:
                result.append({
                    "hash": lines[0],
                    "author": lines[1],
                    "email": lines[2],
                    "date": lines[3],
                    "message": lines[4],
                    "body": "\n".join(lines[5:]) if len(lines) > 5 else ""
                })

        return result

    @staticmethod
    def get_diff(
        document_name: str,
        revision: str | None = None,
        context_lines: int = 3
    ) -> str:
        """Get diff for a revision.

        Args:
            document_name: Name of document
            revision: Git revision (None = unstaged changes)
            context_lines: Lines of context

        Returns:
            Diff output as string
        """
        if not GitHistoryService.ensure_git_initialized(document_name):
            return ""

        doc_path = _get_document_path(document_name)
        return git_diff(doc_path, revision, context_lines)

    @staticmethod
    def restore_revision(
        document_name: str,
        revision: str
    ) -> OperationStatus:
        """Restore document to a previous revision.

        Args:
            document_name: Name of document
            revision: Git revision to restore to

        Returns:
            OperationStatus with success/failure details
        """
        if not GitHistoryService.ensure_git_initialized(document_name):
            return OperationStatus(
                success=False,
                message="Git repository not available",
                details={}
            )

        doc_path = _get_document_path(document_name)
        user = get_current_user() or "document-mcp"
        user_email = f"{user}@document-mcp.local"

        # Reset to revision
        if not git_reset(doc_path, revision):
            return OperationStatus(
                success=False,
                message=f"Failed to reset to revision {revision}",
                details={"revision": revision}
            )

        # Create restoration commit
        msg = f"restore: Restored to {revision}\n\nUser: {user}\nTimestamp: {datetime.datetime.now().isoformat()}\n"
        success, error = git_commit(
            doc_path,
            msg,
            user,
            user_email,
            allow_empty=True
        )

        if success:
            return OperationStatus(
                success=True,
                message=f"Document restored to {revision}",
                details={
                    "revision": revision,
                    "action": "restore"
                }
            )
        else:
            return OperationStatus(
                success=False,
                message=f"Failed to commit restoration: {error}",
                details={"revision": revision, "error": error}
            )

    @staticmethod
    def get_status(document_name: str) -> dict[str, Any]:
        """Get current repository status.

        Args:
            document_name: Name of document

        Returns:
            Dict with has_changes, modified_files, staged_files
        """
        if not GitHistoryService.ensure_git_initialized(document_name):
            return {
                "has_changes": False,
                "modified_files": [],
                "staged_files": []
            }

        doc_path = _get_document_path(document_name)
        return git_status(doc_path)
```

---

## Part 2: Tool Modifications

### 2.1 Update Auto-Commit Decorator

Update `/Users/clchinkc/Documents/GitHub/document-mcp/document_mcp/utils/decorators.py`:

```python
"""Decorators for automatic operations on documents."""

from functools import wraps
from typing import Any, Callable

from ..services.git_history_service import GitHistoryService


def auto_commit(
    scope: str = "document",
    operation: str = "edit"
) -> Callable:
    """Decorator to automatically record changes to Git.

    Usage:
        @auto_commit(scope="chapter", operation="edit")
        def modify_chapter(document_name: str, chapter_name: str, ...):
            # Implementation
            pass

    Args:
        scope: Scope of change (document, chapter, paragraph)
        operation: Type of operation (create, edit, delete, replace, restore)

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Execute the original operation
            result = func(*args, **kwargs)

            # Extract document name and optional chapter/paragraph info
            document_name = args[0] if args else kwargs.get("document_name")
            if not document_name:
                return result

            chapter_name = None
            paragraph_index = None

            if scope == "chapter":
                chapter_name = args[1] if len(args) > 1 else kwargs.get("chapter_name")
            elif scope == "paragraph":
                chapter_name = args[1] if len(args) > 1 else kwargs.get("chapter_name")
                paragraph_index = args[2] if len(args) > 2 else kwargs.get("paragraph_index")

            # Record the change
            GitHistoryService.record_change(
                document_name,
                operation,
                scope,
                chapter_name,
                paragraph_index
            )

            return result

        return wrapper
    return decorator
```

### 2.2 Update manage_history Tool

Create new version in `/Users/clchinkc/Documents/GitHub/document-mcp/document_mcp/tools/history_tools.py`:

```python
"""Version history management tools using Git backend."""

from typing import Any

from ..logger_config import ErrorCategory, log_mcp_call, log_structured_error
from ..models import OperationStatus
from ..services.git_history_service import GitHistoryService
from ..utils.validation import validate_document_name


def register_history_tools(mcp_server) -> None:
    """Register history management tools."""

    @mcp_server.tool()
    @log_mcp_call
    def manage_history(
        document_name: str,
        action: str,  # "log", "diff", "restore", "status"
        revision_id: str | None = None,
        context_lines: int = 3,
        max_commits: int = 50,
    ) -> Any:
        """Manage document version history using Git backend.

        This unified tool provides comprehensive version control operations:
        - log: View commit history
        - diff: Show changes in specific commit
        - restore: Restore to previous version
        - status: Show uncommitted changes

        Parameters:
            document_name (str): Name of document directory
            action (str): Operation to perform
            revision_id (Optional[str]): Git revision (commit hash, HEAD~N, etc)
            context_lines (int): Lines of context in diffs (default: 3)
            max_commits (int): Maximum commits to retrieve (default: 50)

        Returns:
            OperationStatus with structured details

        Example Usage:
            ```json
            {
                "name": "manage_history",
                "arguments": {
                    "document_name": "my_document",
                    "action": "log",
                    "max_commits": 20
                }
            }
            ```

            ```json
            {
                "name": "manage_history",
                "arguments": {
                    "document_name": "my_document",
                    "action": "diff",
                    "revision_id": "abc123"
                }
            }
            ```

            ```json
            {
                "name": "manage_history",
                "arguments": {
                    "document_name": "my_document",
                    "action": "restore",
                    "revision_id": "abc123"
                }
            }
            ```
        """
        # Validate document name
        is_valid, error_msg = validate_document_name(document_name)
        if not is_valid:
            return OperationStatus(
                success=False,
                message=f"Invalid document name: {error_msg}",
                details={"action": action}
            )

        try:
            if action == "log":
                history = GitHistoryService.get_history(document_name, max_commits)
                return OperationStatus(
                    success=True,
                    message=f"Retrieved {len(history)} commits",
                    details={
                        "commits": history,
                        "total_count": len(history),
                        "action": "log"
                    }
                )

            elif action == "diff":
                if not revision_id:
                    return OperationStatus(
                        success=False,
                        message="revision_id is required for diff action",
                        details={"action": "diff"}
                    )

                diff_output = GitHistoryService.get_diff(
                    document_name,
                    revision_id,
                    context_lines
                )

                return OperationStatus(
                    success=True,
                    message="Diff generated successfully",
                    details={
                        "diff_text": diff_output,
                        "revision": revision_id,
                        "context_lines": context_lines,
                        "action": "diff"
                    }
                )

            elif action == "restore":
                if not revision_id:
                    return OperationStatus(
                        success=False,
                        message="revision_id is required for restore action",
                        details={"action": "restore"}
                    )

                return GitHistoryService.restore_revision(document_name, revision_id)

            elif action == "status":
                status = GitHistoryService.get_status(document_name)
                return OperationStatus(
                    success=True,
                    message="Status retrieved successfully",
                    details={
                        **status,
                        "action": "status"
                    }
                )

            else:
                return OperationStatus(
                    success=False,
                    message=f"Unknown action: {action}",
                    details={
                        "action": action,
                        "valid_actions": ["log", "diff", "restore", "status"]
                    }
                )

        except Exception as e:
            log_structured_error(
                ErrorCategory.ERROR,
                f"Failed to manage history: {e}",
                {
                    "document_name": document_name,
                    "action": action,
                    "error": str(e)
                }
            )
            return OperationStatus(
                success=False,
                message=f"History operation failed: {str(e)}",
                details={"action": action, "error": str(e)}
            )
```

---

## Part 3: Migration Tools

### 3.1 Create Migration Service

Create `/Users/clchinkc/Documents/GitHub/document-mcp/document_mcp/services/migration_service.py`:

```python
"""Service for migrating documents from snapshot to Git backend."""

import datetime
import json
import shutil
from pathlib import Path
from typing import Any

from ..helpers import _get_document_path, _get_snapshots_path
from ..logger_config import ErrorCategory, log_structured_error
from ..services.git_history_service import GitHistoryService


class MigrationService:
    """Service for managing document migrations to Git backend."""

    @staticmethod
    def validate_migration_readiness(document_name: str) -> dict[str, Any]:
        """Check if document can be safely migrated.

        Returns:
            Dict with:
            - ready: bool
            - issues: list of problems
            - warnings: list of warnings
        """
        doc_path = _get_document_path(document_name)
        issues = []
        warnings = []

        # Check document exists
        if not doc_path.exists():
            issues.append(f"Document directory does not exist: {doc_path}")
            return {
                "ready": False,
                "issues": issues,
                "warnings": warnings,
                "document_path": str(doc_path)
            }

        # Check write permissions
        if not Path(doc_path).is_dir():
            issues.append("Document path is not a directory")

        if not os.access(doc_path, os.W_OK):
            issues.append("No write permission to document directory")

        # Check for existing Git
        if (doc_path / ".git").exists():
            warnings.append("Git repository already initialized")

        # Check for snapshots
        snapshots_path = _get_snapshots_path(document_name)
        if snapshots_path.exists():
            snapshot_count = len(list(snapshots_path.glob("*.snapshot")))
            if snapshot_count > 0:
                warnings.append(f"Found {snapshot_count} existing snapshots")

        return {
            "ready": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "document_path": str(doc_path)
        }

    @staticmethod
    def migrate_document(
        document_name: str,
        preserve_snapshots: bool = True,
        auto_cleanup: bool = False
    ) -> dict[str, Any]:
        """Migrate a document to Git backend.

        Args:
            document_name: Name of document to migrate
            preserve_snapshots: Keep old snapshots in archive
            auto_cleanup: Remove old snapshots after migration

        Returns:
            Dict with migration status and details
        """
        # Validate readiness
        readiness = MigrationService.validate_migration_readiness(document_name)
        if not readiness["ready"]:
            return {
                "success": False,
                "message": "Document not ready for migration",
                "issues": readiness["issues"]
            }

        doc_path = _get_document_path(document_name)
        migration_log = []

        try:
            # Step 1: Initialize Git
            migration_log.append("Initializing Git repository...")
            if not GitHistoryService.ensure_git_initialized(document_name):
                raise Exception("Failed to initialize Git repository")
            migration_log.append("✓ Git repository initialized")

            # Step 2: Archive snapshots
            snapshots_path = _get_snapshots_path(document_name)
            snapshot_archive = None

            if snapshots_path.exists() and preserve_snapshots:
                migration_log.append("Archiving existing snapshots...")
                snapshot_archive = doc_path / ".snapshot_archive"
                snapshot_archive.mkdir(exist_ok=True)

                snapshot_count = 0
                for snapshot_file in snapshots_path.glob("*.snapshot"):
                    shutil.copy2(snapshot_file, snapshot_archive / snapshot_file.name)
                    snapshot_count += 1

                migration_log.append(f"✓ Archived {snapshot_count} snapshots")

            # Step 3: Record migration metadata
            migration_metadata = {
                "migration_date": datetime.datetime.now().isoformat(),
                "previous_system": "filesystem_snapshots",
                "new_system": "git",
                "preserved_snapshots": preserve_snapshots,
                "snapshot_archive": str(snapshot_archive) if snapshot_archive else None
            }

            metadata_file = doc_path / ".migration_metadata.json"
            metadata_file.write_text(
                json.dumps(migration_metadata, indent=2),
                encoding="utf-8"
            )
            migration_log.append("✓ Migration metadata recorded")

            # Step 4: Cleanup if requested
            if auto_cleanup and snapshots_path.exists():
                migration_log.append("Cleaning up old snapshots...")
                shutil.rmtree(snapshots_path)
                migration_log.append("✓ Old snapshots removed")

            return {
                "success": True,
                "message": f"Successfully migrated {document_name}",
                "details": {
                    "document_name": document_name,
                    "migration_log": migration_log,
                    "metadata": migration_metadata,
                    "git_repo_initialized": True
                }
            }

        except Exception as e:
            migration_log.append(f"✗ Migration failed: {e}")
            log_structured_error(
                ErrorCategory.ERROR,
                f"Migration failed: {e}",
                {
                    "document_name": document_name,
                    "migration_log": migration_log
                }
            )
            return {
                "success": False,
                "message": f"Migration failed: {str(e)}",
                "details": {
                    "document_name": document_name,
                    "migration_log": migration_log,
                    "error": str(e)
                }
            }
```

---

## Part 4: Integration Points

### 4.1 Apply Decorator to Existing Tools

Update existing tools to use `@auto_commit`:

```python
# In document_mcp/tools/chapter_tools.py

from ..utils.decorators import auto_commit

@mcp_server.tool()
@log_mcp_call
@auto_commit(scope="chapter", operation="create")
def create_chapter(document_name: str, chapter_name: str, content: str) -> Any:
    """Create a new chapter."""
    # Existing implementation
    ...

@mcp_server.tool()
@log_mcp_call
@auto_commit(scope="chapter", operation="edit")
def edit_chapter(document_name: str, chapter_name: str, content: str) -> Any:
    """Edit existing chapter."""
    # Existing implementation
    ...
```

### 4.2 Register History Tools

Update `/Users/clchinkc/Documents/GitHub/document-mcp/document_mcp/doc_tool_server.py`:

```python
def create_server():
    """Create and configure the MCP server."""
    mcp_server = Server("document-mcp")

    # Register all tool categories
    from .tools.history_tools import register_history_tools
    register_history_tools(mcp_server)  # Add this line

    # ... other registrations ...

    return mcp_server
```

---

## Part 5: Testing Strategy

### 5.1 Unit Tests for Git Backend

Create `/Users/clchinkc/Documents/GitHub/document-mcp/tests/unit/test_git_backend.py`:

```python
"""Tests for Git backend operations."""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from document_mcp.utils.git_backend import (
    git_add,
    git_commit,
    git_init,
    is_git_available,
)


class TestGitBackend:
    """Test low-level Git operations."""

    @pytest.fixture
    def temp_repo(self):
        """Create temporary Git repository for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Initialize repo
            assert git_init(repo_path)

            # Configure user
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )

            yield repo_path

    def test_git_available(self):
        """Test Git availability check."""
        assert is_git_available()

    def test_git_init(self, temp_repo):
        """Test Git repository initialization."""
        git_dir = temp_repo / ".git"
        assert git_dir.exists()

    def test_git_add_and_commit(self, temp_repo):
        """Test adding and committing files."""
        # Create a test file
        test_file = temp_repo / "test.txt"
        test_file.write_text("test content")

        # Add and commit
        assert git_add(temp_repo)
        success, _ = git_commit(temp_repo, "Test commit", "Test User", "test@example.com")
        assert success

    def test_multiple_commits(self, temp_repo):
        """Test creating multiple commits."""
        for i in range(3):
            test_file = temp_repo / f"file{i}.txt"
            test_file.write_text(f"content {i}")

            git_add(temp_repo)
            success, _ = git_commit(
                temp_repo,
                f"Commit {i}",
                "Test User",
                "test@example.com"
            )
            assert success
```

### 5.2 Integration Tests

Create `/Users/clchinkc/Documents/GitHub/document-mcp/tests/integration/test_git_history.py`:

```python
"""Integration tests for Git history service."""

import pytest

from document_mcp.services.git_history_service import GitHistoryService


class TestGitHistoryService:
    """Test high-level history operations."""

    @pytest.fixture
    def sample_document(self, test_docs_root, sample_document):
        """Create test document with Git history."""
        doc_name = sample_document.name
        assert GitHistoryService.ensure_git_initialized(doc_name)
        return sample_document

    def test_ensure_git_initialized(self, sample_document):
        """Test Git initialization."""
        doc_path = sample_document
        git_dir = doc_path / ".git"
        assert git_dir.exists()

    def test_record_change(self, sample_document):
        """Test recording document changes."""
        doc_name = sample_document.name

        # Record a change
        result = GitHistoryService.record_change(
            doc_name,
            operation="edit",
            scope="document"
        )

        assert result or len(GitHistoryService.get_history(doc_name)) > 0

    def test_get_history(self, sample_document):
        """Test retrieving commit history."""
        doc_name = sample_document.name

        history = GitHistoryService.get_history(doc_name)
        assert isinstance(history, list)

    def test_restore_revision(self, sample_document):
        """Test restoring to previous revision."""
        doc_name = sample_document.name

        history = GitHistoryService.get_history(doc_name, max_count=2)
        if len(history) > 1:
            target_rev = history[1]["hash"]
            result = GitHistoryService.restore_revision(doc_name, target_rev)
            assert result.success
```

---

## Deployment Checklist

- [ ] Create all new modules
- [ ] Update existing decorators
- [ ] Add history tools
- [ ] Create migration service
- [ ] Update doc_tool_server.py
- [ ] Add unit tests
- [ ] Add integration tests
- [ ] Document migration procedures
- [ ] Test backward compatibility
- [ ] Performance baseline measurements
- [ ] Monitoring and alerting setup
- [ ] User documentation
- [ ] Release notes

---

## Quick Reference

### Environment Variables

```bash
# Enable/disable Git backend
export GIT_BACKEND_ENABLED=true

# Git operation timeout (seconds)
export GIT_COMMIT_TIMEOUT=30

# Automatic GC threshold (commits)
export GIT_GC_THRESHOLD=1000

# Aggressive GC (0/1)
export GIT_AGGRESSIVE_GC=0
```

### Common Operations

```python
from document_mcp.services.git_history_service import GitHistoryService

# Initialize Git for a document
GitHistoryService.ensure_git_initialized("my_document")

# Record a change
GitHistoryService.record_change(
    "my_document",
    operation="edit",
    scope="chapter",
    chapter_name="01-intro.md"
)

# Get history
history = GitHistoryService.get_history("my_document", max_count=20)

# Restore a version
result = GitHistoryService.restore_revision("my_document", "abc123")

# Check status
status = GitHistoryService.get_status("my_document")
```

