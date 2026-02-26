"""Git-based version control tools for document management.

This module provides tools for managing document version history using Git:
- get_version_history: Retrieve commit history for a document
- checkout_version: Restore document to a specific commit
- compare_versions: Generate diff between two commits
"""

from typing import Any

from ..logger_config import ErrorCategory
from ..logger_config import log_mcp_call
from ..logger_config import log_structured_error
from ..models import OperationStatus
from ..models import VersionComparisonResult
from ..models import VersionDiff
from ..models import VersionHistory
from ..utils.git_manager import GitManager
from ..utils.validation import validate_document_name


def register_version_tools(mcp_server) -> None:
    """Register all version control tools with the MCP server."""

    @mcp_server.tool()
    @log_mcp_call
    def get_version_history(
        document_name: str,
        limit: int = 10,
    ) -> Any:
        """Retrieve Git commit history for a document.

        Returns the most recent commits in the document's Git repository,
        including author, timestamp, and commit messages.

        Parameters:
            document_name (str): Name of the document directory
            limit (int): Maximum number of commits to return (default: 10, max: 100)

        Returns:
            Dict[str, Any]: VersionHistory with commit information:
                - document_name: Name of the document
                - total_commits: Total number of commits in history
                - commits: List of CommitInfo objects with hash, author, timestamp, message, summary
                - time_window: Always "all" for complete history

        Raises:
            ValidationError: If document_name is invalid
            GitError: If version history cannot be retrieved

        Example Usage:
            ```json
            {
                "name": "get_version_history",
                "arguments": {
                    "document_name": "my_novel",
                    "limit": 20
                }
            }
            ```

        Response Example:
            ```json
            {
                "document_name": "my_novel",
                "total_commits": 42,
                "commits": [
                    {
                        "hash": "a1b2c3d4...",
                        "author": "Alice <alice@example.com>",
                        "timestamp": "2026-02-25T15:30:00+00:00",
                        "message": "edit chapter: revised opening scene",
                        "summary": "edit chapter: revised opening scene"
                    },
                    ...
                ],
                "time_window": "all"
            }
            ```
        """
        # Validate document name
        is_valid, error_msg = validate_document_name(document_name)
        if not is_valid:
            return OperationStatus(
                success=False,
                message=f"Invalid document name: {error_msg}",
                details={"operation": "get_version_history", "document_name": document_name},
            )

        # Validate limit
        if limit < 1 or limit > 100:
            return OperationStatus(
                success=False,
                message="limit must be between 1 and 100",
                details={"operation": "get_version_history", "limit": limit},
            )

        try:
            from ..helpers import _get_document_path

            doc_path = _get_document_path(document_name)

            if not doc_path.exists():
                return OperationStatus(
                    success=False,
                    message=f"Document '{document_name}' not found",
                    details={"operation": "get_version_history", "document_name": document_name},
                )

            # Initialize Git manager
            git_manager = GitManager(doc_path)

            # Get commit history
            commits = git_manager.get_version_history(limit=limit)

            # Convert to response format
            commit_dicts = [c.to_dict() for c in commits]

            return VersionHistory(
                document_name=document_name,
                total_commits=len(commits),
                commits=[
                    {
                        "hash": c["hash"],
                        "author": c["author"],
                        "timestamp": c["timestamp"],
                        "message": c["message"],
                        "summary": c["summary"],
                    }
                    for c in commit_dicts
                ],
                time_window="all",
            )

        except Exception as e:
            log_structured_error(
                ErrorCategory.ERROR,
                f"Failed to retrieve version history for '{document_name}': {e}",
                {
                    "operation": "get_version_history",
                    "document_name": document_name,
                    "error": str(e),
                },
            )
            return OperationStatus(
                success=False,
                message=f"Failed to retrieve version history: {str(e)}",
                details={
                    "operation": "get_version_history",
                    "error": str(e),
                },
            )

    @mcp_server.tool()
    @log_mcp_call
    def checkout_version(
        document_name: str,
        version_hash: str,
    ) -> Any:
        """Restore document to a specific Git commit.

        Restores the document files to their state at the specified commit.
        This operation modifies the working tree but preserves Git history.

        Parameters:
            document_name (str): Name of the document directory
            version_hash (str): Commit SHA hash or tag name to restore

        Returns:
            Dict[str, Any]: OperationStatus indicating success or failure:
                - success: Boolean indicating whether restoration succeeded
                - message: Human-readable status message
                - details: Operation metadata including version_hash and files_restored

        Raises:
            ValidationError: If document_name is invalid
            GitError: If version checkout fails (e.g., invalid hash)

        Example Usage:
            ```json
            {
                "name": "checkout_version",
                "arguments": {
                    "document_name": "my_novel",
                    "version_hash": "a1b2c3d4e5f6..."
                }
            }
            ```

        Response Example:
            ```json
            {
                "success": true,
                "message": "Version a1b2c3d4e5f6... restored successfully",
                "details": {
                    "operation": "checkout_version",
                    "document_name": "my_novel",
                    "version_hash": "a1b2c3d4e5f6...",
                    "files_restored": 5
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
                details={"operation": "checkout_version", "document_name": document_name},
            )

        # Validate version_hash is provided
        if not version_hash or not version_hash.strip():
            return OperationStatus(
                success=False,
                message="version_hash is required",
                details={"operation": "checkout_version"},
            )

        try:
            from ..helpers import _get_document_path

            doc_path = _get_document_path(document_name)

            if not doc_path.exists():
                return OperationStatus(
                    success=False,
                    message=f"Document '{document_name}' not found",
                    details={"operation": "checkout_version", "document_name": document_name},
                )

            # Initialize Git manager
            git_manager = GitManager(doc_path)

            # Perform checkout
            git_manager.checkout_version(version_hash)

            # Count files in document (simplified)
            files_restored = len(list(doc_path.glob("*.md")))

            return OperationStatus(
                success=True,
                message=f"Version {version_hash[:8]}... restored successfully",
                details={
                    "operation": "checkout_version",
                    "document_name": document_name,
                    "version_hash": version_hash,
                    "files_restored": files_restored,
                },
            )

        except Exception as e:
            log_structured_error(
                ErrorCategory.ERROR,
                f"Failed to checkout version for '{document_name}': {e}",
                {
                    "operation": "checkout_version",
                    "document_name": document_name,
                    "version_hash": version_hash,
                    "error": str(e),
                },
            )
            return OperationStatus(
                success=False,
                message=f"Failed to checkout version: {str(e)}",
                details={
                    "operation": "checkout_version",
                    "version_hash": version_hash,
                    "error": str(e),
                },
            )

    @mcp_server.tool()
    @log_mcp_call
    def compare_versions(
        document_name: str,
        version1_hash: str,
        version2_hash: str | None = None,
        stat_only: bool = False,
    ) -> Any:
        """Generate detailed diff between two Git commits.

        Compares two versions of a document and returns the differences in unified
        diff format along with statistics about changes.

        Parameters:
            document_name (str): Name of the document directory
            version1_hash (str): Source commit SHA hash or tag
            version2_hash (Optional[str]): Target commit SHA hash or tag (defaults to HEAD)
            stat_only (bool): Return only statistics, not full diff text (default: False)

        Returns:
            Dict[str, Any]: VersionComparisonResult with diff information:
                - document_name: Name of the document
                - version1: Source version hash
                - version2: Target version hash
                - diff: VersionDiff with diff_text, files_changed, insertions, deletions
                - has_changes: Boolean indicating whether changes exist
                - summary: Human-readable comparison summary

        Raises:
            ValidationError: If document_name is invalid
            GitError: If comparison fails (e.g., invalid hashes)

        Example Usage:
            ```json
            {
                "name": "compare_versions",
                "arguments": {
                    "document_name": "my_novel",
                    "version1_hash": "a1b2c3d4...",
                    "version2_hash": "b2c3d4e5...",
                    "stat_only": false
                }
            }
            ```

        Response Example:
            ```json
            {
                "document_name": "my_novel",
                "version1": "a1b2c3d4...",
                "version2": "b2c3d4e5...",
                "diff": {
                    "source_hash": "a1b2c3d4...",
                    "target_hash": "b2c3d4e5...",
                    "diff_text": "--- a/01-chapter.md\\n+++ b/01-chapter.md\\n...",
                    "files_changed": 1,
                    "insertions": 45,
                    "deletions": 12
                },
                "has_changes": true,
                "summary": "1 file changed, 45 insertions, 12 deletions"
            }
            ```
        """
        # Validate document name
        is_valid, error_msg = validate_document_name(document_name)
        if not is_valid:
            return OperationStatus(
                success=False,
                message=f"Invalid document name: {error_msg}",
                details={"operation": "compare_versions", "document_name": document_name},
            )

        # Validate version1_hash is provided
        if not version1_hash or not version1_hash.strip():
            return OperationStatus(
                success=False,
                message="version1_hash is required",
                details={"operation": "compare_versions"},
            )

        try:
            from ..helpers import _get_document_path

            doc_path = _get_document_path(document_name)

            if not doc_path.exists():
                return OperationStatus(
                    success=False,
                    message=f"Document '{document_name}' not found",
                    details={"operation": "compare_versions", "document_name": document_name},
                )

            # Initialize Git manager
            git_manager = GitManager(doc_path)

            # Perform comparison
            git_diff = git_manager.compare_versions(
                version_hash=version1_hash,
                target_hash=version2_hash,
                stat_only=stat_only,
            )

            # Build response
            target = version2_hash or "HEAD"
            has_changes = (
                git_diff.stats.get("insertions", 0) > 0
                or git_diff.stats.get("deletions", 0) > 0
                or git_diff.stats.get("files_changed", 0) > 0
            )

            summary = (
                f"{git_diff.stats.get('files_changed', 0)} file(s) changed, "
                f"{git_diff.stats.get('insertions', 0)} insertion(s), "
                f"{git_diff.stats.get('deletions', 0)} deletion(s)"
            )

            return VersionComparisonResult(
                document_name=document_name,
                version1=version1_hash,
                version2=target,
                diff=VersionDiff(
                    source_hash=version1_hash,
                    target_hash=target,
                    diff_text=git_diff.diff_text if not stat_only else "",
                    files_changed=git_diff.stats.get("files_changed", 0),
                    insertions=git_diff.stats.get("insertions", 0),
                    deletions=git_diff.stats.get("deletions", 0),
                ),
                has_changes=has_changes,
                summary=summary,
            )

        except Exception as e:
            log_structured_error(
                ErrorCategory.ERROR,
                f"Failed to compare versions for '{document_name}': {e}",
                {
                    "operation": "compare_versions",
                    "document_name": document_name,
                    "version1_hash": version1_hash,
                    "version2_hash": version2_hash,
                    "error": str(e),
                },
            )
            return OperationStatus(
                success=False,
                message=f"Failed to compare versions: {str(e)}",
                details={
                    "operation": "compare_versions",
                    "error": str(e),
                },
            )
