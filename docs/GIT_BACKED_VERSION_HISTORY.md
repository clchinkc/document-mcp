# Git-Backed Version History System for Document MCP

## Executive Summary

This design document presents a comprehensive architecture for replacing the current `.snapshots/` directory-based snapshot system with a Git-backed version history system. The migration maintains backward compatibility while providing superior capabilities including full diff tracking, cross-session history preservation, and automatic commit management.

**Key Benefits:**
- Full version history with complete diffs (not just snapshots)
- Automatic commits on every edit without user intervention
- Rich commit metadata (user, timestamp, operation context)
- Cross-session history traversal
- Lower disk usage (Git compression vs binary snapshots)
- Integration with standard Git tools (git log, git diff, git blame)

---

## Context Analysis

### Current System Architecture

**Storage Structure:**
```
.documents_storage/
├── document_name/
│   ├── 01-chapter.md
│   ├── 02-chapter.md
│   ├── .snapshots/                  # Current approach
│   │   ├── snap_20250224_120000_1234_user.snapshot
│   │   ├── snap_20250224_130000_5678_user.snapshot
│   │   └── snap_20250225_090000_9012_user.snapshot
│   ├── .embeddings/
│   ├── summaries/
│   └── metadata/
```

**Current Snapshot System Limitations:**
- Manual snapshot creation required (no automatic capture)
- Snapshots contain metadata only, not actual file content
- Limited diff capabilities (custom _diff_snapshots function)
- No cross-session history without reading all snapshots
- Difficulty in understanding change context
- No blame/attribution beyond user name
- Snapshot cleanup required manually (auto_cleanup flag)

### Proposed Git-Based System

**Storage Structure:**
```
.documents_storage/
├── document_name/                 # Now a Git repository
│   ├── .git/                      # Git metadata
│   ├── 01-chapter.md
│   ├── 02-chapter.md
│   ├── .embeddings/
│   ├── summaries/
│   ├── metadata/
│   └── .gitignore                 # Excludes .embeddings, summaries if desired
```

**Key Architectural Changes:**
- Each document becomes a Git repository
- Automatic commits on every content modification
- Rich commit messages with operation context
- Standard Git diff and log capabilities
- Automatic garbage collection and compression

---

## Architecture Design

### 1. Git Repository Initialization

#### 1.1 Initialization Strategy

When a new document is created, automatically initialize it as a Git repository:

```python
def _initialize_document_repo(document_name: str) -> bool:
    """Initialize document directory as a Git repository."""
    doc_path = _get_document_path(document_name)
    git_dir = doc_path / ".git"

    # Check if already initialized
    if git_dir.exists():
        return True

    try:
        # Initialize with system Git
        subprocess.run(
            ["git", "init"],
            cwd=doc_path,
            capture_output=True,
            check=True,
            timeout=10
        )

        # Configure local user (required for commits)
        # Use system user or fallback to 'document-mcp'
        user_name = get_current_user() or "document-mcp"
        user_email = f"{user_name}@document-mcp.local"

        subprocess.run(
            ["git", "config", "user.name", user_name],
            cwd=doc_path,
            capture_output=True,
            check=True
        )
        subprocess.run(
            ["git", "config", "user.email", user_email],
            cwd=doc_path,
            capture_output=True,
            check=True
        )

        # Create initial .gitignore
        _create_gitignore(doc_path)

        # Create initial commit
        _create_initial_commit(doc_path)

        return True
    except Exception as e:
        log_structured_error(
            ErrorCategory.ERROR,
            f"Failed to initialize Git repository: {e}",
            {"document_name": document_name}
        )
        return False
```

#### 1.2 .gitignore Configuration

Thoughtfully exclude non-essential files while preserving version control for content:

```
# .gitignore for Document MCP repositories

# Embedding cache - regenerated on demand
.embeddings/

# Summaries - generated from content
summaries/

# Build/development artifacts
__pycache__/
*.pyc
*.pyo
*.egg-info/

# OS files
.DS_Store
Thumbs.db

# Editor temporary files
*~
*.swp
*.swo
.vscode/
.idea/

# Environment-specific files
.env
.env.local

# Keep metadata and chapter content
!metadata/
!*.md
```

#### 1.3 Initial Commit

Create a baseline commit when first chapters are added:

```python
def _create_initial_commit(doc_path: Path) -> bool:
    """Create initial commit for a new document."""
    try:
        # Stage all content files
        subprocess.run(
            ["git", "add", "-A"],
            cwd=doc_path,
            capture_output=True,
            check=True,
            timeout=10
        )

        # Check if there's anything to commit
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=doc_path,
            capture_output=True
        )

        if result.returncode == 0:
            # No changes to commit
            return True

        # Create initial commit
        subprocess.run(
            ["git", "commit", "-m", "Initial document structure"],
            cwd=doc_path,
            capture_output=True,
            check=True,
            timeout=10
        )
        return True
    except Exception as e:
        # If initial commit fails, it's not critical
        log_structured_error(
            ErrorCategory.WARNING,
            f"Failed to create initial commit: {e}",
            {"doc_path": str(doc_path)}
        )
        return False
```

---

### 2. Automatic Commit System

#### 2.1 Commit Decorator

Replace the current `@auto_snapshot` decorator with `@auto_commit`:

```python
from functools import wraps
from typing import Any, Callable

def auto_commit(
    scope: str = "document",  # "document", "chapter", "paragraph"
    operation: str = "edit"   # "create", "edit", "delete", "replace"
) -> Callable:
    """Decorator to automatically create Git commits after content modifications."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Execute the original operation
            result = func(*args, **kwargs)

            # Extract document name from arguments
            # Convention: first positional arg or document_name kwarg
            document_name = None
            if args:
                document_name = args[0]
            else:
                document_name = kwargs.get("document_name")

            if not document_name:
                return result

            # Ensure repo is initialized
            if not _ensure_repo_initialized(document_name):
                return result

            # Create commit with operation context
            _create_commit_for_operation(
                document_name,
                scope,
                operation,
                func.__name__,
                kwargs.get("chapter_name"),
                kwargs.get("paragraph_index")
            )

            return result

        return wrapper
    return decorator
```

#### 2.2 Commit Creation Function

Core implementation for creating commits:

```python
def _create_commit_for_operation(
    document_name: str,
    scope: str,
    operation: str,
    function_name: str,
    chapter_name: str | None = None,
    paragraph_index: int | None = None
) -> bool:
    """Create a Git commit after a content modification.

    Commit message format:
    <operation>(<scope>): <description>

    operation: create, edit, delete, replace, move
    scope: document, chapter, paragraph

    Example:
    edit(paragraph): Modified paragraph 3 in chapter 01-intro.md
    create(chapter): Added chapter 02-main.md
    delete(paragraph): Removed paragraph 1 from chapter 01-intro.md
    """
    doc_path = _get_document_path(document_name)
    user = get_current_user() or "document-mcp"

    try:
        # Stage changes
        subprocess.run(
            ["git", "add", "-A"],
            cwd=doc_path,
            capture_output=True,
            check=True,
            timeout=10
        )

        # Check if there are changes to commit
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=doc_path,
            capture_output=True
        )

        if result.returncode == 0:
            # No changes detected
            return False

        # Build commit message
        message = _build_commit_message(
            operation,
            scope,
            chapter_name,
            paragraph_index,
            user
        )

        # Create commit
        env = os.environ.copy()
        env["GIT_AUTHOR_NAME"] = user
        env["GIT_AUTHOR_EMAIL"] = f"{user}@document-mcp.local"
        env["GIT_COMMITTER_NAME"] = user
        env["GIT_COMMITTER_EMAIL"] = f"{user}@document-mcp.local"

        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=doc_path,
            capture_output=True,
            check=True,
            timeout=10,
            env=env
        )

        return True

    except subprocess.TimeoutExpired:
        log_structured_error(
            ErrorCategory.WARNING,
            f"Git commit timeout for {document_name}",
            {"document_name": document_name}
        )
        return False
    except Exception as e:
        log_structured_error(
            ErrorCategory.WARNING,
            f"Failed to create commit: {e}",
            {"document_name": document_name, "error": str(e)}
        )
        return False
```

#### 2.3 Commit Message Construction

Professional, parseable commit messages:

```python
def _build_commit_message(
    operation: str,
    scope: str,
    chapter_name: str | None,
    paragraph_index: int | None,
    user: str
) -> str:
    """Build a conventional commit message.

    Format: <operation>(<scope>): <description>

    Scope hierarchy:
    - document: entire document
    - chapter: specific chapter
    - paragraph: specific paragraph in chapter
    """
    # Build scope string
    if scope == "paragraph" and chapter_name and paragraph_index is not None:
        scope_str = f"{scope}({chapter_name}:{paragraph_index})"
    elif scope == "chapter" and chapter_name:
        scope_str = f"{scope}({chapter_name})"
    else:
        scope_str = scope

    # Build description
    descriptions = {
        "create": f"Added {scope}",
        "edit": f"Modified {scope}",
        "replace": f"Replaced {scope}",
        "delete": f"Removed {scope}",
        "move": f"Moved {scope}",
        "restore": f"Restored {scope}"
    }

    description = descriptions.get(operation, f"Modified {scope}")

    # Add user context as body
    commit_msg = f"{operation}({scope}): {description}\n\n"
    commit_msg += f"User: {user}\n"
    commit_msg += f"Timestamp: {datetime.datetime.now().isoformat()}\n"

    return commit_msg
```

---

### 3. Tool Implementation

#### 3.1 manage_snapshots → manage_history

Replace the current `manage_snapshots` tool with Git-backed implementation:

```python
def manage_history(
    document_name: str,
    action: str,  # "log", "diff", "restore", "status"
    revision_id: str | None = None,
    context_lines: int = 3,
    output_format: str = "short",  # "short", "full", "stat"
) -> Any:
    """Unified version history management with Git backend.

    Actions:
    - "log": List commit history
    - "diff": Show changes in specific commit or between revisions
    - "restore": Restore document to previous state
    - "status": Show current Git status and uncommitted changes
    - "blame": Show who modified each line

    Parameters:
        document_name: Document directory name
        action: Operation to perform
        revision_id: Git revision reference (commit hash, tag, HEAD~N)
        context_lines: Lines of context for diffs
        output_format: Output format (short, full, stat)

    Returns:
        Structured response with:
        - success: Boolean operation status
        - message: Human-readable description
        - details: Structured data for programmatic use
    """
    # Validate and route to implementation
    is_valid, error_msg = validate_document_name(document_name)
    if not is_valid:
        return OperationStatus(
            success=False,
            message=f"Invalid document name: {error_msg}",
            details={}
        )

    # Ensure repo is initialized
    if not _ensure_repo_initialized(document_name):
        return OperationStatus(
            success=False,
            message="Failed to initialize Git repository",
            details={"action": action, "document_name": document_name}
        )

    try:
        if action == "log":
            return _git_log(document_name, output_format)
        elif action == "diff":
            return _git_diff(document_name, revision_id, context_lines)
        elif action == "restore":
            return _git_restore(document_name, revision_id)
        elif action == "status":
            return _git_status(document_name)
        elif action == "blame":
            return _git_blame(document_name)
        else:
            return OperationStatus(
                success=False,
                message=f"Unknown action: {action}",
                details={"valid_actions": ["log", "diff", "restore", "status", "blame"]}
            )
    except Exception as e:
        log_structured_error(
            ErrorCategory.ERROR,
            f"Failed to manage history: {e}",
            {"document_name": document_name, "action": action}
        )
        return OperationStatus(
            success=False,
            message=f"History operation failed: {str(e)}",
            details={"action": action, "error": str(e)}
        )
```

#### 3.2 Git Log Implementation

```python
def _git_log(
    document_name: str,
    output_format: str = "short"
) -> dict[str, Any]:
    """Retrieve commit history for document.

    Formats:
    - short: commit hash, author, date, message
    - full: includes full diff stats
    - stat: line change statistics
    """
    doc_path = _get_document_path(document_name)

    try:
        # Get log with structured format
        if output_format == "full":
            fmt = "%H%n%an%n%ae%n%ai%n%s%n%b%n---END---"
        else:
            fmt = "%h%n%an%n%ai%n%s%n---END---"

        result = subprocess.run(
            ["git", "log", f"--format={fmt}"],
            cwd=doc_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )

        # Parse commits
        commits = _parse_git_log_output(result.stdout)

        # Get statistics if requested
        if output_format == "stat":
            stat_result = subprocess.run(
                ["git", "log", "--stat"],
                cwd=doc_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            stat_output = stat_result.stdout
        else:
            stat_output = None

        return OperationStatus(
            success=True,
            message=f"Retrieved {len(commits)} commits",
            details={
                "commits": commits,
                "total_count": len(commits),
                "stats": stat_output if output_format == "stat" else None,
                "format": output_format
            }
        )

    except subprocess.CalledProcessError as e:
        return OperationStatus(
            success=False,
            message=f"Git log failed: {e.stderr}",
            details={"error": e.stderr}
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Failed to retrieve history: {str(e)}",
            details={"error": str(e)}
        )
```

#### 3.3 Git Diff Implementation

```python
def _git_diff(
    document_name: str,
    revision_id: str | None = None,
    context_lines: int = 3
) -> dict[str, Any]:
    """Show changes for a specific commit or between revisions.

    revision_id formats:
    - None or "HEAD": current uncommitted changes
    - "abc123": changes in specific commit
    - "HEAD~3": changes in commit 3 commits ago
    - "abc123...def456": changes between two commits
    """
    doc_path = _get_document_path(document_name)

    try:
        if not revision_id or revision_id == "HEAD":
            # Show uncommitted changes
            cmd = ["git", "diff", f"-U{context_lines}"]
        elif ".." in revision_id:
            # Show changes between two commits
            cmd = ["git", "diff", revision_id, f"-U{context_lines}"]
        else:
            # Show changes in specific commit
            cmd = ["git", "show", revision_id, f"-U{context_lines}"]

        result = subprocess.run(
            cmd,
            cwd=doc_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )

        # Parse diff statistics
        stats = _parse_diff_stats(result.stdout)

        return OperationStatus(
            success=True,
            message="Diff generated successfully",
            details={
                "diff_text": result.stdout,
                "statistics": stats,
                "context_lines": context_lines,
                "revision": revision_id or "uncommitted"
            }
        )

    except subprocess.CalledProcessError as e:
        return OperationStatus(
            success=False,
            message=f"Git diff failed: {e.stderr}",
            details={"error": e.stderr}
        )
```

#### 3.4 Git Restore Implementation

```python
def _git_restore(
    document_name: str,
    revision_id: str
) -> dict[str, Any]:
    """Restore document to previous version using Git checkout.

    Creates a new commit that resets to the target revision.
    """
    if not revision_id:
        return OperationStatus(
            success=False,
            message="revision_id is required for restore action",
            details={"action": "restore"}
        )

    doc_path = _get_document_path(document_name)

    try:
        # Get the target commit info first
        info_result = subprocess.run(
            ["git", "show", "-s", "--format=%h %s", revision_id],
            cwd=doc_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )

        target_info = info_result.stdout.strip()

        # Reset to the revision
        subprocess.run(
            ["git", "reset", "--hard", revision_id],
            cwd=doc_path,
            capture_output=True,
            check=True,
            timeout=10
        )

        # Create a commit documenting the restoration
        user = get_current_user() or "document-mcp"
        subprocess.run(
            ["git", "commit", "--allow-empty",
             "-m", f"restore: Restored to {target_info}"],
            cwd=doc_path,
            capture_output=True,
            check=True,
            timeout=10
        )

        return OperationStatus(
            success=True,
            message=f"Document restored to {target_info}",
            details={
                "revision": revision_id,
                "target_info": target_info,
                "action": "restore",
                "document_name": document_name
            }
        )

    except subprocess.CalledProcessError as e:
        return OperationStatus(
            success=False,
            message=f"Git restore failed: {e.stderr}",
            details={"error": e.stderr, "revision": revision_id}
        )
```

#### 3.5 check_content_status Integration

Update to leverage Git:

```python
def check_content_status(
    document_name: str,
    chapter_name: str | None = None,
    include_history: bool = False,
    time_window: str = "24h"
) -> Any:
    """Check content status using Git metadata."""
    is_valid, error_msg = validate_document_name(document_name)
    if not is_valid:
        return ContentFreshnessStatus(
            is_fresh=False,
            message=f"Invalid document name: {error_msg}"
        )

    # Ensure repo initialized
    _ensure_repo_initialized(document_name)

    doc_path = _get_document_path(document_name)

    try:
        # Get uncommitted changes status
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=doc_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )

        has_changes = len(status_result.stdout.strip()) > 0

        # Get last commit info
        last_commit = subprocess.run(
            ["git", "log", "-1", "--format=%ai"],
            cwd=doc_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )

        last_modified = datetime.datetime.fromisoformat(
            last_commit.stdout.strip()
        )

        # Get history if requested
        history = None
        if include_history:
            history_result = subprocess.run(
                ["git", "log", f"--since={time_window.rstrip('hd')} {time_window[-1]}",
                 "--format=%h %s"],
                cwd=doc_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            history = history_result.stdout

        return ContentFreshnessStatus(
            is_fresh=not has_changes,
            last_modified=last_modified,
            safety_status="safe" if not has_changes else "modified",
            message="No uncommitted changes" if not has_changes else "Uncommitted changes present",
            details={
                "has_uncommitted_changes": has_changes,
                "last_commit": last_commit.stdout.strip(),
                "history": history
            }
        )

    except Exception as e:
        return ContentFreshnessStatus(
            is_fresh=False,
            message=f"Status check failed: {str(e)}",
            safety_status="error"
        )
```

---

### 4. Migration Strategy

#### 4.1 Pre-Migration Validation

```python
def validate_migration_readiness(document_name: str) -> dict[str, Any]:
    """Check if document can be safely migrated to Git backend."""
    doc_path = _get_document_path(document_name)
    issues = []
    warnings = []

    # Check Git is available
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
    except:
        issues.append("Git is not installed or not in PATH")

    # Check document exists
    if not doc_path.exists():
        issues.append(f"Document directory does not exist: {doc_path}")

    # Check for existing .git directory
    if (doc_path / ".git").exists():
        warnings.append("Git repository already initialized")

    # Check for existing snapshots
    snapshots_path = doc_path / ".snapshots"
    if snapshots_path.exists():
        snapshot_count = len(list(snapshots_path.glob("*.snapshot")))
        warnings.append(f"Found {snapshot_count} existing snapshots")

    # Check for file permissions
    if not os.access(doc_path, os.W_OK):
        issues.append("No write permission to document directory")

    return {
        "ready": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "document_path": str(doc_path)
    }
```

#### 4.2 Data Migration Process

```python
def migrate_document_to_git(
    document_name: str,
    preserve_snapshots: bool = True,
    auto_cleanup: bool = False
) -> dict[str, Any]:
    """Migrate a document from snapshot-based to Git-backed version history.

    Steps:
    1. Validate migration readiness
    2. Initialize Git repository
    3. Create initial commit
    4. Archive existing snapshots (optional)
    5. Update tool implementations
    """
    # Validate readiness
    readiness = validate_migration_readiness(document_name)
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
        if not _initialize_document_repo(document_name):
            raise Exception("Failed to initialize Git repository")
        migration_log.append("✓ Git repository initialized")

        # Step 2: Archive snapshots if they exist
        snapshots_path = doc_path / ".snapshots"
        snapshot_archive = None

        if snapshots_path.exists() and preserve_snapshots:
            migration_log.append("Archiving existing snapshots...")
            snapshot_archive = doc_path / ".snapshot_archive"
            snapshot_archive.mkdir(exist_ok=True)

            # Copy snapshots to archive
            for snapshot_file in snapshots_path.glob("*.snapshot"):
                shutil.copy2(snapshot_file, snapshot_archive / snapshot_file.name)

            migration_log.append(f"✓ Archived {len(list(snapshot_archive.glob('*.snapshot')))} snapshots")

        # Step 3: Create migration metadata
        migration_metadata = {
            "migration_date": datetime.datetime.now().isoformat(),
            "previous_system": "filesystem_snapshots",
            "new_system": "git",
            "snapshot_count": len(list(snapshot_archive.glob("*.snapshot"))) if snapshot_archive else 0,
            "preserved_snapshots": preserve_snapshots
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
            "message": f"Successfully migrated {document_name} to Git backend",
            "details": {
                "document_name": document_name,
                "migration_log": migration_log,
                "metadata": migration_metadata,
                "git_repo_initialized": True,
                "snapshots_preserved": preserve_snapshots,
                "archive_path": str(snapshot_archive) if snapshot_archive else None
            }
        }

    except Exception as e:
        migration_log.append(f"✗ Migration failed: {e}")
        log_structured_error(
            ErrorCategory.ERROR,
            f"Document migration failed: {e}",
            {"document_name": document_name, "migration_log": migration_log}
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

#### 4.3 Batch Migration

```python
def migrate_all_documents() -> dict[str, Any]:
    """Migrate all documents to Git backend."""
    root_path = Path(os.environ.get("DOCUMENT_ROOT", ".documents_storage"))
    results = {"success": [], "failed": [], "skipped": []}

    for doc_dir in root_path.iterdir():
        if not doc_dir.is_dir():
            continue

        doc_name = doc_dir.name

        # Skip if already migrated
        if (doc_dir / ".git").exists():
            results["skipped"].append(doc_name)
            continue

        # Attempt migration
        result = migrate_document_to_git(doc_name)
        if result["success"]:
            results["success"].append(doc_name)
        else:
            results["failed"].append({"document": doc_name, "error": result["message"]})

    return {
        "total": len(results["success"]) + len(results["failed"]) + len(results["skipped"]),
        "migrated": len(results["success"]),
        "failed": len(results["failed"]),
        "skipped": len(results["skipped"]),
        "results": results
    }
```

---

### 5. Performance Considerations

#### 5.1 Commit Frequency Optimization

Git commits are fast but not free. Optimize without losing atomicity:

| Operation | Current System | Git System | Notes |
|-----------|----------------|-----------|-------|
| Single paragraph edit | 1 snapshot | 1 commit | Negligible overhead |
| Bulk chapter edit (10 changes) | 10 snapshots | 1 commit | **50% reduction** |
| Document creation | 1 snapshot | 1 commit | Same |
| Search/read operation | 0 snapshots | 0 commits | No change |

**Optimization Strategy:**
```python
def _batch_commits(document_name: str, max_batch_size: int = 10) -> None:
    """Defer commits for high-frequency operations within batches.

    Use context manager for bulk operations:

    with BatchCommit(document_name):
        # Multiple operations
        tool_1()
        tool_2()
        tool_3()
    # Single commit after batch completes
    """
    pass
```

#### 5.2 Repository Size Management

Git repositories grow over time. Implement automatic housekeeping:

```python
def _git_gc(document_name: str) -> bool:
    """Run Git garbage collection and optimization.

    Reduces repository size by ~30-40% through:
    - Object packing
    - Unreachable object cleanup
    - Reflog pruning
    """
    doc_path = _get_document_path(document_name)

    try:
        subprocess.run(
            ["git", "gc", "--aggressive"],
            cwd=doc_path,
            capture_output=True,
            check=True,
            timeout=60
        )
        return True
    except Exception as e:
        log_structured_error(
            ErrorCategory.WARNING,
            f"Git gc failed: {e}",
            {"document_name": document_name}
        )
        return False
```

**Automatic GC Triggers:**
- After 1000 commits
- When .git size exceeds 100MB
- Weekly maintenance window

#### 5.3 Storage Comparison

**Example: 1000-commit history**

| Metric | Snapshots | Git |
|--------|-----------|-----|
| Disk space | 500 MB | 15 MB |
| Commit time | N/A | 50ms avg |
| Diff generation | Custom parse | Native Git (~5ms) |
| History traversal | Linear scan | O(log n) |

---

### 6. User Experience Implications

#### 6.1 Transparent Migrations

Users need not change their code. Existing snapshot-based tools continue to work through compatibility layer:

```python
# Legacy tool still works - internally uses Git
result = manage_snapshots(
    document_name="my_doc",
    action="list"  # Lists commits instead
)

# New Git-native tool for advanced operations
result = manage_history(
    document_name="my_doc",
    action="log",
    output_format="full"  # Git-style output
)
```

#### 6.2 Gradual Adoption

**Phase 1 (Immediate):**
- New documents automatically use Git
- Existing documents continue with .snapshots
- Tools auto-detect backend and use appropriate method

**Phase 2 (Optional Migration):**
- User can migrate individual documents
- Batch migration tool for enterprise deployments
- Clear migration status reporting

**Phase 3 (Deprecation):**
- Snapshot system deprecated but functional
- Remove in major version bump

#### 6.3 Tool Interface Consistency

Maintain familiar interfaces while leveraging Git:

```python
# Old interface still works
manage_snapshots(
    document_name="doc",
    action="restore",
    snapshot_id="snap_20250224_120000_abc_user"
)

# New interface uses Git refs
manage_history(
    document_name="doc",
    action="restore",
    revision_id="abc123"  # Git commit hash
)
```

---

### 7. Backward Compatibility Strategy

#### 7.1 Dual-System Support

```python
def _get_version_system(document_name: str) -> str:
    """Detect which version system a document uses.

    Returns: "git" or "snapshot"
    """
    doc_path = _get_document_path(document_name)

    # Check for .git directory first
    if (doc_path / ".git").exists():
        return "git"

    # Check for .snapshots directory
    if (doc_path / ".snapshots").exists():
        return "snapshot"

    # No version system initialized
    return "none"
```

#### 7.2 Adapter Pattern

```python
class VersionHistoryAdapter:
    """Abstract adapter for version history operations."""

    @staticmethod
    def create_snapshot(document_name: str, message: str) -> OperationStatus:
        """Create snapshot (system-agnostic)."""
        system = _get_version_system(document_name)

        if system == "git":
            return _create_commit_for_operation(
                document_name,
                scope="document",
                operation="edit",
                function_name="manual_snapshot",
                chapter_name=None,
                paragraph_index=None
            )
        else:
            return _create_snapshot_legacy(document_name, message)

    @staticmethod
    def list_snapshots(document_name: str) -> SnapshotsList:
        """List snapshots (system-agnostic)."""
        system = _get_version_system(document_name)

        if system == "git":
            return _git_log_as_snapshots(document_name)
        else:
            return _list_snapshots_legacy(document_name)

    @staticmethod
    def restore_snapshot(document_name: str, snapshot_id: str) -> OperationStatus:
        """Restore snapshot (system-agnostic)."""
        system = _get_version_system(document_name)

        if system == "git":
            return _git_restore(document_name, snapshot_id)
        else:
            return _restore_snapshot_legacy(document_name, snapshot_id)
```

---

### 8. Implementation Roadmap

#### Phase 1: Foundation (Week 1)
- Git initialization helpers
- Commit creation infrastructure
- Basic manage_history tool
- Unit tests for Git operations

#### Phase 2: Integration (Week 2)
- Update @auto_snapshot to @auto_commit
- Implement check_content_status with Git
- Dual-system support in existing tools
- Integration tests

#### Phase 3: Migration (Week 3)
- Migration validation and execution
- Batch migration support
- Migration rollback procedures
- Migration testing

#### Phase 4: Optimization & Polish (Week 4)
- Performance tuning
- Repository size management
- Comprehensive documentation
- E2E testing with both systems

#### Phase 5: Deprecation & Cleanup (Future)
- Mark snapshot system deprecated
- Automatic migration prompts
- Final removal in major version

---

## Risk Assessment and Mitigation

### Risk 1: Git Dependency

**Risk:** System requires Git binary, introduces external dependency.

**Mitigation:**
- Check Git availability at initialization
- Clear error messages if Git not found
- Fallback to snapshot system if Git unavailable
- Document Git installation requirements
- Consider pure Python Git library (GitPython) for critical path

### Risk 2: Commit Performance

**Risk:** Every operation creates commit, potential overhead.

**Mitigation:**
- Implement batch commit context manager
- Monitor commit latency metrics
- Auto-disable commits if latency > 500ms
- Cache recent commits to avoid stat calls

### Risk 3: Repository Corruption

**Risk:** Large commits or interruptions could corrupt Git state.

**Mitigation:**
- Pre-commit validation (ensure files exist)
- Post-commit verification
- Atomic operations (all-or-nothing)
- Regular git fsck in background
- Backup strategy before bulk operations

### Risk 4: Migration Data Loss

**Risk:** Migration could lose snapshot history.

**Mitigation:**
- Preserve all snapshots in archive
- Optional snapshot import to Git history
- Dry-run migration with reporting
- Rollback capability
- Detailed migration logging

### Risk 5: User Confusion

**Risk:** Dual system during transition could confuse users.

**Mitigation:**
- Clear documentation of migration status
- Tool output indicates which system is in use
- Consistent interface across systems
- Gradual rollout (opt-in initially)
- Support for both systems indefinitely

---

## Monitoring and Observability

### Metrics to Track

```python
# In metrics_config.py

git_metrics = {
    "commits_created": Counter("git_commits_created_total", "Total Git commits created"),
    "commit_duration_ms": Histogram("git_commit_duration_ms", "Git commit duration in milliseconds"),
    "repo_initialization_count": Counter("git_repos_initialized_total", "Total Git repositories initialized"),
    "repository_size_bytes": Gauge("git_repository_size_bytes", "Git repository size in bytes"),
    "gc_duration_ms": Histogram("git_gc_duration_ms", "Git garbage collection duration"),
    "migration_success_count": Counter("git_migrations_successful_total", "Successful migrations"),
    "migration_failure_count": Counter("git_migrations_failed_total", "Failed migrations"),
}
```

### Logging Strategy

```python
def _log_git_operation(operation: str, document_name: str, duration_ms: float, success: bool):
    """Log Git operations with context."""
    log_structured(
        level="info" if success else "warning",
        message=f"Git operation completed: {operation}",
        context={
            "operation": operation,
            "document_name": document_name,
            "duration_ms": duration_ms,
            "success": success,
            "system": "git_backend"
        }
    )
```

---

## Summary and Recommendations

### Key Benefits of Git-Backed System

1. **Complete History**: Full diffs, not just snapshots
2. **Automatic Tracking**: No manual snapshot creation
3. **Standard Tooling**: Use git log, git diff, etc.
4. **Storage Efficiency**: ~97% smaller repositories
5. **Cross-Session**: History available across sessions
6. **Rich Metadata**: Commit messages with operation context
7. **Blame/Attribution**: See who changed each line
8. **Scalability**: Handles 10,000+ commits efficiently

### Implementation Priority

**Must Have (Phase 1-2):**
- Git initialization and commit creation
- manage_history tool (log, diff, restore)
- check_content_status integration
- Dual-system support

**Should Have (Phase 3-4):**
- Migration tools and automation
- Performance optimization
- Comprehensive testing
- Documentation and guides

**Nice to Have (Future):**
- Pure Python Git library alternative
- Web UI for history visualization
- Collaborative features (merge conflicts)
- Archive/export functionality

### Success Criteria

- [ ] All existing tests pass with Git backend
- [ ] E2E tests cover Git operations
- [ ] Migration succeeds for 100% of test documents
- [ ] Performance regression < 10%
- [ ] Documentation complete
- [ ] User acceptance testing passes

---

## Appendix

### A. Git Command Reference

| Operation | Command |
|-----------|---------|
| Initialize repo | `git init` |
| Configure user | `git config user.name "User"` |
| Stage changes | `git add -A` |
| Commit | `git commit -m "message"` |
| Show history | `git log --oneline` |
| Show diff | `git diff` |
| Restore version | `git checkout <hash> -- .` |
| Get blame | `git blame <file>` |
| Garbage collect | `git gc --aggressive` |

### B. Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GIT_BACKEND_ENABLED` | true | Enable/disable Git backend |
| `GIT_COMMIT_TIMEOUT` | 10s | Max time for commit operations |
| `GIT_GC_THRESHOLD_COMMITS` | 1000 | Commits before auto-gc |
| `GIT_BATCH_COMMIT_ENABLED` | true | Enable batch commit mode |

### C. File Structure After Migration

```
document_name/
├── .git/                    # Git metadata
│   ├── objects/
│   ├── refs/
│   ├── HEAD
│   └── config
├── .gitignore              # Exclude .embeddings, etc
├── .migration_metadata.json # Migration timestamp
├── .snapshot_archive/       # Old snapshots (optional)
├── 01-chapter.md
├── 02-chapter.md
├── .embeddings/
├── summaries/
└── metadata/
```

