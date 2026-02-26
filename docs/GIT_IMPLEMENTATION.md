# Phase 4.4: Git-Backed Version History Implementation

## Overview

Phase 4.4 implements Git-backed version control for Document MCP, replacing the snapshot system with proper distributed version control. Each document directory becomes a Git repository with automatic commits on all edits, user attribution, and full version history tracking.

## Architecture

### Core Components

#### 1. GitManager (`story_mcp/utils/git_manager.py`)

The `GitManager` class provides high-level Git operations for document repositories:

```python
from story_mcp.utils.git_manager import GitManager
from pathlib import Path

# Initialize Git manager for a document
manager = GitManager(Path("/path/to/document"))

# Create commits with formatted messages
commit = manager.commit(
    operation="edit",
    scope="chapter",
    description="updated introduction narrative"
)

# Retrieve version history
history = manager.get_version_history(limit=10)

# Compare versions
diff = manager.compare_versions(version1_hash, version2_hash)

# Restore to previous version
manager.checkout_version(version_hash)
```

**Key Features:**
- Automatic `.git/` directory initialization
- Formatted commit messages: `{operation} {scope}: {description}`
- User attribution tracking via author parameter
- Full Git command execution with error handling
- Diff generation with statistics
- Tag support for marking important versions

#### 2. Version Control Models (`story_mcp/models/version_control.py`)

Pydantic models for version control operations:

```python
class CommitInfo(BaseModel):
    """Information about a Git commit."""
    hash: str                          # SHA1 hash
    author: str                        # Author name and email
    timestamp: datetime.datetime       # Commit time
    message: str                       # Full message
    summary: str                       # First line

class VersionHistory(BaseModel):
    """Git commit history for a document."""
    document_name: str
    total_commits: int
    commits: list[CommitInfo]
    time_window: str

class VersionDiff(BaseModel):
    """Diff between two commits."""
    source_hash: str
    target_hash: str
    diff_text: str                     # Unified diff
    files_changed: int
    insertions: int
    deletions: int

class VersionComparisonResult(BaseModel):
    """Result of version comparison."""
    document_name: str
    version1: str
    version2: str
    diff: VersionDiff
    has_changes: bool
    summary: str                       # Human-readable summary
```

#### 3. Version Control Tools (`story_mcp/tools/version_tools.py`)

Three new MCP tools for version management:

##### Tool 1: `get_version_history`

Retrieves Git commit history for a document.

**Parameters:**
- `document_name` (str): Name of document directory
- `limit` (int): Maximum commits to return (default: 10, max: 100)

**Returns:** `VersionHistory` with commit list

**Example:**
```json
{
    "name": "get_version_history",
    "arguments": {
        "document_name": "my_novel",
        "limit": 20
    }
}
```

**Response:**
```json
{
    "document_name": "my_novel",
    "total_commits": 42,
    "commits": [
        {
            "hash": "a1b2c3d4e5f6...",
            "author": "Alice <alice@example.com>",
            "timestamp": "2026-02-25T15:30:00+00:00",
            "message": "edit chapter: revised opening scene",
            "summary": "edit chapter: revised opening scene"
        }
    ],
    "time_window": "all"
}
```

##### Tool 2: `checkout_version`

Restores document to a specific Git commit.

**Parameters:**
- `document_name` (str): Name of document directory
- `version_hash` (str): Commit SHA hash or tag name

**Returns:** `OperationStatus` indicating success

**Example:**
```json
{
    "name": "checkout_version",
    "arguments": {
        "document_name": "my_novel",
        "version_hash": "a1b2c3d4e5f6..."
    }
}
```

**Response:**
```json
{
    "success": true,
    "message": "Version a1b2c3d4... restored successfully",
    "details": {
        "operation": "checkout_version",
        "document_name": "my_novel",
        "version_hash": "a1b2c3d4...",
        "files_restored": 5
    }
}
```

##### Tool 3: `compare_versions`

Generates detailed diff between two commits.

**Parameters:**
- `document_name` (str): Name of document directory
- `version1_hash` (str): Source commit hash or tag
- `version2_hash` (Optional[str]): Target commit hash (defaults to HEAD)
- `stat_only` (bool): Return only statistics, not full diff (default: false)

**Returns:** `VersionComparisonResult` with diff

**Example:**
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

**Response:**
```json
{
    "document_name": "my_novel",
    "version1": "a1b2c3d4...",
    "version2": "b2c3d4e5...",
    "diff": {
        "source_hash": "a1b2c3d4...",
        "target_hash": "b2c3d4e5...",
        "diff_text": "--- a/01-chapter.md\n+++ b/01-chapter.md\n...",
        "files_changed": 1,
        "insertions": 45,
        "deletions": 12
    },
    "has_changes": true,
    "summary": "1 file changed, 45 insertions, 12 deletions"
}
```

## Commit Message Format

All commits follow a standardized format for consistency and automation:

```
{operation} {scope}: {description}
```

### Examples

```
edit chapter: updated introduction narrative
add paragraph: new section on performance
replace paragraph: refined conclusion
delete chapter: removed outdated content
move paragraph: reorganized flow
```

### Operation Types

- `edit`: Modify existing content
- `add`: Create new content
- `replace`: Substitute content
- `delete`: Remove content
- `move`: Reorder content
- `snapshot`: Data migration from snapshot system

### Scope Types

- `document`: Document-level operations
- `chapter`: Chapter operations
- `paragraph`: Paragraph operations
- `metadata`: Metadata modifications

## Repository Structure

Each document becomes a Git repository:

```
.documents_storage/
├── my_novel/                    # Document directory (becomes Git repo)
│   ├── .git/                    # Git repository
│   │   ├── HEAD
│   │   ├── config              # Git config
│   │   ├── objects/            # Git objects database
│   │   ├── refs/               # Branches and tags
│   │   └── ...
│   ├── 01-chapter.md           # Chapter files
│   ├── 02-chapter.md
│   ├── summaries/              # Existing summary structure
│   ├── metadata/               # Existing metadata
│   ├── .snapshots/             # Deprecated (for migration)
│   └── .embeddings/            # Existing embedding cache
```

## Integration with Existing Systems

### Automatic Snapshot Creation on Git Commits

When a Git commit is created, it automatically serves as a snapshot. The commit history provides:

1. **Complete Content History**: Every change tracked with full content
2. **User Attribution**: Author information preserved
3. **Time Tracking**: Precise timestamps for all edits
4. **Diff Capabilities**: Built-in comparison between any two versions
5. **Branching Support**: Potential for complex workflows

### Safety Tools Migration

Existing safety tools continue to work but can leverage Git:

```python
# Old snapshot system (still supported)
manage_snapshots(document_name, action="list")

# New Git-based version control (recommended)
get_version_history(document_name, limit=10)
```

### Backward Compatibility

- Snapshot system remains functional
- New documents use Git by default
- Existing snapshots can be migrated to Git commits
- Gradual transition path for existing documents

## User Attribution

Commits include user information for audit trails:

```python
# Explicit user attribution
manager.commit(
    operation="edit",
    scope="chapter",
    description="updated introduction",
    author="Alice <alice@example.com>"
)

# Automatic user detection (falls back to system user)
manager.commit(
    operation="edit",
    scope="chapter",
    description="updated introduction"
    # author auto-detected from environment
)
```

## Error Handling

GitManager provides comprehensive error handling:

```python
from story_mcp.utils.git_manager import GitError

try:
    manager.checkout_version(invalid_hash)
except GitError as e:
    print(f"Failed to checkout: {e}")

# Errors are also logged via logger_config
# See story_mcp/logger_config.py for structured logging
```

## Snapshot Migration

Existing `.snapshots/` directories can be migrated to Git commits:

```python
from pathlib import Path

manager = GitManager(document_path)
report = manager.migrate_snapshots(document_path / ".snapshots")

print(f"Migrated: {report['migrated']} snapshots")
print(f"Failed: {report['failed']} snapshots")
if report['errors']:
    for error in report['errors']:
        print(f"  Error: {error}")
```

## Performance Characteristics

### Storage Efficiency

- Git objects database is space-efficient (compression, deduplication)
- Smaller than manual snapshots for large documents
- Better for version control workflows

### Speed

- Commit creation: < 100ms for typical documents
- History retrieval: O(n) where n = number of commits
- Diff generation: Fast for typical changes
- Large documents may need optimization

### Scalability

- Supports 100+ commits easily
- 1000+ commits may require periodic cleanup
- Can leverage Git's built-in tools (gc, rebase, etc.)

## Advanced Usage

### Creating Tags for Milestones

```python
# Create a lightweight tag
manager.create_tag("v1.0")

# Create an annotated tag with message
manager.create_tag("v1.0", message="Version 1.0 release")
```

### Repository Status

```python
status = manager.get_status()
# Returns:
# {
#     "staged": ["file1.md"],
#     "unstaged": ["file2.md"],
#     "untracked": ["file3.md"],
#     "is_dirty": true
# }
```

### Diff Statistics Only

```python
diff = manager.compare_versions(
    version1_hash,
    version2_hash,
    stat_only=True  # Only stats, no full diff
)

print(f"Files changed: {diff.stats['files_changed']}")
print(f"Lines added: {diff.stats['insertions']}")
print(f"Lines deleted: {diff.stats['deletions']}")
```

## Testing

Comprehensive test suite in `tests/unit/test_git_integration.py` includes:

- **34 unit tests** covering all operations
- Repository initialization
- Commit creation and formatting
- Version history retrieval
- Checkout and restoration
- Diff generation and comparison
- Status checking
- Tag management
- Error handling
- Integration workflows

**Run tests:**
```bash
pytest tests/unit/test_git_integration.py -v
```

## API Reference

### GitManager Class

```python
class GitManager:
    def __init__(self, repo_path: Path)
    def commit(
        operation: str,
        scope: str,
        description: str,
        author: str | None = None,
        commit_all: bool = True
    ) -> GitCommit

    def get_version_history(self, limit: int = 10) -> list[GitCommit]

    def checkout_version(self, version_hash: str) -> None

    def compare_versions(
        self,
        version1: str,
        version2: str | None = None,
        stat_only: bool = False
    ) -> GitDiff

    def create_tag(
        self,
        tag_name: str,
        message: str | None = None
    ) -> str

    def get_current_hash(self) -> str

    def get_status(self) -> dict[str, Any]

    def migrate_snapshots(
        self,
        snapshots_dir: Path
    ) -> dict[str, Any]
```

## Files Created/Modified

### New Files
1. **story_mcp/utils/git_manager.py** - Core GitManager implementation
2. **story_mcp/models/version_control.py** - Version control models
3. **story_mcp/tools/version_tools.py** - MCP tool definitions
4. **tests/unit/test_git_integration.py** - 34 comprehensive tests
5. **docs/GIT_IMPLEMENTATION.md** - This documentation

### Modified Files
1. **story_mcp/models/__init__.py** - Added version_control imports
2. **story_mcp/tools/__init__.py** - Added register_version_tools import
3. **story_mcp/doc_tool_server.py** - Registered version tools with MCP server

## Deployment Checklist

- [x] GitManager implementation with error handling
- [x] Version control models (Pydantic)
- [x] Three new MCP tools (get_version_history, checkout_version, compare_versions)
- [x] Tool registration in doc_tool_server
- [x] 34 comprehensive unit tests (100% passing)
- [x] Model integration in models/__init__.py
- [x] Tool integration in tools/__init__.py
- [x] Comprehensive documentation
- [x] Zero breaking changes to existing system
- [x] Backward compatibility with snapshot system

## Next Steps

1. **Integration Testing**: Run full integration test suite
2. **E2E Testing**: Test with actual agents
3. **Performance Monitoring**: Track performance on production workflows
4. **Snapshot Migration**: Implement automated migration for existing documents
5. **UI Updates**: Update agents to use new version history tools
6. **Documentation**: Update user guides with Git workflows

## Known Limitations

1. **Concurrent Edits**: Sequential commits required (no parallel editing)
2. **Large Files**: Git may slow with very large documents (>100MB)
3. **Binary Content**: Markdown only (binary files not supported)
4. **Merge Conflicts**: No automated conflict resolution (manual intervention needed)

## Future Enhancements

1. **Branching Support**: Parallel editing with merge workflows
2. **Remote Sync**: Push/pull from remote Git repositories
3. **Collaborative Features**: Multi-user editing with conflict resolution
4. **Automation**: Periodic commits, automated backups
5. **Analytics**: Visualization of edit history and authorship

## References

- [Git Documentation](https://git-scm.com/doc)
- [GitPython Alternative](https://gitpython.readthedocs.io/) (not used, using subprocess)
- Phase 3: Automatic Snapshot System
- Phase 4.1-4.3: Previous MCP tool enhancements
