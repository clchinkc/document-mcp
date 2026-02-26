# Story-MCP Migration Guide

## Phase 4.5: Story MCP Rename Strategy

This document provides guidance for migrating from `document-mcp` to `story-mcp`. The rename reflects the project's evolution from generic document management to specialized story and narrative management.

### What Changed?

| Component | Old Name | New Name | Status |
|-----------|----------|----------|--------|
| **Python Package** | `document_mcp` | `story_mcp` | ✅ New |
| **PyPI Package** | `document-mcp` | `story-mcp` | ✅ New (old kept as deprecated alias) |
| **CLI Command** | `document-mcp` | `story-mcp` | ✅ New (old still works) |
| **Class Names** | `DocumentMCP*` | `StoryMCP*` | ✅ Updated in code |
| **Default Behavior** | Unchanged | Unchanged | ✅ Fully backward compatible |

### Migration Timeline

**Deprecation Period: 6 months (approximately)**

```
v0.0.5 (Now)          → Both work, old shows warnings
  ↓
v0.0.6-0.0.9         → Both work, old shows warnings
  ↓
v1.0.0               → Old removed, only new available
```

### Why the Rename?

The original `document-mcp` name was too generic. As the project evolved, it became clear that the system excels at managing **structured narratives** - novels, screenplays, interactive stories, and technical documentation with narrative structure.

The new name `story-mcp` better reflects:
- Focus on narrative and storytelling
- Multi-chapter document structures
- Creative and technical writing workflows
- Rich content management for structured stories

### Zero Breaking Changes

**Everything works exactly the same.** This is a pure rename with full backward compatibility:

- All APIs are identical
- All file formats are identical
- All functionality is identical
- Old imports still work (with deprecation warnings)
- Old CLI command still works (with deprecation warnings)

### Migration Paths

Choose the migration path that fits your usage:

#### Option 1: Immediate Migration (Recommended)

If you're starting fresh or actively maintaining your code:

**1. Update Python imports:**

```python
# Old (still works, but deprecated)
from document_mcp import tools
from document_mcp.models import Document

# New (recommended)
from story_mcp import tools
from story_mcp.models import Document
```

**2. Update CLI usage:**

```bash
# Old (still works, shows deprecation warning)
document-mcp stdio

# New (recommended)
story-mcp stdio
```

**3. Update installation:**

```bash
# Old (still works as alias)
pip install document-mcp

# New (recommended)
pip install story-mcp
```

**4. Update MCP configuration:**

If you use Claude Code or other MCP clients:

```json
{
  "mcpServers": {
    "story-mcp": {
      "command": "story-mcp",
      "args": ["stdio"]
    }
  }
}
```

#### Option 2: Gradual Migration

If you have a large codebase or want to migrate incrementally:

**Phase 1: Install new package alongside old**

```bash
# Install both (both work in parallel)
pip install story-mcp
# document-mcp will still work via compatibility shim
```

**Phase 2: Update imports gradually**

You can update imports one file at a time. Both old and new imports work:

```python
# Old code still works
from document_mcp import tools

# New code
from story_mcp import tools

# Both can coexist during migration
```

**Phase 3: Remove old references**

Once all code is updated, uninstall the old package:

```bash
pip uninstall document-mcp
```

#### Option 3: Lazy Migration

If your code is stable and you don't actively develop:

Just keep using the old imports. They'll continue to work during the deprecation period.

**⚠️ Note:** After v1.0.0, the old imports won't work. Plan your migration accordingly.

### Common Migration Scenarios

#### Scenario 1: Simple Agent Using story-mcp

**Before:**
```python
from document_mcp import tools
from document_mcp.models import Document

def process_document(name: str) -> Document:
    return tools.read_document(name)
```

**After:**
```python
from story_mcp import tools
from story_mcp.models import Document

def process_document(name: str) -> Document:
    return tools.read_document(name)
```

#### Scenario 2: Custom MCP Integration

**Before:**
```python
from document_mcp.doc_tool_server import MCP_TOOLS
from document_mcp.config import get_storage_root

server = MCPServer()
for tool in MCP_TOOLS:
    server.register_tool(tool)
```

**After:**
```python
from story_mcp.doc_tool_server import MCP_TOOLS
from story_mcp.config import get_storage_root

server = MCPServer()
for tool in MCP_TOOLS:
    server.register_tool(tool)
```

#### Scenario 3: Direct CLI Usage

**Before:**
```bash
document-mcp stdio
# or
python -m document_mcp.doc_tool_server stdio
```

**After:**
```bash
story-mcp stdio
# or
python -m story_mcp.doc_tool_server stdio
```

#### Scenario 4: Claude Code MCP Configuration

**Before:**
```json
{
  "mcpServers": {
    "document-mcp": {
      "command": "document-mcp",
      "args": ["stdio"]
    }
  }
}
```

**After:**
```json
{
  "mcpServers": {
    "story-mcp": {
      "command": "story-mcp",
      "args": ["stdio"]
    }
  }
}
```

### Backward Compatibility Details

The old `document_mcp` package name is maintained as a **compatibility shim** that automatically redirects to `story_mcp`:

```python
# This still works:
from document_mcp import tools

# Under the hood, it's equivalent to:
from story_mcp import tools
```

**How it works:**
1. `document_mcp/__init__.py` imports everything from `story_mcp`
2. The old CLI command `document-mcp` is aliased to the new command
3. Both import paths work identically
4. Deprecation warnings inform users of the change

**When it stops working:**
- In version 1.0.0, the compatibility shim will be removed
- After that, only `story_mcp` imports and `story-mcp` CLI will work
- Attempting to use old imports will raise ImportError

### Deprecation Warnings

When you use the old names, you'll see warnings:

```python
>>> from document_mcp import tools
DeprecationWarning: The 'document_mcp' package is deprecated and will be removed
in version 1.0. Please migrate to 'story_mcp' (e.g., 'from story_mcp import ...'
instead of 'from document_mcp import ...'). See docs/STORY_MCP_MIGRATION.md for
migration guide.
```

These warnings:
- Only appear once per import
- Include the migration guide URL
- Don't affect functionality
- Help you identify what needs updating

### Automated Migration Helper

For large codebases, use a simple find-and-replace to update imports:

**Using sed (macOS/Linux):**
```bash
# Update all Python files
find . -name "*.py" -type f -exec sed -i '' 's/from document_mcp/from story_mcp/g' {} \;
find . -name "*.py" -type f -exec sed -i '' 's/import document_mcp/import story_mcp/g' {} \;
```

**Using PowerShell (Windows):**
```powershell
Get-ChildItem -Path . -Filter "*.py" -Recurse |
  ForEach-Object {
    (Get-Content $_) -replace 'from document_mcp', 'from story_mcp' | Set-Content $_
    (Get-Content $_) -replace 'import document_mcp', 'import story_mcp' | Set-Content $_
  }
```

**Using regex in your IDE:**
- Find: `(from|import) document_mcp`
- Replace: `$1 story_mcp`

### Troubleshooting Migration

#### Q: Will the old imports continue to work after v1.0?

**A:** No. The deprecation period lasts approximately 6 months. After v1.0.0, old imports will not work. Plan your migration before then.

#### Q: Can I use both old and new imports in the same project?

**A:** Yes! During the deprecation period, old and new imports work together. Mix and match as needed during migration.

#### Q: What if I see multiple deprecation warnings?

**A:** This is normal. You'll see one warning per import statement. Use an IDE search to find all occurrences and batch-update them.

#### Q: Do I need to update my data storage?

**A:** No. The `.documents_storage/` directory and all stored documents are unchanged. The rename is purely in the Python package namespace.

#### Q: Should I update my pyproject.toml?

**A:** Yes, update the dependency:
```toml
# Old
dependencies = ["document-mcp>=0.0.5"]

# New
dependencies = ["story-mcp>=0.0.5"]
```

#### Q: What about GitHub Actions / CI/CD?

**A:** Update your workflows to use `pip install story-mcp` instead of `pip install document-mcp`. Both will work during the deprecation period.

### Support

**Questions about migration?**
- Review this guide for common scenarios
- Check the inline docstrings in `story_mcp/` modules
- See `/docs/` for comprehensive documentation
- File an issue on GitHub with migration questions

### Version-Specific Information

#### v0.0.5 (Current)

- Both `document-mcp` and `story-mcp` available
- Old imports show deprecation warnings
- Full backward compatibility
- 6-month deprecation period begins

#### v0.0.6-0.0.9

- Both names continue to work
- Deprecation warnings persist
- No breaking changes

#### v1.0.0 (Estimated: August 2026)

- `document-mcp` package removed
- `document_mcp` Python package removed
- Old imports will raise ImportError
- Only `story-mcp` and `story_mcp` available

### Deprecation Notice

```
⚠️ DEPRECATION NOTICE

The 'document-mcp' package is deprecated as of v0.0.5.
It will be removed completely in v1.0.0 (approximately 6 months).

Migration required:
  - Update imports: document_mcp → story_mcp
  - Update CLI: document-mcp → story-mcp
  - Update config: references to document-mcp → story-mcp

Timeline:
  - Now: Deprecation period starts (warnings shown)
  - v1.0.0: Complete removal (no old names available)

For detailed guidance, see: docs/STORY_MCP_MIGRATION.md
```

---

**Last Updated:** February 25, 2026
**Deprecation Period:** 6 months (approx. through August 2026)
**Full Removal:** v1.0.0
