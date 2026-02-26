# Deprecation Notice: document-mcp → story-mcp

## Timeline

| Version | Date | Status | Action |
|---------|------|--------|--------|
| **v0.0.5+** | Now | Current | Both names work (old shows warnings) |
| **v0.0.6 - v0.0.9** | Ongoing | Maintenance | Both names work (old shows warnings) |
| **v1.0.0** | ~Aug 2026 | Final | Old names removed |

## What's Changing?

The `document-mcp` project is being renamed to `story-mcp` to better reflect its focus on narrative and storytelling management.

### What's Affected

| Item | Old | New |
|------|-----|-----|
| PyPI package name | `document-mcp` | `story-mcp` |
| Python module | `document_mcp` | `story_mcp` |
| CLI command | `document-mcp` | `story-mcp` |
| GitHub repo | `document-mcp` | `story-mcp` |
| MCP server name | `document-mcp` | `story-mcp` |

### What's NOT Changing

- All functionality is identical
- All file formats are unchanged
- All APIs are the same
- Document storage (`.documents_storage/`) is unchanged
- Configuration options are unchanged

## Migration Status

### Phase 1 (v0.0.5): ✅ Complete
- Package renamed to `story-mcp`
- Backward compatibility layer created
- Deprecation warnings activated

### Phase 2 (v0.0.6+): 🔄 In Progress
- Documentation migration
- User communication
- Monitoring usage of old names

### Phase 3 (v1.0.0): 📋 Planned
- Complete removal of old names
- Final cleanup

## Action Items by User Type

### For Active Developers

**Action:** Migrate immediately to new names

```diff
- from document_mcp import tools
+ from story_mcp import tools

- pip install document-mcp
+ pip install story-mcp

- "command": "document-mcp"
+ "command": "story-mcp"
```

**Timeline:** Do this before v1.0.0 (approx. August 2026)

### For Maintenance-Mode Projects

**Action:** Plan migration within next 6 months

You have until v1.0.0. Set a calendar reminder for July 2026 to migrate.

### For Inactive/Archived Projects

**Action:** No immediate action required

If you reactivate development before v1.0.0, migrate at that time.

## Backward Compatibility Details

### During Deprecation Period (v0.0.5 - v0.0.9)

Both old and new names work:

```python
# These work identically
from document_mcp import tools        # Shows deprecation warning
from story_mcp import tools           # Recommended

# Both work in CLI
document-mcp stdio                    # Shows deprecation warning
story-mcp stdio                       # Recommended
```

### After Deprecation (v1.0.0+)

Only new names work:

```python
from story_mcp import tools           # Works
from document_mcp import tools        # ImportError
```

## Why the Rename?

The original name `document-mcp` was generic. As the project evolved, it became clear that the system excels at managing **structured narratives**:

- Novels and screenplays
- Interactive fiction
- Technical documentation with narrative structure
- Research papers with chapter organization
- Any multi-chapter content

The new name `story-mcp` better reflects this focus while remaining accessible for non-fiction and technical writing.

## Migration Guide

See **[docs/STORY_MCP_MIGRATION.md](STORY_MCP_MIGRATION.md)** for:
- Detailed migration instructions
- Code examples
- Common scenarios
- Automated migration helpers
- Troubleshooting

## Support

**Have questions?**
1. Review [docs/STORY_MCP_MIGRATION.md](STORY_MCP_MIGRATION.md)
2. Check inline docstrings in module files
3. File an issue on GitHub
4. Check existing issues for common problems

## FAQs

### Q: Will my existing projects break?

**A:** No. During the deprecation period (v0.0.5 - v0.0.9), old imports work fine. You'll see deprecation warnings, but everything functions normally.

### Q: How long do I have?

**A:** The deprecation period lasts approximately 6 months, ending with v1.0.0 (estimated August 2026). After that, old names won't work.

### Q: Can I mix old and new names?

**A:** Yes, during the deprecation period, you can use both in the same project as you migrate incrementally.

### Q: Do I need to change my data?

**A:** No. Your document storage (`.documents_storage/`) and all stored files are completely unchanged.

### Q: What about CI/CD pipelines?

**A:** Update them to use `pip install story-mcp` instead of `pip install document-mcp`. Both work during the deprecation period.

## Deprecation Warning Examples

When using old names, you'll see warnings like:

```
DeprecationWarning: The 'document_mcp' package is deprecated and will be removed
in version 1.0. Please migrate to 'story_mcp'...
```

These warnings:
- Only appear once per import (not cluttering your output)
- Include migration guide references
- Don't affect functionality
- Help identify what needs updating

## Related Documentation

- **[Story MCP Migration Guide](STORY_MCP_MIGRATION.md)** - Detailed migration instructions
- **[Release Notes](../README.md)** - Version history and changes
- **[Technical Reference](./docs/)** - Full documentation

---

**Version:** 1.0
**Last Updated:** February 25, 2026
**Deprecation Timeline:** 6 months (through approximately August 2026)
