# Architecture Ideas & Future Patterns

**Status**: Research Notes | **Last Updated**: December 27, 2024

This document captures architecture patterns and research findings for potential future enhancements. These are ideas explored but **not currently prioritized** for implementation.

---

## 1. Dynamic Tool Execution (Code Mode)

**Concept**: Instead of exposing 32 tool schemas, agent writes code that calls tools dynamically.

**How It Works**:
```python
# Instead of: agent.call_tool("insert_paragraph_after", {...params})
# Agent generates:
async def execute_task():
    tools = await mcp.list_tools()
    paragraph_tool = find_tool(tools, "insert_paragraph")
    return await mcp.call_tool(paragraph_tool.name, {
        "document_name": "novel",
        "chapter_name": "01-intro.md",
        "paragraph_index": 2,
        "new_content": "New text"
    })
```

**Token Reduction Analysis**:
| Approach | Tokens per Request | Reduction |
|----------|-------------------|-----------|
| All 32 tools | ~15,000 | Baseline |
| Core + Deferred | ~2,500 | 83% |
| Code Mode | ~200 | 98.7% |

**Why NOT Recommended for Document MCP**:

1. **Security Concerns**: Executing LLM-generated code requires sandboxing
2. **Current Solution Sufficient**: Progressive disclosure already achieves 83% reduction
3. **Complexity**: Requires code execution environment, error handling, validation
4. **Debugging**: Harder to trace issues in generated code vs direct tool calls

**When It Would Be Valuable**:
- Batch operations across many documents
- Complex multi-step workflows
- Power users who want programmatic control

**Decision**: Not implementing. Progressive disclosure (`search_tool` + deferred loading) provides sufficient token optimization without code execution complexity.

---

## 2. Meta-Tool Pattern

Inspired by [programmatic-mcp-prototype](https://github.com/domdomegg/programmatic-mcp-prototype).

**Current State**: 32 tools exposed directly to agent.

**Meta-Tool Pattern**: Reduce to 2-3 meta-tools:
```
search_tool(query)                → Already have ✅
execute_tool(tool_name, params)   → Not implemented ❌
```

**How It Would Work**:
```python
# Instead of agent selecting from 32 tools directly:
# 1. Agent searches for relevant tool
result = search_tool("insert paragraph")
# Returns: ["insert_paragraph_before", "insert_paragraph_after", "append_paragraph_to_chapter"]

# 2. Agent executes chosen tool dynamically
execute_tool("insert_paragraph_after", {
    "document_name": "novel",
    "chapter_name": "01-intro.md",
    "paragraph_index": 2,
    "new_content": "New paragraph text"
})
```

**Benefits**:
- Reduced token overhead (only 2 tool schemas vs 32)
- Agent discovers tools on-demand
- Flexible extension without agent retraining

**Risks**:
- Loss of type safety in tool parameters
- Additional round-trip for discovery + execution
- More complex error handling

**Decision**: Only implement if benchmarks show tool selection is a bottleneck.

---

## 3. Batch Operations & Skills

**Problem**: Multi-step workflows require many round-trips.

**Example Workflows**:
```
"Update status of all draft chapters to revised"
→ list_chapters → loop → write_metadata (N times)

"Move all mentions of Marcus to a new chapter"
→ find_entity → loop → read_content + append + delete (N×3 times)

"Generate summaries for all chapters"
→ list_chapters → loop → read_content + write_summary (N×2 times)
```

**Possible Solutions**:

### Option A: Batch Parameter Support
Add batch mode to existing tools:
```python
write_metadata(
    document_name="novel",
    scope="chapter",
    targets=["01-intro.md", "02-middle.md", "03-end.md"],  # Batch!
    status="revised"
)
```

### Option B: Workflow/Skill Tools
Pre-built multi-step operations:
```python
batch_update_chapter_status(document_name, from_status, to_status)
move_entity_mentions(document_name, entity_name, to_chapter)
generate_all_summaries(document_name)
```

### Option C: Programmatic Composition
Agent writes code that chains tools:
```typescript
const chapters = await list_chapters("novel");
for (const ch of chapters) {
    if (ch.status === "draft") {
        await write_metadata("novel", "chapter", ch.name, { status: "revised" });
    }
}
```

**Trade-off Analysis**:
| Approach | Complexity | Flexibility | Safety | Value |
|----------|------------|-------------|--------|-------|
| Batch params | Low | Medium | High | Medium |
| Skill tools | Medium | Low (fixed workflows) | High | Medium |
| Code composition | High | Very High | Low (code exec) | High for complex ops |

**Recommendation**: Start with batch parameter support (Option A) for common operations. Monitor usage to identify which skill tools would be valuable.

---

## 4. Workspace State Persistence

**Current**: Only snapshots (version control) persist between calls. Existing observability via `@log_mcp_call` decorator captures tool usage, duration, and errors.

**Concept**: Workspace = Session from Claude-Mem. A workspace persists across multiple tool calls within a logical working session on a document.

### Workspace Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    WORKSPACE LIFECYCLE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. WORKSPACE START                                             │
│     └── Triggered by: First tool call on a document            │
│     └── Creates: .workspace/ directory with metadata.json      │
│     └── Records: session_id, start_time, document_name         │
│                                                                 │
│  2. ACTIVE WORKSPACE                                            │
│     └── Each tool call: Updates workspace state                 │
│     └── Caches: Search results, entity analysis, statistics    │
│     └── Tracks: Pending operations, undo stack                  │
│                                                                 │
│  3. WORKSPACE END (Optional)                                    │
│     └── Triggered by: Explicit close OR timeout (30 min idle)  │
│     └── Creates: workspace_summary.md (optional, see below)    │
│     └── Cleans up: Temporary caches, retains key state         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
.documents_storage/
├── document_name/
│   └── .workspace/                    # Active workspace state
│       ├── metadata.json              # Session metadata
│       ├── context_cache.json         # Cached search results, entity analysis
│       ├── pending_operations.json    # Multi-step operation tracking
│       └── workspace_summary.md       # Optional: Generated on workspace end
```

### metadata.json Schema

```json
{
  "workspace_id": "ws_20251226_143052_abc123",
  "document_name": "my-novel",
  "created_at": "2025-12-26T14:30:52Z",
  "last_activity": "2025-12-26T15:45:00Z",
  "tool_calls_count": 15,
  "state": "active"
}
```

### Context Cache Schema

```json
{
  "search_results": {
    "find_entity:Marcus": {
      "result": [...],
      "cached_at": "2025-12-26T14:35:00Z",
      "ttl_minutes": 30
    }
  },
  "statistics": {
    "document_outline": {
      "result": {...},
      "cached_at": "2025-12-26T14:32:00Z"
    }
  }
}
```

### Integration with Existing Observability

**Decision**: Do NOT add separate post-tool observation hooks. The existing `@log_mcp_call` decorator already captures:
- Tool name, duration, success/error status
- OpenTelemetry traces with span context
- Local Prometheus metrics via `instrument_tool`

**Workspace adds semantic layer on top**:
- `@log_mcp_call`: Low-level telemetry (timing, errors, traces)
- Workspace: High-level context (what user is working on, cached results)

These are complementary, not duplicative.

### Progressive Disclosure Pattern

When workspace context is loaded into agent prompts, use progressive disclosure (~800 tokens max):

```
[WORKSPACE CONTEXT]
Document: my-novel (15 chapters, 45K words)
Session: Active for 15 min, 8 tool calls
Recent: find_entity("Marcus") → 12 mentions across 5 chapters
Cached: document_outline (valid), entity_list (valid)
```

This provides just enough context for LLM to make informed decisions without overwhelming the prompt.

### Workspace Summary (Optional)

**Purpose**: Generate a human-readable summary when workspace ends, useful for resuming work later.

**Format**:
```markdown
# Workspace Summary: my-novel
Session: 2025-12-26 14:30 - 15:45 (1h 15min)

## Key Actions
- Searched for entity "Marcus" (12 mentions found)
- Moved 3 paragraphs in chapter 5
- Updated chapter 6 metadata (status: revised)

## Pending Work
- Chapter 7 has unresolved timeline conflict
- Entity "Elena" mentioned but not in entities.yaml
```

**Note**: This is OPTIONAL and should be user-configurable.

### Implementation Priority

| Feature | Priority | Rationale |
|---------|----------|-----------|
| metadata.json + lifecycle | P0 | Foundation for all workspace features |
| context_cache.json | P1 | Immediate value: reduces repeated expensive queries |
| pending_operations.json | P2 | Enables multi-step workflow tracking |
| workspace_summary.md | P3 | Optional, nice-to-have for session continuity |
| Progressive disclosure | P1 | Critical for LLM context efficiency |

### Privacy Note

**No privacy filtering required for workspace data**. The workspace only contains:
- Document names and metadata (already visible to user)
- Cached query results (already returned by tools)
- Operation history (already in git snapshots)

All workspace data is local to the user's machine.

---

## References

- [Programmatic MCP Prototype](https://github.com/domdomegg/programmatic-mcp-prototype) - Tool composition patterns
- [Claude-Mem](https://github.com/anthropics/claude-mem) - Session/workspace patterns
