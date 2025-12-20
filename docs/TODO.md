# Document MCP Development Roadmap

**Last Updated**: December 17, 2025
**Status**: Production Ready (352 tests passing) - Enhancement Phase

This document outlines the development priorities for evolving Document MCP into a comprehensive story writing and document management platform.

---

## Design Philosophy

### Writer Mental Model Alignment

Writers think in **two parallel tracks**:
1. **Narrative Content** - Prose organized hierarchically (Acts/Parts → Chapters → Scenes → Paragraphs)
2. **Story Bible** - Reference metadata (Characters, Locations, Timeline, Worldbuilding Rules)

Document MCP currently excels at Track 1. Track 2 (Story Bible / Codex) is an enhancement opportunity.

### Key Writer Needs Addressed

| Writer Need | Current MCP Feature | Status |
|-------------|---------------------|--------|
| Hierarchical organization | Document → Chapter → Paragraph | ✅ Strong |
| Version control | Automatic snapshots on every edit | ✅ Strong |
| Content search | `find_text` with scope targeting | ✅ Strong |
| Semantic discovery | `find_similar_text` with embeddings | ✅ Strong |
| Progressive loading | Pagination (50K chars/page) | ✅ Strong |
| Chapter summaries | `summaries/` directory | ✅ Strong |
| Entity tracking | "When did I last mention X?" | ❌ Gap |
| Story Bible storage | Characters, timeline, worldbuilding | ❌ Gap |
| Outline view | All chapter synopses at once | ❌ Gap |

---

## Phase 1: MCP Standards Compliance (Priority: Critical)

### 1.1 Streamable HTTP Transport Migration
**Effort**: 2-3 days | **Value**: Critical

The current SSE transport was **deprecated on March 26, 2025**. Migration to Streamable HTTP (MCP spec 2025-06-18) is required for compatibility with Claude Desktop and other MCP clients.

**Current State**:
- Uses FastMCP with HTTP SSE transport
- Two endpoints (`/sse`, `/messages`)

**Target State**:
- Single `/mcp` endpoint handling both POST and GET
- Session management via `Mcp-Session-Id` header
- Origin validation for security

**Implementation Details**:
```
POST /mcp  → Client sends JSON-RPC requests
GET /mcp   → Client listens for server-initiated messages (Accept: text/event-stream)
```

**Key Changes**:
- [ ] Update transport configuration in `doc_tool_server.py`
- [ ] Implement session ID handling
- [ ] Add Origin header validation middleware
- [ ] Update connection documentation
- [ ] Test with Claude Desktop

**References**:
- [MCP Spec 2025-06-18](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports)
- [Why MCP deprecated SSE](https://blog.fka.dev/blog/2025-06-06-why-mcp-deprecated-sse-and-go-with-streamable-http)

### 1.2 Tool Description Optimization
**Effort**: 1 day | **Value**: High

Audit and rewrite all 26 tool descriptions for LLM agent comprehension.

**Current Issue**: Tool descriptions may be written for human developers rather than LLM agents.

**Optimization Format** (per tool):
```
WHAT: Primary action + target object
WHEN: Trigger conditions / user intents that warrant this tool
RETURNS: Expected output + key information fields
AUTO: Edge cases handled automatically
```

**Example Before**:
```
"find_text_in_document: Finds text within a document or specific scope"
```

**Example After**:
```
"find_text_in_document: Search for exact text matches within a document, chapter, or paragraph. Use when user asks 'where did I mention X?' or needs to locate specific content. Returns match locations with surrounding context (paragraph index, chapter name). Automatically handles case-insensitive matching and partial word boundaries."
```

**Tasks**:
- [ ] Audit all 26 tool descriptions in `tool_descriptions.py`
- [ ] Apply WHAT/WHEN/RETURNS/AUTO format
- [ ] Test with both Simple and ReAct agents
- [ ] Measure token usage before/after

### 1.3 Defer Loading Implementation
**Effort**: 1-2 days | **Value**: High

Implement Claude's advanced tool use pattern to reduce context consumption by **85%**.

**Concept**: Mark advanced tools with `defer_loading: true` so they're not loaded into context until needed. Provide a `search_available_tools` meta-tool for discovery.

**Tool Categorization**:

| Category | Tools | Loading |
|----------|-------|---------|
| **Core** (always loaded) | list_documents, read_chapter_content, create_document | Immediate |
| **Discovery** | search_available_tools | Immediate |
| **Content** (defer) | find_text, find_similar_text, replace_text, get_statistics | Deferred |
| **Paragraph** (defer) | All 7 paragraph tools | Deferred |
| **Safety** (defer) | manage_snapshots, diff_content, get_modification_history | Deferred |
| **Advanced** (defer) | All remaining tools | Deferred |

**New Tool**: `search_available_tools`
```python
@mcp.tool()
def search_available_tools(
    query: str,
    category: str = None
) -> list[dict]:
    """
    Search available tools by capability.
    Returns tool names, descriptions, and categories matching the query.
    Use this to discover tools before calling them.
    """
```

**Tasks**:
- [ ] Add `defer_loading` field to tool metadata
- [ ] Implement `search_available_tools` tool
- [ ] Categorize all 26 tools by loading priority
- [ ] Update tool registration in `doc_tool_server.py`
- [ ] Test that deferred tools are loaded on-demand

**Reference**: [Anthropic Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)

---

## Phase 2: Writer-Focused Enhancements (Priority: High)

### 2.1 Entity Tracking Tool
**Effort**: 2 days | **Value**: High

Writers constantly ask "When did I last mention this character?" Create a dedicated tool for tracking entity occurrences.

**New Tool**: `track_entity`

**Input**:
```json
{
  "document_name": "string",
  "entity_name": "string",
  "entity_type": "character|location|object|concept|any"
}
```

**Output**:
```json
{
  "entity": "Marcus",
  "total_mentions": 47,
  "first_mention": {
    "chapter": "01-intro",
    "paragraph": 3,
    "context": "...Marcus entered the room..."
  },
  "last_mention": {
    "chapter": "12-climax",
    "paragraph": 8,
    "context": "...Marcus knew this was the end..."
  },
  "by_chapter": [
    {"chapter": "01-intro", "count": 5, "first_paragraph": 3},
    {"chapter": "02-rising", "count": 12, "first_paragraph": 1}
  ],
  "total_chapters_mentioned": 8
}
```

**Implementation**:
- [ ] Create `track_entity` tool in `content_tools.py`
- [ ] Add case-insensitive name matching
- [ ] Include context snippets (±50 chars)
- [ ] Support fuzzy matching for name variants (e.g., "Marcus" / "Marc")
- [ ] Add unit tests
- [ ] Update tool descriptions

### 2.2 Story Bible / Codex Storage
**Effort**: 2 days | **Value**: Medium-High

Extend the storage model to support structured metadata separate from narrative content.

**Current Storage**:
```
document_name/
├── 01-chapter.md
├── summaries/
├── .snapshots/
└── .embeddings/
```

**Proposed Storage**:
```
document_name/
├── 01-chapter.md
├── summaries/
├── codex/                    # NEW: Story Bible
│   ├── characters.yaml       # Character profiles
│   ├── locations.yaml        # Setting details
│   ├── timeline.yaml         # Event chronology
│   ├── worldbuilding.yaml    # Rules, magic systems, etc.
│   └── glossary.yaml         # Terms, names, spelling
├── .snapshots/
└── .embeddings/
```

**New Tools**:
- `read_codex_item(document_name, category, item_name)` → Read a codex entry
- `write_codex_item(document_name, category, item_name, content)` → Create/update entry
- `list_codex_items(document_name, category?)` → List all items, optionally filtered
- `search_codex(document_name, query)` → Semantic search across codex

**YAML Schema Example** (characters.yaml):
```yaml
characters:
  - name: "Marcus Chen"
    aliases: ["Marc", "The Detective"]
    role: "protagonist"
    physical:
      age: 35
      appearance: "Tall, dark hair, scar on left cheek"
    personality:
      traits: ["determined", "cynical", "secretly kind"]
      motivations: ["Find his sister's killer", "Redemption"]
    relationships:
      - target: "Sarah Chen"
        type: "sister (deceased)"
      - target: "Detective Wong"
        type: "partner"
    first_appearance: "01-intro"
    arc_notes: "Starts cynical, learns to trust again"
```

**Tasks**:
- [ ] Define YAML schemas for each codex category
- [ ] Implement `codex/` directory management
- [ ] Create 4 new codex tools
- [ ] Add validation for codex entries
- [ ] Integrate with semantic search (reuse embedding system)
- [ ] Add unit and integration tests

### 2.3 Outline View Tool
**Effort**: 1 day | **Value**: Medium

Provide a high-level structural overview of the document, similar to Scrivener's corkboard view.

**New Tool**: `get_document_outline`

**Output**:
```json
{
  "document_name": "My Novel",
  "total_chapters": 12,
  "total_words": 45000,
  "chapters": [
    {
      "name": "01-opening",
      "synopsis": "Hero discovers the ancient map in grandmother's attic",
      "word_count": 2500,
      "paragraph_count": 15,
      "status": "revised",
      "pov_character": "Marcus",
      "timeline_position": "Day 1"
    },
    {
      "name": "02-journey-begins",
      "synopsis": "Travel to the mountain begins, first obstacle encountered",
      "word_count": 3200,
      "paragraph_count": 18,
      "status": "draft",
      "pov_character": "Marcus",
      "timeline_position": "Day 3"
    }
  ]
}
```

**Tasks**:
- [ ] Create `get_document_outline` tool
- [ ] Aggregate chapter summaries if available
- [ ] Include word counts and paragraph counts
- [ ] Add optional metadata fields (status, pov, timeline)
- [ ] Format output for easy LLM consumption

### 2.4 Chapter Metadata Extension
**Effort**: 1 day | **Value**: Medium

Extend chapter files to support optional metadata header.

**Format** (Markdown frontmatter):
```markdown
---
status: draft|revised|complete
pov_character: Marcus
tags: [flashback, action, dialogue-heavy]
timeline_position: Day 3, Morning
word_target: 3000
---

# Chapter Title

Chapter content here...
```

**New Tools/Extensions**:
- `read_chapter_metadata(document_name, chapter_name)` → Get metadata only
- `write_chapter_metadata(document_name, chapter_name, metadata)` → Update metadata
- `list_chapters_by_status(document_name, status)` → Filter by status
- `list_chapters_by_tag(document_name, tag)` → Filter by tag

**Tasks**:
- [ ] Implement YAML frontmatter parsing in chapter tools
- [ ] Create metadata-specific tools
- [ ] Update existing read_chapter_content to optionally include metadata
- [ ] Add filtering capabilities
- [ ] Preserve metadata during chapter edits

---

## Phase 3: Advanced Features (Priority: Medium)

### 3.1 FastMCP 2.0 Upgrade
**Effort**: 1 day | **Value**: Medium

Upgrade to FastMCP 2.0+ (current: 2.14.1) for production features.

**Benefits**:
- Server composition (proxy multiple MCP servers)
- Auto-generated REST API from MCP tools
- Enterprise auth integration (OAuth, OIDC)
- Built-in testing utilities

**Tasks**:
- [ ] Evaluate FastMCP 2.0 migration path
- [ ] Test compatibility with existing tools
- [ ] Update dependencies in `pyproject.toml`
- [ ] Document new capabilities

### 3.2 Consistency Checking Tools
**Effort**: 3 days | **Value**: Medium

Leverage semantic search for automated consistency checking.

**New Tools**:
- `check_character_consistency(document_name, character_name)` → Find description contradictions
- `check_timeline_consistency(document_name)` → Detect temporal paradoxes
- `check_name_spelling(document_name)` → Find spelling variations of proper nouns

**Example Output** (character consistency):
```json
{
  "character": "Marcus",
  "inconsistencies": [
    {
      "type": "physical_description",
      "chapter_a": "01-intro",
      "text_a": "His blue eyes narrowed",
      "chapter_b": "08-revelation",
      "text_b": "Marcus's brown eyes widened",
      "severity": "high"
    }
  ],
  "consistent_attributes": ["height", "age", "voice"]
}
```

**Tasks**:
- [ ] Design consistency checking algorithms
- [ ] Implement character consistency tool
- [ ] Implement timeline consistency tool
- [ ] Implement name spelling tool
- [ ] Add severity scoring
- [ ] Integration with codex data

### 3.3 Tool Consolidation Experiment
**Effort**: 2 days | **Value**: Low-Medium

Test whether consolidating 26 tools to 10-15 improves agent performance.

**Consolidation Candidates**:
- Multiple paragraph tools → `modify_paragraph(action: replace|insert|delete|move, ...)`
- Multiple chapter tools → `manage_chapter(action: create|read|write|delete, ...)`
- Multiple content tools → `query_content(type: read|search|stats|similar, ...)`

**Approach**:
- [ ] Create consolidated tool variants
- [ ] A/B test with both agent types
- [ ] Measure success rate, token usage, latency
- [ ] Document findings
- [ ] Only adopt if measurably better

---

## Hosting Strategy

### Deployment Options

#### Option A: Developer-Hosted (Recommended for Firstory Integration)

A central server hosted by the developer that all users connect to.

**Architecture**:
```
User's Claude Desktop/Web
         │
         │ HTTPS (Streamable HTTP)
         ▼
┌─────────────────────────┐
│  Hosted MCP Server      │
│  (FastAPI + Uvicorn)    │
│                         │
│  /mcp endpoint          │
│  ├─ Authentication      │
│  ├─ User isolation      │
│  └─ Rate limiting       │
│                         │
│  Storage: Per-user dirs │
│  or Cloud storage       │
└─────────────────────────┘
```

**Pros**:
- Users need zero setup beyond Claude configuration
- Centralized updates and maintenance
- Consistent experience across all users
- Easier integration with Firstory (same server serves both UI and MCP)
- Usage analytics and monitoring

**Cons**:
- Hosting costs scale with users
- Data stored on developer's infrastructure (privacy considerations)
- Single point of failure

**Best For**: Firstory integration where both the web UI and LLM use the same MCP server.

#### Option B: Local Installation (Current Model)

Users install the package and run the server locally.

**Architecture**:
```
User's Machine
├── Claude Desktop
│   └── MCP Connection (stdio or localhost)
│
├── document-mcp server
│   └── Local storage (~/.documents_storage)
│
└── User's files stay local
```

**Pros**:
- User data stays on their machine
- No hosting costs for developer
- Works offline
- Privacy-first

**Cons**:
- Users must install Python, pip, and configure
- Harder to provide support
- Version fragmentation

**Best For**: Privacy-conscious users, offline use, developer/power users.

#### Recommendation

For **Firstory integration**: Use Option A (hosted)
- Single MCP server serves both Firstory UI and LLM
- Both user interactions and AI operations go through same MCP
- Unified data store, no sync issues

For **standalone document-mcp**: Continue supporting Option B (local)
- Keep `pip install document-mcp` working
- Document both deployment options

---

## Integration with Firstory

### Key Architectural Principle

**Both the user AND the LLM use the SAME MCP to manipulate data.**

This unified approach means:
- MCP server is the **single source of truth** for story content
- Firstory UI makes MCP calls for all document operations (via REST API wrapper or direct MCP)
- Claude/LLM makes MCP calls through the same interface
- No separate data stores, no sync complexity, no dual-write patterns

### Unified Data Flow Architecture
```
┌──────────────────────┐         ┌──────────────────────┐
│   Firstory Web UI    │         │   Claude Desktop     │
│   (User Actions)     │         │   (AI Actions)       │
└──────────┬───────────┘         └──────────┬───────────┘
           │                                 │
           │ HTTP/HTTPS                      │ Streamable HTTP
           │ (REST or MCP)                   │ (MCP Protocol)
           │                                 │
           ▼                                 ▼
┌─────────────────────────────────────────────────────────┐
│         Hosted MCP Server (try-document-mcp)            │
│  ┌─────────────────────────────────────────────────┐   │
│  │  /mcp endpoint (Streamable HTTP - 2025-06-18)   │   │
│  │  - Both UI and LLM use this                     │   │
│  │  - Single source of truth                       │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  /api/* endpoints (Optional REST layer)         │   │
│  │  - More efficient for web UI bulk operations    │   │
│  │  - Calls same MCP tools internally              │   │
│  └─────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              Document MCP Core (this package)           │
│  - 26 Tools for document/chapter/paragraph operations   │
│  - Automatic snapshots (write-safety)                   │
│  - Semantic search with embedding cache                 │
│  - Pagination system (50K chars/page)                   │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    Storage Layer                        │
│  Per-user isolated storage:                             │
│  documents_storage/{user_id}/{story_id}/                │
└─────────────────────────────────────────────────────────┘
```

### Integration Points

| Firstory Feature | MCP Tool(s) | Implementation Status | Notes |
|------------------|-------------|-----------------------|-------|
| Story creation | `create_document` | ✅ Ready | Initialize document structure |
| Act management | `create_chapter`, `list_chapters` | ✅ Ready | Acts as chapters |
| Sequence organization | Chapter sections | ✅ Ready | Sections within chapters |
| Scene/Beat editing | `read_paragraph_content`, `replace_paragraph` | ✅ Ready | Atomic paragraph operations |
| Codex storage | `*_codex_item` tools | 📋 Planned (Phase 2) | Story Bible in codex/ directory |
| Version history | `manage_snapshots`, `diff_content` | ✅ Ready | Automatic snapshot system |
| Consistency checking | `check_*_consistency` tools | 📋 Planned (Phase 3) | AI-powered validation |
| Semantic search | `find_similar_text` | ✅ Ready | Embedding-based search |
| Outline view | `get_document_outline` | 📋 Planned (Phase 2) | Chapter summaries |
| Entity tracking | `track_entity` | 📋 Planned (Phase 2) | Character mention tracking |

### Mapping Firstory Structure to MCP

```
Firstory Structure          MCP Structure                    Implementation
─────────────────          ──────────────                   ──────────────
Story                  →    Document                        create_document()
  ├── Metadata         →    Document metadata               In frontmatter
  ├── Acts             →    Chapters (01-act-1.md, etc.)   create_chapter()
  │   ├── Metadata     →    Chapter frontmatter             YAML header
  │   └── Sequences    →    Sections within chapters       Markdown headings
  │       └── Scenes   →    Paragraphs                     Paragraph tools
  │           └── Beats →   Sub-paragraphs or content      In paragraph text
  └── Codex            →    codex/ directory               Phase 2 feature
      ├── Characters   →    codex/characters.yaml          *_codex_item tools
      ├── Locations    →    codex/locations.yaml           (when implemented)
      ├── Themes       →    codex/themes.yaml
      └── Concepts     →    codex/concepts.yaml
```

### Why This Architecture Works for Firstory

**Single Source of Truth**:
- No data sync issues between UI and AI operations
- Snapshots protect against both user and AI errors
- Version history tracks all changes regardless of source

**Scalability**:
- MCP server can be scaled independently
- Storage can move from filesystem → cloud → database as needed
- REST API layer optional but beneficial for UI performance

**Developer Experience**:
- Firstory devs work with familiar REST patterns (if REST layer used)
- Or directly use MCP protocol (same as AI)
- TypeScript MCP client available for type safety

**User Experience**:
- Seamless integration between manual edits and AI operations
- Real-time collaboration possible (Phase 3)
- Offline support via IndexedDB + sync queue (already in Firstory)

### Firstory Documentation Reference

The Firstory codebase already documents this integration:
- `docs/ai-and-integrations.md` - "Document MCP Integration" section
- `docs/todo/development-tasks-roadmap-reference.md` - Lists "MCP Document Operations [3 days]"

This Document MCP TODO serves as the implementation roadmap for those planned features.

---

## Testing Strategy

### Current State (352 tests, 100% passing)

| Tier | Count | Purpose |
|------|-------|---------|
| Unit | 181 | Isolated component testing |
| Integration | 155 | Agent-MCP communication |
| E2E | 6 | Full system with real APIs |
| Evaluation | 4 | Performance benchmarking |
| Metrics | 6 | OpenTelemetry validation |

### Testing Requirements for New Features

**Each new tool must have**:
- Unit tests for core logic
- Integration tests for MCP communication
- Documentation with examples

**Phase 1 (Standards Compliance)**:
- Test Streamable HTTP transport with MCP Inspector
- Verify Claude Desktop connectivity
- Test defer loading with mock LLM calls

**Phase 2 (Writer Features)**:
- Entity tracking accuracy tests
- Codex YAML validation tests
- Outline generation tests

---

## Development Priorities Summary

### Essential (Do First)
1. ✅ Streamable HTTP Transport (2-3 days) - Required for Claude compatibility
2. ✅ Tool Description Audit (1 day) - Improve agent success rate
3. ✅ Defer Loading (1-2 days) - 85% token reduction
4. ✅ Entity Tracking Tool (2 days) - High-value writer feature

### Important (Do Second)
5. Story Bible / Codex Storage (2 days) - Track 2 of writer mental model
6. Outline View Tool (1 day) - Structural overview
7. Chapter Metadata Extension (1 day) - Status/tag tracking
8. FastMCP 2.0 Upgrade (1 day) - Production features

### Nice to Have (Do If Time)
9. Consistency Checking Tools (3 days) - Advanced validation
10. Tool Consolidation Experiment (2 days) - Only if issues arise

### Total Estimated Effort
- **Essential**: 6-8 days
- **Important**: 5 days
- **Nice to Have**: 5 days
- **Grand Total**: 16-18 days

---

## References

- [MCP Specification 2025-06-18](https://modelcontextprotocol.io/specification/2025-06-18)
- [Anthropic Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)
- [FastMCP Documentation](https://www.prefect.io/fastmcp)
- [MCP Design Patterns](./MCP_DESIGN_PATTERNS.md)
- [Writer Mental Models Research](https://blog.reedsy.com/guide/story-structure/)
- [Scrivener Binder Patterns](https://www.literatureandlatte.com/blog/integrating-scriveners-binder-corkboard-and-outliner)
