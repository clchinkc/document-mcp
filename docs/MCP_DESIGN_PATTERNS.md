# MCP Design Patterns for Context Management

**Document Version**: 1.0
**Last Updated**: 2025-01-24

This guide provides production-ready design patterns for building MCP servers that effectively manage context offloading and partial hydration, based on industry best practices and the official MCP specification.

## Table of Contents

- [Core Concepts](#core-concepts)
- [Essential MCP References](#essential-mcp-references)
- [Reference-by-Handle Pattern](#reference-by-handle-pattern)
- [Implementation Guide](#implementation-guide)
- [Security Best Practices](#security-best-practices)
- [Common Pitfalls](#common-pitfalls)
- [Production Checklist](#production-checklist)

## Core Concepts

### The Problem: Context Window Limitations

AI models have finite context windows. Large documents, extensive conversation histories, or comprehensive knowledge bases can exceed these limits, forcing developers to choose between:

1. **Truncating content** - Losing important context
2. **Inlining everything** - Wasting tokens and hitting limits
3. **Manual chunking** - Complex, error-prone code

### The Solution: Reference-by-Handle

MCP provides two powerful primitives for elegant context management:

- **Tools**: Structured calls with JSON schemas for storing/retrieving context slices
- **Resources**: URI-addressable, read-only content the model can pull on demand

**Key Insight**: Instead of inlining massive blobs into chat, return compact handles the model can hydrate when needed.

## Essential MCP References

### Official Documentation

1. **[Build an MCP server](https://modelcontextprotocol.io/docs/develop/build-server)** - Quickstart guide covering core concepts (Tools, Resources, Prompts) and transport gotchas
2. **[MCP Specification (2025-06-18)](https://modelcontextprotocol.io/specification/2025-06-18/index)** - Authoritative protocol types and behavior
3. **[GitHub: modelcontextprotocol](https://github.com/modelcontextprotocol)** - Official schemas, SDKs, and examples
4. **[Resources Concept](https://modelcontextprotocol.io/docs/concepts/resources)** - URI-addressable, file-like context patterns
5. **[Tools Concept](https://modelcontextprotocol.io/docs/concepts/tools)** - Structured JSON schema tool definitions
6. **[Example Servers](https://modelcontextprotocol.io/examples)** - Reference implementations
7. **[MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector)** - Official testing UI for rapid iteration

### Additional Resources

8. **[Anthropic Announcement](https://www.anthropic.com/news/model-context-protocol)** - Protocol introduction and positioning
9. **[Memory Server Patterns](https://glama.ai/mcp/servers/@grizzlypeaksoftware/mcp-memory-server)** - Community examples for state persistence
10. **[Security Considerations](https://www.techradar.com/pro/mcps-biggest-security-loophole-is-identity-fragmentation)** - Identity and secret management best practices

## Reference-by-Handle Pattern

### Architecture Overview

Split your MCP server into three distinct layers:

```
┌─────────────────────────────────────────────────────┐
│  Layer 1: Storage (Tools)                          │
│  ├─ store_context → returns { artifact_id }        │
│  └─ search_context → returns handles + summaries   │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│  Layer 2: Hydration (Resources)                    │
│  └─ context://<artifact_id>?params → content slices│
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│  Layer 3: Metadata & Lifecycle                     │
│  ├─ Scoping (user, thread, project)                │
│  ├─ TTL and eviction policies                      │
│  └─ Token budgeting and pagination                 │
└─────────────────────────────────────────────────────┘
```

### Why This Design?

**Tools** are perfect for:
- Mutations (store, update, delete)
- Indexed search operations
- Side-effecting operations with retry safety

**Resources** are ideal for:
- Cheap, idempotent reads
- On-demand content hydration
- Deterministic, cacheable responses
- Token-aware progressive disclosure

## Implementation Guide

### Tool Definitions

#### 1. Store Context Tool

**Purpose**: Write raw state, create compact summary, return durable handle

**Input Schema**:
```json
{
  "payload": {
    "type": "string",
    "description": "The content to store (text, JSON, etc.)"
  },
  "tags": {
    "type": "array",
    "items": {"type": "string"},
    "description": "Optional metadata tags for filtering"
  },
  "scope": {
    "type": "object",
    "properties": {
      "user_id": {"type": "string"},
      "thread_id": {"type": "string"},
      "project_id": {"type": "string"}
    },
    "description": "Scoping information for access control and lifecycle"
  },
  "ttl_seconds": {
    "type": "integer",
    "minimum": 60,
    "description": "Time-to-live for automatic cleanup"
  }
}
```

**Output Schema**:
```json
{
  "artifact_id": {
    "type": "string",
    "description": "Unique identifier for this stored content"
  },
  "bytes": {
    "type": "integer",
    "description": "Size of stored content in bytes"
  },
  "checksum": {
    "type": "string",
    "description": "Content hash for integrity verification"
  },
  "resource_uri": {
    "type": "string",
    "description": "URI for resource-based hydration",
    "example": "context://abc123"
  }
}
```

#### 2. Search Context Tool

**Purpose**: Find stored content by query, return ranked handles with summaries

**Input Schema**:
```json
{
  "query": {
    "type": "string",
    "description": "Search query (semantic or keyword-based)"
  },
  "top_k": {
    "type": "integer",
    "default": 5,
    "minimum": 1,
    "maximum": 50,
    "description": "Number of results to return"
  },
  "tags": {
    "type": "array",
    "items": {"type": "string"},
    "description": "Filter by tags"
  },
  "scope": {
    "type": "object",
    "description": "Scope filtering (user, thread, project)"
  }
}
```

**Output Schema**:
```json
{
  "results": {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "artifact_id": {"type": "string"},
        "score": {"type": "number"},
        "summary": {"type": "string"},
        "resource_uri": {"type": "string"},
        "metadata": {
          "type": "object",
          "properties": {
            "created_at": {"type": "string"},
            "size_bytes": {"type": "integer"},
            "tags": {"type": "array"}
          }
        }
      }
    }
  }
}
```

### Resource Implementation

#### Context Resource URI Pattern

```
context://<artifact_id>?select=<selector>&limitTokens=<budget>&page=<n>
```

**URI Parameters**:
- `artifact_id`: Unique identifier from `store_context`
- `select`: Content selector (e.g., `summary`, `raw`, `slice:7,9,10`)
- `limitTokens`: Maximum tokens to return (client-specified budget)
- `page`: Pagination cursor for large content

**Example URIs**:
```
context://abc123?select=summary&limitTokens=500
context://abc123?select=raw&limitTokens=2000&page=1
context://abc123?select=slice:intro,conclusion&limitTokens=1500
```

**Resource Response**:
```json
{
  "content": "The actual content slice requested",
  "artifact_id": "abc123",
  "selector": "summary",
  "tokens_used": 487,
  "pagination": {
    "current_page": 1,
    "total_pages": 3,
    "has_more": true,
    "next_page": 2
  }
}
```

### Token-Aware Hydration

**Core Principle**: Support token budgeting at every layer

**Implementation Strategy**:

1. **On Write** (store_context):
   - Store **raw** content (complete, unprocessed)
   - Generate **distilled** summary + embeddings
   - Pre-chunk into **slices** with stable IDs
   - Persist metadata (token counts, chunk boundaries)

2. **On Read** (resource hydration):
   - Parse `limitTokens` parameter from URI
   - Select appropriate slices to fit budget
   - Return pagination hints for continuation
   - Never exceed client-specified token limit

**Example Pre-Processing**:
```python
def store_context(payload: str, scope: dict, ttl_seconds: int) -> dict:
    """Store content with multi-level indexing"""
    artifact_id = generate_id()

    # Layer 1: Raw storage
    raw_bytes = payload.encode('utf-8')

    # Layer 2: Distilled summary
    summary = generate_summary(payload)  # LLM call
    embeddings = generate_embeddings(payload)  # Vector API

    # Layer 3: Pre-chunked slices
    slices = chunk_content(payload, max_tokens=500)
    slice_metadata = [
        {"id": f"slice_{i}", "tokens": count_tokens(s), "checksum": hash(s)}
        for i, s in enumerate(slices)
    ]

    # Persist all layers
    db.store(artifact_id, {
        "raw": raw_bytes,
        "summary": summary,
        "embeddings": embeddings,
        "slices": slices,
        "metadata": {
            "scope": scope,
            "ttl": ttl_seconds,
            "created_at": now(),
            "slice_metadata": slice_metadata
        }
    })

    return {
        "artifact_id": artifact_id,
        "bytes": len(raw_bytes),
        "checksum": hash(raw_bytes),
        "resource_uri": f"context://{artifact_id}"
    }
```

### Scoping and Lifecycle

**Scope Model**:
```python
@dataclass
class ContentScope:
    user_id: str          # Isolation boundary
    thread_id: str        # Conversation context
    project_id: str       # Workspace or domain

    def to_key(self) -> str:
        return f"{self.user_id}:{self.thread_id}:{self.project_id}"
```

**Lifecycle Management**:
- **TTL-based eviction**: Automatic cleanup after `ttl_seconds`
- **Size-based eviction**: LRU eviction when hitting `max_bytes` per scope
- **Explicit cleanup**: Tool for manual deletion by handle or scope
- **Capabilities**: Return limits in tool descriptions to guide model choices

**Example Eviction Policy**:
```python
def enforce_storage_limits(scope: ContentScope):
    """LRU eviction when scope exceeds size limits"""
    scope_key = scope.to_key()
    total_bytes = db.get_scope_size(scope_key)
    max_bytes = config.MAX_BYTES_PER_SCOPE

    if total_bytes > max_bytes:
        # Get artifacts ordered by last access time
        artifacts = db.list_artifacts(scope_key, order_by="last_accessed_at")

        # Evict oldest until under limit
        while total_bytes > max_bytes and artifacts:
            oldest = artifacts.pop(0)
            db.delete(oldest["artifact_id"])
            total_bytes -= oldest["bytes"]

        logger.info(f"Evicted {len(artifacts)} artifacts from scope {scope_key}")
```

## Security Best Practices

### 1. Authentication and Authorization

**Problem**: Static API keys in config files are a security risk

**Solution**: Ephemeral, scoped authentication

```python
# BAD: Static keys in config
config = {
    "mcp_api_key": "sk-static-key-12345"  # ❌ Long-lived secret
}

# GOOD: OAuth/OIDC with short-lived tokens
def get_mcp_token(user_context):
    """Exchange user auth for short-lived MCP token"""
    response = oauth_client.exchange_token(
        user_token=user_context.access_token,
        scopes=["mcp:read", "mcp:write"],
        expires_in=3600  # 1 hour
    )
    return response.access_token
```

### 2. Logging and Observability

**Critical Rule**: Never log to stdout on stdio transports (breaks JSON-RPC framing)

```python
# BAD: stdout logging on stdio transport
print(f"Processing request: {request}")  # ❌ Breaks protocol

# GOOD: stderr logging only
import sys
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stderr))
logger.info(f"Processing request: {request.id}")
```

### 3. Content Security

**Best Practices**:
- Validate all inputs with JSON schemas
- Sanitize content before storage (prevent injection attacks)
- Use checksums for integrity verification
- Audit all mutations with user tracking
- Implement rate limiting per scope

**Example Input Validation**:
```python
from pydantic import BaseModel, Field, validator

class StoreContextInput(BaseModel):
    payload: str = Field(..., max_length=1_000_000)  # 1MB limit
    tags: list[str] = Field(default_factory=list, max_items=20)
    scope: ContentScope
    ttl_seconds: int = Field(ge=60, le=2_592_000)  # 1 min to 30 days

    @validator('payload')
    def sanitize_payload(cls, v):
        # Strip control characters, validate encoding
        return v.encode('utf-8', errors='replace').decode('utf-8')
```

### 4. Secret Management

**Key Principles**:
- Never commit secrets to version control
- Use environment variables or secret managers
- Rotate credentials regularly
- Use JIT (just-in-time) access for production

**Example**:
```python
import os
from typing import Optional

def get_db_credentials() -> dict:
    """Fetch credentials from secure source"""
    # Development: .env file
    if os.getenv("ENV") == "development":
        return {
            "host": os.getenv("DB_HOST"),
            "password": os.getenv("DB_PASSWORD")
        }

    # Production: Secret manager
    secret_name = os.getenv("DB_SECRET_NAME")
    return secret_manager.get_secret(secret_name)
```

## Common Pitfalls

### 1. Returning Giant Inline Blobs

**Problem**: Wasting tokens and losing tool-usage steering

```python
# BAD: Return full content inline
def search_context(query: str) -> dict:
    results = db.search(query)
    return {
        "results": [
            {"id": r.id, "content": r.full_text}  # ❌ Blows context budget
            for r in results
        ]
    }

# GOOD: Return handles with summaries
def search_context(query: str, top_k: int = 5) -> dict:
    results = db.search(query, limit=top_k)
    return {
        "results": [
            {
                "artifact_id": r.id,
                "score": r.relevance_score,
                "summary": r.summary[:200],  # ✅ Compact preview
                "resource_uri": f"context://{r.id}"
            }
            for r in results
        ]
    }
```

### 2. Non-Deterministic Resources

**Problem**: Resources must be idempotent and side-effect free

```python
# BAD: Resource with side effects
def get_resource(uri: str) -> str:
    artifact_id = parse_uri(uri)
    db.increment_view_count(artifact_id)  # ❌ Side effect
    return db.get_content(artifact_id)

# GOOD: Pure read operation
def get_resource(uri: str) -> str:
    artifact_id = parse_uri(uri)
    return db.get_content(artifact_id)  # ✅ Idempotent
```

### 3. Ignoring Token Budgets

**Problem**: Not respecting client-specified limits

```python
# BAD: Return arbitrary amount
def get_resource(uri: str) -> str:
    params = parse_uri_params(uri)
    artifact_id = params["artifact_id"]
    return db.get_full_content(artifact_id)  # ❌ Ignores limitTokens

# GOOD: Respect token budget
def get_resource(uri: str) -> dict:
    params = parse_uri_params(uri)
    artifact_id = params["artifact_id"]
    limit_tokens = int(params.get("limitTokens", 2000))

    content = db.get_content(artifact_id)
    slices = chunk_to_fit_tokens(content, limit_tokens)

    return {
        "content": slices[0],
        "tokens_used": count_tokens(slices[0]),
        "pagination": {"has_more": len(slices) > 1, "next_page": 2}
    }
```

### 4. Poor Error Handling

**Problem**: Generic errors without actionable information

```python
# BAD: Vague error messages
def store_context(payload: str, scope: dict) -> dict:
    try:
        return db.store(payload, scope)
    except Exception as e:
        raise Exception("Storage failed")  # ❌ No context

# GOOD: Structured error codes
from enum import Enum

class ErrorCode(Enum):
    QUOTA_EXCEEDED = "quota_exceeded"
    INVALID_SCOPE = "invalid_scope"
    CONTENT_TOO_LARGE = "content_too_large"

def store_context(payload: str, scope: dict) -> dict:
    # Validate scope
    if not is_valid_scope(scope):
        raise MCPError(
            code=ErrorCode.INVALID_SCOPE,
            message=f"Invalid scope: {scope}",
            details={"scope": scope, "required_fields": ["user_id", "thread_id"]}
        )

    # Check quota
    if exceeds_quota(scope, len(payload)):
        raise MCPError(
            code=ErrorCode.QUOTA_EXCEEDED,
            message="Storage quota exceeded for scope",
            details={
                "scope": scope.to_key(),
                "current_usage": get_usage(scope),
                "limit": config.MAX_BYTES_PER_SCOPE
            }
        )
```

## Production Checklist

### Before Deployment

- [ ] **Authentication**: Ephemeral tokens, no static keys
- [ ] **Logging**: All logs to stderr (never stdout on stdio transport)
- [ ] **Error Handling**: Structured error codes with retry hints
- [ ] **Input Validation**: JSON schemas for all tool inputs
- [ ] **Resource Determinism**: All resources are idempotent, read-only
- [ ] **Token Budgeting**: Respect `limitTokens` parameter in all resources
- [ ] **Scoping**: Proper user/thread/project isolation
- [ ] **Lifecycle**: TTL-based cleanup and size limits
- [ ] **Security Audit**: No secrets in code, proper sanitization
- [ ] **Testing**: MCP Inspector validation of all tools and resources

### Testing with MCP Inspector

The [MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector) is essential for validating your implementation:

```bash
# Install and run inspector
npx @modelcontextprotocol/inspector

# Connect to your MCP server
# Test each tool and resource
# Verify JSON schema validation
# Check error handling edge cases
```

**Key Test Cases**:
1. Store content with various sizes and scopes
2. Search with different query types (semantic, keyword)
3. Hydrate resources with token budgets (under, at, over limit)
4. Test pagination for large content
5. Verify TTL expiration
6. Test quota enforcement
7. Validate error codes and messages

### Monitoring

**Essential Metrics**:
- Request latency (p50, p95, p99)
- Storage utilization per scope
- Eviction rate (LRU, TTL)
- Error rates by error code
- Token usage per request
- Resource cache hit rate

**Example OpenTelemetry Integration**:
```python
from opentelemetry import metrics

meter = metrics.get_meter(__name__)

# Request latency
request_latency = meter.create_histogram(
    "mcp.request.latency",
    unit="ms",
    description="Request processing time"
)

# Storage metrics
storage_bytes = meter.create_up_down_counter(
    "mcp.storage.bytes",
    unit="bytes",
    description="Total storage used"
)

# Error tracking
error_counter = meter.create_counter(
    "mcp.errors.total",
    description="Total error count by code"
)

@request_latency.time()
def handle_request(request):
    try:
        result = process_request(request)
        storage_bytes.add(len(result.content), {"scope": request.scope.to_key()})
        return result
    except MCPError as e:
        error_counter.add(1, {"error_code": e.code.value})
        raise
```

## Real-World Examples

### Example 1: Document Management System

**Use Case**: Managing large technical documents with partial loading

```python
# Store a large document
response = store_context(
    payload=open("technical_spec.md").read(),
    tags=["documentation", "v2.0"],
    scope={"user_id": "user123", "thread_id": "doc-thread", "project_id": "proj-alpha"},
    ttl_seconds=86400  # 24 hours
)
# Returns: {"artifact_id": "doc_abc123", "resource_uri": "context://doc_abc123"}

# Search for specific sections
results = search_context(
    query="authentication flow",
    top_k=3,
    tags=["documentation"]
)
# Returns handles with summaries

# Hydrate specific section
content = get_resource("context://doc_abc123?select=slice:auth_section&limitTokens=1500")
# Returns just the auth section within token budget
```

### Example 2: Conversation Memory

**Use Case**: Maintaining multi-turn conversation context

```python
# Store conversation turn
turn_id = store_context(
    payload=json.dumps({
        "user_message": "How do I configure authentication?",
        "assistant_response": "To configure authentication...",
        "timestamp": "2025-01-24T10:30:00Z"
    }),
    tags=["conversation", "turn_5"],
    scope={"user_id": "user123", "thread_id": "chat-abc", "project_id": "support"},
    ttl_seconds=3600  # 1 hour
)

# Search conversation history
relevant_turns = search_context(
    query="authentication configuration",
    scope={"user_id": "user123", "thread_id": "chat-abc"},
    top_k=5
)

# Hydrate relevant context for next turn
for turn in relevant_turns["results"]:
    context = get_resource(f"{turn['resource_uri']}?select=summary&limitTokens=500")
```

## Conclusion

The reference-by-handle pattern transforms MCP from a simple tool protocol into a powerful context management system. By separating **storage** (tools) from **hydration** (resources), you enable models to:

- Access unlimited context through compact handles
- Control token budgets with selective hydration
- Maintain conversation state across sessions
- Search and retrieve relevant context on demand

**Key Takeaways**:
1. Use **tools** for mutations, **resources** for reads
2. Always return handles + summaries, not full content
3. Support token budgeting at every layer
4. Implement proper scoping and lifecycle management
5. Follow security best practices (ephemeral auth, stderr logging)
6. Test thoroughly with MCP Inspector

**Next Steps**:
- Review the [official MCP documentation](https://modelcontextprotocol.io/docs/develop/build-server)
- Explore [example servers](https://modelcontextprotocol.io/examples)
- Test your implementation with [MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector)
- Join the MCP community for support and best practices

---

**Document Credits**: Based on external research and MCP community best practices
**Specification Version**: MCP 2025-06-18
**Last Reviewed**: 2025-01-24
