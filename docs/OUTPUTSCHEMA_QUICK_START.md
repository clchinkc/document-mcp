# OutputSchema Implementation - Quick Start Guide

This guide provides a quick reference for implementing outputSchema in Document MCP tools. For comprehensive details, see [OUTPUTSCHEMA_IMPLEMENTATION_PLAN.md](./OUTPUTSCHEMA_IMPLEMENTATION_PLAN.md).

## What is outputSchema?

OutputSchema is an MCP 2025-06-18 feature that allows tools to declare their response structure using JSON Schema. This enables:
- Automatic response validation
- Better client error handling
- Machine-readable documentation
- IDE autocomplete support

## Current Status

- **Total Tools**: 28
- **Schemas Generated**: Schema generator utility created ✓
- **Tests Written**: 40+ unit tests created ✓
- **Implementation Ready**: Top 10 tools ready to integrate

## 5-Minute Implementation Example

### Step 1: Generate Schema from Pydantic Model

```python
from document_mcp.utils.schema_generator import pydantic_model_to_json_schema
from document_mcp.models import DocumentInfo

# Generate schema
schema = pydantic_model_to_json_schema(DocumentInfo)
print(schema)
```

### Step 2: Create Central Schema Registry

File: `document_mcp/tools/schemas.py`

```python
from document_mcp.utils.schema_generator import (
    pydantic_list_to_json_schema,
    pydantic_model_to_json_schema,
)
from document_mcp.models import (
    DocumentInfo,
    OperationStatus,
    ChapterContent,
)

# Generate and register schemas
LIST_DOCUMENTS_SCHEMA = pydantic_list_to_json_schema(DocumentInfo)
CREATE_DOCUMENT_SCHEMA = pydantic_model_to_json_schema(OperationStatus)
READ_CHAPTER_SCHEMA = pydantic_model_to_json_schema(ChapterContent)

# Central registry
TOOL_SCHEMAS = {
    "list_documents": LIST_DOCUMENTS_SCHEMA,
    "create_document": CREATE_DOCUMENT_SCHEMA,
    "read_chapter_content": READ_CHAPTER_SCHEMA,
    # ... 25 more tools
}
```

### Step 3: Apply Schema to Tool

File: `document_mcp/tools/document_tools.py`

```python
from .schemas import LIST_DOCUMENTS_SCHEMA

def register_document_tools(mcp_server: FastMCP) -> None:

    @mcp_server.tool(outputSchema=LIST_DOCUMENTS_SCHEMA)  # Add this
    @log_mcp_call
    def list_documents(include_chapters: bool = False) -> list[DocumentInfo]:
        """List all available documents."""
        # Implementation unchanged
        ...
```

### Step 4: Test

```python
# tests/unit/test_output_schemas.py
from document_mcp.tools.schemas import TOOL_SCHEMAS
from jsonschema import Draft202012Validator

def test_list_documents_schema_valid():
    schema = TOOL_SCHEMAS['list_documents']
    validator = Draft202012Validator(schema)
    assert validator.is_valid(schema)  # Schema is valid JSON Schema
```

## Schema Patterns

### Simple Objects (OperationStatus)

```python
OPERATION_SCHEMA = pydantic_model_to_json_schema(OperationStatus)

# Usage: create_document, delete_document, write_summary, etc.
@mcp_server.tool(outputSchema=OPERATION_SCHEMA)
def create_document(document_name: str) -> OperationStatus:
    ...
```

### Arrays (list[DocumentInfo])

```python
LIST_SCHEMA = pydantic_list_to_json_schema(DocumentInfo)

# Usage: list_documents, list_chapters, list_summaries
@mcp_server.tool(outputSchema=LIST_SCHEMA)
def list_documents() -> list[DocumentInfo]:
    ...
```

### Union Types (T | None)

```python
UNION_SCHEMA = pydantic_union_to_json_schema(
    DocumentInfo,
    allow_null=True
)

# Usage: read_summary, read_metadata, get_document_outline
@mcp_server.tool(outputSchema=UNION_SCHEMA)
def read_summary(...) -> DocumentSummary | None:
    ...
```

### Multiple Returns

```python
MULTI_SCHEMA = pydantic_union_to_json_schema(
    PaginatedContent,
    ChapterContent,
    ParagraphDetail,
    allow_null=True
)

# Usage: read_content (returns different types based on scope parameter)
@mcp_server.tool(outputSchema=MULTI_SCHEMA)
def read_content(...) -> PaginatedContent | ChapterContent | ParagraphDetail | None:
    ...
```

## Available Schema Generators

```python
from document_mcp.utils.schema_generator import (
    # Core generators
    pydantic_model_to_json_schema,      # Single model
    pydantic_list_to_json_schema,       # Array of models
    pydantic_union_to_json_schema,      # Union of models

    # Flexible types
    create_flexible_object_schema,      # dict[str, Any]
    create_oneOf_schema,                # Exactly one alternative

    # Registry
    register_tool_schema,               # Register schema
    get_tool_schema,                    # Retrieve schema
    get_all_tool_schemas,               # Get all schemas

    # Validation
    validate_schema_against_json_schema_spec,  # Validate schema
)
```

## Common Issues & Solutions

### Issue: "FastMCP doesn't accept outputSchema parameter"

**Solution**: Use fallback injection in tool registration:

```python
@mcp_server.tool()
def my_tool() -> MyModel:
    ...

# After registration, inject schema
if hasattr(mcp_server._tools['my_tool'], 'outputSchema'):
    mcp_server._tools['my_tool'].outputSchema = MY_SCHEMA
```

### Issue: "Tool returns 'Any' type"

**Solution**: Inspect implementation and create proper model:

```python
# Instead of: -> Any
# Create:
class MyResponse(BaseModel):
    results: list[dict]
    count: int

# Then use:
MY_SCHEMA = pydantic_model_to_json_schema(MyResponse)
```

### Issue: "Tool returns dict with arbitrary keys"

**Solution**: Use flexible object schema:

```python
from document_mcp.utils.schema_generator import create_flexible_object_schema

FLEXIBLE_SCHEMA = create_flexible_object_schema(
    description="Configuration object with tool-specific fields"
)

@mcp_server.tool(outputSchema=FLEXIBLE_SCHEMA)
def get_config() -> dict[str, Any]:
    ...
```

## Rollout Checklist

### Phase 1: Infrastructure (Week 1)
- [x] Create schema generator utility
- [x] Write unit tests (40+ tests)
- [ ] Document patterns
- [ ] CI/CD integration

### Phase 2: Top 10 Tools (Week 2-3)
- [ ] list_documents
- [ ] read_content
- [ ] create_document
- [ ] find_text
- [ ] get_statistics
- [ ] find_similar_text
- [ ] check_content_freshness
- [ ] get_modification_history
- [ ] get_document_outline
- [ ] delete_document

### Phase 3: Remaining Tools (Week 4)
- [ ] Batch A: Simple (5 tools)
- [ ] Batch B: Nested (6 tools)
- [ ] Batch C: Union (5 tools)
- [ ] Batch D: Untyped (2 tools)

### Phase 4: Validation (Week 5)
- [ ] E2E testing
- [ ] Documentation review
- [ ] Release preparation

## Files Created

| File | Purpose |
|------|---------|
| `document_mcp/utils/schema_generator.py` | Core schema generation utilities (340 lines) |
| `tests/unit/test_schema_generator.py` | Comprehensive unit tests (500+ lines) |
| `document_mcp/tools/schemas.py` | Central schema registry (to be created) |
| `docs/OUTPUTSCHEMA_IMPLEMENTATION_PLAN.md` | Complete implementation guide |
| `docs/OUTPUTSCHEMA_QUICK_START.md` | This file |

## Next Steps

1. **Verify FastMCP Support** (Day 1)
   ```bash
   python3 -c "from mcp.server import FastMCP; import inspect; print(inspect.signature(FastMCP.tool))"
   ```

2. **Create Central Registry** (Day 1-2)
   - Create `document_mcp/tools/schemas.py`
   - Import all models and schema generators
   - Define all 28 schemas

3. **Test Schema Generator** (Day 2)
   ```bash
   pytest tests/unit/test_schema_generator.py -v
   ```

4. **Implement Top 10 Tools** (Week 2-3)
   - Apply schemas to tool decorators
   - Run integration tests
   - Validate E2E

5. **Complete Remaining Tools** (Week 4)
   - Batch process by pattern
   - Continuous testing

## Reference: Tool-to-Schema Mapping

| Tool | Return Type | Schema Pattern |
|------|-------------|---|
| list_documents | list[DocumentInfo] | List |
| read_content | PaginatedContent \| ChapterContent \| ParagraphDetail \| None | Union |
| create_document | OperationStatus | Simple |
| find_text | Any | Inspect/Define |
| get_statistics | StatisticsReport \| None | Union |
| find_similar_text | SemanticSearchResponse \| None | Union |
| check_content_freshness | ContentFreshnessStatus | Simple |
| get_modification_history | ModificationHistory | Nested |
| get_document_outline | dict[str, Any] \| None | Flexible |
| delete_document | OperationStatus | Simple |

## Support & Questions

- **Implementation Details**: See [OUTPUTSCHEMA_IMPLEMENTATION_PLAN.md](./OUTPUTSCHEMA_IMPLEMENTATION_PLAN.md)
- **Troubleshooting**: See Part 11-12 of implementation plan
- **Schema Validation**: Use `validate_schema_against_json_schema_spec()`
- **Testing**: Run `pytest tests/unit/test_schema_generator.py`

## Related Documentation

- [MCP 2025-06-18 Spec](https://spec.modelcontextprotocol.io/)
- [Pydantic JSON Schema Docs](https://docs.pydantic.dev/latest/concepts/json_schema/)
- [JSON Schema Specification](https://json-schema.org/understanding-json-schema/)
