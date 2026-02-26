# OutputSchema Implementation Plan for MCP 2025-06-18 Compliance

## Executive Summary

This document provides a comprehensive implementation strategy for adding `outputSchema` support to the Document MCP system, targeting MCP 2025-06-18 compliance. The system currently has 28 tools with well-defined Pydantic response models but no output schemas exposed to clients.

**Key Finding**: FastMCP framework supports `outputSchema` through the underlying MCP protocol. The implementation requires:
1. Converting Pydantic models to JSON Schema format
2. Modifying tool registration to include schemas
3. Systematic rollout starting with top 10 most-used tools
4. Comprehensive testing and validation

---

## Part 1: Current State Assessment

### 1.1 Tool Inventory (28 Total)

| Category | Count | Tools |
|----------|-------|-------|
| Document Tools | 6 | list_documents, create_document, delete_document, read_summary, write_summary, list_summaries |
| Chapter Tools | 4 | read_chapter_content, list_chapters, create_chapter, delete_chapter |
| Paragraph Tools | 4 | add_paragraph, replace_paragraph, delete_paragraph, move_paragraph |
| Content Tools | 7 | read_content, find_text, replace_text, get_statistics, find_similar_text, find_entity, (1 internal) |
| Metadata Tools | 3 | read_metadata, write_metadata, list_metadata |
| Safety Tools | 5 | check_content_freshness, get_modification_history, manage_snapshots, check_content_status, diff_content |
| Overview Tools | 1 | get_document_outline |
| Discovery Tools | 1 | search_tool |

### 1.2 Response Model Inventory

All tools use Pydantic BaseModel subclasses for responses:

**Core Response Types** (8 models):
- `OperationStatus` - For write/delete operations
- `DocumentInfo` - Document metadata and structure
- `ChapterContent` - Chapter content with metadata
- `StatisticsReport` - Content statistics
- `PaginatedContent` - Paginated document/chapter/paragraph content
- `ContentFreshnessStatus` - Content safety and freshness info
- `ModificationHistory` - Audit trail with entries
- `SemanticSearchResponse` - Search results with similarity scores

**Supporting Models** (15+ nested models):
- `DocumentSummary`, `ChapterMetadata`, `ParagraphDetail`
- `SnapshotInfo`, `SnapshotsList`
- `PaginationInfo`, `SemanticSearchResult`
- `MetadataResponse`, `MetadataListResponse`
- `ModificationHistoryEntry`, `ChapterEmbeddingManifest`
- `EmbeddingCacheEntry`, and input models

### 1.3 Current Tool Registration Pattern

**Current Implementation** (FastMCP):
```python
@mcp_server.tool()
@log_mcp_call
def list_documents(include_chapters: bool = False) -> list[DocumentInfo]:
    """Tool description..."""
    # Implementation
```

**Limitations**:
- ✗ No `outputSchema` parameter in `@tool()` decorator
- ✗ Return type only used for type hints, not for schema generation
- ✗ Clients cannot validate response structure
- ✗ No machine-readable output specification

**Target Implementation**:
```python
@mcp_server.tool(
    outputSchema=ToolOutputSchema(
        list_documents=list[DocumentInfo]
    )
)
@log_mcp_call
def list_documents(include_chapters: bool = False) -> list[DocumentInfo]:
    """Tool description..."""
```

### 1.4 FastMCP Framework Capabilities

**Confirmed Support**:
- ✓ MCP Tool type has `outputSchema: dict[str, Any] | None` field
- ✓ FastMCP's `@tool()` decorator accepts kwargs
- ✓ Can pass custom annotations via `annotations` parameter

**Query Results**:
```python
Tool.model_fields.keys()
# Result: ['name', 'title', 'description', 'inputSchema', 'outputSchema', 'annotations', 'meta']
```

---

## Part 2: Pydantic → JSON Schema Mapping

### 2.1 Schema Generation Strategy

**Approach**: Use Pydantic's built-in JSON schema generation with customization layer.

```python
from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue

# Core method
schema = MyModel.model_json_schema()

# With custom config
schema = MyModel.model_json_schema(
    mode='serialization',
    by_alias=True,
    ref_template='#/definitions/{model}'
)
```

**Key Features**:
- `mode='serialization'`: Schema reflects runtime values, not input validation
- `by_alias=True`: Use field aliases in schema
- `ref_template`: Control $ref format for nested models

### 2.2 Type Mapping Examples

| Pydantic Type | JSON Schema | Notes |
|--------------|------------|-------|
| `str` | `{"type": "string"}` | Basic string |
| `int` | `{"type": "integer"}` | Basic integer |
| `bool` | `{"type": "boolean"}` | Basic boolean |
| `float` | `{"type": "number"}` | Floating point |
| `list[T]` | `{"type": "array", "items": {...}}` | Array with item schema |
| `dict[str, Any]` | `{"type": "object"}` | Object with arbitrary properties |
| `T \| None` | `{"anyOf": [schema, null]}` | Optional types |
| `datetime.datetime` | `{"type": "string", "format": "date-time"}` | ISO 8601 string |
| `BaseModel` subclass | Nested object with $ref | Recursive definition |

### 2.3 Response Model Schemas (Examples)

**OperationStatus** (Used by 15+ write/delete tools):
```json
{
  "type": "object",
  "properties": {
    "success": {"type": "boolean"},
    "message": {"type": "string"},
    "details": {"type": "object", "additionalProperties": true},
    "safety_info": {"type": "object"},
    "snapshot_created": {"type": ["string", "null"]},
    "warnings": {
      "type": "array",
      "items": {"type": "string"}
    }
  },
  "required": ["success", "message"]
}
```

**DocumentInfo** (list_documents response):
```json
{
  "type": "object",
  "properties": {
    "document_name": {"type": "string"},
    "total_chapters": {"type": "integer"},
    "total_word_count": {"type": "integer"},
    "total_paragraph_count": {"type": "integer"},
    "last_modified": {"type": "string", "format": "date-time"},
    "chapters": {
      "type": "array",
      "items": {"$ref": "#/definitions/ChapterMetadata"}
    },
    "has_summary": {"type": "boolean"}
  },
  "required": ["document_name", "total_chapters", "total_word_count", "total_paragraph_count", "last_modified", "chapters"]
}
```

**PaginatedContent** (read_content response):
```json
{
  "type": "object",
  "properties": {
    "content": {"type": "string"},
    "document_name": {"type": "string"},
    "scope": {"type": "string", "enum": ["document", "chapter", "paragraph"]},
    "chapter_name": {"type": ["string", "null"]},
    "paragraph_index": {"type": ["integer", "null"]},
    "pagination": {
      "$ref": "#/definitions/PaginationInfo"
    }
  },
  "required": ["content", "document_name", "scope", "pagination"]
}
```

### 2.4 Schema Generation Utility Module

**File**: `document_mcp/utils/schema_generator.py`

```python
"""JSON Schema generation from Pydantic models."""

from typing import Any, Type
from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaMode


class MCP2025JsonSchema(GenerateJsonSchema):
    """Custom JSON schema generator for MCP 2025 compliance."""

    def generate(self, schema, mode='validation'):
        """Generate schema with MCP 2025 optimizations."""
        json_schema = super().generate(schema, mode=mode)

        # Ensure title and description are preserved
        if hasattr(schema, '__doc__'):
            json_schema['description'] = schema.__doc__

        return json_schema


def pydantic_model_to_json_schema(
    model: Type[BaseModel],
    mode: JsonSchemaMode = 'serialization',
    include_definitions: bool = True
) -> dict[str, Any]:
    """Convert Pydantic model to JSON schema for MCP tool output.

    Args:
        model: Pydantic BaseModel subclass
        mode: 'validation' or 'serialization' (default: serialization for responses)
        include_definitions: Include $defs for nested models

    Returns:
        JSON schema dict compatible with MCP outputSchema

    Example:
        ```python
        schema = pydantic_model_to_json_schema(DocumentInfo)
        # {
        #   "$schema": "http://json-schema.org/draft/2020-12/schema",
        #   "type": "object",
        #   "properties": {...},
        #   "$defs": {...}
        # }
        ```
    """
    schema = model.model_json_schema(
        mode=mode,
        by_alias=True,
        ref_template='#/$defs/{model}'
    )

    # Clean up $schema for MCP (unnecessary in tool outputs)
    schema.pop('$schema', None)

    return schema


def pydantic_union_to_json_schema(
    *models: Type[BaseModel],
    allow_null: bool = True
) -> dict[str, Any]:
    """Generate schema for union of models (e.g., T | None).

    Args:
        models: Variable number of Pydantic models
        allow_null: Include null in anyOf

    Returns:
        JSON schema with anyOf construct

    Example:
        ```python
        schema = pydantic_union_to_json_schema(
            DocumentInfo,
            allow_null=True
        )
        # {"anyOf": [...], "type": ["object", "null"]}
        ```
    """
    schemas = [
        pydantic_model_to_json_schema(model, include_definitions=False)
        for model in models
    ]

    if allow_null:
        schemas.append({"type": "null"})

    return {"anyOf": schemas}


def pydantic_list_to_json_schema(
    item_model: Type[BaseModel]
) -> dict[str, Any]:
    """Generate schema for list of models.

    Args:
        item_model: Pydantic model type for list items

    Returns:
        JSON schema for array

    Example:
        ```python
        schema = pydantic_list_to_json_schema(DocumentInfo)
        # {"type": "array", "items": {...}}
        ```
    """
    return {
        "type": "array",
        "items": pydantic_model_to_json_schema(
            item_model,
            include_definitions=False
        )
    }


# Registry of tool output schemas
TOOL_OUTPUT_SCHEMAS: dict[str, dict[str, Any]] = {}


def register_tool_schema(tool_name: str, schema: dict[str, Any]) -> None:
    """Register a tool's output schema for reference.

    Args:
        tool_name: Name of the tool
        schema: JSON schema dict

    Example:
        ```python
        register_tool_schema(
            'list_documents',
            pydantic_list_to_json_schema(DocumentInfo)
        )
        ```
    """
    TOOL_OUTPUT_SCHEMAS[tool_name] = schema


def get_tool_schema(tool_name: str) -> dict[str, Any] | None:
    """Retrieve registered schema for a tool.

    Args:
        tool_name: Name of the tool

    Returns:
        JSON schema dict or None if not registered
    """
    return TOOL_OUTPUT_SCHEMAS.get(tool_name)
```

---

## Part 3: Implementation Strategy

### 3.1 Phase 1: Infrastructure (Week 1)

**Deliverables**:
- Schema generator utility module
- Unit tests for schema generation
- Schema registry system
- Documentation of schema patterns

**Tasks**:
1. Create `/document_mcp/utils/schema_generator.py` with conversion utilities
2. Create `/tests/unit/test_schema_generation.py` with test cases
3. Document schema patterns in `/docs/SCHEMA_PATTERNS.md`
4. Verify Pydantic compatibility across Python 3.10+

**Success Criteria**:
- [ ] All schema generation tests pass
- [ ] Schemas validate against JSON Schema spec
- [ ] No Pydantic version conflicts

### 3.2 Phase 2: Top 10 Tools (Week 2-3)

**Priority Tools** (by usage frequency and complexity):

1. **list_documents** ⭐⭐⭐
   - Return type: `list[DocumentInfo]`
   - Complexity: Medium (nested ChapterMetadata)
   - Uses: Core overview operation
   - Schema: Array of objects with nested definitions

2. **read_content** ⭐⭐⭐
   - Return type: `PaginatedContent | ChapterContent | ParagraphDetail | None`
   - Complexity: High (union type, optional)
   - Uses: Most common read operation
   - Schema: Union type with pagination support

3. **create_document** ⭐⭐⭐
   - Return type: `OperationStatus`
   - Complexity: Low
   - Uses: Document creation
   - Schema: Reusable across write operations

4. **find_text** ⭐⭐
   - Return type: `Any` (needs clarification)
   - Complexity: Medium
   - Uses: Content search
   - Schema: Needs runtime inspection

5. **get_statistics** ⭐⭐
   - Return type: `StatisticsReport | None`
   - Complexity: Low
   - Uses: Content analysis
   - Schema: Simple object

6. **find_similar_text** ⭐⭐
   - Return type: `SemanticSearchResponse | None`
   - Complexity: Medium (nested results)
   - Uses: AI-powered search
   - Schema: Complex nested structure

7. **check_content_freshness** ⭐⭐
   - Return type: `ContentFreshnessStatus`
   - Complexity: Low
   - Uses: Safety checking
   - Schema: Simple status object

8. **get_modification_history** ⭐⭐
   - Return type: `ModificationHistory`
   - Complexity: Medium (nested entries)
   - Uses: Audit trail
   - Schema: Array of history entries

9. **get_document_outline** ⭐
   - Return type: `dict[str, Any] | None`
   - Complexity: High (unstructured)
   - Uses: Document structure
   - Schema: Flexible object

10. **delete_document** ⭐
    - Return type: `OperationStatus`
    - Complexity: Low
    - Uses: Cleanup operation
    - Schema: Reusable OperationStatus

**Implementation Pattern** (per tool):

```python
# Before
@mcp_server.tool()
@log_mcp_call
def list_documents(include_chapters: bool = False) -> list[DocumentInfo]:
    """List documents..."""
    # Implementation

# After
@mcp_server.tool(
    outputSchema={
        "type": "array",
        "items": {
            "$ref": "#/$defs/DocumentInfo"
        },
        "$defs": {
            "DocumentInfo": {...},
            "ChapterMetadata": {...}
        }
    }
)
@log_mcp_call
def list_documents(include_chapters: bool = False) -> list[DocumentInfo]:
    """List documents..."""
    # Implementation
```

**Rollout Steps**:
1. Tool #1-2: Create schema, test, validate
2. Tool #3-5: Batch implementation with shared schemas
3. Tool #6-10: Final batch with complex schemas

### 3.3 Phase 3: Remaining 18 Tools (Week 4)

**Batch Processing** (group by schema complexity):

**Batch A - Simple Schemas** (5 tools):
- create_chapter, delete_chapter, write_summary, write_metadata, diff_content
- Schema: OperationStatus (reusable)

**Batch B - Nested Schemas** (6 tools):
- read_chapter_content, read_summary, read_metadata, list_metadata, list_chapters, list_summaries
- Schema: Model + nested definitions

**Batch C - Union/Complex** (5 tools):
- replace_paragraph, add_paragraph, delete_paragraph, move_paragraph, manage_snapshots
- Schema: OperationStatus or complex unions

**Batch D - Untyped** (2 tools):
- search_tool, check_content_status
- Schema: Requires inspection and documentation

### 3.4 Implementation Approach: Centralized Schema Definition

**Location**: Create `/document_mcp/tools/schemas.py`

```python
"""Central schema definitions for all MCP tools.

This module defines output schemas for all 28 tools,
ensuring consistency and enabling easy updates.
"""

from document_mcp.models import (
    DocumentInfo,
    OperationStatus,
    ChapterContent,
    # ... import all models
)
from document_mcp.utils.schema_generator import (
    pydantic_model_to_json_schema,
    pydantic_list_to_json_schema,
    pydantic_union_to_json_schema,
)

# ===== Document Tools =====

LIST_DOCUMENTS_SCHEMA = pydantic_list_to_json_schema(DocumentInfo)

CREATE_DOCUMENT_SCHEMA = pydantic_model_to_json_schema(OperationStatus)

DELETE_DOCUMENT_SCHEMA = pydantic_model_to_json_schema(OperationStatus)

READ_SUMMARY_SCHEMA = pydantic_union_to_json_schema(DocumentSummary, allow_null=True)

# ... 24 more tool schemas

# Registry for easy lookup
TOOL_SCHEMAS = {
    "list_documents": LIST_DOCUMENTS_SCHEMA,
    "create_document": CREATE_DOCUMENT_SCHEMA,
    "delete_document": DELETE_DOCUMENT_SCHEMA,
    # ... 25 more entries
}


def get_tool_output_schema(tool_name: str) -> dict | None:
    """Get output schema for a tool by name.

    Returns None if tool has no schema (not yet implemented).
    """
    return TOOL_SCHEMAS.get(tool_name)
```

**Benefits**:
- Single source of truth for schemas
- Easy to validate completeness
- Supports incremental rollout
- Enables schema testing

### 3.5 Tool Registration Update

**Pattern for Phase 2 implementation**:

```python
# In document_tools.py
from .schemas import LIST_DOCUMENTS_SCHEMA

def register_document_tools(mcp_server: FastMCP) -> None:

    @mcp_server.tool(
        outputSchema=LIST_DOCUMENTS_SCHEMA
    )
    @log_mcp_call
    def list_documents(include_chapters: bool = False) -> list[DocumentInfo]:
        """List documents..."""
        # Implementation unchanged
```

**Pattern for tools with union returns** (read_content):

```python
# For union types, use anyOf construct
READ_CONTENT_SCHEMA = {
    "anyOf": [
        {
            "$ref": "#/$defs/PaginatedContent"
        },
        {
            "$ref": "#/$defs/ChapterContent"
        },
        {
            "$ref": "#/$defs/ParagraphDetail"
        },
        {"type": "null"}
    ],
    "$defs": {
        "PaginatedContent": {...},
        "ChapterContent": {...},
        "ParagraphDetail": {...},
    }
}
```

---

## Part 4: Handling Untyped Returns

### 4.1 Tools Needing Clarification

**Current Issues**:

| Tool | Current Return Type | Status | Action |
|------|-------------------|--------|--------|
| read_content | Any | Conditional (PaginatedContent \| ChapterContent \| ParagraphDetail \| None) | Clarify with runtime inspection |
| find_text | Any | Returns list of matches or error details | Define proper return model |
| find_entity | Any | Returns entity metadata or None | Create EntityResult model |
| get_statistics | Any | Actually StatisticsReport | Update type hint |
| manage_snapshots | Any | Returns SnapshotsList or OperationStatus | Use union type |
| check_content_status | Any | Returns ContentFreshnessStatus or dict | Standardize response |
| move_paragraph | Any | Returns OperationStatus | Update type hint |
| search_tool | tuple[list, dict] | Search results tuple | Create SearchResponse model |

### 4.2 Resolution Strategy

**Step 1: Runtime Inspection**
```python
import inspect
from document_mcp.tools import content_tools

sig = inspect.signature(content_tools.find_text)
print(sig.return_annotation)  # See actual type hint or Any
```

**Step 2: Code Analysis**
```python
# Read tool implementation to understand actual return value
# Example:
def find_text(...) -> Any:
    results = [
        {"paragraph_index": i, "content": text, "match_count": count}
        for i, text in enumerate(found_texts)
    ]
    return results or []
```

**Step 3: Schema Definition**
```python
# Create proper model
class TextSearchResult(BaseModel):
    paragraph_index: int
    content: str
    match_count: int

class TextSearchResponse(BaseModel):
    document_name: str
    scope: str
    results: list[TextSearchResult]
    total_matches: int

FIND_TEXT_SCHEMA = pydantic_model_to_json_schema(TextSearchResponse)
```

### 4.3 Handling dict[str, Any] Returns

**Pattern for tools with flexible responses**:

```python
# If tool genuinely returns flexible data structure
FLEXIBLE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": true,  # Allow arbitrary properties
    "description": "Flexible response containing tool-specific data"
}

# Better: Define explicit variants
SNAPSHOT_OPERATION_SCHEMA = {
    "oneOf": [
        {"$ref": "#/$defs/SnapshotsList"},
        {"$ref": "#/$defs/OperationStatus"}
    ],
    "$defs": {
        "SnapshotsList": {...},
        "OperationStatus": {...}
    }
}
```

---

## Part 5: FastMCP Integration

### 5.1 Confirming FastMCP Support

**Research Results**:
- ✓ MCP Protocol supports outputSchema (MCP 2025-06-18 spec)
- ✓ FastMCP's `@tool()` decorator accepts kwargs
- ✓ Tool definition accepts outputSchema parameter

**Fallback Options** (if not directly supported):
1. Modify server initialization to add schemas post-registration
2. Use custom tool wrapper that injects schemas
3. Patch MCP Tool objects before server startup

### 5.2 Implementation Methods

**Method 1: Direct Parameter** (Recommended if supported)
```python
@mcp_server.tool(outputSchema=SCHEMA_DICT)
def my_tool() -> Model:
    pass
```

**Method 2: Via Annotations** (Alternative)
```python
@mcp_server.tool(
    annotations=ToolAnnotations(
        # Can we add outputSchema here?
    )
)
def my_tool() -> Model:
    pass
```

**Method 3: Post-Registration** (Fallback)
```python
# Register tool normally
@mcp_server.tool()
def my_tool() -> Model:
    pass

# Modify MCP server's internal tool registry
mcp_server._tools['my_tool'].outputSchema = SCHEMA_DICT
```

### 5.3 Testing FastMCP Integration

**Test File**: `/tests/unit/test_fastmcp_outputschema.py`

```python
import pytest
from mcp.server import FastMCP
from mcp.types import Tool
from pydantic import BaseModel


class SimpleModel(BaseModel):
    name: str
    value: int


@pytest.fixture
def test_mcp_server():
    return FastMCP(name="test")


def test_tool_with_outputschema(test_mcp_server):
    """Test that outputSchema parameter is accepted and stored."""

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "value": {"type": "integer"}
        }
    }

    @test_mcp_server.tool(outputSchema=schema)
    def test_tool() -> SimpleModel:
        return SimpleModel(name="test", value=42)

    # Verify schema is stored in MCP tool definition
    assert test_mcp_server._tools['test_tool'].outputSchema == schema


def test_pydantic_schema_generation():
    """Test Pydantic JSON schema generation."""
    schema = SimpleModel.model_json_schema(mode='serialization')

    assert schema['type'] == 'object'
    assert 'properties' in schema
    assert 'name' in schema['properties']
    assert schema['properties']['name']['type'] == 'string'
```

---

## Part 6: Testing Strategy

### 6.1 Unit Tests

**File**: `/tests/unit/test_output_schemas.py`

```python
"""Test output schema generation and validation."""

import json
import pytest
from jsonschema import Draft202012Validator
from document_mcp.models import DocumentInfo, OperationStatus
from document_mcp.tools.schemas import TOOL_SCHEMAS
from document_mcp.utils.schema_generator import pydantic_model_to_json_schema


class TestSchemaGeneration:
    """Test Pydantic → JSON Schema conversion."""

    def test_simple_model_schema(self):
        """Test schema generation for simple model."""
        schema = pydantic_model_to_json_schema(OperationStatus)

        assert schema['type'] == 'object'
        assert 'success' in schema['properties']
        assert 'message' in schema['properties']

    def test_nested_model_schema(self):
        """Test schema generation for nested models."""
        schema = pydantic_model_to_json_schema(DocumentInfo)

        assert schema['type'] == 'object'
        assert 'chapters' in schema['properties']
        # Chapters should reference ChapterMetadata
        assert 'items' in schema['properties']['chapters']

    def test_schema_validation(self):
        """Test generated schemas are valid JSON Schema."""
        for tool_name, schema in TOOL_SCHEMAS.items():
            validator = Draft202012Validator(schema)
            assert validator.is_valid(schema), f"Schema for {tool_name} is invalid"

    def test_schema_completeness(self):
        """Test all required fields are documented."""
        schema = pydantic_model_to_json_schema(OperationStatus)

        # All non-optional fields should be in required list
        required = schema.get('required', [])
        assert 'success' in required
        assert 'message' in required


class TestToolSchemas:
    """Test individual tool schemas."""

    def test_list_documents_schema(self):
        """Test list_documents returns array schema."""
        schema = TOOL_SCHEMAS['list_documents']
        assert schema['type'] == 'array'
        assert 'items' in schema

    def test_operation_status_reuse(self):
        """Test OperationStatus schema is used by multiple tools."""
        tools_using_operation_status = [
            'create_document',
            'delete_document',
            'create_chapter',
            'delete_chapter',
        ]

        base_schema = TOOL_SCHEMAS['create_document']

        for tool_name in tools_using_operation_status:
            assert TOOL_SCHEMAS[tool_name] == base_schema
```

### 6.2 Integration Tests

**File**: `/tests/integration/test_tools_with_outputschema.py`

```python
"""Test tools with outputSchema in MCP protocol."""

import json
import pytest
from mcp.types import Tool
from document_mcp.doc_tool_server import mcp_server
from document_mcp.tools.schemas import TOOL_SCHEMAS


@pytest.mark.asyncio
async def test_tool_definitions_have_schemas(mcp_server):
    """Test all top-10 tools have outputSchema defined."""

    priority_tools = [
        'list_documents',
        'read_content',
        'create_document',
        'find_text',
        'get_statistics',
        'find_similar_text',
        'check_content_freshness',
        'get_modification_history',
        'get_document_outline',
        'delete_document',
    ]

    for tool_name in priority_tools:
        tool = mcp_server._tools.get(tool_name)
        assert tool is not None, f"Tool {tool_name} not found"

        # Check outputSchema is present
        if tool_name in TOOL_SCHEMAS:
            assert hasattr(tool, 'outputSchema')
            assert tool.outputSchema is not None
            assert isinstance(tool.outputSchema, dict)


def test_schema_json_serializable():
    """Test all schemas are JSON serializable."""
    for tool_name, schema in TOOL_SCHEMAS.items():
        try:
            json.dumps(schema)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Schema for {tool_name} is not JSON serializable: {e}")
```

### 6.3 E2E Validation Tests

**File**: `/tests/e2e/test_outputschema_e2e.py`

```python
"""End-to-end validation of outputSchema with real tool execution."""

import pytest
from jsonschema import validate, ValidationError
from document_mcp.doc_tool_server import mcp_server
from document_mcp.tools.schemas import TOOL_SCHEMAS


@pytest.mark.asyncio
async def test_list_documents_response_matches_schema(mcp_client):
    """Test actual list_documents response matches its outputSchema."""

    schema = TOOL_SCHEMAS['list_documents']

    # Create a test document first
    await mcp_client.call_tool('create_document', {'document_name': 'test_doc'})

    # Call list_documents
    response = await mcp_client.call_tool('list_documents', {})

    # Validate response against schema
    try:
        validate(instance=response, schema=schema)
    except ValidationError as e:
        pytest.fail(f"Response does not match schema: {e}")


@pytest.mark.asyncio
async def test_create_document_response_matches_schema(mcp_client):
    """Test actual create_document response matches its outputSchema."""

    schema = TOOL_SCHEMAS['create_document']

    response = await mcp_client.call_tool(
        'create_document',
        {'document_name': 'test_doc_2'}
    )

    try:
        validate(instance=response, schema=schema)
    except ValidationError as e:
        pytest.fail(f"Response does not match schema: {e}")
```

---

## Part 7: Fallback Strategy

### 7.1 If FastMCP Doesn't Support outputSchema Parameter

**Detection**:
```python
import inspect
from mcp.server import FastMCP

sig = inspect.signature(FastMCP.tool)
has_output_schema = 'outputSchema' in str(sig)
```

**Fallback 1: Post-Registration Modification** (Recommended)

```python
# In tool registration
def register_document_tools(mcp_server: FastMCP) -> None:

    @mcp_server.tool()
    @log_mcp_call
    def list_documents(include_chapters: bool = False) -> list[DocumentInfo]:
        """..."""
        # Implementation

    # Inject schema post-registration
    if hasattr(mcp_server, '_tools'):
        tool_obj = mcp_server._tools.get('list_documents')
        if tool_obj:
            tool_obj.outputSchema = LIST_DOCUMENTS_SCHEMA
```

**Fallback 2: Custom Tool Wrapper**

```python
from mcp.types import Tool

def register_tool_with_schema(
    mcp_server: FastMCP,
    name: str,
    description: str,
    input_schema: dict,
    output_schema: dict,
    func: callable
) -> None:
    """Register tool with explicit schema support."""

    # Create Tool definition manually
    tool = Tool(
        name=name,
        description=description,
        inputSchema=input_schema,
        outputSchema=output_schema
    )

    # Register with server
    mcp_server._tools[name] = tool
    # Also register callable
    setattr(mcp_server, f'_tool_{name}', func)
```

**Fallback 3: Server Initialization Hook**

```python
def apply_output_schemas(mcp_server: FastMCP) -> None:
    """Apply all output schemas to registered tools post-initialization."""

    for tool_name, schema in TOOL_SCHEMAS.items():
        if tool_name in mcp_server._tools:
            mcp_server._tools[tool_name].outputSchema = schema
        else:
            print(f"Warning: Tool {tool_name} not registered")

# Call after all tools are registered
register_document_tools(mcp_server)
register_chapter_tools(mcp_server)
# ... etc
apply_output_schemas(mcp_server)  # Inject all schemas
```

### 7.2 Testing Fallback Mechanisms

```python
def test_schema_injection_fallback():
    """Test schema injection works if direct parameter not supported."""

    mcp = FastMCP(name="test")

    @mcp.tool()
    def test_tool() -> dict:
        return {"result": "ok"}

    # Attempt fallback injection
    schema = {"type": "object"}
    if hasattr(mcp._tools['test_tool'], 'outputSchema'):
        mcp._tools['test_tool'].outputSchema = schema

    # Verify injection succeeded
    assert mcp._tools['test_tool'].outputSchema == schema
```

---

## Part 8: Quality Checklist

### 8.1 Schema Quality Standards

- [ ] All schemas are valid JSON Schema (Draft 2020-12)
- [ ] Schema descriptions match docstrings
- [ ] Required fields are correctly marked
- [ ] Type constraints are enforced
- [ ] Nested definitions use $ref correctly
- [ ] datetime fields marked with `"format": "date-time"`
- [ ] Enums specified where applicable
- [ ] No hardcoded null schemas without reason

### 8.2 Implementation Checklist

- [ ] Schema generator utility created and tested
- [ ] All 28 tool schemas defined and registered
- [ ] Top 10 tools integrated with outputSchema
- [ ] Remaining 18 tools updated
- [ ] No regressions in existing functionality
- [ ] Documentation updated
- [ ] CI/CD tests passing

### 8.3 Validation Checklist

- [ ] Unit tests for schema generation (>90% coverage)
- [ ] Integration tests for tool registration
- [ ] E2E tests validating responses against schemas
- [ ] Fallback mechanism tested if needed
- [ ] MCP 2025-06-18 compliance verified

---

## Part 9: Documentation Requirements

### 9.1 Schema Documentation File

**File**: `/docs/OUTPUTSCHEMA_SPECIFICATION.md`

```markdown
# Output Schema Specification

This document lists the output schema for each of the 28 MCP tools.

## Document Tools

### list_documents
Returns: Array of DocumentInfo objects
Schema: [schema JSON]
Example: [example JSON]

### read_content
Returns: Union of PaginatedContent, ChapterContent, ParagraphDetail, null
Schema: [schema JSON]

... (26 more tools)
```

### 9.2 Developer Guide Addition

**File**: `/docs/ADDING_NEW_TOOLS.md` (section addition)

```markdown
## Step 5: Define Output Schema

When creating a new tool with a Pydantic response model:

1. Add the schema to `document_mcp/tools/schemas.py`:
   ```python
   MY_TOOL_SCHEMA = pydantic_model_to_json_schema(MyResponseModel)
   TOOL_SCHEMAS['my_tool'] = MY_TOOL_SCHEMA
   ```

2. Apply to tool registration:
   ```python
   @mcp_server.tool(outputSchema=MY_TOOL_SCHEMA)
   def my_tool(...) -> MyResponseModel:
       pass
   ```

3. Add test case in `tests/unit/test_output_schemas.py`
```

### 9.3 Changelog Entry

```markdown
## [Unreleased]

### Added
- OutputSchema support for all 28 MCP tools (MCP 2025-06-18 compliance)
- JSON schema generator utility for Pydantic models
- Centralized tool schema registry in `document_mcp/tools/schemas.py`
- Schema validation tests ensuring all schemas are valid JSON Schema Draft 2020-12

### Changed
- Tool definitions now include `outputSchema` parameter
- Tool responses now machine-readable with strict validation
- Schema generation from Pydantic models automated

### Technical Details
- Added `document_mcp/utils/schema_generator.py`
- Added `document_mcp/tools/schemas.py` with 28 tool schemas
- Added schema validation in test suite
- No breaking changes to tool behavior or input schemas
```

---

## Part 10: Implementation Timeline

### Week 1: Infrastructure
- Days 1-2: Create schema generator utility
- Days 3-4: Create central schema registry
- Day 5: Documentation and unit test setup

**Deliverable**: `/document_mcp/utils/schema_generator.py`, `/tests/unit/test_schema_generation.py`

### Week 2-3: Phase 2 (Top 10 Tools)
- Week 2:
  - Mon-Tue: Tools #1-2 (list_documents, read_content)
  - Wed-Thu: Tools #3-5 (create_document, find_text, get_statistics)
  - Fri: Testing and documentation

- Week 3:
  - Mon-Tue: Tools #6-8 (find_similar_text, check_content_freshness, get_modification_history)
  - Wed-Thu: Tools #9-10 (get_document_outline, delete_document)
  - Fri: Integration testing and validation

**Deliverable**: All top 10 tools with outputSchema, passing tests

### Week 4: Phase 3 (Remaining 18 Tools)
- Mon-Tue: Batch A (5 tools) + Batch B (6 tools)
- Wed-Thu: Batch C (5 tools) + Batch D (2 tools)
- Fri: Final testing and documentation

**Deliverable**: All 28 tools with outputSchema, comprehensive documentation

### Week 5: Validation & Release
- Mon-Tue: E2E testing, performance validation
- Wed-Thu: Documentation review, examples
- Fri: Release preparation, changelog

**Deliverable**: MCP 2025-06-18 compliance achieved

---

## Part 11: Code Examples

### Example 1: Simple Tool (OperationStatus)

**Before**:
```python
@mcp_server.tool()
@log_mcp_call
def delete_document(document_name: str) -> OperationStatus:
    """Delete document and all chapters."""
    # Implementation
```

**After**:
```python
from .schemas import DELETE_DOCUMENT_SCHEMA

@mcp_server.tool(outputSchema=DELETE_DOCUMENT_SCHEMA)
@log_mcp_call
def delete_document(document_name: str) -> OperationStatus:
    """Delete document and all chapters."""
    # Implementation unchanged
```

**Schema Definition**:
```python
# In document_mcp/tools/schemas.py
DELETE_DOCUMENT_SCHEMA = pydantic_model_to_json_schema(OperationStatus)

# Which generates:
{
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "message": {"type": "string"},
        "details": {"type": "object", "additionalProperties": true},
        "safety_info": {},
        "snapshot_created": {"type": ["string", "null"]},
        "warnings": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["success", "message"]
}
```

### Example 2: Complex Tool (Union Return)

**Before**:
```python
@mcp_server.tool()
@log_mcp_call
def read_content(
    document_name: str,
    scope: str = "document",
    chapter_name: str | None = None,
    paragraph_index: int | None = None,
    page: int = 1,
    page_size: int = 50000,
) -> Any:
    """Unified content reading..."""
    if scope == "document":
        return PaginatedContent(...)
    elif scope == "chapter":
        return ChapterContent(...)
    elif scope == "paragraph":
        return ParagraphDetail(...)
    return None
```

**After**:
```python
from .schemas import READ_CONTENT_SCHEMA

@mcp_server.tool(outputSchema=READ_CONTENT_SCHEMA)
@log_mcp_call
def read_content(
    document_name: str,
    scope: str = "document",
    chapter_name: str | None = None,
    paragraph_index: int | None = None,
    page: int = 1,
    page_size: int = 50000,
) -> PaginatedContent | ChapterContent | ParagraphDetail | None:
    """Unified content reading..."""
    # Implementation unchanged
```

**Schema Definition**:
```python
# In document_mcp/tools/schemas.py
READ_CONTENT_SCHEMA = {
    "anyOf": [
        {"$ref": "#/$defs/PaginatedContent"},
        {"$ref": "#/$defs/ChapterContent"},
        {"$ref": "#/$defs/ParagraphDetail"},
        {"type": "null"}
    ],
    "$defs": {
        "PaginatedContent": pydantic_model_to_json_schema(
            PaginatedContent,
            include_definitions=False
        ),
        "ChapterContent": pydantic_model_to_json_schema(
            ChapterContent,
            include_definitions=False
        ),
        "ParagraphDetail": pydantic_model_to_json_schema(
            ParagraphDetail,
            include_definitions=False
        ),
    }
}
```

### Example 3: Array Return Type

**Before**:
```python
@mcp_server.tool()
@log_mcp_call
def list_documents(include_chapters: bool = False) -> list[DocumentInfo]:
    """List documents..."""
    # Implementation
```

**After**:
```python
from .schemas import LIST_DOCUMENTS_SCHEMA

@mcp_server.tool(outputSchema=LIST_DOCUMENTS_SCHEMA)
@log_mcp_call
def list_documents(include_chapters: bool = False) -> list[DocumentInfo]:
    """List documents..."""
    # Implementation unchanged
```

**Schema Definition**:
```python
# In document_mcp/tools/schemas.py
LIST_DOCUMENTS_SCHEMA = pydantic_list_to_json_schema(DocumentInfo)

# Which generates:
{
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "document_name": {"type": "string"},
            "total_chapters": {"type": "integer"},
            "total_word_count": {"type": "integer"},
            "total_paragraph_count": {"type": "integer"},
            "last_modified": {"type": "string", "format": "date-time"},
            "chapters": {
                "type": "array",
                "items": {"$ref": "#/$defs/ChapterMetadata"}
            },
            "has_summary": {"type": "boolean"}
        },
        "required": [...],
        "$defs": {
            "ChapterMetadata": {...}
        }
    }
}
```

---

## Part 12: Risk Assessment & Mitigation

### 12.1 Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| FastMCP doesn't support outputSchema param | Medium | High | Fallback mechanisms, early testing |
| Pydantic schema generation produces invalid JSON Schema | Low | High | Comprehensive unit tests, JSON Schema validator |
| Breaking changes to existing tools | Low | High | Comprehensive testing, no behavior changes |
| Performance impact from schema generation | Low | Medium | Lazy schema loading, schema caching |
| Incomplete type hints in existing code | Medium | Medium | Code analysis, manual schema definition |
| MCP client incompatibility | Low | Medium | Version testing, fallback compatibility mode |

### 12.2 Mitigation Strategies

1. **Early Testing**: Week 1 - Test FastMCP support immediately
2. **Fallback Options**: Develop fallback 1-3 in parallel
3. **Validation Framework**: Comprehensive JSON Schema validation
4. **Gradual Rollout**: Top 10 tools first, validate, then complete
5. **Documentation**: Clear schema patterns for future tools
6. **Testing Coverage**: >90% test coverage for schema code

---

## Conclusion

This implementation plan provides a systematic approach to adding MCP 2025-06-18 outputSchema support to Document MCP. The phased approach, comprehensive testing strategy, and fallback mechanisms ensure production-ready deployment with minimal risk.

**Key Success Factors**:
- Week 1 infrastructure creates reusable utilities
- Top 10 tools serve as reference implementation
- Centralized schema registry enables scalability
- Comprehensive testing validates correctness
- Fallback mechanisms ensure compatibility

**Expected Outcome**: All 28 tools with machine-readable output schemas, enabling MCP clients to validate responses and provide better error handling.

