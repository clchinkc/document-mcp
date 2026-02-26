# FastMCP OutputSchema Integration Patterns

This document provides technical patterns for integrating outputSchema with FastMCP framework in Document MCP.

## Research Findings

### FastMCP Compatibility

**FastMCP Version**: Latest (supports MCP 2025-06-18)

**Tool Decorator Signature**:
```python
@tool(
    name: str | None = None,
    title: str | None = None,
    description: str | None = None,
    annotations: ToolAnnotations | None = None,
    structured_output: bool | None = None,
    # Potential: outputSchema parameter
) -> Callable
```

**MCP Tool Type**:
```python
from mcp.types import Tool

Tool(
    name: str,
    description: str,
    inputSchema: dict[str, Any],
    outputSchema: dict[str, Any] | None = None,  # Supported!
    annotations: ToolAnnotations | None = None,
)
```

**Key Finding**: The underlying MCP protocol supports `outputSchema`, but FastMCP's `@tool()` decorator may need adaptation.

## Pattern 1: Direct Parameter (Preferred)

### Assumption
FastMCP accepts `outputSchema` as a decorator parameter.

### Implementation

**File**: `document_mcp/tools/document_tools.py`

```python
from .schemas import LIST_DOCUMENTS_SCHEMA

def register_document_tools(mcp_server: FastMCP) -> None:

    @mcp_server.tool(outputSchema=LIST_DOCUMENTS_SCHEMA)
    @log_mcp_call
    def list_documents(include_chapters: bool = False) -> list[DocumentInfo]:
        """List all available documents."""
        docs_info = []
        docs_root = DOCS_ROOT_PATH.resolve()

        if not docs_root.exists():
            return docs_info

        for doc_dir in docs_root.iterdir():
            if not doc_dir.is_dir():
                continue

            doc_name = doc_dir.name
            chapters = _get_ordered_chapter_files(doc_name)

            if not chapters:
                continue

            # ... rest of implementation
```

### Testing

```python
def test_tool_has_outputschema():
    """Test that tool decorator accepts outputSchema parameter."""
    from document_mcp.doc_tool_server import mcp_server

    tool = mcp_server._tools.get('list_documents')
    assert tool is not None
    assert hasattr(tool, 'outputSchema')
    assert tool.outputSchema == LIST_DOCUMENTS_SCHEMA
```

## Pattern 2: Post-Registration Injection

### Assumption
FastMCP doesn't accept `outputSchema` directly, but tools can be modified after registration.

### Implementation

**File**: `document_mcp/tools/__init__.py`

```python
from mcp.server import FastMCP
from .schemas import TOOL_SCHEMAS


def inject_output_schemas(mcp_server: FastMCP) -> None:
    """Inject outputSchema into registered tools.

    Call after all tools are registered but before server starts.
    """
    for tool_name, schema in TOOL_SCHEMAS.items():
        if tool_name not in mcp_server._tools:
            print(f"Warning: Tool '{tool_name}' not registered")
            continue

        tool = mcp_server._tools[tool_name]
        tool.outputSchema = schema

        print(f"✓ Added schema to {tool_name}")


# Usage in doc_tool_server.py
register_document_tools(mcp_server)
register_chapter_tools(mcp_server)
# ... register all tools ...

# Then inject schemas
inject_output_schemas(mcp_server)
```

### Testing

```python
@pytest.mark.asyncio
async def test_schema_injection():
    """Test schema injection works correctly."""
    from document_mcp.tools import inject_output_schemas
    from document_mcp.doc_tool_server import mcp_server

    # After injection
    inject_output_schemas(mcp_server)

    # Verify schemas are present
    for tool_name in ['list_documents', 'create_document']:
        tool = mcp_server._tools[tool_name]
        assert hasattr(tool, 'outputSchema')
        assert tool.outputSchema is not None
        assert isinstance(tool.outputSchema, dict)
```

## Pattern 3: Custom Tool Wrapper

### Assumption
Need more control over tool registration and schema attachment.

### Implementation

**File**: `document_mcp/utils/tool_wrapper.py`

```python
"""Tool wrapper for MCP 2025-06-18 outputSchema support."""

from typing import Any, Callable, TypeVar
from mcp.server import FastMCP
from mcp.types import Tool, ToolAnnotations

F = TypeVar('F', bound=Callable[..., Any])


class ToolWithSchema:
    """Wrapper for registering tools with outputSchema."""

    def __init__(
        self,
        mcp_server: FastMCP,
        name: str,
        description: str,
        output_schema: dict[str, Any],
        annotations: ToolAnnotations | None = None,
    ):
        self.mcp_server = mcp_server
        self.name = name
        self.description = description
        self.output_schema = output_schema
        self.annotations = annotations

    def __call__(self, func: F) -> F:
        """Decorator to register tool with schema."""
        # Register tool normally first
        self.mcp_server.tool(
            name=self.name,
            description=self.description,
            annotations=self.annotations,
        )(func)

        # Then inject schema
        if self.name in self.mcp_server._tools:
            tool = self.mcp_server._tools[self.name]
            tool.outputSchema = self.output_schema

        return func


# Usage
def register_document_tools(mcp_server: FastMCP) -> None:
    from .schemas import LIST_DOCUMENTS_SCHEMA

    @ToolWithSchema(
        mcp_server,
        name="list_documents",
        description="List all available documents",
        output_schema=LIST_DOCUMENTS_SCHEMA,
    )
    def list_documents(include_chapters: bool = False) -> list[DocumentInfo]:
        """Implementation..."""
        ...
```

## Pattern 4: Schema Server Extension

### Assumption
Want to extend FastMCP server with schema capabilities.

### Implementation

**File**: `document_mcp/server.py`

```python
"""Extended MCP server with outputSchema support."""

from mcp.server import FastMCP
from mcp.types import Tool


class FastMCPWithSchema(FastMCP):
    """FastMCP extension with outputSchema support."""

    def tool_with_schema(
        self,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        outputSchema: dict | None = None,
    ):
        """Register tool with outputSchema support.

        Args:
            name: Tool name
            title: Human-readable title
            description: Tool description
            outputSchema: JSON schema for output
        """

        def decorator(func):
            # Register with standard tool decorator
            decorated = self.tool(
                name=name,
                title=title,
                description=description,
            )(func)

            # Inject outputSchema
            tool_name = name or func.__name__
            if tool_name in self._tools and outputSchema:
                self._tools[tool_name].outputSchema = outputSchema

            return decorated

        return decorator


# Usage
from document_mcp.server import FastMCPWithSchema
from document_mcp.tools.schemas import LIST_DOCUMENTS_SCHEMA

mcp_server = FastMCPWithSchema(name="DocumentManagementTools")


@mcp_server.tool_with_schema(outputSchema=LIST_DOCUMENTS_SCHEMA)
def list_documents(include_chapters: bool = False) -> list[DocumentInfo]:
    """List documents..."""
    ...
```

## Pattern 5: Centralized Registry with Post-Processing

### Assumption
Want single source of truth for schemas with automatic validation.

### Implementation

**File**: `document_mcp/tools/schemas.py`

```python
"""Centralized schema registry for all MCP tools."""

from typing import Any
from document_mcp.models import DocumentInfo, OperationStatus, ChapterContent
from document_mcp.utils.schema_generator import (
    pydantic_model_to_json_schema,
    pydantic_list_to_json_schema,
    validate_schema_against_json_schema_spec,
)

# Schema definitions
_SCHEMAS = {
    "list_documents": pydantic_list_to_json_schema(DocumentInfo),
    "create_document": pydantic_model_to_json_schema(OperationStatus),
    "read_chapter_content": pydantic_model_to_json_schema(ChapterContent),
    # ... 25 more tools
}

# Validate all schemas at import time
for tool_name, schema in _SCHEMAS.items():
    is_valid, errors = validate_schema_against_json_schema_spec(schema)
    if not is_valid:
        raise ValueError(
            f"Invalid schema for {tool_name}: {errors}"
        )

# Export
TOOL_SCHEMAS = _SCHEMAS


def apply_schemas_to_server(mcp_server) -> None:
    """Apply all schemas to registered tools.

    Args:
        mcp_server: FastMCP server instance
    """
    for tool_name, schema in TOOL_SCHEMAS.items():
        if tool_name not in mcp_server._tools:
            print(f"Warning: Tool '{tool_name}' not registered")
            continue

        tool = mcp_server._tools[tool_name]
        tool.outputSchema = schema
        print(f"✓ Registered schema for {tool_name}")


# Usage in doc_tool_server.py
from document_mcp.tools.schemas import apply_schemas_to_server

# ... register all tools ...
apply_schemas_to_server(mcp_server)
```

## Testing Patterns

### Test 1: Verify Schema Parameter Support

```python
import inspect
from mcp.server import FastMCP

# Check if outputSchema is supported
sig = inspect.signature(FastMCP.tool)
params = list(sig.parameters.keys())

if 'outputSchema' in params:
    print("✓ FastMCP supports outputSchema parameter")
else:
    print("✗ FastMCP does not support outputSchema parameter")
    print("  Using fallback injection method")
```

### Test 2: End-to-End Schema Integration

```python
@pytest.mark.asyncio
async def test_tool_schema_in_mcp_protocol():
    """Test that schema appears in MCP protocol messages."""
    from document_mcp.doc_tool_server import mcp_server

    # Get tool definition (as MCP would see it)
    tool = mcp_server._tools.get('list_documents')

    assert tool is not None
    assert hasattr(tool, 'outputSchema')

    # Verify schema is valid
    from jsonschema import Draft202012Validator
    validator = Draft202012Validator(tool.outputSchema)
    assert validator.is_valid(tool.outputSchema)
```

### Test 3: Response Validation Against Schema

```python
@pytest.mark.asyncio
async def test_response_matches_schema():
    """Test actual tool response matches declared schema."""
    from document_mcp.mcp_client import MCPClient
    from document_mcp.doc_tool_server import mcp_server
    from jsonschema import validate

    async with MCPClient(mcp_server) as client:
        # Call tool
        response = await client.call_tool('list_documents', {})

        # Get schema
        schema = mcp_server._tools['list_documents'].outputSchema

        # Validate response
        validate(instance=response, schema=schema)
        # Should not raise
```

## Fallback Detection and Handling

### Detect Which Pattern to Use

```python
def detect_fastmcp_schema_support() -> str:
    """Detect which pattern FastMCP supports.

    Returns:
        - 'direct': FastMCP accepts outputSchema parameter
        - 'injection': Supports post-registration injection
        - 'wrapper': Need custom wrapper
    """
    import inspect
    from mcp.server import FastMCP

    # Check 1: Direct parameter support
    sig = inspect.signature(FastMCP.tool)
    if 'outputSchema' in sig.parameters:
        return 'direct'

    # Check 2: Can inject into _tools dict
    test_server = FastMCP(name="test")

    @test_server.tool()
    def test_tool() -> dict:
        return {}

    try:
        test_tool_obj = test_server._tools.get('test_tool')
        test_tool_obj.outputSchema = {"type": "object"}
        if test_tool_obj.outputSchema == {"type": "object"}:
            return 'injection'
    except (AttributeError, TypeError):
        pass

    # Check 3: Need wrapper
    return 'wrapper'


# Usage
pattern = detect_fastmcp_schema_support()
print(f"Using pattern: {pattern}")
```

### Automatic Pattern Selection

```python
def register_all_tools_with_schemas(mcp_server: FastMCP) -> None:
    """Register all tools with appropriate schema pattern."""
    from document_mcp.tools import (
        register_document_tools,
        register_chapter_tools,
        # ... all other tool modules
    )
    from document_mcp.tools.schemas import (
        TOOL_SCHEMAS,
        apply_schemas_to_server,
    )

    # Register all tools
    register_document_tools(mcp_server)
    register_chapter_tools(mcp_server)
    # ... register all categories ...

    # Detect and apply schemas
    pattern = detect_fastmcp_schema_support()

    if pattern == 'direct':
        print("✓ Using direct parameter pattern")
        # Schemas should already be applied via decorator
    elif pattern == 'injection':
        print("✓ Using injection pattern")
        apply_schemas_to_server(mcp_server)
    else:
        print("✗ Need custom wrapper implementation")
        # Implement wrapper pattern
```

## Complete Integration Example

Here's the complete flow for one tool using Pattern 2 (Post-Registration Injection):

```python
# File: document_mcp/tools/schemas.py
from document_mcp.utils.schema_generator import pydantic_list_to_json_schema
from document_mcp.models import DocumentInfo

LIST_DOCUMENTS_SCHEMA = pydantic_list_to_json_schema(DocumentInfo)

TOOL_SCHEMAS = {
    "list_documents": LIST_DOCUMENTS_SCHEMA,
    # ... more schemas
}


# File: document_mcp/tools/document_tools.py
from .schemas import LIST_DOCUMENTS_SCHEMA

def register_document_tools(mcp_server: FastMCP) -> None:

    @mcp_server.tool()  # No outputSchema parameter yet
    @log_mcp_call
    def list_documents(include_chapters: bool = False) -> list[DocumentInfo]:
        """List all available documents."""
        # Implementation unchanged
        ...


# File: document_mcp/doc_tool_server.py
from .tools import register_document_tools, register_chapter_tools, ...
from .tools.schemas import TOOL_SCHEMAS

# Register all tools
register_document_tools(mcp_server)
register_chapter_tools(mcp_server)
# ... register all categories ...

# Inject schemas (fallback if direct parameter not supported)
for tool_name, schema in TOOL_SCHEMAS.items():
    if tool_name in mcp_server._tools:
        mcp_server._tools[tool_name].outputSchema = schema
```

## Migration Path

If starting with Pattern 2 (Injection) and later FastMCP adds direct support:

```python
# Old code (Pattern 2 - Injection)
@mcp_server.tool()
def list_documents(...) -> list[DocumentInfo]:
    ...

# Later: FastMCP adds outputSchema support
# New code (Pattern 1 - Direct)
from .schemas import LIST_DOCUMENTS_SCHEMA

@mcp_server.tool(outputSchema=LIST_DOCUMENTS_SCHEMA)
def list_documents(...) -> list[DocumentInfo]:
    ...

# Post-registration injection code can be removed
```

## Debugging

### Check if Schema is Applied

```python
from document_mcp.doc_tool_server import mcp_server

def check_schemas():
    """Debug: Check which tools have schemas."""
    for name, tool in mcp_server._tools.items():
        has_schema = hasattr(tool, 'outputSchema') and tool.outputSchema is not None
        status = "✓" if has_schema else "✗"
        print(f"{status} {name}")

check_schemas()
```

### Validate Schema Quality

```python
from document_mcp.doc_tool_server import mcp_server
from document_mcp.utils.schema_generator import validate_schema_against_json_schema_spec

def validate_all_schemas():
    """Debug: Validate all schemas."""
    for name, tool in mcp_server._tools.items():
        if not hasattr(tool, 'outputSchema') or tool.outputSchema is None:
            print(f"⚠ {name}: No schema")
            continue

        is_valid, errors = validate_schema_against_json_schema_spec(tool.outputSchema)
        if is_valid:
            print(f"✓ {name}: Valid")
        else:
            print(f"✗ {name}: Invalid")
            for error in errors:
                print(f"   - {error}")

validate_all_schemas()
```

## Summary

| Pattern | Complexity | Support | Recommendation |
|---------|-----------|---------|-----------------|
| 1: Direct Parameter | Low | FastMCP future release | Use if supported |
| 2: Post-Registration | Low | Current FastMCP | Use now (best) |
| 3: Custom Wrapper | Medium | Full control | Use if 1-2 fail |
| 4: Server Extension | High | Most flexible | Use for advanced needs |
| 5: Registry + Processing | Medium | Production-ready | Use for batch operations |

**Recommended**: Start with Pattern 2 (Post-Registration Injection), migrate to Pattern 1 if FastMCP adds support.
