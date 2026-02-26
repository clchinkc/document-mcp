"""JSON Schema generation from Pydantic models for MCP tools.

This module provides utilities to convert Pydantic BaseModel subclasses
to JSON schemas compatible with MCP 2025-06-18 outputSchema specification.

Key Features:
- Automatic JSON schema generation from Pydantic models
- Support for union types (e.g., T | None)
- Support for list types (e.g., list[T])
- Nested model definitions with $ref references
- MCP 2025-06-18 compliance
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema
from pydantic.json_schema import JsonSchemaMode


class MCP2025JsonSchema(GenerateJsonSchema):
    """Custom JSON schema generator for MCP 2025 compliance.

    Customizes Pydantic's JSON schema generation to:
    - Preserve model docstrings as descriptions
    - Use MCP-compatible reference format
    - Optimize schema structure for tools
    """

    def generate(self, schema: Any, mode: JsonSchemaMode = "validation") -> dict[str, Any]:
        """Generate schema with MCP 2025 optimizations.

        Args:
            schema: Pydantic model or type to generate schema for
            mode: 'validation' or 'serialization' (default: validation)

        Returns:
            JSON schema dict
        """
        json_schema = super().generate(schema, mode=mode)

        # Ensure title and description are preserved
        if hasattr(schema, "__doc__") and schema.__doc__:
            json_schema["description"] = schema.__doc__.strip()

        return json_schema


def pydantic_model_to_json_schema(
    model: type[BaseModel],
    mode: JsonSchemaMode = "serialization",
    include_definitions: bool = True,
    by_alias: bool = True,
) -> dict[str, Any]:
    """Convert Pydantic model to JSON schema for MCP tool output.

    This function generates a JSON schema that represents the structure
    of a Pydantic model's serialized output. It's the primary utility for
    creating outputSchema definitions for MCP tools.

    Args:
        model: Pydantic BaseModel subclass to convert
        mode: 'validation' (input schema) or 'serialization' (output schema).
            Default: 'serialization' for tool responses.
        include_definitions: Include $defs for nested models.
            Default: True for standalone schemas, False for embedding in unions.
        by_alias: Use field aliases in schema. Default: True.

    Returns:
        JSON schema dict compatible with MCP outputSchema

    Example:
        ```python
        from story_mcp.models import DocumentInfo
        from story_mcp.utils.schema_generator import pydantic_model_to_json_schema

        schema = pydantic_model_to_json_schema(DocumentInfo)
        # Returns:
        # {
        #   "type": "object",
        #   "properties": {
        #     "document_name": {"type": "string"},
        #     "total_chapters": {"type": "integer"},
        #     ...
        #   },
        #   "$defs": {
        #     "ChapterMetadata": {...}
        #   }
        # }
        ```
    """
    schema = model.model_json_schema(
        mode=mode,
        by_alias=by_alias,
        ref_template="#/$defs/{model}",
    )

    # Remove $schema URI for MCP (unnecessary in tool outputs)
    schema.pop("$schema", None)

    # If not including definitions, extract and remove $defs
    if not include_definitions:
        schema.pop("$defs", None)

    return schema


def pydantic_union_to_json_schema(
    *models: type[BaseModel],
    allow_null: bool = True,
) -> dict[str, Any]:
    """Generate schema for union of models (e.g., T1 | T2 | T3 | None).

    Creates a JSON schema using anyOf construct to represent a union of
    multiple possible response types, with optional null support.

    Args:
        models: Variable number of Pydantic model types
        allow_null: Include null in anyOf. Default: True.

    Returns:
        JSON schema with anyOf construct and merged $defs

    Example:
        ```python
        from story_mcp.models import DocumentInfo, OperationStatus
        from story_mcp.utils.schema_generator import pydantic_union_to_json_schema

        schema = pydantic_union_to_json_schema(
            DocumentInfo,
            OperationStatus,
            allow_null=True
        )
        # Returns:
        # {
        #   "anyOf": [
        #     {"$ref": "#/$defs/DocumentInfo"},
        #     {"$ref": "#/$defs/OperationStatus"},
        #     {"type": "null"}
        #   ],
        #   "$defs": {
        #     "DocumentInfo": {...},
        #     "OperationStatus": {...}
        #   }
        # }
        ```
    """
    schemas = []
    all_defs: dict[str, Any] = {}

    # Generate schema for each model
    for model in models:
        model_schema = pydantic_model_to_json_schema(
            model,
            include_definitions=True,
        )

        # Extract $defs from this model
        if "$defs" in model_schema:
            all_defs.update(model_schema.pop("$defs"))

        # Add the model itself to definitions
        all_defs[model.__name__] = model_schema

        # Add reference to anyOf
        schemas.append({"$ref": f"#/$defs/{model.__name__}"})

    # Add null if requested
    if allow_null:
        schemas.append({"type": "null"})

    # Build final schema
    result: dict[str, Any] = {"anyOf": schemas}

    # Add merged definitions
    if all_defs:
        result["$defs"] = all_defs

    return result


def pydantic_list_to_json_schema(
    item_model: type[BaseModel],
) -> dict[str, Any]:
    """Generate schema for list of models (e.g., list[T]).

    Creates a JSON schema representing an array of objects where each
    object conforms to the item_model schema.

    Args:
        item_model: Pydantic model type for array items

    Returns:
        JSON schema for array with merged $defs

    Example:
        ```python
        from story_mcp.models import DocumentInfo
        from story_mcp.utils.schema_generator import pydantic_list_to_json_schema

        schema = pydantic_list_to_json_schema(DocumentInfo)
        # Returns:
        # {
        #   "type": "array",
        #   "items": {
        #     "$ref": "#/$defs/DocumentInfo"
        #   },
        #   "$defs": {
        #     "DocumentInfo": {...},
        #     "ChapterMetadata": {...}
        #   }
        # }
        ```
    """
    item_schema = pydantic_model_to_json_schema(
        item_model,
        include_definitions=True,
    )

    # Extract definitions
    defs = item_schema.pop("$defs", {})

    result: dict[str, Any] = {
        "type": "array",
        "items": item_schema,
    }

    # Add definitions if present
    if defs:
        result["$defs"] = defs

    return result


def create_flexible_object_schema(
    description: str = "Flexible object with arbitrary properties",
) -> dict[str, Any]:
    """Create schema for flexible object (dict[str, Any]).

    Use this when a tool returns a dictionary with arbitrary keys/values
    that cannot be represented by a Pydantic model.

    Args:
        description: Description of what the object contains

    Returns:
        JSON schema allowing arbitrary properties

    Example:
        ```python
        schema = create_flexible_object_schema(
            "Tool output with configuration options"
        )
        # Returns:
        # {
        #   "type": "object",
        #   "additionalProperties": true,
        #   "description": "Tool output with configuration options"
        # }
        ```
    """
    return {
        "type": "object",
        "additionalProperties": True,
        "description": description,
    }


def create_oneOf_schema(
    *schemas: dict[str, Any],
) -> dict[str, Any]:
    """Create schema for exactly one of multiple possibilities (oneOf).

    Use this for union types where exactly one alternative applies,
    unlike anyOf where multiple could apply.

    Args:
        schemas: Variable number of JSON schemas

    Returns:
        JSON schema with oneOf construct

    Example:
        ```python
        schema = create_oneOf_schema(
            pydantic_model_to_json_schema(SnapshotsList),
            pydantic_model_to_json_schema(OperationStatus)
        )
        # Returns:
        # {
        #   "oneOf": [
        #     {...SnapshotsList schema...},
        #     {...OperationStatus schema...}
        #   ]
        # }
        ```
    """
    return {"oneOf": list(schemas)}


# Registry of tool output schemas
_TOOL_OUTPUT_SCHEMAS: dict[str, dict[str, Any]] = {}


def register_tool_schema(
    tool_name: str,
    schema: dict[str, Any],
    overwrite: bool = False,
) -> None:
    """Register a tool's output schema for reference.

    Maintains a central registry of all tool schemas for validation,
    documentation, and testing purposes.

    Args:
        tool_name: Name of the MCP tool (must be unique)
        schema: JSON schema dict (must be valid JSON Schema Draft 2020-12)
        overwrite: Allow overwriting existing schema. Default: False.

    Raises:
        ValueError: If tool_name already registered and overwrite=False
        TypeError: If schema is not a dict

    Example:
        ```python
        from story_mcp.utils.schema_generator import (
            register_tool_schema,
            pydantic_list_to_json_schema
        )
        from story_mcp.models import DocumentInfo

        schema = pydantic_list_to_json_schema(DocumentInfo)
        register_tool_schema('list_documents', schema)
        ```
    """
    if not isinstance(schema, dict):
        raise TypeError(f"schema must be dict, got {type(schema)}")

    if tool_name in _TOOL_OUTPUT_SCHEMAS and not overwrite:
        raise ValueError(
            f"Tool '{tool_name}' schema already registered. "
            f"Use overwrite=True to replace."
        )

    _TOOL_OUTPUT_SCHEMAS[tool_name] = schema


def get_tool_schema(tool_name: str) -> dict[str, Any] | None:
    """Retrieve registered schema for a tool by name.

    Args:
        tool_name: Name of the tool

    Returns:
        JSON schema dict or None if tool not registered

    Example:
        ```python
        schema = get_tool_schema('list_documents')
        if schema:
            print(f"Found schema for list_documents")
        ```
    """
    return _TOOL_OUTPUT_SCHEMAS.get(tool_name)


def get_all_tool_schemas() -> dict[str, dict[str, Any]]:
    """Get all registered tool schemas.

    Returns:
        Dict mapping tool names to their schemas

    Example:
        ```python
        all_schemas = get_all_tool_schemas()
        print(f"Registered schemas for {len(all_schemas)} tools")
        ```
    """
    return _TOOL_OUTPUT_SCHEMAS.copy()


def validate_schema_against_json_schema_spec(
    schema: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Validate that a schema conforms to JSON Schema Draft 2020-12.

    Args:
        schema: Schema to validate

    Returns:
        Tuple of (is_valid, error_messages)

    Note:
        This performs basic validation. For comprehensive validation,
        use the jsonschema library's Draft202012Validator.

    Example:
        ```python
        from story_mcp.utils.schema_generator import (
            pydantic_model_to_json_schema,
            validate_schema_against_json_schema_spec
        )

        schema = pydantic_model_to_json_schema(DocumentInfo)
        is_valid, errors = validate_schema_against_json_schema_spec(schema)
        if not is_valid:
            for error in errors:
                print(f"Schema error: {error}")
        ```
    """
    errors: list[str] = []

    if not isinstance(schema, dict):
        errors.append(f"Schema must be dict, got {type(schema)}")
        return False, errors

    # Check for required type field
    if (
        "type" not in schema
        and "anyOf" not in schema
        and "$ref" not in schema
        and "items" not in schema
    ):  # Special case: root-level array
        errors.append(
            "Schema must have 'type', 'anyOf', '$ref', or 'items' field"
        )

    # Check oneOf/anyOf/allOf consistency
    exclusive_fields = ("oneOf", "anyOf", "allOf")
    if sum(field in schema for field in exclusive_fields) > 1:
        errors.append("Schema has multiple exclusive fields: oneOf, anyOf, allOf")

    # Validate nested schemas
    if "properties" in schema and isinstance(schema["properties"], dict):
        for prop_name, prop_schema in schema["properties"].items():
            if isinstance(prop_schema, dict) and prop_schema:
                _, prop_errors = validate_schema_against_json_schema_spec(prop_schema)
                for error in prop_errors:
                    errors.append(f"Property '{prop_name}': {error}")

    # Validate items schema
    if "items" in schema and isinstance(schema["items"], dict):
        _, items_errors = validate_schema_against_json_schema_spec(schema["items"])
        for error in items_errors:
            errors.append(f"Array items: {error}")

    return len(errors) == 0, errors


__all__ = [
    "MCP2025JsonSchema",
    "pydantic_model_to_json_schema",
    "pydantic_union_to_json_schema",
    "pydantic_list_to_json_schema",
    "create_flexible_object_schema",
    "create_oneOf_schema",
    "register_tool_schema",
    "get_tool_schema",
    "get_all_tool_schemas",
    "validate_schema_against_json_schema_spec",
]
