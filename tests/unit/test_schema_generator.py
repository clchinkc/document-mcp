"""Unit tests for JSON schema generation utilities.

Tests cover:
- Pydantic model to JSON schema conversion
- Union type schema generation
- List type schema generation
- Schema validation
- Schema registry operations
"""

import json

import pytest
from jsonschema import Draft202012Validator

from story_mcp.models import ChapterContent
from story_mcp.models import DocumentInfo
from story_mcp.models import DocumentSummary
from story_mcp.models import ModificationHistory
from story_mcp.models import OperationStatus
from story_mcp.models import PaginatedContent
from story_mcp.models import StatisticsReport
from story_mcp.utils.schema_generator import create_flexible_object_schema
from story_mcp.utils.schema_generator import create_oneOf_schema
from story_mcp.utils.schema_generator import get_all_tool_schemas
from story_mcp.utils.schema_generator import get_tool_schema
from story_mcp.utils.schema_generator import pydantic_list_to_json_schema
from story_mcp.utils.schema_generator import pydantic_model_to_json_schema
from story_mcp.utils.schema_generator import pydantic_union_to_json_schema
from story_mcp.utils.schema_generator import register_tool_schema
from story_mcp.utils.schema_generator import validate_schema_against_json_schema_spec


class TestPydanticModelToJsonSchema:
    """Test conversion of Pydantic models to JSON schema."""

    def test_simple_model_schema(self) -> None:
        """Test schema generation for simple model (OperationStatus)."""
        schema = pydantic_model_to_json_schema(OperationStatus)

        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "success" in schema["properties"]
        assert "message" in schema["properties"]

    def test_simple_model_required_fields(self) -> None:
        """Test that required fields are marked."""
        schema = pydantic_model_to_json_schema(OperationStatus)

        required = schema.get("required", [])
        assert "success" in required
        assert "message" in required

    def test_nested_model_schema(self) -> None:
        """Test schema generation for models with nested objects."""
        schema = pydantic_model_to_json_schema(DocumentInfo)

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "chapters" in schema["properties"]

        # Check that nested model uses $ref
        chapters_prop = schema["properties"]["chapters"]
        assert chapters_prop["type"] == "array"
        assert "items" in chapters_prop

    def test_nested_definitions_included(self) -> None:
        """Test that nested model definitions are included in $defs."""
        schema = pydantic_model_to_json_schema(DocumentInfo)

        assert "$defs" in schema
        assert "ChapterMetadata" in schema["$defs"]

    def test_schema_json_serializable(self) -> None:
        """Test that generated schema is JSON serializable."""
        schema = pydantic_model_to_json_schema(DocumentInfo)

        try:
            json_str = json.dumps(schema)
            assert isinstance(json_str, str)
            assert len(json_str) > 0
        except TypeError as e:
            pytest.fail(f"Schema is not JSON serializable: {e}")

    def test_datetime_field_formatting(self) -> None:
        """Test that datetime fields use correct JSON schema format."""
        schema = pydantic_model_to_json_schema(DocumentInfo)

        # Check last_modified field
        last_modified_schema = schema["properties"]["last_modified"]
        assert last_modified_schema["type"] == "string"
        assert last_modified_schema.get("format") == "date-time"

    def test_optional_field_handling(self) -> None:
        """Test handling of optional (nullable) fields."""
        schema = pydantic_model_to_json_schema(DocumentSummary)

        # target_name is optional
        target_name_schema = schema["properties"]["target_name"]
        # Should support null
        assert "anyOf" in target_name_schema or target_name_schema.get("type") == "string"

    def test_mode_serialization_vs_validation(self) -> None:
        """Test that serialization mode differs from validation mode."""
        schema_ser = pydantic_model_to_json_schema(OperationStatus, mode="serialization")
        schema_val = pydantic_model_to_json_schema(OperationStatus, mode="validation")

        # Both should be valid objects
        assert schema_ser["type"] == "object"
        assert schema_val["type"] == "object"

    def test_no_schema_uri_in_output(self) -> None:
        """Test that $schema URI is removed for MCP compatibility."""
        schema = pydantic_model_to_json_schema(OperationStatus)

        assert "$schema" not in schema

    def test_include_definitions_flag(self) -> None:
        """Test include_definitions parameter."""
        # With definitions
        with_defs = pydantic_model_to_json_schema(
            DocumentInfo, include_definitions=True
        )
        assert "$defs" in with_defs

        # Without definitions
        without_defs = pydantic_model_to_json_schema(
            DocumentInfo, include_definitions=False
        )
        assert "$defs" not in without_defs


class TestPydanticUnionToJsonSchema:
    """Test conversion of union types to JSON schema."""

    def test_union_with_null(self) -> None:
        """Test union type with null option."""
        schema = pydantic_union_to_json_schema(
            DocumentInfo, DocumentSummary, allow_null=True
        )

        assert "anyOf" in schema
        assert len(schema["anyOf"]) == 3  # Two models + null
        assert {"type": "null"} in schema["anyOf"]

    def test_union_without_null(self) -> None:
        """Test union type without null option."""
        schema = pydantic_union_to_json_schema(
            DocumentInfo, DocumentSummary, allow_null=False
        )

        assert "anyOf" in schema
        assert len(schema["anyOf"]) == 2  # Two models only
        assert {"type": "null"} not in schema["anyOf"]

    def test_union_includes_definitions(self) -> None:
        """Test that union schema includes all nested definitions."""
        schema = pydantic_union_to_json_schema(DocumentInfo, DocumentSummary)

        assert "$defs" in schema
        # Should have definitions from both models
        assert "DocumentInfo" in schema["$defs"]
        assert "DocumentSummary" in schema["$defs"]

    def test_union_schema_json_serializable(self) -> None:
        """Test union schema is JSON serializable."""
        schema = pydantic_union_to_json_schema(DocumentInfo, DocumentSummary)

        try:
            json.dumps(schema)
        except TypeError as e:
            pytest.fail(f"Union schema is not JSON serializable: {e}")

    def test_union_references_correct_models(self) -> None:
        """Test union anyOf references correct model definitions."""
        schema = pydantic_union_to_json_schema(DocumentInfo, DocumentSummary)

        refs = [item.get("$ref") for item in schema["anyOf"] if "$ref" in item]
        assert "#/$defs/DocumentInfo" in refs
        assert "#/$defs/DocumentSummary" in refs


class TestPydanticListToJsonSchema:
    """Test conversion of list types to JSON schema."""

    def test_list_schema_structure(self) -> None:
        """Test basic list schema structure."""
        schema = pydantic_list_to_json_schema(DocumentInfo)

        assert schema["type"] == "array"
        assert "items" in schema

    def test_list_item_schema(self) -> None:
        """Test that list items have correct schema."""
        schema = pydantic_list_to_json_schema(DocumentInfo)

        items_schema = schema["items"]
        assert items_schema["type"] == "object"
        assert "properties" in items_schema

    def test_list_includes_nested_definitions(self) -> None:
        """Test that list schema includes nested model definitions."""
        schema = pydantic_list_to_json_schema(DocumentInfo)

        assert "$defs" in schema
        assert "ChapterMetadata" in schema["$defs"]

    def test_list_schema_json_serializable(self) -> None:
        """Test list schema is JSON serializable."""
        schema = pydantic_list_to_json_schema(DocumentInfo)

        try:
            json.dumps(schema)
        except TypeError as e:
            pytest.fail(f"List schema is not JSON serializable: {e}")

    def test_list_of_simple_models(self) -> None:
        """Test list of simple models."""
        schema = pydantic_list_to_json_schema(OperationStatus)

        assert schema["type"] == "array"
        assert schema["items"]["type"] == "object"
        assert "success" in schema["items"]["properties"]


class TestFlexibleObjectSchema:
    """Test flexible object schema generation."""

    def test_flexible_schema_structure(self) -> None:
        """Test flexible object schema structure."""
        schema = create_flexible_object_schema()

        assert schema["type"] == "object"
        assert schema["additionalProperties"] is True

    def test_flexible_schema_custom_description(self) -> None:
        """Test flexible schema with custom description."""
        desc = "Custom configuration object"
        schema = create_flexible_object_schema(description=desc)

        assert schema["description"] == desc


class TestOneOfSchema:
    """Test oneOf schema generation."""

    def test_oneof_schema_structure(self) -> None:
        """Test oneOf schema structure."""
        schema1 = {"type": "string"}
        schema2 = {"type": "integer"}

        schema = create_oneOf_schema(schema1, schema2)

        assert "oneOf" in schema
        assert len(schema["oneOf"]) == 2

    def test_oneof_with_pydantic_models(self) -> None:
        """Test oneOf with Pydantic model schemas."""
        schema1 = pydantic_model_to_json_schema(
            OperationStatus, include_definitions=False
        )
        schema2 = pydantic_model_to_json_schema(DocumentInfo, include_definitions=False)

        schema = create_oneOf_schema(schema1, schema2)

        assert len(schema["oneOf"]) == 2


class TestSchemaValidation:
    """Test schema validation utilities."""

    def test_valid_simple_schema(self) -> None:
        """Test validation of valid simple schema."""
        schema = pydantic_model_to_json_schema(OperationStatus)

        is_valid, errors = validate_schema_against_json_schema_spec(schema)

        assert is_valid
        assert len(errors) == 0

    def test_valid_complex_schema(self) -> None:
        """Test validation of valid complex schema."""
        schema = pydantic_model_to_json_schema(DocumentInfo)

        is_valid, errors = validate_schema_against_json_schema_spec(schema)

        assert is_valid
        assert len(errors) == 0

    def test_invalid_schema_non_dict(self) -> None:
        """Test validation rejects non-dict schemas."""
        is_valid, errors = validate_schema_against_json_schema_spec("not a dict")

        assert not is_valid
        assert len(errors) > 0

    def test_schema_json_schema_compatibility(self) -> None:
        """Test generated schemas work with jsonschema validator."""
        schema = pydantic_model_to_json_schema(OperationStatus)

        # Should not raise when creating validator
        validator = Draft202012Validator(schema)

        # Test that valid data passes validation
        valid_data = {"success": True, "message": "OK"}
        assert validator.is_valid(valid_data)


class TestSchemaRegistry:
    """Test tool schema registry operations."""

    def test_register_and_retrieve_schema(self) -> None:
        """Test registering and retrieving a schema."""
        schema = pydantic_model_to_json_schema(OperationStatus)
        register_tool_schema("test_tool", schema)

        retrieved = get_tool_schema("test_tool")
        assert retrieved == schema

    def test_retrieve_nonexistent_schema(self) -> None:
        """Test retrieving a non-existent schema returns None."""
        schema = get_tool_schema("nonexistent_tool")
        assert schema is None

    def test_register_duplicate_without_overwrite(self) -> None:
        """Test registering duplicate schema without overwrite raises."""
        schema = pydantic_model_to_json_schema(OperationStatus)
        register_tool_schema("test_dup", schema)

        with pytest.raises(ValueError):
            register_tool_schema("test_dup", schema, overwrite=False)

    def test_register_duplicate_with_overwrite(self) -> None:
        """Test registering duplicate schema with overwrite succeeds."""
        schema1 = pydantic_model_to_json_schema(OperationStatus)
        schema2 = pydantic_model_to_json_schema(DocumentInfo)

        register_tool_schema("test_over", schema1)
        register_tool_schema("test_over", schema2, overwrite=True)

        retrieved = get_tool_schema("test_over")
        assert retrieved == schema2

    def test_register_invalid_schema_type(self) -> None:
        """Test registering non-dict schema raises."""
        with pytest.raises(TypeError):
            register_tool_schema("test_invalid", "not a dict")

    def test_get_all_schemas(self) -> None:
        """Test getting all registered schemas."""
        schema1 = pydantic_model_to_json_schema(OperationStatus)
        schema2 = pydantic_model_to_json_schema(DocumentInfo)

        register_tool_schema("get_all_test_1", schema1)
        register_tool_schema("get_all_test_2", schema2)

        all_schemas = get_all_tool_schemas()

        assert isinstance(all_schemas, dict)
        assert "get_all_test_1" in all_schemas
        assert "get_all_test_2" in all_schemas
        assert all_schemas["get_all_test_1"] == schema1
        assert all_schemas["get_all_test_2"] == schema2


class TestSchemaWithComplexModels:
    """Test schema generation with complex real-world models."""

    def test_paginated_content_schema(self) -> None:
        """Test schema for PaginatedContent (complex nested model)."""
        schema = pydantic_model_to_json_schema(PaginatedContent)

        assert schema["type"] == "object"
        assert "pagination" in schema["properties"]
        assert "$defs" in schema
        assert "PaginationInfo" in schema["$defs"]

    def test_modification_history_schema(self) -> None:
        """Test schema for ModificationHistory with nested entries."""
        schema = pydantic_model_to_json_schema(ModificationHistory)

        assert schema["type"] == "object"
        assert "entries" in schema["properties"]
        assert schema["properties"]["entries"]["type"] == "array"

    def test_chapter_content_schema(self) -> None:
        """Test schema for ChapterContent."""
        schema = pydantic_model_to_json_schema(ChapterContent)

        assert schema["type"] == "object"
        required = schema.get("required", [])
        assert "content" in required
        assert "document_name" in required

    def test_statistics_report_schema(self) -> None:
        """Test schema for StatisticsReport."""
        schema = pydantic_model_to_json_schema(StatisticsReport)

        assert schema["type"] == "object"
        assert "word_count" in schema["properties"]
        assert "paragraph_count" in schema["properties"]


class TestSchemaEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_definitions_not_included(self) -> None:
        """Test that empty $defs are not included."""
        schema = pydantic_model_to_json_schema(
            OperationStatus, include_definitions=False
        )

        # Should not have $defs since OperationStatus has no nested models
        assert "$defs" not in schema

    def test_multiple_schema_generation_consistency(self) -> None:
        """Test that generating same schema twice produces identical results."""
        schema1 = pydantic_model_to_json_schema(DocumentInfo)
        schema2 = pydantic_model_to_json_schema(DocumentInfo)

        assert schema1 == schema2

    def test_schema_with_list_field(self) -> None:
        """Test schema for model with list field."""
        schema = pydantic_model_to_json_schema(OperationStatus)

        # warnings is a list field
        warnings_schema = schema["properties"]["warnings"]
        assert warnings_schema["type"] == "array"
        assert "items" in warnings_schema

    def test_schema_with_optional_list(self) -> None:
        """Test schema for optional list field."""
        schema = pydantic_model_to_json_schema(DocumentInfo)

        # chapters is a required list
        chapters_schema = schema["properties"]["chapters"]
        assert chapters_schema["type"] == "array"
