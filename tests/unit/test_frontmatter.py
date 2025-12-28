"""Unit tests for YAML frontmatter parsing and writing utilities."""

from document_mcp.utils.frontmatter import get_content_without_frontmatter
from document_mcp.utils.frontmatter import has_frontmatter
from document_mcp.utils.frontmatter import parse_frontmatter
from document_mcp.utils.frontmatter import update_frontmatter
from document_mcp.utils.frontmatter import write_frontmatter


class TestParseFrontmatter:
    """Tests for parse_frontmatter function."""

    def test_parse_valid_frontmatter(self):
        """Test parsing content with valid frontmatter."""
        content = """---
status: draft
pov_character: Marcus
tags:
  - action
  - dialogue
---

# Chapter One

This is the chapter content."""

        metadata, body = parse_frontmatter(content)

        assert metadata["status"] == "draft"
        assert metadata["pov_character"] == "Marcus"
        assert metadata["tags"] == ["action", "dialogue"]
        assert body.strip().startswith("# Chapter One")

    def test_parse_no_frontmatter(self):
        """Test parsing content without frontmatter."""
        content = """# Chapter One

This is just regular content."""

        metadata, body = parse_frontmatter(content)

        assert metadata == {}
        assert body == content

    def test_parse_empty_frontmatter(self):
        """Test parsing content with empty frontmatter block."""
        content = """---
---

# Chapter One"""

        metadata, body = parse_frontmatter(content)

        assert metadata == {}
        assert "# Chapter One" in body

    def test_parse_invalid_yaml(self):
        """Test parsing content with invalid YAML in frontmatter."""
        content = """---
invalid: [unclosed bracket
---

# Chapter One"""

        metadata, body = parse_frontmatter(content)

        # Should return empty metadata and full content when YAML is invalid
        assert metadata == {}
        assert body == content

    def test_parse_empty_content(self):
        """Test parsing empty content."""
        metadata, body = parse_frontmatter("")

        assert metadata == {}
        assert body == ""


class TestWriteFrontmatter:
    """Tests for write_frontmatter function."""

    def test_write_new_frontmatter(self):
        """Test writing frontmatter to content without existing frontmatter."""
        content = "# Chapter One\n\nContent here."
        metadata = {"status": "draft", "pov_character": "Marcus"}

        result = write_frontmatter(content, metadata)

        assert result.startswith("---\n")
        assert "status: draft" in result
        assert "pov_character: Marcus" in result
        assert "# Chapter One" in result

    def test_write_replace_frontmatter(self):
        """Test replacing existing frontmatter."""
        content = """---
status: draft
---

# Chapter One"""
        metadata = {"status": "complete", "pov_character": "Sarah"}

        result = write_frontmatter(content, metadata)

        assert "status: complete" in result
        assert "pov_character: Sarah" in result
        assert "status: draft" not in result

    def test_write_empty_metadata(self):
        """Test that empty metadata removes frontmatter."""
        content = """---
status: draft
---

# Chapter One"""

        result = write_frontmatter(content, {})

        assert not result.startswith("---")
        assert "# Chapter One" in result

    def test_write_preserves_content(self):
        """Test that writing frontmatter preserves body content."""
        original_body = "# Chapter\n\nParagraph one.\n\nParagraph two."
        metadata = {"status": "revised"}

        result = write_frontmatter(original_body, metadata)
        _, new_body = parse_frontmatter(result)

        assert "Paragraph one." in new_body
        assert "Paragraph two." in new_body


class TestUpdateFrontmatter:
    """Tests for update_frontmatter function."""

    def test_update_existing_fields(self):
        """Test updating existing frontmatter fields."""
        content = """---
status: draft
pov_character: Marcus
---

# Chapter"""

        result = update_frontmatter(content, {"status": "revised"})
        metadata, _ = parse_frontmatter(result)

        assert metadata["status"] == "revised"
        assert metadata["pov_character"] == "Marcus"  # Preserved

    def test_add_new_fields(self):
        """Test adding new fields to frontmatter."""
        content = """---
status: draft
---

# Chapter"""

        result = update_frontmatter(content, {"tags": ["action"]})
        metadata, _ = parse_frontmatter(result)

        assert metadata["status"] == "draft"
        assert metadata["tags"] == ["action"]

    def test_remove_fields_with_none(self):
        """Test removing fields by setting them to None."""
        content = """---
status: draft
pov_character: Marcus
---

# Chapter"""

        result = update_frontmatter(content, {"pov_character": None})
        metadata, _ = parse_frontmatter(result)

        assert metadata["status"] == "draft"
        assert "pov_character" not in metadata

    def test_update_no_existing_frontmatter(self):
        """Test adding frontmatter to content without any."""
        content = "# Chapter\n\nContent."

        result = update_frontmatter(content, {"status": "draft"})
        metadata, body = parse_frontmatter(result)

        assert metadata["status"] == "draft"
        assert "# Chapter" in body


class TestHasFrontmatter:
    """Tests for has_frontmatter function."""

    def test_has_frontmatter_true(self):
        """Test detection of valid frontmatter."""
        content = """---
status: draft
---

# Chapter"""

        assert has_frontmatter(content) is True

    def test_has_frontmatter_false(self):
        """Test detection when no frontmatter."""
        content = "# Chapter\n\nNo frontmatter here."

        assert has_frontmatter(content) is False

    def test_has_frontmatter_empty(self):
        """Test detection with empty content."""
        assert has_frontmatter("") is False

    def test_has_frontmatter_dashes_not_at_start(self):
        """Test that dashes in middle of content don't trigger false positive."""
        content = "Some content\n---\nMore content"

        assert has_frontmatter(content) is False


class TestGetContentWithoutFrontmatter:
    """Tests for get_content_without_frontmatter function."""

    def test_strip_frontmatter(self):
        """Test stripping frontmatter from content."""
        content = """---
status: draft
---

# Chapter One

Content here."""

        result = get_content_without_frontmatter(content)

        assert not result.startswith("---")
        assert "# Chapter One" in result
        assert "Content here." in result

    def test_no_frontmatter(self):
        """Test content without frontmatter is unchanged."""
        content = "# Chapter One\n\nContent here."

        result = get_content_without_frontmatter(content)

        assert result == content
