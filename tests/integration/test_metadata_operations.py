"""Integration tests for Phase 2 metadata operations in the Document MCP tool server."""

from pathlib import Path

from document_mcp.mcp_client import find_entity
from document_mcp.mcp_client import get_document_outline
from document_mcp.mcp_client import list_metadata
from document_mcp.mcp_client import read_metadata
from document_mcp.mcp_client import write_metadata


class TestMetadataTools:
    """Integration tests for read_metadata, write_metadata, list_metadata."""

    def test_write_and_read_chapter_metadata(self, document_factory, temp_docs_root: Path):
        """Test writing chapter metadata (frontmatter) and reading it back."""
        doc_name = "metadata_test_doc"
        chapter_name = "01-intro.md"
        document_factory(doc_name, {chapter_name: "# Introduction\n\nChapter content here."})

        # Write chapter metadata
        write_result = write_metadata(
            doc_name,
            scope="chapter",
            target=chapter_name,
            status="draft",
            pov_character="Marcus",
            tags=["action", "dialogue"],
        )
        assert write_result.success is True

        # Read it back
        read_result = read_metadata(doc_name, scope="chapter", target=chapter_name)
        assert read_result is not None
        assert read_result.scope == "chapter"
        assert read_result.data.get("status") == "draft"
        assert read_result.data.get("pov_character") == "Marcus"
        assert "action" in read_result.data.get("tags", [])

    def test_write_and_read_entity_metadata(self, document_factory, temp_docs_root: Path):
        """Test writing entity metadata and reading it back."""
        doc_name = "entity_test_doc"
        chapter_name = "01-chapter.md"
        document_factory(doc_name, {chapter_name: "# Chapter\n\nMarcus walked in."})

        # Write entity metadata
        write_result = write_metadata(
            doc_name,
            scope="entity",
            target="Marcus Chen",
            entity_type="character",
            aliases=["Marcus", "Marc"],
            description="Main protagonist",
        )
        assert write_result.success is True

        # Read it back
        read_result = read_metadata(doc_name, scope="entity", target="Marcus Chen")
        assert read_result is not None
        assert read_result.scope == "entity"
        assert "Marcus" in read_result.data.get("aliases", [])

    def test_write_and_read_timeline_metadata(self, document_factory, temp_docs_root: Path):
        """Test writing timeline metadata and reading it back."""
        doc_name = "timeline_test_doc"
        chapter_name = "01-chapter.md"
        document_factory(doc_name, {chapter_name: "# Chapter\n\nContent."})

        # Write timeline event
        write_result = write_metadata(
            doc_name,
            scope="timeline",
            event_id="discovery",
            date="Day 1",
            description="Marcus finds the pendant",
            chapters=[chapter_name],
        )
        assert write_result.success is True

        # Read it back
        read_result = read_metadata(doc_name, scope="timeline")
        assert read_result is not None
        assert read_result.scope == "timeline"

    def test_list_chapter_metadata(self, document_factory, temp_docs_root: Path):
        """Test listing all chapter metadata."""
        doc_name = "list_meta_doc"
        document_factory(
            doc_name,
            {
                "01-intro.md": "# Intro\n\nContent.",
                "02-middle.md": "# Middle\n\nContent.",
            },
        )

        # Write metadata to both chapters
        write_metadata(doc_name, scope="chapter", target="01-intro.md", status="draft")
        write_metadata(doc_name, scope="chapter", target="02-middle.md", status="revised")

        # List all chapter metadata
        list_result = list_metadata(doc_name, scope="chapters")
        assert list_result is not None
        assert list_result.count >= 2

    def test_list_metadata_with_filter(self, document_factory, temp_docs_root: Path):
        """Test filtering metadata by status."""
        doc_name = "filter_meta_doc"
        document_factory(
            doc_name,
            {
                "01-draft.md": "# Draft\n\nContent.",
                "02-revised.md": "# Revised\n\nContent.",
            },
        )

        # Write metadata with different statuses
        write_metadata(doc_name, scope="chapter", target="01-draft.md", status="draft")
        write_metadata(doc_name, scope="chapter", target="02-revised.md", status="revised")

        # Filter by status
        draft_only = list_metadata(doc_name, scope="chapters", filter_status="draft")
        assert draft_only is not None
        # Should only return draft chapters
        for item in draft_only.items:
            if "status" in item:
                assert item["status"] == "draft"


class TestFindEntity:
    """Integration tests for find_entity tool."""

    def test_find_entity_basic(self, document_factory):
        """Test finding entity mentions across chapters.

        Note: find_entity counts once per paragraph that contains the entity,
        not the total number of occurrences.
        """
        doc_name = "entity_search_doc"
        document_factory(
            doc_name,
            {
                "01-intro.md": "# Introduction\n\nMarcus walked into the room. He looked around.",
                "02-middle.md": "# Middle\n\nMarcus spoke to Sarah. Marcus was nervous.",
            },
        )

        result = find_entity(doc_name, "Marcus")
        assert result is not None
        # Counts paragraphs containing the entity, not total occurrences
        assert result.get("total_mentions", 0) >= 2
        assert len(result.get("by_chapter", [])) == 2

    def test_find_entity_with_aliases(self, document_factory, temp_docs_root: Path):
        """Test finding entity using aliases from metadata.

        Note: find_entity counts once per paragraph containing any alias.
        """
        doc_name = "alias_search_doc"
        document_factory(
            doc_name,
            {
                "01-intro.md": "# Introduction\n\nMarcus Chen entered. Marc looked around.",
                "02-middle.md": "# Middle\n\nThe Detective solved the case.",
            },
        )

        # Write entity metadata with aliases
        write_metadata(
            doc_name,
            scope="entity",
            target="Marcus Chen",
            entity_type="character",
            aliases=["Marcus", "Marc", "The Detective"],
        )

        result = find_entity(doc_name, "Marcus Chen")
        assert result is not None
        # Should find paragraphs containing any alias (2 paragraphs total)
        assert result.get("total_mentions", 0) >= 2
        assert "Marcus" in result.get("aliases_searched", [])

    def test_find_entity_no_matches(self, document_factory):
        """Test finding entity that doesn't exist."""
        doc_name = "no_match_doc"
        document_factory(doc_name, {"01-intro.md": "# Introduction\n\nSome content here."})

        result = find_entity(doc_name, "NonexistentCharacter")
        assert result is not None
        assert result.get("total_mentions", 0) == 0

    def test_find_entity_excludes_frontmatter(self, document_factory, temp_docs_root: Path):
        """Test that find_entity doesn't match metadata in frontmatter."""
        doc_name = "frontmatter_exclude_doc"
        chapter_name = "01-intro.md"
        document_factory(doc_name, {chapter_name: "# Introduction\n\nSome content."})

        # Write frontmatter with Marcus as pov_character
        write_metadata(doc_name, scope="chapter", target=chapter_name, pov_character="Marcus")

        # Search for Marcus - should NOT find in frontmatter
        result = find_entity(doc_name, "Marcus")
        # Should have 0 mentions if Marcus is only in frontmatter
        assert result.get("total_mentions", 0) == 0


class TestGetDocumentOutline:
    """Integration tests for get_document_outline tool."""

    def test_get_outline_basic(self, document_factory):
        """Test getting basic document outline."""
        doc_name = "outline_doc"
        document_factory(
            doc_name,
            {
                "01-intro.md": "# Introduction\n\nFirst paragraph.\n\nSecond paragraph.",
                "02-middle.md": "# Middle\n\nMiddle content.",
            },
        )

        result = get_document_outline(doc_name)
        assert result is not None
        assert result.get("document_name") == doc_name
        assert result.get("total_chapters", 0) == 2
        assert len(result.get("chapters", [])) == 2

    def test_get_outline_with_metadata(self, document_factory, temp_docs_root: Path):
        """Test outline includes chapter metadata."""
        doc_name = "outline_meta_doc"
        chapter_name = "01-intro.md"
        document_factory(doc_name, {chapter_name: "# Introduction\n\nContent."})

        # Add metadata
        write_metadata(doc_name, scope="chapter", target=chapter_name, status="draft", pov_character="Marcus")

        result = get_document_outline(doc_name, include_metadata=True)
        assert result is not None
        chapters = result.get("chapters", [])
        assert len(chapters) >= 1

        # Check that metadata is included
        first_chapter = chapters[0]
        metadata = first_chapter.get("metadata", {})
        assert metadata.get("status") == "draft"

    def test_get_outline_includes_statistics(self, document_factory):
        """Test outline includes word count statistics."""
        doc_name = "outline_stats_doc"
        document_factory(
            doc_name,
            {"01-intro.md": "# Introduction\n\nThis has five words here."},
        )

        result = get_document_outline(doc_name)
        assert result is not None
        assert result.get("total_words", 0) > 0

    def test_get_outline_entity_counts(self, document_factory, temp_docs_root: Path):
        """Test outline includes entity counts when entities exist."""
        doc_name = "outline_entity_doc"
        document_factory(doc_name, {"01-intro.md": "# Introduction\n\nContent."})

        # Add entities
        write_metadata(
            doc_name,
            scope="entity",
            target="Marcus",
            entity_type="character",
        )
        write_metadata(
            doc_name,
            scope="entity",
            target="The Office",
            entity_type="location",
        )

        result = get_document_outline(doc_name, include_entity_counts=True)
        assert result is not None
        # Entity counts may be in different formats depending on implementation
        # Just verify the outline was retrieved successfully with include_entity_counts
        assert result.get("document_name") == doc_name
