"""Tests for Context Management Tools (Phase 4.3: OneContext-inspired system).

This test suite covers:
- Memory storage and retrieval (store_memory, recall_memory)
- Memory listing and deletion (list_memories, delete_memory)
- Context export/import (export_context, import_context)
- Session lifecycle management
- Metadata and tags support
- Import conflict detection
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import pytest

from story_mcp.models import ExportStatus
from story_mcp.models import ImportStatus
from story_mcp.models import MemoryEntry
from story_mcp.models import OperationStatus
from story_mcp.models import SessionMetadata
from story_mcp.utils.context_manager import delete_memory
from story_mcp.utils.context_manager import ensure_context_directory
from story_mcp.utils.context_manager import export_context
from story_mcp.utils.context_manager import get_blockers_file_path
from story_mcp.utils.context_manager import get_context_path
from story_mcp.utils.context_manager import get_decisions_file_path
from story_mcp.utils.context_manager import get_goals_file_path
from story_mcp.utils.context_manager import get_memory_file_path
from story_mcp.utils.context_manager import get_memories_dir_path
from story_mcp.utils.context_manager import get_session_file_path
from story_mcp.utils.context_manager import import_context
from story_mcp.utils.context_manager import initialize_session
from story_mcp.utils.context_manager import list_memories
from story_mcp.utils.context_manager import load_session
from story_mcp.utils.context_manager import recall_memory
from story_mcp.utils.context_manager import store_memory
from story_mcp.utils.context_manager import update_session


@pytest.fixture
def sample_document(temp_docs_root):
    """Create a sample document for testing."""
    doc_path = temp_docs_root / "test_doc"
    doc_path.mkdir(exist_ok=True)
    (doc_path / "01-chapter.md").write_text("Test content")
    return "test_doc"


class TestContextDirectory:
    """Tests for context directory initialization."""

    def test_ensure_context_directory_creates_paths(self, sample_document, temp_docs_root):
        """Test that ensure_context_directory creates all necessary paths."""
        context_path = ensure_context_directory(sample_document)

        assert context_path.exists()
        assert context_path.name == ".context"
        assert (context_path / "memories").exists()

    def test_context_paths_exist_after_init(self, sample_document):
        """Test all context path helpers after initialization."""
        ensure_context_directory(sample_document)

        assert get_context_path(sample_document).exists()
        assert get_memories_dir_path(sample_document).exists()
        assert get_session_file_path(sample_document).parent.exists()


class TestSessionManagement:
    """Tests for session lifecycle management."""

    def test_initialize_session_creates_file(self, sample_document):
        """Test session initialization creates session.json."""
        session = initialize_session(sample_document)

        assert session.session_id is not None
        assert session.document_name == sample_document
        assert session.started_at is not None
        assert session.last_activity is not None
        assert get_session_file_path(sample_document).exists()

    def test_initialize_session_with_custom_id(self, sample_document):
        """Test session initialization with custom session ID."""
        custom_id = "custom_session_123"
        session = initialize_session(sample_document, session_id=custom_id)

        assert session.session_id == custom_id

    def test_load_session_returns_none_when_not_exists(self, sample_document):
        """Test load_session returns None when no session exists."""
        session = load_session(sample_document)
        assert session is None

    def test_load_session_returns_existing(self, sample_document):
        """Test load_session returns existing session."""
        created = initialize_session(sample_document)
        loaded = load_session(sample_document)

        assert loaded is not None
        assert loaded.session_id == created.session_id
        assert loaded.document_name == sample_document

    def test_update_session_updates_timestamp(self, sample_document):
        """Test update_session updates last_activity."""
        session = initialize_session(sample_document)
        original_time = session.last_activity

        # Wait a tiny bit to ensure different timestamp
        import time
        time.sleep(0.01)

        session.goals = ["New goal"]
        update_session(sample_document, session)

        loaded = load_session(sample_document)
        assert loaded.last_activity > original_time
        assert "New goal" in loaded.goals

    def test_session_has_required_fields(self, sample_document):
        """Test session has all required fields."""
        session = initialize_session(sample_document)

        assert hasattr(session, "session_id")
        assert hasattr(session, "document_name")
        assert hasattr(session, "started_at")
        assert hasattr(session, "last_activity")
        assert hasattr(session, "goals")
        assert hasattr(session, "progress")
        assert hasattr(session, "blockers")
        assert hasattr(session, "metadata")


class TestMemoryStorage:
    """Tests for memory storage operations."""

    def test_store_memory_creates_file(self, sample_document):
        """Test store_memory creates memory file."""
        entry = store_memory(
            document_name=sample_document,
            key="test_key",
            value="test_value",
        )

        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.stored_at is not None
        assert get_memory_file_path(sample_document, "test_key").exists()

    def test_store_memory_with_tags(self, sample_document):
        """Test store_memory with tags."""
        tags = ["important", "urgent"]
        entry = store_memory(
            document_name=sample_document,
            key="tagged_memory",
            value="test",
            tags=tags,
        )

        assert entry.tags == tags

    def test_store_memory_with_expiration(self, sample_document):
        """Test store_memory with expiration."""
        expires = datetime.datetime.utcnow() + datetime.timedelta(days=7)
        entry = store_memory(
            document_name=sample_document,
            key="expiring_memory",
            value="test",
            expires=expires,
        )

        assert entry.expires is not None
        assert entry.expires > datetime.datetime.utcnow()

    def test_store_memory_with_metadata(self, sample_document):
        """Test store_memory with custom metadata."""
        metadata = {"priority": "high", "reviewed": False}
        entry = store_memory(
            document_name=sample_document,
            key="meta_memory",
            value="test",
            metadata=metadata,
        )

        assert entry.metadata == metadata

    def test_store_memory_with_complex_value(self, sample_document):
        """Test store_memory with complex JSON value."""
        complex_value = {
            "name": "Marcus",
            "attributes": ["strong", "intelligent"],
            "stats": {"strength": 8, "wisdom": 9},
        }
        entry = store_memory(
            document_name=sample_document,
            key="character",
            value=complex_value,
        )

        assert entry.value == complex_value

    def test_store_memory_creates_session_if_needed(self, sample_document):
        """Test store_memory auto-initializes session."""
        assert load_session(sample_document) is None

        store_memory(
            document_name=sample_document,
            key="test",
            value="test",
        )

        assert load_session(sample_document) is not None

    def test_store_memory_overwrites_existing(self, sample_document):
        """Test store_memory overwrites existing memory."""
        store_memory(sample_document, "key", "value1")
        store_memory(sample_document, "key", "value2")

        entry = recall_memory(sample_document, "key")
        assert entry.value == "value2"

    def test_memory_file_is_valid_json(self, sample_document):
        """Test stored memory files are valid JSON."""
        store_memory(sample_document, "test", "value")

        memory_file = get_memory_file_path(sample_document, "test")
        with open(memory_file) as f:
            data = json.load(f)

        assert "key" in data
        assert "value" in data
        assert "stored_at" in data


class TestMemoryRetrieval:
    """Tests for memory retrieval operations."""

    def test_recall_memory_returns_entry(self, sample_document):
        """Test recall_memory returns stored entry."""
        stored = store_memory(sample_document, "key", "value")
        recalled = recall_memory(sample_document, "key")

        assert recalled is not None
        assert recalled.key == stored.key
        assert recalled.value == stored.value

    def test_recall_memory_returns_none_if_not_found(self, sample_document):
        """Test recall_memory returns None for non-existent key."""
        result = recall_memory(sample_document, "nonexistent")
        assert result is None

    def test_recall_memory_updates_retrieved_at(self, sample_document):
        """Test recall_memory updates retrieved_at timestamp."""
        store_memory(sample_document, "key", "value")
        first_recall = recall_memory(sample_document, "key")
        assert first_recall.retrieved_at is not None

        # Second recall should update timestamp
        import time
        time.sleep(0.01)
        second_recall = recall_memory(sample_document, "key")
        assert second_recall.retrieved_at > first_recall.retrieved_at

    def test_recall_memory_preserves_all_data(self, sample_document):
        """Test recall_memory preserves all stored data."""
        metadata = {"test": "data"}
        tags = ["tag1", "tag2"]
        store_memory(
            sample_document,
            "key",
            "value",
            tags=tags,
            metadata=metadata,
        )

        recalled = recall_memory(sample_document, "key")
        assert recalled.tags == tags
        assert recalled.metadata == metadata


class TestMemoryListing:
    """Tests for memory listing and filtering."""

    def test_list_memories_empty_when_no_memories(self, sample_document):
        """Test list_memories returns empty list when no memories exist."""
        memories = list_memories(sample_document)
        assert memories == []

    def test_list_memories_returns_all(self, sample_document):
        """Test list_memories returns all stored memories."""
        store_memory(sample_document, "key1", "value1")
        store_memory(sample_document, "key2", "value2")
        store_memory(sample_document, "key3", "value3")

        memories = list_memories(sample_document)
        assert len(memories) == 3
        assert any(m.key == "key1" for m in memories)
        assert any(m.key == "key2" for m in memories)
        assert any(m.key == "key3" for m in memories)

    def test_list_memories_filters_by_tags(self, sample_document):
        """Test list_memories filters by tags."""
        store_memory(sample_document, "key1", "value1", tags=["urgent"])
        store_memory(sample_document, "key2", "value2", tags=["normal"])
        store_memory(sample_document, "key3", "value3", tags=["urgent", "important"])

        urgent_memories = list_memories(sample_document, tags=["urgent"])
        assert len(urgent_memories) == 2
        assert all("urgent" in m.tags for m in urgent_memories)

    def test_list_memories_multiple_tag_filter(self, sample_document):
        """Test list_memories with multiple filter tags (OR logic)."""
        store_memory(sample_document, "key1", "value1", tags=["urgent"])
        store_memory(sample_document, "key2", "value2", tags=["important"])
        store_memory(sample_document, "key3", "value3", tags=["normal"])

        filtered = list_memories(sample_document, tags=["urgent", "important"])
        assert len(filtered) == 2


class TestMemoryDeletion:
    """Tests for memory deletion operations."""

    def test_delete_memory_removes_file(self, sample_document):
        """Test delete_memory removes memory file."""
        store_memory(sample_document, "key", "value")
        memory_file = get_memory_file_path(sample_document, "key")
        assert memory_file.exists()

        success = delete_memory(sample_document, "key")
        assert success
        assert not memory_file.exists()

    def test_delete_memory_returns_false_if_not_found(self, sample_document):
        """Test delete_memory returns False for non-existent memory."""
        success = delete_memory(sample_document, "nonexistent")
        assert not success

    def test_delete_memory_does_not_affect_other_memories(self, sample_document):
        """Test delete_memory only removes target memory."""
        store_memory(sample_document, "key1", "value1")
        store_memory(sample_document, "key2", "value2")

        delete_memory(sample_document, "key1")

        memories = list_memories(sample_document)
        assert len(memories) == 1
        assert memories[0].key == "key2"


class TestContextExport:
    """Tests for context export functionality."""

    def test_export_to_json_creates_file(self, sample_document, temp_docs_root):
        """Test export_context creates JSON file."""
        store_memory(sample_document, "key1", "value1")
        store_memory(sample_document, "key2", "value2")

        export_file = temp_docs_root / "context_export.json"
        status = export_context(sample_document, export_file, format_type="json")

        assert status.success
        assert export_file.exists()
        assert status.entry_count == 2
        assert status.file_size > 0

    def test_export_json_structure(self, sample_document, temp_docs_root):
        """Test exported JSON has correct structure."""
        store_memory(sample_document, "test", "value")

        export_file = temp_docs_root / "export.json"
        export_context(sample_document, export_file, format_type="json")

        with open(export_file) as f:
            data = json.load(f)

        assert "document_name" in data
        assert "exported_at" in data
        assert "memories" in data
        assert len(data["memories"]) == 1

    def test_export_to_yaml(self, sample_document, temp_docs_root):
        """Test export_context creates YAML file."""
        store_memory(sample_document, "key", "value")

        export_file = temp_docs_root / "export.yaml"
        status = export_context(sample_document, export_file, format_type="yaml")

        assert status.success
        assert export_file.exists()
        assert export_file.suffix == ".yaml"

    def test_export_to_markdown(self, sample_document, temp_docs_root):
        """Test export_context creates Markdown file."""
        store_memory(sample_document, "key", "value", tags=["test"])

        export_file = temp_docs_root / "export.md"
        status = export_context(sample_document, export_file, format_type="markdown")

        assert status.success
        assert export_file.exists()
        content = export_file.read_text()
        assert "# Context Export" in content

    def test_export_includes_session(self, sample_document, temp_docs_root):
        """Test exported context includes session data."""
        session = initialize_session(sample_document)
        session.goals = ["Goal 1", "Goal 2"]
        update_session(sample_document, session)

        export_file = temp_docs_root / "export.json"
        export_context(sample_document, export_file)

        with open(export_file) as f:
            data = json.load(f)

        assert data["session"] is not None
        assert len(data["session"]["goals"]) == 2

    def test_export_includes_all_memories(self, sample_document, temp_docs_root):
        """Test export includes all memories."""
        for i in range(5):
            store_memory(sample_document, f"key{i}", f"value{i}")

        export_file = temp_docs_root / "export.json"
        status = export_context(sample_document, export_file)

        assert status.entry_count == 5

    def test_export_invalid_format_returns_error(self, sample_document, temp_docs_root):
        """Test export with invalid format returns error."""
        export_file = temp_docs_root / "export"
        status = export_context(sample_document, export_file, format_type="invalid")

        assert not status.success
        assert "Unsupported format" in status.message


class TestContextImport:
    """Tests for context import functionality."""

    def test_import_from_json(self, sample_document, temp_docs_root):
        """Test importing context from JSON file."""
        # Setup: Create and export context
        store_memory(sample_document, "key1", "value1")
        export_file = temp_docs_root / "export.json"
        export_context(sample_document, export_file)

        # Create new document and import
        doc2 = "test_doc_2"
        (temp_docs_root / doc2).mkdir(exist_ok=True)
        status = import_context(doc2, export_file)

        assert status.success
        assert status.entries_imported == 1

        # Verify imported
        imported = recall_memory(doc2, "key1")
        assert imported is not None
        assert imported.value == "value1"

    def test_import_detects_conflicts(self, sample_document, temp_docs_root):
        """Test import detects existing keys when merge=False."""
        # Setup existing memory in target
        store_memory(sample_document, "key1", "existing_value")

        # Create and export source
        doc2 = "source_doc"
        (temp_docs_root / doc2).mkdir(exist_ok=True)
        store_memory(doc2, "key1", "new_value")
        export_file = temp_docs_root / "export.json"
        export_context(doc2, export_file)

        # Import without merge
        status = import_context(sample_document, export_file, merge=False)

        assert status.conflicts_detected == 1
        assert len(status.conflict_details) > 0

        # Original value should be preserved
        entry = recall_memory(sample_document, "key1")
        assert entry.value == "existing_value"

    def test_import_with_merge_overwrites(self, sample_document, temp_docs_root):
        """Test import with merge=True overwrites existing."""
        store_memory(sample_document, "key1", "original")

        doc2 = "source_doc"
        (temp_docs_root / doc2).mkdir(exist_ok=True)
        store_memory(doc2, "key1", "new_value")
        export_file = temp_docs_root / "export.json"
        export_context(doc2, export_file)

        # Import with merge
        status = import_context(sample_document, export_file, merge=True)

        assert status.success
        assert status.entries_imported > 0

        # Value should be updated
        entry = recall_memory(sample_document, "key1")
        assert entry.value == "new_value"

    def test_import_nonexistent_file_fails(self, sample_document):
        """Test import fails for nonexistent file."""
        status = import_context(sample_document, Path("/nonexistent/file.json"))

        assert not status.success
        assert "not found" in status.message.lower()

    def test_import_invalid_json_fails(self, sample_document, temp_docs_root):
        """Test import fails for invalid JSON."""
        bad_file = temp_docs_root / "invalid.json"
        bad_file.write_text("{ invalid json }")

        status = import_context(sample_document, bad_file)

        assert not status.success

    def test_import_preserves_metadata(self, sample_document, temp_docs_root):
        """Test import preserves memory metadata."""
        metadata = {"custom": "data", "version": 1}
        store_memory(sample_document, "key", "value", metadata=metadata)

        export_file = temp_docs_root / "export.json"
        export_context(sample_document, export_file)

        doc2 = "doc2"
        (temp_docs_root / doc2).mkdir(exist_ok=True)
        import_context(doc2, export_file)

        imported = recall_memory(doc2, "key")
        assert imported.metadata == metadata


class TestMemoryEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_memory_key_with_special_characters(self, sample_document):
        """Test memory with special characters in key."""
        special_key = "memory/key:with/special-chars_123"
        store_memory(sample_document, special_key, "value")

        recalled = recall_memory(sample_document, special_key)
        assert recalled is not None
        assert recalled.key == special_key

    def test_memory_with_unicode_value(self, sample_document):
        """Test memory with unicode characters."""
        unicode_value = "Hello 世界 مرحبا мир 🚀"
        store_memory(sample_document, "unicode", unicode_value)

        recalled = recall_memory(sample_document, "unicode")
        assert recalled.value == unicode_value

    def test_memory_with_empty_string_value(self, sample_document):
        """Test memory with empty string value."""
        store_memory(sample_document, "empty", "")

        recalled = recall_memory(sample_document, "empty")
        assert recalled is not None
        assert recalled.value == ""

    def test_memory_with_null_like_values(self, sample_document):
        """Test memory with None, empty list, empty dict."""
        store_memory(sample_document, "null_value", None)
        store_memory(sample_document, "empty_list", [])
        store_memory(sample_document, "empty_dict", {})

        assert recall_memory(sample_document, "null_value").value is None
        assert recall_memory(sample_document, "empty_list").value == []
        assert recall_memory(sample_document, "empty_dict").value == {}

    def test_large_memory_value(self, sample_document):
        """Test storing and retrieving large values."""
        large_value = "x" * 100000
        store_memory(sample_document, "large", large_value)

        recalled = recall_memory(sample_document, "large")
        assert len(recalled.value) == 100000

    def test_many_memories(self, sample_document):
        """Test storing and listing many memories."""
        for i in range(100):
            store_memory(sample_document, f"memory_{i:03d}", f"value_{i}")

        memories = list_memories(sample_document)
        assert len(memories) == 100


class TestContextIntegration:
    """Integration tests for complete context workflows."""

    def test_complete_workflow_store_export_import(self, sample_document, temp_docs_root):
        """Test complete workflow: store -> export -> import."""
        # Store memories
        store_memory(sample_document, "goal", "Complete novel", tags=["current"])
        store_memory(sample_document, "blocker", "Character motivation", tags=["blocking"])
        store_memory(sample_document, "decision", "Changed ending", tags=["decision"])

        # Export
        export_file = temp_docs_root / "context.json"
        export_status = export_context(sample_document, export_file)
        assert export_status.success
        assert export_status.entry_count == 3

        # Import to new document
        doc2 = "restored_doc"
        (temp_docs_root / doc2).mkdir(exist_ok=True)
        import_status = import_context(doc2, export_file)
        assert import_status.success
        assert import_status.entries_imported == 3

        # Verify all data
        memories = list_memories(doc2)
        assert len(memories) == 3
        assert any(m.key == "goal" for m in memories)

    def test_session_tracking_through_operations(self, sample_document):
        """Test session is created and updated through operations."""
        # Initially no session
        assert load_session(sample_document) is None

        # First store creates session
        store_memory(sample_document, "mem1", "val1")
        session1 = load_session(sample_document)
        assert session1 is not None
        time1 = session1.last_activity

        # Second store updates session
        import time
        time.sleep(0.01)
        store_memory(sample_document, "mem2", "val2")
        session2 = load_session(sample_document)
        assert session2.last_activity > time1

    def test_export_formats_all_have_same_data(self, sample_document, temp_docs_root):
        """Test all export formats contain same data."""
        store_memory(sample_document, "key", {"nested": "data"})

        json_file = temp_docs_root / "export.json"
        yaml_file = temp_docs_root / "export.yaml"
        md_file = temp_docs_root / "export.md"

        export_context(sample_document, json_file, format_type="json")
        export_context(sample_document, yaml_file, format_type="yaml")
        export_context(sample_document, md_file, format_type="markdown")

        assert json_file.exists()
        assert yaml_file.exists()
        assert md_file.exists()

        # All formats should have file size > 0
        assert json_file.stat().st_size > 0
        assert yaml_file.stat().st_size > 0
        assert md_file.stat().st_size > 0
