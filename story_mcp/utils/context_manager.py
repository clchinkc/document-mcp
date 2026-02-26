"""Context Management utilities for OneContext-inspired document system.

This module provides utilities for managing document context including memory storage,
retrieval, session tracking, import/export, and lifecycle management.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import yaml

from ..models import ExportStatus
from ..models import ImportStatus
from ..models import MemoryEntry
from ..models import SessionMetadata
from .file_operations import get_document_path


def get_context_path(document_name: str) -> Path:
    """Get the .context directory path for a document."""
    doc_path = get_document_path(document_name)
    context_path = doc_path / ".context"
    return context_path


def get_session_file_path(document_name: str) -> Path:
    """Get the session.json file path for a document."""
    return get_context_path(document_name) / "session.json"


def get_memories_dir_path(document_name: str) -> Path:
    """Get the memories directory path for a document."""
    return get_context_path(document_name) / "memories"


def get_memory_file_path(document_name: str, key: str) -> Path:
    """Get the file path for a specific memory entry."""
    memories_dir = get_memories_dir_path(document_name)
    safe_filename = _sanitize_filename(key)
    return memories_dir / f"{safe_filename}.json"


def get_decisions_file_path(document_name: str) -> Path:
    """Get the decisions.md file path for a document."""
    return get_context_path(document_name) / "decisions.md"


def get_blockers_file_path(document_name: str) -> Path:
    """Get the blockers.md file path for a document."""
    return get_context_path(document_name) / "blockers.md"


def get_goals_file_path(document_name: str) -> Path:
    """Get the goals.md file path for a document."""
    return get_context_path(document_name) / "goals.md"


def _sanitize_filename(key: str) -> str:
    """Convert a memory key to a safe filename."""
    safe = key.replace("/", "_").replace("\\", "_").replace(":", "_")
    safe = safe.replace(" ", "_").replace("\n", "_")
    return safe[:200]  # Limit filename length


def ensure_context_directory(document_name: str) -> Path:
    """Create context directory structure if it doesn't exist.

    Creates:
    - .context/
    - .context/memories/

    Returns:
        Path to .context directory
    """
    context_path = get_context_path(document_name)
    context_path.mkdir(parents=True, exist_ok=True)

    memories_dir = get_memories_dir_path(document_name)
    memories_dir.mkdir(parents=True, exist_ok=True)

    return context_path


def initialize_session(document_name: str, session_id: str | None = None) -> SessionMetadata:
    """Initialize a new session for a document.

    Creates session.json if it doesn't exist. Auto-generates session_id if not provided.

    Args:
        document_name: Document to create session for
        session_id: Optional session ID (auto-generated if not provided)

    Returns:
        SessionMetadata object for the new session
    """
    ensure_context_directory(document_name)

    session_file = get_session_file_path(document_name)
    generated_id = session_id or _generate_session_id()

    now = datetime.datetime.utcnow()
    session = SessionMetadata(
        session_id=generated_id,
        document_name=document_name,
        started_at=now,
        last_activity=now,
        goals=[],
        progress={},
        blockers=[],
        metadata={},
    )

    # Only write if file doesn't exist (don't overwrite existing session)
    if not session_file.exists():
        _write_session_file(session_file, session)

    return session


def load_session(document_name: str) -> SessionMetadata | None:
    """Load existing session metadata.

    Returns:
        SessionMetadata if session exists, None otherwise
    """
    session_file = get_session_file_path(document_name)
    if not session_file.exists():
        return None

    return _load_session_file(session_file)


def update_session(document_name: str, session: SessionMetadata) -> None:
    """Update session metadata.

    Args:
        document_name: Document to update session for
        session: SessionMetadata object with updates
    """
    ensure_context_directory(document_name)
    session.last_activity = datetime.datetime.utcnow()
    session_file = get_session_file_path(document_name)
    _write_session_file(session_file, session)


def store_memory(
    document_name: str,
    key: str,
    value: Any,
    tags: list[str] | None = None,
    expires: datetime.datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> MemoryEntry:
    """Store a memory entry in the context system.

    Automatically initializes session if needed. Creates or overwrites memory entry.

    Args:
        document_name: Document to store memory in
        key: Memory key (unique identifier)
        value: Value to store (string, JSON, etc)
        tags: Optional tags for organizing memories
        expires: Optional expiration timestamp
        metadata: Optional additional metadata

    Returns:
        MemoryEntry object that was stored
    """
    ensure_context_directory(document_name)

    # Auto-initialize session on first memory store
    session = load_session(document_name)
    if not session:
        initialize_session(document_name)

    now = datetime.datetime.utcnow()
    entry = MemoryEntry(
        key=key,
        value=value,
        stored_at=now,
        retrieved_at=None,
        tags=tags or [],
        expires=expires,
        metadata=metadata or {},
    )

    memory_file = get_memory_file_path(document_name, key)
    with open(memory_file, "w") as f:
        json.dump(entry.model_dump(mode="json"), f, indent=2, default=str)

    # Update session last_activity
    if session:
        update_session(document_name, session)

    return entry


def recall_memory(
    document_name: str,
    key: str,
    pattern: str | None = None,
) -> MemoryEntry | None:
    """Retrieve a memory entry from the context system.

    Updates retrieved_at timestamp when memory is recalled.

    Args:
        document_name: Document to recall memory from
        key: Memory key to retrieve
        pattern: Optional pattern matching (not implemented yet)

    Returns:
        MemoryEntry if found, None otherwise
    """
    memory_file = get_memory_file_path(document_name, key)
    if not memory_file.exists():
        return None

    with open(memory_file) as f:
        data = json.load(f)

    entry = MemoryEntry(**data)
    entry.retrieved_at = datetime.datetime.utcnow()

    # Update file with new retrieved_at
    with open(memory_file, "w") as f:
        json.dump(entry.model_dump(mode="json"), f, indent=2, default=str)

    return entry


def list_memories(document_name: str, tags: list[str] | None = None) -> list[MemoryEntry]:
    """List all memory entries for a document.

    Optionally filters by tags.

    Args:
        document_name: Document to list memories from
        tags: Optional list of tags to filter by

    Returns:
        List of MemoryEntry objects
    """
    memories_dir = get_memories_dir_path(document_name)
    if not memories_dir.exists():
        return []

    memories = []
    for memory_file in memories_dir.glob("*.json"):
        with open(memory_file) as f:
            data = json.load(f)
        entry = MemoryEntry(**data)

        if tags:
            if any(tag in entry.tags for tag in tags):
                memories.append(entry)
        else:
            memories.append(entry)

    return memories


def delete_memory(document_name: str, key: str) -> bool:
    """Delete a memory entry.

    Args:
        document_name: Document to delete memory from
        key: Memory key to delete

    Returns:
        True if deleted, False if not found
    """
    memory_file = get_memory_file_path(document_name, key)
    if not memory_file.exists():
        return False

    memory_file.unlink()
    return True


def export_context(
    document_name: str,
    export_path: Path | str,
    format_type: str = "json",
) -> ExportStatus:
    """Export document context to a file.

    Supports json, yaml, and markdown formats.

    Args:
        document_name: Document to export context from
        export_path: Path where to export context
        format_type: Export format ('json', 'yaml', 'markdown')

    Returns:
        ExportStatus with results
    """
    export_path = Path(export_path)

    try:
        # Collect all data
        session = load_session(document_name)
        memories = list_memories(document_name)
        context_dir = get_context_path(document_name)

        decisions = ""
        blockers = ""
        goals = ""

        if (context_dir / "decisions.md").exists():
            decisions = (context_dir / "decisions.md").read_text()
        if (context_dir / "blockers.md").exists():
            blockers = (context_dir / "blockers.md").read_text()
        if (context_dir / "goals.md").exists():
            goals = (context_dir / "goals.md").read_text()

        # Prepare export data
        export_data = {
            "document_name": document_name,
            "exported_at": datetime.datetime.utcnow().isoformat(),
            "session": session.model_dump(mode="json") if session else None,
            "memories": [m.model_dump(mode="json") for m in memories],
            "decisions": decisions,
            "blockers": blockers,
            "goals": goals,
        }

        # Write in requested format
        export_path.parent.mkdir(parents=True, exist_ok=True)

        if format_type == "json":
            with open(export_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
        elif format_type == "yaml":
            with open(export_path, "w") as f:
                yaml.dump(export_data, f, default_flow_style=False)
        elif format_type == "markdown":
            with open(export_path, "w") as f:
                f.write(_export_to_markdown(export_data))
        else:
            return ExportStatus(
                success=False,
                message=f"Unsupported format: {format_type}",
                format_used=format_type,
                entry_count=0,
            )

        file_size = export_path.stat().st_size if export_path.exists() else 0

        return ExportStatus(
            success=True,
            exported_file_path=str(export_path),
            file_size=file_size,
            entry_count=len(memories),
            format_used=format_type,
            message=f"Context exported successfully to {export_path}",
        )

    except Exception as e:
        return ExportStatus(
            success=False,
            message=f"Export failed: {str(e)}",
            format_used=format_type,
            entry_count=0,
        )


def import_context(
    document_name: str,
    context_file: Path | str,
    merge: bool = False,
) -> ImportStatus:
    """Import context from a file.

    Supports json, yaml, and markdown formats. Can merge with existing or replace.

    Args:
        document_name: Document to import context into
        context_file: Path to context file
        merge: If True, merge with existing; if False, replace

    Returns:
        ImportStatus with results
    """
    context_file = Path(context_file)

    try:
        if not context_file.exists():
            return ImportStatus(
                success=False,
                message=f"Context file not found: {context_file}",
            )

        # Detect format and load
        suffix = context_file.suffix.lower()
        if suffix == ".json":
            with open(context_file) as f:
                import_data = json.load(f)
        elif suffix in [".yaml", ".yml"]:
            with open(context_file) as f:
                import_data = yaml.safe_load(f)
        else:
            return ImportStatus(
                success=False,
                message=f"Unsupported file format: {suffix}",
            )

        # Validate structure
        if not isinstance(import_data, dict):
            return ImportStatus(
                success=False,
                message="Invalid context file structure",
            )

        ensure_context_directory(document_name)

        # Handle session
        imported_session = import_data.get("session")
        if imported_session:
            session = SessionMetadata(**imported_session)
            update_session(document_name, session)

        # Handle memories
        imported_memories = import_data.get("memories", [])
        conflicts = []
        imported_count = 0

        for memory_data in imported_memories:
            memory = MemoryEntry(**memory_data)
            memory_file = get_memory_file_path(document_name, memory.key)

            if memory_file.exists() and not merge:
                conflicts.append({
                    "key": memory.key,
                    "reason": "File exists (merge=False)",
                })
            else:
                with open(memory_file, "w") as f:
                    json.dump(memory.model_dump(mode="json"), f, indent=2, default=str)
                imported_count += 1

        # Handle markdown files
        context_dir = get_context_path(document_name)
        for key in ["decisions", "blockers", "goals"]:
            if key in import_data and import_data[key]:
                file_path = context_dir / f"{key}.md"
                if file_path.exists() and not merge:
                    conflicts.append({
                        "key": f"{key}.md",
                        "reason": "File exists (merge=False)",
                    })
                else:
                    file_path.write_text(import_data[key])

        return ImportStatus(
            success=len(conflicts) == 0,
            entries_imported=imported_count,
            conflicts_detected=len(conflicts),
            conflict_details=conflicts,
            merge_mode_used=merge,
            message=(
                f"Imported {imported_count} memories"
                if len(conflicts) == 0
                else f"Imported {imported_count} memories with {len(conflicts)} conflicts"
            ),
        )

    except Exception as e:
        return ImportStatus(
            success=False,
            message=f"Import failed: {str(e)}",
        )


def _generate_session_id() -> str:
    """Generate a unique session ID."""
    from uuid import uuid4
    return f"session_{uuid4().hex[:8]}"


def _write_session_file(file_path: Path, session: SessionMetadata) -> None:
    """Write session to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(session.model_dump(mode="json"), f, indent=2, default=str)


def _load_session_file(file_path: Path) -> SessionMetadata | None:
    """Load session from JSON file."""
    try:
        with open(file_path) as f:
            data = json.load(f)
        return SessionMetadata(**data)
    except Exception:
        return None


def _export_to_markdown(data: dict[str, Any]) -> str:
    """Convert export data to markdown format."""
    lines = [
        f"# Context Export: {data['document_name']}",
        f"\n**Exported at:** {data['exported_at']}\n",
    ]

    if data.get("session"):
        lines.extend([
            "## Session",
            f"- **ID:** {data['session'].get('session_id')}",
            f"- **Started:** {data['session'].get('started_at')}",
            f"- **Goals:** {len(data['session'].get('goals', []))} items",
            f"- **Blockers:** {len(data['session'].get('blockers', []))} items",
            "",
        ])

    if data.get("memories"):
        lines.extend([
            "## Memories",
            f"\nTotal: {len(data['memories'])} entries\n",
        ])
        for memory in data["memories"]:
            lines.extend([
                f"### {memory.get('key')}",
                f"- **Tags:** {', '.join(memory.get('tags', []))}",
                f"- **Stored:** {memory.get('stored_at')}",
                "",
            ])

    if data.get("goals"):
        lines.extend(["## Goals\n", data["goals"], ""])

    if data.get("decisions"):
        lines.extend(["## Decisions\n", data["decisions"], ""])

    if data.get("blockers"):
        lines.extend(["## Blockers\n", data["blockers"], ""])

    return "\n".join(lines)
