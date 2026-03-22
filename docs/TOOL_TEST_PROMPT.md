# Story MCP Tool Test Prompt

Use this prompt with Claude to systematically test all 37 MCP tools. Copy the entire prompt below and paste it into a Claude conversation that has the story-mcp server connected.

---

## Test Prompt

I want to test all Story MCP tools systematically. Please execute the following test sequence, reporting results for each step. Create test documents as needed and clean up afterward.

### Phase 1: Document Management (6 tools)

**1. create_document** - Create a test document
```
Create a document called "test_novel" for testing purposes.
```

**2. list_documents** - Verify document was created
```
List all documents to verify "test_novel" exists.
```

**3. create_chapter with metadata** - Create chapters with frontmatter
```
Create these chapters in "test_novel":
- "01-intro.md" with content "# Introduction\n\nMarcus Chen walked into the dimly lit office."
  with frontmatter: status="draft", pov_character="Marcus"
- "02-discovery.md" with content "# The Discovery\n\nThe pendant glowed with an otherworldly light. Marcus picked it up carefully."
  with frontmatter: status="draft", pov_character="Marcus", tags=["action", "mystery"]
- "03-confrontation.md" with content "# Confrontation\n\nSarah confronted Marcus about the pendant. 'Where did you find it?' she demanded."
  with frontmatter: status="draft", pov_character="Sarah"
```

Note: create_chapter uses individual parameters (status, pov_character, tags, notes) for Gemini API compatibility.

**4. write_summary** - Create document and chapter summaries
```
Write a document-level summary for "test_novel": "A mystery novel following Marcus Chen as he discovers a mysterious pendant."
Then write a chapter summary for "01-intro.md": "Marcus arrives at his office."
```

**5. read_summary** - Read the summaries back
```
Read the document-level summary for "test_novel".
```

**6. list_summaries** - List all summaries
```
List all summaries in "test_novel".
```

### Phase 2: Chapter Management (3 tools — create_chapter tested above)

**7. list_chapters with metadata** - List chapters with frontmatter
```
List all chapters in "test_novel" with include_metadata=true.
```

**8. write_chapter_content** - Update chapter content (preserves frontmatter)
```
Update the content of "01-intro.md" to: "# Introduction\n\nMarcus Chen walked into the dimly lit office. The air smelled of old books and coffee."
Verify the frontmatter was preserved.
```

**9. delete_chapter** - Delete a chapter
```
First create a temporary chapter "99-temp.md" with content "Temporary chapter for testing."
Then delete it.
```

### Phase 3: Paragraph Operations (4 tools)

**10. add_paragraph** - Add paragraph (position: end / before / after)
```
Add a new paragraph to "02-discovery.md" at position="end": "He wondered what secrets it held."
Add another paragraph at position="before", target_index=0: "The room was silent."
```

**11. replace_paragraph** - Replace a specific paragraph
```
Replace paragraph 1 in "02-discovery.md" with: "The ancient pendant glowed with a mysterious blue light."
```

**12. delete_paragraph** - Delete a paragraph
```
Delete paragraph 2 from "02-discovery.md".
```

**13. move_paragraph** - Move paragraph to new position
```
Move paragraph 0 to before paragraph 2 in "02-discovery.md".
```

### Phase 4: Scope-based Content Access (6 tools)

**14. read_content** - Read at different scopes
```
Read the full document "test_novel" (scope="document", page=1).
Then read just chapter "03-confrontation.md" (scope="chapter").
Then read paragraph 0 from "01-intro.md" (scope="paragraph").
```

**15. find_text** - Search for text
```
Search for "Marcus" across the entire "test_novel" document (scope="document").
Then search for "pendant" in just chapter "02-discovery.md" (scope="chapter").
```

**16. replace_text** - Find and replace
```
Replace "pendant" with "amulet" in chapter "02-discovery.md" only.
```

**17. get_statistics** - Get word counts
```
Get statistics for the entire "test_novel" document.
Then get statistics for just "01-intro.md".
```

**18. find_similar_text** - Semantic search (requires GEMINI_API_KEY)
```
If GEMINI_API_KEY is available, search for content similar to "mysterious artifact" in "test_novel".
```

**19. find_entity** - Entity mention tracking
```
First write entity metadata for "test_novel" with character "Marcus Chen" and aliases ["Marcus", "Chen"].
Then use find_entity to find all mentions of "Marcus" across the document.
```

### Phase 5: Metadata Management (3 tools)

**20. write_metadata** - Write chapter and entity metadata
```
Update the frontmatter for "01-intro.md": scope="chapter", target="01-intro.md", status="revised"
Add a timeline event: scope="timeline", event_id="discovery", date="Day 1", description="Marcus finds the pendant", chapters=["02-discovery.md"]
```

**21. read_metadata** - Read metadata back
```
Read the metadata for chapter "01-intro.md".
Read the timeline metadata.
```

**22. list_metadata** - List and filter metadata
```
List all chapter metadata in "test_novel".
List all timeline events.
```

### Phase 6: Overview & Discovery (2 tools)

**23. get_document_outline** - Get comprehensive outline
```
Get the full document outline for "test_novel" with include_metadata=true and include_entity_counts=true.
```

**24. search_tool** - Tool discovery
```
Search for tools related to "metadata".
Search for tools in category "Version Control".
```

### Phase 7: Safety/Snapshots (3 tools)

**25. manage_snapshots** - Create and list snapshots
```
Create a snapshot of "test_novel" with the description "Before major edits".
List all snapshots for "test_novel".
```

**26. check_content_status** - Check modification status
```
Check the content status for chapter "01-intro.md" to see if it's been modified.
```

**27. diff_content** - Compare against snapshot
```
Make a small change to "01-intro.md", then diff against the last snapshot to see what changed.
```

### Phase 8: Context Management (6 tools — Phase 4.3)

**28. store_memory** - Store cross-session context
```
Store a memory in "test_novel": key="protagonist_notes", value="Marcus is a detective in his 40s, cynical but perceptive", tags=["character", "notes"]
```

**29. recall_memory** - Retrieve stored memory
```
Recall the memory with key="protagonist_notes" from "test_novel".
```

**30. list_memories** - List all memories
```
List all memories in "test_novel". Filter by tag="character".
```

**31. store_memory (second)** - Store another memory to test deletion
```
Store another memory in "test_novel": key="temp_note", value="Temporary note for testing deletion."
```

**32. delete_memory** - Delete a memory
```
Delete the memory with key="temp_note" from "test_novel".
```

**33. export_context** - Export context as JSON
```
Export the context for "test_novel" in format="json".
```

**34. import_context** - Import context
```
Import the previously exported context back into "test_novel" with conflict_strategy="skip".
```

### Phase 9: Git Version History (3 tools — Phase 4.4)

**35. get_version_history** - List commit history
```
Get the version history for "test_novel" (limit=10).
```

**36. compare_versions** - Diff between versions
```
Get the two most recent commit hashes from get_version_history, then compare them.
```

**37. checkout_version** - Restore to a previous version
```
List version history to find a suitable commit, then checkout "test_novel" to that version (use dry_run=true first to preview).
```

### Cleanup

```
Delete the "test_novel" document to clean up after testing.
```

---

## Expected Results

After running all 37 tool tests, you should have verified:

1. **Document lifecycle**: create, list, delete
2. **Chapter operations**: create with frontmatter, list, update (preserving frontmatter), delete
3. **Paragraph operations**: add (3 positions), replace, delete, move
4. **Content access**: multi-scope reading (document/chapter/paragraph), search, replace, statistics, semantic search, entity search
5. **Metadata**: chapter frontmatter, entity tracking, timeline events
6. **Overview & discovery**: outline with metadata, tool search
7. **Safety**: snapshots, status checking, diffing
8. **Context management**: store/recall/list/delete memories, export/import context
9. **Git version history**: history, compare, checkout (dry run)
