# Document MCP Tool Test Prompt

Use this prompt with Claude to systematically test all 30 MCP tools. Copy the entire prompt below and paste it into a Claude conversation that has the Document MCP server connected.

---

## Test Prompt

I want to test all Document MCP tools systematically. Please execute the following test sequence, reporting results for each step. Create test documents as needed and clean up afterward.

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

**Note**: create_chapter now uses individual parameters (status, pov_character, tags, notes) instead of a metadata dict for Gemini API compatibility.

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

### Phase 2: Chapter Management (4 tools)

**7. list_chapters with metadata** - List chapters with frontmatter
```
List all chapters in "test_novel" with include_metadata=true.
```

**8. write_chapter_content** - Update chapter content (should preserve frontmatter)
```
Update the content of "01-intro.md" to: "# Introduction\n\nMarcus Chen walked into the dimly lit office. The air smelled of old books and coffee. He knew something was wrong."
Verify the frontmatter was preserved.
```

**9. delete_chapter** - Delete a chapter
```
First create a temporary chapter "99-temp.md" with content "Temporary chapter for testing."
Then delete it.
```

### Phase 3: Paragraph Operations (7 tools)

**10. append_paragraph_to_chapter** - Add paragraph at end
```
Append a new paragraph to "02-discovery.md": "He wondered what secrets it held."
```

**11. replace_paragraph** - Replace a specific paragraph
```
Replace paragraph 1 (the second paragraph) in "02-discovery.md" with: "The ancient pendant glowed with a mysterious blue light. Marcus felt drawn to it."
```

**12. insert_paragraph_before** - Insert paragraph before index
```
Insert before paragraph 1 in "02-discovery.md": "The room fell silent."
```

**13. insert_paragraph_after** - Insert paragraph after index
```
Insert after paragraph 0 in "02-discovery.md": "Outside, thunder rumbled ominously."
```

**14. delete_paragraph** - Delete a paragraph
```
Delete paragraph 2 from "02-discovery.md".
```

**15. move_paragraph_before** - Move paragraph to new position
```
Move paragraph 3 to before paragraph 1 in "02-discovery.md".
```

**16. move_paragraph_to_end** - Move paragraph to end
```
Move paragraph 0 to the end of "02-discovery.md".
```

### Phase 4: Scope-based Content Access (6 tools)

**17. read_content** - Read at different scopes
```
Read the full document "test_novel" (scope="document", page=1).
Then read just chapter "03-confrontation.md" (scope="chapter").
Then read paragraph 0 from "01-intro.md" (scope="paragraph").
```

**18. find_text** - Search for text
```
Search for "Marcus" across the entire "test_novel" document.
Then search for "pendant" in just chapter "02-discovery.md".
```

**19. replace_text** - Find and replace
```
Replace "pendant" with "amulet" in chapter "02-discovery.md" only.
```

**20. get_statistics** - Get word counts
```
Get statistics for the entire "test_novel" document.
Then get statistics for just "01-intro.md".
```

**21. find_similar_text** - Semantic search (requires GEMINI_API_KEY)
```
If GEMINI_API_KEY is available, search for content similar to "mysterious artifact" in "test_novel".
```

**22. find_entity** - Entity mention tracking
```
First, create entities metadata for "test_novel":
- Add character "Marcus Chen" with aliases ["Marcus", "Chen"]
- Add character "Sarah" with aliases ["Sarah"]
- Add item "The Pendant" with aliases ["pendant", "amulet", "artifact"]

Then use find_entity to find all mentions of "Marcus" across the document.
```

### Phase 5: Metadata Management (5 tools)

**23. write_metadata** - Write chapter and entity metadata
```
Update the metadata for "01-intro.md":
  - scope="chapter", target="01-intro.md", status="revised"

Add a timeline event:
  - scope="timeline", event_id="discovery", date="Day 1", description="Marcus finds the pendant", chapters=["02-discovery.md"]

Add/update entity metadata:
  - scope="entity", target="Marcus Chen", entity_type="character", aliases=["Marcus", "Chen"], description="Main protagonist"
```

**Note**: All metadata tools use individual typed parameters (status, pov_character, tags, etc.) rather than dict parameters for Gemini API compatibility.

**24. read_metadata** - Read metadata back
```
Read the metadata for chapter "01-intro.md".
Read the entity metadata for "Marcus Chen".
Read the timeline metadata.
```

**25. list_metadata** - List and filter metadata
```
List all chapter metadata in "test_novel".
List only chapters with status="draft".
List all entities.
List all timeline events.
```

**26. get_document_outline** - Get comprehensive outline
```
Get the full document outline for "test_novel" with include_metadata=true and include_entity_counts=true.
```

### Phase 6: Version Control (3 tools)

**27. manage_snapshots** - Create and list snapshots
```
Create a snapshot of "test_novel" with the description "Before major edits".
List all snapshots for "test_novel".
```

**28. check_content_status** - Check modification history
```
Check the content status for chapter "01-intro.md" to see if it's been modified.
```

**29. diff_content** - Compare versions
```
Make a small change to "01-intro.md", then diff against the snapshot to see what changed.
```

### Phase 7: Discovery (1 tool)

**30. search_tool** - Tool discovery
```
Search for tools related to "metadata".
Search for tools in category "Version Control".
```

### Cleanup

```
Delete the "test_novel" document to clean up after testing.
```

---

## Expected Results

After running all tests, you should have verified:

1. **Document lifecycle**: Create, list, delete documents
2. **Chapter operations**: Create with metadata, list with frontmatter, update while preserving frontmatter
3. **Paragraph manipulation**: All atomic operations (append, replace, insert, delete, move)
4. **Content access**: Multi-scope reading (document/chapter/paragraph), search, replace, statistics
5. **Metadata system**: Chapter frontmatter, entity tracking, timeline events
6. **Version control**: Snapshots, status checking, diffing
7. **Tool discovery**: Search and category filtering

## Tool Count Summary

| Category | Tools |
|----------|-------|
| Document Management | 6 |
| Chapter Management | 4 |
| Paragraph Operations | 8 |
| Scope-based Content Access | 6 |
| Metadata Management | 3 |
| Version Control | 3 |
| Overview | 1 |
| Discovery | 1 |
| **Total** | **32** |

Note: Total registered MCP tools = 32 (including search_tool and get_document_outline).
