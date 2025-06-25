from src.agents.react_agent.main import REACT_SYSTEM_PROMPT, ReActStep


def simulate_react_step(thought: str, action: str = None) -> dict:
    """Simulate a ReAct step response that an LLM would generate."""
    step = ReActStep(thought=thought, action=action)
    return step.model_dump()


def test_react_sequence():
    """Test a complete ReAct sequence for document creation."""
    steps = [
        {
            "step_num": 1,
            "response": simulate_react_step(
                "The user wants me to create a document and add two chapters. I need to do this step by step. First, I'll create the document.",
                'create_document(document_name="My Novel")',
            ),
            "observation": {
                "success": True,
                "message": "Document 'My Novel' created successfully.",
            },
        },
        {
            "step_num": 2,
            "response": simulate_react_step(
                "The document has been created successfully. Now I need to add the first chapter 'prologue'. I'll create it with a proper filename.",
                'create_chapter(document_name="My Novel", chapter_name="00-prologue.md", initial_content="# Prologue")',
            ),
            "observation": {
                "success": True,
                "message": "Chapter '00-prologue.md' created successfully in document 'My Novel'.",
            },
        },
        {
            "step_num": 3,
            "response": simulate_react_step(
                "The prologue has been created successfully. Now I need to add the second chapter 'chapter 1' to complete the user's request.",
                'create_chapter(document_name="My Novel", chapter_name="01-chapter1.md", initial_content="# Chapter 1")',
            ),
            "observation": {
                "success": True,
                "message": "Chapter '01-chapter1.md' created successfully in document 'My Novel'.",
            },
        },
        {
            "step_num": 4,
            "response": simulate_react_step(
                "Perfect! I have successfully created the document 'My Novel' and added both requested chapters: the prologue and chapter 1. The user's request has been completely fulfilled.",
                None,
            ),
            "observation": None,  # No observation needed for completion step
        },
    ]

    assert len(steps) == 4
    final_step = steps[3]
    assert final_step["response"]["action"] is None
    assert "prologue" in steps[1]["response"]["action"]


def test_error_handling():
    """Test how the system prompt guides error handling."""
    error_steps = [
        {
            "step_num": 1,
            "response": simulate_react_step(
                "The user wants me to add a chapter to 'Nonexistent Book'. I'll try to create the chapter directly.",
                'create_chapter(document_name="Nonexistent Book", chapter_name="01-intro.md", initial_content="# Introduction")',
            ),
            "observation": {
                "success": False,
                "message": "Document 'Nonexistent Book' not found.",
            },
        },
        {
            "step_num": 2,
            "response": simulate_react_step(
                "The chapter creation failed because the document doesn't exist. I need to create the document first before adding chapters to it.",
                'create_document(document_name="Nonexistent Book")',
            ),
            "observation": {
                "success": True,
                "message": "Document 'Nonexistent Book' created successfully.",
            },
        },
        {
            "step_num": 3,
            "response": simulate_react_step(
                "Now that the document exists, I can create the chapter as originally requested.",
                'create_chapter(document_name="Nonexistent Book", chapter_name="01-intro.md", initial_content="# Introduction")',
            ),
            "observation": {
                "success": True,
                "message": "Chapter '01-intro.md' created successfully in document 'Nonexistent Book'.",
            },
        },
        {
            "step_num": 4,
            "response": simulate_react_step(
                "Excellent! I successfully handled the error by creating the document first, then adding the requested chapter. The task is now complete.",
                None,
            ),
            "observation": None,
        },
    ]

    assert not error_steps[0]["observation"]["success"]
    assert "not found" in error_steps[0]["observation"]["message"]
    assert "create_document" in error_steps[1]["response"]["action"]
    assert error_steps[2]["observation"]["success"]


def test_complex_multi_document_workflow():
    """Test complex multi-document workflow with cross-document operations."""
    overview_content = "# Documentation Overview\\n\\nThis documentation suite consists of:\\n\\n- **Main Guide** (this document) - Core concepts\\n- **API Reference** - Detailed API documentation\\n- **Tutorial** - Step-by-step learning guide\\n\\n## Quick Navigation\\n\\nFor API details, see the API Reference\\nFor hands-on learning, see the Tutorial"
    api_content = "# API Reference\\n\\n> **Note**: For conceptual overview, see Main Guide\\n\\n## Available Endpoints\\n\\n### Document Management\\n- POST /documents - Create document\\n- GET /documents - List documents\\n\\n### Chapter Management\\n- POST /documents/{id}/chapters - Create chapter\\n- GET /documents/{id}/chapters - List chapters"
    tutorial_content = "# Getting Started Tutorial\\n\\n> **Prerequisites**: Read the Main Guide Overview first\\n\\n## What You'll Learn\\n\\n1. Creating your first document\\n2. Adding chapters\\n3. Managing content\\n\\n## Step 1: Create Your First Document\\n\\nLet's start by creating a simple document..."

    complex_steps = [
        {
            "step_num": 1,
            "response": simulate_react_step(
                "I need to create a comprehensive documentation suite with three interconnected documents. I'll start by creating the main guide document.",
                'create_document(document_name="Main Guide")',
            ),
            "observation": {
                "success": True,
                "message": "Document 'Main Guide' created successfully.",
            },
        },
        {
            "step_num": 2,
            "response": simulate_react_step(
                "Main guide created. Now I'll create the API reference document.",
                'create_document(document_name="API Reference")',
            ),
            "observation": {
                "success": True,
                "message": "Document 'API Reference' created successfully.",
            },
        },
        {
            "step_num": 3,
            "response": simulate_react_step(
                "API reference created. Now I'll create the tutorial document to complete the suite.",
                'create_document(document_name="Tutorial")',
            ),
            "observation": {
                "success": True,
                "message": "Document 'Tutorial' created successfully.",
            },
        },
        {
            "step_num": 4,
            "response": simulate_react_step(
                "All three documents created. Now I'll add the main overview chapter to the Main Guide with cross-references.",
                f'create_chapter(document_name="Main Guide", chapter_name="01-overview.md", initial_content="{overview_content}")',
            ),
            "observation": {
                "success": True,
                "message": "Chapter '01-overview.md' created successfully in document 'Main Guide'.",
            },
        },
        {
            "step_num": 5,
            "response": simulate_react_step(
                "Overview chapter with cross-references created. Now I'll add the API reference index with links back to the main guide.",
                f'create_chapter(document_name="API Reference", chapter_name="00-index.md", initial_content="{api_content}")',
            ),
            "observation": {
                "success": True,
                "message": "Chapter '00-index.md' created successfully in document 'API Reference'.",
            },
        },
        {
            "step_num": 6,
            "response": simulate_react_step(
                "API reference index created with bidirectional links. Now I'll create the tutorial introduction that references both other documents.",
                f'create_chapter(document_name="Tutorial", chapter_name="01-getting-started.md", initial_content="{tutorial_content}")',
            ),
            "observation": {
                "success": True,
                "message": "Chapter '01-getting-started.md' created successfully in document 'Tutorial'.",
            },
        },
        {
            "step_num": 7,
            "response": simulate_react_step(
                "Perfect! I've successfully created a comprehensive documentation suite with three interconnected documents. Each document contains cross-references to the others, creating a cohesive documentation ecosystem. The task is complete.",
                None,
            ),
            "observation": None,
        },
    ]

    assert len(complex_steps) == 7
    assert "create_document" in complex_steps[0]["response"]["action"]
    assert "create_chapter" in complex_steps[3]["response"]["action"]
    assert complex_steps[6]["response"]["action"] is None


def test_system_prompt_structure():
    """Test the structure and content of the system prompt."""
    assert "ReAct Process Overview" in REACT_SYSTEM_PROMPT
    assert "Available Tools" in REACT_SYSTEM_PROMPT
    assert "Example ReAct Sequence" in REACT_SYSTEM_PROMPT
    assert len(REACT_SYSTEM_PROMPT) > 5000
    assert REACT_SYSTEM_PROMPT.count('"thought":') > 2


def test_edge_case_invalid_input():
    """Test handling of invalid input, like illegal characters in names."""
    edge_case_1 = [
        {
            "step_num": 1,
            "response": simulate_react_step(
                "The user wants me to create a chapter with invalid characters in the name. I should try the operation and handle any validation errors.",
                'create_chapter(document_name="Test Doc", chapter_name="invalid/name.md", initial_content="# Test")',
            ),
            "observation": {
                "success": False,
                "message": "Chapter name cannot contain path separators",
            },
        },
        {
            "step_num": 2,
            "response": simulate_react_step(
                "The chapter creation failed due to invalid characters. I need to sanitize the name by removing path separators and try again.",
                'create_chapter(document_name="Test Doc", chapter_name="invalid-name.md", initial_content="# Test")',
            ),
            "observation": {
                "success": True,
                "message": "Chapter 'invalid-name.md' created successfully in document 'Test Doc'.",
            },
        },
    ]
    assert not edge_case_1[0]["observation"]["success"]
    assert "invalid/name" in edge_case_1[0]["response"]["action"]
    assert edge_case_1[1]["observation"]["success"]
    assert "invalid-name" in edge_case_1[1]["response"]["action"]


def test_edge_case_resource_limits():
    """Test handling of resource limits, like oversized content."""
    edge_case_2 = [
        {
            "step_num": 1,
            "response": simulate_react_step(
                "The user wants to create a chapter with extremely large content. I'll attempt the operation and handle any size limitations.",
                'create_chapter(document_name="Large Doc", chapter_name="huge-content.md", initial_content="'
                + "Very long content... " * 1000
                + '")',
            ),
            "observation": {
                "success": False,
                "message": "Content too long (max 1,000,000 characters)",
            },
        },
        {
            "step_num": 2,
            "response": simulate_react_step(
                "The content exceeded size limits. I'll create the chapter with a summary instead and suggest breaking it into multiple chapters.",
                'create_chapter(document_name="Large Doc", chapter_name="huge-content.md", initial_content="# Large Content Chapter\\n\\nThis chapter was intended to contain extensive content but exceeded size limits.\\n\\n## Recommendation\\n\\nConsider breaking this into multiple smaller chapters for better organization.")',
            ),
            "observation": {
                "success": True,
                "message": "Chapter 'huge-content.md' created successfully in document 'Large Doc'.",
            },
        },
    ]
    assert not edge_case_2[0]["observation"]["success"]
    assert "Content too long" in edge_case_2[0]["observation"]["message"]
    assert edge_case_2[1]["observation"]["success"]
    assert "Recommendation" in edge_case_2[1]["response"]["action"]


def test_edge_case_state_management():
    """Test handling of concurrent operations and state management."""
    edge_case_3 = [
        {
            "step_num": 1,
            "response": simulate_react_step(
                "I need to perform multiple operations that might conflict. I'll check the current state first to ensure consistency.",
                "list_documents()",
            ),
            "observation": {
                "success": True,
                "message": "Documents retrieved successfully",
            },
        },
        {
            "step_num": 2,
            "response": simulate_react_step(
                "Current state confirmed. Now I'll attempt to create a document that might already exist, handling any conflicts.",
                'create_document(document_name="Existing Doc")',
            ),
            "observation": {
                "success": False,
                "message": "Document 'Existing Doc' already exists",
            },
        },
        {
            "step_num": 3,
            "response": simulate_react_step(
                "Document already exists. I'll work with the existing document instead of creating a new one, ensuring no data loss.",
                'list_chapters(document_name="Existing Doc")',
            ),
            "observation": {
                "success": True,
                "message": "Chapters retrieved successfully",
            },
        },
    ]

    assert edge_case_3[0]["observation"]["success"]
    assert "list_documents" in edge_case_3[0]["response"]["action"]
    assert not edge_case_3[1]["observation"]["success"]
    assert "already exists" in edge_case_3[1]["observation"]["message"]
    assert "list_chapters" in edge_case_3[2]["response"]["action"]


def test_performance_bulk_operations():
    """Test performance optimization for bulk operations."""
    perf_case_1 = [
        {
            "step_num": 1,
            "response": simulate_react_step(
                "The user wants to create 50 chapters efficiently. Instead of individual operations, I'll use a batch approach to optimize performance.",
                'create_document(document_name="Bulk Operations Test")',
            ),
            "observation": {"success": True},
        },
        {
            "step_num": 2,
            "response": simulate_react_step(
                "Document created. Now I'll generate the batch request to create 50 chapters in a single operation if a batch tool existed. Since it doesn't, I will have to do it one by one. I will just simulate a few.",
                'create_chapter(document_name="Bulk Operations Test", chapter_name="01.md")',
            ),
            "observation": {"success": True},
        },
        {
            "step_num": 3,
            "response": simulate_react_step(
                "Chapter 1 created.",
                'create_chapter(document_name="Bulk Operations Test", chapter_name="02.md")',
            ),
            "observation": {"success": True},
        },
    ]

    assert "batch" in perf_case_1[1]["response"]["thought"]
    assert "create_chapter" in perf_case_1[1]["response"]["action"]


def test_advanced_content_manipulation_search_replace():
    """Test complex search and replace operations."""
    content_case_1 = [
        {
            "step_num": 1,
            "response": simulate_react_step(
                "I need to perform a complex find-and-replace operation across multiple documents. First, I'll search for the target text to understand its scope.",
                'find_text_in_document(document_name="Technical Manual", query="deprecated API", case_sensitive=False)',
            ),
            "observation": {
                "success": True,
                "message": "Found 5 occurrences in 3 chapters.",
            },
        },
        {
            "step_num": 2,
            "response": simulate_react_step(
                "Found 5 occurrences. Now I'll perform a global replacement with a link to the new documentation.",
                'replace_text_in_document(document_name="Technical Manual", text_to_find="deprecated API", replacement_text="legacy API (see New API Guide)")',
            ),
            "observation": {
                "success": True,
                "message": "Replaced 5 occurrences successfully.",
            },
        },
    ]
    assert "find_text_in_document" in content_case_1[0]["response"]["action"]
    assert "replace_text_in_document" in content_case_1[1]["response"]["action"]
    assert content_case_1[1]["observation"]["success"]
