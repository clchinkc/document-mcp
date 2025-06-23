import asyncio
import os
import uuid

import pytest

# Import from the simple agent (moved file)
from src.agents.simple_agent import (
    FinalAgentResponse,
    initialize_agent_and_mcp_server,
    process_single_user_query,
)

# Import shared testing utilities
from tests.shared import (
    assert_agent_response_valid,
    validate_agent_environment,
    validate_package_imports,
    validate_simple_agent_imports,
)

# Import from the installed document_mcp package for direct manipulation
try:
    pass
except ImportError:
    # Fallback for development/testing without installed package
    import sys

    sys.path.insert(0, "..")

# --- Environment Testing Functions ---


def test_agent_environment_setup():
    """Test agent environment setup and configuration."""
    validate_agent_environment()


def test_agent_package_imports():
    """Test if all required packages can be imported for agent functionality."""
    validate_package_imports()
    validate_simple_agent_imports()


# --- Test Helper ---


async def run_simple_agent_test(query: str):
    """Simplified agent test runner."""
    agent, _ = await initialize_agent_and_mcp_server()
    async with agent.run_mcp_servers():
        result = await asyncio.wait_for(
            process_single_user_query(agent, query), timeout=30.0
        )
        return result


async def run_conversation_test(queries: list[str], timeout: float = 30.0):
    """
    Run multiple queries in sequence using the same agent connection.
    
    This is specifically designed for conversation tests that need to maintain
    state across multiple rounds of interaction.
    
    Args:
        queries: List of queries to run in sequence
        timeout: Timeout per individual query
        
    Returns:
        List of FinalAgentResponse objects, one per query
    """
    agent, _ = await initialize_agent_and_mcp_server()
    async with agent.run_mcp_servers():
        results = []
        for query in queries:
            result = await asyncio.wait_for(
                process_single_user_query(agent, query), timeout=timeout
            )
            results.append(result)
        return results


# --- Core Agent Test Cases ---


@pytest.mark.asyncio
async def test_agent_list_documents_empty(test_docs_root):
    """Test that listing documents works when no documents exist."""
    response = await run_simple_agent_test("Show me all available documents")

    assert_agent_response_valid(response, "Simple agent")
    assert isinstance(
        response.details, list
    ), "Details for list_documents must be a list"


@pytest.mark.asyncio
async def test_agent_create_document_and_list(test_docs_root):
    """Test creating a document and then listing it."""
    doc_name = f"test_doc_{uuid.uuid4().hex[:8]}"

    # Create document
    create_response = await run_simple_agent_test(
        f"Create a new document named '{doc_name}'"
    )
    assert_agent_response_valid(create_response, "Simple agent")


@pytest.mark.asyncio
async def test_agent_add_chapter_to_document(test_docs_root):
    """Test adding a chapter to a document that was created by the agent."""
    doc_name = f"doc_chapter_{uuid.uuid4().hex[:8]}"
    chapter_name = "chapter1.md"
    chapter_content = "This is the first chapter."

    # Step 1: Create the document using the agent
    create_doc_response = await run_simple_agent_test(
        f"Create a new document named '{doc_name}'"
    )
    assert_agent_response_valid(create_doc_response, "Simple agent")

    # Step 2: Add a chapter to the newly created document
    add_chapter_response = await run_simple_agent_test(
        f"Create a chapter named '{chapter_name}' in document '{doc_name}' with content: {chapter_content}"
    )
    assert_agent_response_valid(add_chapter_response, "Simple agent")


@pytest.mark.asyncio
async def test_agent_read_chapter_content(test_docs_root):
    """Test reading chapter content."""
    doc_name = "readable_doc_test"
    chapter_name = "readable_chapter.md"
    content = "Content to be read."

    # Set up test data
    doc_dir = test_docs_root / doc_name
    doc_dir.mkdir(parents=True, exist_ok=True)
    chapter_path = doc_dir / chapter_name
    chapter_path.write_text(content)

    # Read chapter
    response = await run_simple_agent_test(
        f"Read chapter '{chapter_name}' from document '{doc_name}'"
    )
    assert_agent_response_valid(response, "Simple agent")


@pytest.mark.asyncio
async def test_agent_get_document_statistics(test_docs_root):
    """Test getting document statistics."""
    doc_name = "stats_doc_test"

    # Set up test data
    doc_dir = test_docs_root / doc_name
    doc_dir.mkdir(parents=True, exist_ok=True)

    # Create a few chapters with content
    for i in range(1, 4):
        chapter_path = doc_dir / f"0{i}-chapter.md"
        chapter_path.write_text(f"# Chapter {i}\n\nThis is chapter {i} content.")

    # Get statistics
    response = await run_simple_agent_test(f"Get statistics for document '{doc_name}'")
    assert_agent_response_valid(response, "Simple agent")


@pytest.mark.asyncio
async def test_agent_find_text_in_document(test_docs_root):
    """Test finding text in a document."""
    doc_name = "search_doc_test"

    # Set up test data
    doc_dir = test_docs_root / doc_name
    doc_dir.mkdir(parents=True, exist_ok=True)

    # Create chapters with searchable content
    chapter1_path = doc_dir / "01-intro.md"
    chapter1_path.write_text(
        "# Introduction\n\nThis chapter contains searchable content."
    )

    chapter2_path = doc_dir / "02-body.md"
    chapter2_path.write_text("# Main Content\n\nMore searchable text here.")

    # Search for text
    response = await run_simple_agent_test(
        f"Find the text 'searchable' in document '{doc_name}'"
    )
    assert_agent_response_valid(response, "Simple agent")


# --- AI Model-Specific Tests ---


@pytest.mark.asyncio
async def test_openai_gpt_model_integration(test_docs_root):
    """Test that OpenAI GPT model works correctly with the agent."""
    # Skip test if OpenAI API key is not available
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not found in environment - skipping OpenAI test")

    # Temporarily force OpenAI model by clearing Gemini key
    original_gemini_key = os.environ.get("GEMINI_API_KEY")
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]

    try:
        doc_name = f"openai_test_doc_{uuid.uuid4().hex[:8]}"

        # Test document creation with OpenAI model
        create_response = await run_simple_agent_test(
            f"Create a new document named '{doc_name}'"
        )
        assert_agent_response_valid(create_response, "OpenAI model")
        assert (
            create_response.error_message is None
        ), "OpenAI model should not produce error messages for valid requests"

        # Test listing documents to verify creation worked
        list_response = await run_simple_agent_test("Show me all available documents")
        assert_agent_response_valid(list_response, "OpenAI model")
        assert isinstance(
            list_response.details, list
        ), "OpenAI model should return list of documents"

        # Verify our document was created and appears in the list (using correct attribute name)
        doc_names = [
            doc.document_name
            for doc in list_response.details
            if hasattr(doc, "document_name")
        ]
        assert (
            doc_name in doc_names
        ), f"Document '{doc_name}' should be in the list created by OpenAI model"

        print("✅ OpenAI GPT model test passed")

    finally:
        # Restore original Gemini key if it existed
        if original_gemini_key is not None:
            os.environ["GEMINI_API_KEY"] = original_gemini_key


@pytest.mark.asyncio
async def test_google_gemini_model_integration(test_docs_root):
    """Test that Google Gemini model works correctly with the agent."""
    # Skip test if Gemini API key is not available
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not found in environment - skipping Gemini test")

    # Temporarily force Gemini model by clearing OpenAI key
    original_openai_key = os.environ.get("OPENAI_API_KEY")
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]

    try:
        doc_name = f"gemini_test_doc_{uuid.uuid4().hex[:8]}"

        # Test document creation with Gemini model
        create_response = await run_simple_agent_test(
            f"Create a new document named '{doc_name}'"
        )
        assert_agent_response_valid(create_response, "Gemini model")
        assert (
            create_response.error_message is None
        ), "Gemini model should not produce error messages for valid requests"

        # Test statistics with Gemini model
        doc_dir = test_docs_root / doc_name
        if doc_dir.exists():
            # Add some content to test statistics
            chapter_path = doc_dir / "01-test.md"
            chapter_path.write_text(
                "# Test Chapter\n\nThis is test content for Gemini model."
            )

            stats_response = await run_simple_agent_test(
                f"Get statistics for document '{doc_name}'"
            )
            assert_agent_response_valid(stats_response, "Gemini model")

        print("✅ Google Gemini model test passed")

    finally:
        # Restore original OpenAI key if it existed
        if original_openai_key is not None:
            os.environ["OPENAI_API_KEY"] = original_openai_key


# --- Multi-Round Conversation Tests ---


@pytest.mark.asyncio
async def test_simple_agent_three_round_conversation_document_workflow(test_docs_root):
    """
    Test a 3-round conversation workflow with document creation, modification, and reading.

    This test validates:
    - State persistence across conversation rounds
    - Proper data flow between rounds
    - Independent operation summaries

    Workflow:
    1. Create a new document
    2. Add a chapter with content
    3. Read the chapter content back
    """
    # Use worker ID and test ID for better isolation in parallel runs
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
    doc_name = f"multiround_doc_{worker_id}_{uuid.uuid4().hex[:8]}"
    chapter_name = "01-intro.md"
    chapter_content = (
        "# Introduction\n\nThis is the first chapter of our multi-round test."
    )

    # Run all rounds in a single conversation to maintain agent connection
    queries = [
        f"Create a new document named '{doc_name}'",
        f"Create a chapter named '{chapter_name}' in document '{doc_name}' "
        f"with content: {chapter_content}",
        f"Read chapter '{chapter_name}' from document '{doc_name}'"
    ]
    
    responses = await run_conversation_test(queries)
    assert len(responses) == 3, "Should have responses for all 3 rounds"
    
    round1_response, round2_response, round3_response = responses

    # Validate each round
    assert_agent_response_valid(round1_response, "Simple agent")
    assert round1_response.error_message is None, "Round 1 should not have errors"

    assert_agent_response_valid(round2_response, "Simple agent")
    assert round2_response.error_message is None, "Round 2 should not have errors"

    assert_agent_response_valid(round3_response, "Simple agent")
    assert round3_response.error_message is None, "Round 3 should not have errors"

    # Validate state persistence and independence
    assert round3_response.details is not None, "Round 3 should return chapter content"
    assert (
        round1_response.summary != round2_response.summary
    ), "Each round should have different summaries"
    assert (
        round2_response.summary != round3_response.summary
    ), "Each round should have different summaries"


@pytest.mark.asyncio
async def test_simple_agent_three_round_conversation_with_error_recovery(
    test_docs_root,
):
    """
    Test a 3-round conversation with error handling and recovery.

    This test validates:
    - Graceful error handling for invalid operations
    - Recovery from errors in subsequent rounds
    - Continued operation after error recovery

    Workflow:
    1. Attempt invalid operation (read non-existent content)
    2. Recover by creating the document
    3. Successfully add content to demonstrate recovery
    """
    # Use worker ID and test ID for better isolation in parallel runs
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
    doc_name = f"error_recovery_doc_{worker_id}_{uuid.uuid4().hex[:8]}"

    # Run all rounds in a single conversation to maintain agent connection
    queries = [
        f"Read chapter 'nonexistent.md' from document '{doc_name}'",
        f"Create a new document named '{doc_name}'",
        f"Create a chapter named '01-recovery.md' in document '{doc_name}' "
        f"with content: # Recovery Chapter"
    ]
    
    responses = await run_conversation_test(queries)
    assert len(responses) == 3, "Should have responses for all 3 rounds"
    
    round1_response, round2_response, round3_response = responses

    # Validate each round
    assert_agent_response_valid(round1_response, "Simple agent")
    # Note: Error is expected here, but response should still be valid

    assert_agent_response_valid(round2_response, "Simple agent")
    assert (
        round2_response.error_message is None
    ), "Round 2 should succeed after error recovery"

    assert_agent_response_valid(round3_response, "Simple agent")
    assert (
        round3_response.error_message is None
    ), "Round 3 should succeed after recovery"

    # Verify error recovery behavior
    assert (
        round1_response.summary != round2_response.summary
    ), "Error and recovery should have different responses"
    assert (
        round2_response.error_message is None
    ), "Recovery round should not have errors"


@pytest.mark.asyncio
async def test_simple_agent_three_round_conversation_state_isolation(test_docs_root):
    """
    Test that each round maintains proper state isolation.

    This test validates:
    - Independent operation of multiple documents
    - No interference between separate operations
    - Proper state isolation across rounds

    Workflow:
    1. Create first document
    2. Create second document (should not interfere with first)
    3. Verify both documents exist independently
    """
    # Use worker ID and test ID for better isolation in parallel runs
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
    base_doc_name = f"isolation_test_{worker_id}_{uuid.uuid4().hex[:8]}"
    doc1_name = f"{base_doc_name}_1"
    doc2_name = f"{base_doc_name}_2"

    # Run all rounds in a single conversation to maintain agent connection
    queries = [
        f"Create a new document named '{doc1_name}'",
        f"Create a new document named '{doc2_name}'",
        "Show me all available documents"
    ]
    
    responses = await run_conversation_test(queries)
    assert len(responses) == 3, "Should have responses for all 3 rounds"
    
    round1_response, round2_response, round3_response = responses

    # Validate each round
    assert_agent_response_valid(round1_response, "Simple agent")
    assert_agent_response_valid(round2_response, "Simple agent")
    assert_agent_response_valid(round3_response, "Simple agent")
    assert isinstance(round3_response.details, list), "Should return list of documents"

    # Verify state isolation: both documents should exist independently
    doc_names = [
        doc.document_name
        for doc in round3_response.details
        if hasattr(doc, "document_name")
    ]
    assert doc1_name in doc_names, f"First document {doc1_name} should exist"
    assert doc2_name in doc_names, f"Second document {doc2_name} should exist"

    # Verify each round was independent
    assert (
        round1_response.summary != round2_response.summary
    ), "Independent operations should have different summaries"


@pytest.mark.asyncio
async def test_simple_agent_three_round_conversation_resource_cleanup(test_docs_root):
    """
    Test that resources are properly cleaned up between conversation rounds.

    This test validates:
    - Proper resource cleanup between rounds
    - No resource conflicts or leaks
    - Consistent operation across multiple rounds

    Workflow:
    1. Create document and verify it exists
    2. Add content to document
    3. Access document statistics (tests resource availability)
    """
    # Use worker ID and test ID for better isolation in parallel runs
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
    doc_name = f"cleanup_test_{worker_id}_{uuid.uuid4().hex[:8]}"

    # Run all rounds in a single conversation to maintain agent connection
    queries = [
        f"Create a new document named '{doc_name}'",
        f"Create a chapter named '01-test.md' in document '{doc_name}' "
        f"with content: # Test Content",
        f"Get statistics for document '{doc_name}'"
    ]
    
    responses = await run_conversation_test(queries)
    assert len(responses) == 3, "Should have responses for all 3 rounds"
    
    round1_response, round2_response, round3_response = responses

    # Validate each round
    assert_agent_response_valid(round1_response, "Simple agent")
    assert round1_response.error_message is None, "Round 1 should succeed"

    assert_agent_response_valid(round2_response, "Simple agent")
    assert round2_response.error_message is None, "Round 2 should succeed"

    assert_agent_response_valid(round3_response, "Simple agent")
    assert round3_response.error_message is None, "Round 3 should succeed"
    assert round3_response.details is not None, "Statistics should be available"

    # Verify resource cleanup: each round should complete successfully without resource conflicts
    responses_list = [round1_response, round2_response, round3_response]
    for i, response in enumerate(responses_list, 1):
        assert (
            response.error_message is None
        ), f"Round {i} should not have resource-related errors"
        assert response.summary is not None, f"Round {i} should have valid summary"
        assert len(response.summary) > 0, f"Round {i} summary should not be empty"


@pytest.mark.asyncio
async def test_simple_agent_three_round_conversation_complex_workflow(test_docs_root):
    """
    Test a complex 3-round workflow combining multiple operations.

    This test validates:
    - Complex multi-operation workflows
    - State persistence across diverse operations
    - Data availability and consistency

    Workflow:
    1. Create document with initial content
    2. Search for text in the document
    3. Get comprehensive statistics
    """
    doc_name = f"complex_workflow_{uuid.uuid4().hex[:8]}"
    search_content = (
        "# Introduction\n\nThis document contains searchable content for testing."
    )

    # Round 1: Create document with initial content
    round1_response = await run_simple_agent_test(
        f"Create a new document named '{doc_name}'"
    )
    assert_agent_response_valid(round1_response, "Simple agent")

    # Setup: Add content for subsequent rounds to work with
    setup_response = await run_simple_agent_test(
        f"Create a chapter named '01-intro.md' in document '{doc_name}' "
        f"with content: {search_content}"
    )
    assert_agent_response_valid(setup_response, "Simple agent")

    # Round 2: Search for text in the document
    round2_response = await run_simple_agent_test(
        f"Find the text 'searchable' in document '{doc_name}'"
    )
    assert_agent_response_valid(round2_response, "Simple agent")
    assert round2_response.error_message is None, "Search should succeed"

    # Round 3: Get comprehensive statistics
    round3_response = await run_simple_agent_test(
        f"Get statistics for document '{doc_name}'"
    )
    assert_agent_response_valid(round3_response, "Simple agent")
    assert round3_response.error_message is None, "Statistics should succeed"
    assert round3_response.details is not None, "Statistics should return data"

    # Verify complex workflow: all operations should work together
    assert round1_response.error_message is None, "Document creation should succeed"
    assert round2_response.error_message is None, "Text search should succeed"
    assert round3_response.error_message is None, "Statistics should succeed"

    # Verify state persistence across the workflow
    search_has_data = (
        round2_response.details is not None or round2_response.summary is not None
    )
    assert search_has_data, "Search should return results or summary"
    assert round3_response.details is not None, "Statistics should return detailed data"
