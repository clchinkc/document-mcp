import os
import shutil
import tempfile
from pathlib import Path

import pytest

from document_mcp import (  # Import the module itself to modify its global
    doc_tool_server,
)

# Make sure to import the necessary functions and models from doc_tool_server
from document_mcp.doc_tool_server import (
    read_full_document,
)
from document_mcp.doc_tool_server import (
    ChapterContent,
    ChapterMetadata,
    DocumentInfo,
    FullDocumentContent,
    OperationStatus,
    ParagraphDetail,
    StatisticsReport,
    append_paragraph_to_chapter,
    create_chapter,
    create_document,
    delete_chapter,
    delete_document,
    find_text_in_chapter,
    find_text_in_document,
    get_chapter_statistics,
    get_document_statistics,
    list_chapters,
    list_documents,

    read_chapter_content,
    read_document_summary,
    read_paragraph_content,
    replace_text_in_chapter,
    replace_text_in_document,
    write_chapter_content,
    DOCUMENT_SUMMARY_FILE
)

# --- Environment Testing Functions ---


def test_package_imports():
    """Test if all required packages can be imported."""
    try:
        import pydantic_ai

        # Verify pydantic_ai has expected functionality
        assert hasattr(pydantic_ai, "Agent"), "pydantic_ai should provide Agent class"
        assert hasattr(
            pydantic_ai, "RunContext"
        ), "pydantic_ai should provide RunContext class"
    except ImportError:
        pytest.fail("Failed to import pydantic_ai")

    try:
        # Test imports work
        from document_mcp.doc_tool_server import ChapterContent, StatisticsReport

        # Verify imported classes are proper types
        assert isinstance(
            StatisticsReport, type
        ), "StatisticsReport should be a class type"
        assert isinstance(ChapterContent, type), "ChapterContent should be a class type"
        # Verify they are pydantic models
        assert hasattr(
            StatisticsReport, "model_fields"
        ), "StatisticsReport should be a pydantic model"
        assert hasattr(
            ChapterContent, "model_fields"
        ), "ChapterContent should be a pydantic model"
    except ImportError as e:
        pytest.fail(f"Failed to import from doc_tool_server: {e}")


# --- Pytest Fixtures ---


## Pytest fixtures are provided via conftest.py


# --- Helper Functions for Tests ---


def _assert_operation_success(
    status: OperationStatus, expected_message_part: str = None
):
    assert (
        status.success is True
    ), f"Operation should succeed but got failure: {status.message}"
    assert (
        isinstance(status.message, str) and len(status.message) > 0
    ), "Success status should have a meaningful message"
    if expected_message_part:
        assert (
            expected_message_part.lower() in status.message.lower()
        ), f"Expected '{expected_message_part}' in success message: '{status.message}'"


def _assert_operation_failure(
    status: OperationStatus, expected_message_part: str = None
):
    assert (
        status.success is False
    ), f"Operation should fail but got success: {status.message}"
    assert (
        isinstance(status.message, str) and len(status.message) > 0
    ), "Failure status should have a meaningful error message"
    if expected_message_part:
        assert (
            expected_message_part.lower() in status.message.lower()
        ), f"Expected '{expected_message_part}' to be in '{status.message}'"


# --- Comprehensive Integration Tests ---


def test_comprehensive_statistics_functionality(document_factory, test_docs_root: Path, validate_test_data):
    """Test comprehensive statistics functionality for documents and chapters."""
    # Create a document with 5 chapters, each with Lorem ipsum content and 10 paragraphs
    chapters = []
    for i in range(1, 6):
        content = f"# Chapter {i}\n\n"
        # Add 10 paragraphs per chapter
        for j in range(1, 11):
            content += f"Lorem ipsum dolor sit amet, consectetur adipiscing elit paragraph {j}.\n\n"
        chapters.append((f"{i:02d}-chapter.md", content))
    
    doc_name = document_factory(
        doc_type="simple",
        name="test_story_document",
        chapters=chapters
    )
    
    # Validate the document was created correctly
    validate_test_data.document_exists(test_docs_root, doc_name)

    # Test document statistics
    doc_stats = get_document_statistics(document_name=doc_name)
    assert isinstance(
        doc_stats, StatisticsReport
    ), f"Expected StatisticsReport, got {type(doc_stats)}"

    # Check basic fields exist
    assert hasattr(
        doc_stats, "word_count"
    ), "StatisticsReport missing 'word_count' field"
    assert hasattr(
        doc_stats, "paragraph_count"
    ), "StatisticsReport missing 'paragraph_count' field"
    assert hasattr(
        doc_stats, "chapter_count"
    ), "StatisticsReport missing 'chapter_count' field"

    # Check values are reasonable (each chapter has title + 10 content paragraphs = 11 total)
    assert (
        doc_stats.chapter_count == 5
    ), f"Expected 5 chapters, got {doc_stats.chapter_count}"
    assert (
        doc_stats.paragraph_count == 55
    ), f"Expected 55 paragraphs (5*11), got {doc_stats.paragraph_count}"
    assert (
        doc_stats.word_count > 200
    ), f"Expected reasonable word count, got {doc_stats.word_count}"

    # Test chapter statistics for multiple chapters
    for chapter_num in [1, 3, 5]:
        chapter_name = f"{chapter_num:02d}-chapter.md"
        chapter_stats = get_chapter_statistics(
            document_name=doc_name, chapter_name=chapter_name
        )
        assert isinstance(
            chapter_stats, StatisticsReport
        ), f"Expected StatisticsReport for chapter {chapter_num}"
        # Chapter statistics don't necessarily need chapter_count, so don't assert on it
        if chapter_stats.chapter_count is not None:
            assert (
                chapter_stats.chapter_count == 1
            ), f"Chapter stats should show 1 chapter if present, got {chapter_stats.chapter_count}"
        assert (
            chapter_stats.paragraph_count == 11
        ), f"Expected 11 paragraphs (title + 10 content) in chapter {chapter_num}"
        assert (
            chapter_stats.word_count > 20
        ), f"Expected reasonable word count for chapter {chapter_num}"


def test_comprehensive_search_functionality(document_factory, test_docs_root: Path, validate_test_data):
    """Test comprehensive search functionality across different scenarios."""
    # Create a searchable document with Lorem ipsum content
    search_terms = ["Lorem", "paragraph"]
    doc_name = document_factory(
        doc_type="searchable",
        name="search_test_document",
        search_terms=search_terms
    )
    
    # Also create additional content to match the original test expectations
    # We need 5 chapters with 10 "Lorem" occurrences each
    chapters = []
    for i in range(1, 6):
        content = f"# Chapter {i}\n\n"
        # Add 10 paragraphs per chapter, each with "Lorem"
        for j in range(1, 11):
            content += f"Lorem ipsum dolor sit amet, consectetur adipiscing elit paragraph {j}.\n\n"
        chapters.append((f"{i:02d}-chapter.md", content))
    
    # Override the searchable document with our specific structure
    doc_name = document_factory(
        doc_type="simple",
        name="comprehensive_search_doc",
        chapters=chapters
    )
    
    # Validate the document was created correctly
    validate_test_data.document_exists(test_docs_root, doc_name)

    # Test chapter search
    chapter_results = find_text_in_chapter(
        document_name=doc_name, chapter_name="01-chapter.md", query="Lorem"
    )

    # Each chapter has 10 paragraphs, each with "Lorem"
    assert (
        len(chapter_results) == 10
    ), f"Expected 10 'Lorem' occurrences in chapter, got {len(chapter_results)}"

    # Test document-wide search
    doc_results = find_text_in_document(document_name=doc_name, query="Lorem")

    # 5 chapters × 10 paragraphs = 50 occurrences
    assert (
        len(doc_results) == 50
    ), f"Expected 50 'Lorem' occurrences in document, got {len(doc_results)}"

    # Test different search terms
    paragraph_results = find_text_in_document(document_name=doc_name, query="paragraph")
    assert len(paragraph_results) > 0, "Should find paragraphs containing 'paragraph'"

    # Test case sensitivity
    case_sensitive_results = find_text_in_document(
        document_name=doc_name, query="lorem", case_sensitive=True  # lowercase
    )
    assert isinstance(
        case_sensitive_results, list
    ), "Case-sensitive search should return a list"
    assert (
        len(case_sensitive_results) == 0
    ), "Case-sensitive search for 'lorem' should find no matches (all test data uses 'Lorem')"

    # Test search that should return no results
    no_results = find_text_in_document(
        document_name=doc_name, query="nonexistent_unique_term_xyz"
    )
    assert isinstance(
        no_results, list
    ), "Search should return a list even when no results found"
    assert (
        len(no_results) == 0
    ), "Search for nonexistent term should return empty list, not None or other value"


def test_comprehensive_content_operations(document_factory, test_docs_root: Path, validate_test_data):
    """Test comprehensive content reading and manipulation operations."""
    # Create a document with 5 chapters, each with Lorem ipsum content and 10 paragraphs
    chapters = []
    for i in range(1, 6):
        content = f"# Chapter {i}\n\n"
        # Add 10 paragraphs per chapter
        for j in range(1, 11):
            content += f"Lorem ipsum dolor sit amet, consectetur adipiscing elit paragraph {j}.\n\n"
        chapters.append((f"{i:02d}-chapter.md", content))
    
    doc_name = document_factory(
        doc_type="simple",
        name="content_ops_document",
        chapters=chapters
    )
    
    # Validate the document was created correctly
    validate_test_data.document_exists(test_docs_root, doc_name)

    # Test reading individual chapters
    for chapter_num in [1, 3, 5]:
        chapter_name = f"{chapter_num:02d}-chapter.md"
        chapter_content = read_chapter_content(
            document_name=doc_name, chapter_name=chapter_name
        )
        assert (
            chapter_content is not None
        ), f"Should be able to read chapter {chapter_num}"
        assert isinstance(
            chapter_content, ChapterContent
        ), f"Expected ChapterContent for chapter {chapter_num}"
        assert (
            f"Chapter {chapter_num}" in chapter_content.content
        ), f"Chapter {chapter_num} should contain title"
        assert (
            len(chapter_content.content) > 100
        ), f"Chapter {chapter_num} should have substantial content"

    # Test reading full document
    full_doc = read_full_document(document_name=doc_name)
    assert full_doc is not None, "Should be able to read full document"
    assert isinstance(full_doc, FullDocumentContent), "Expected FullDocumentContent"
    assert (
        len(full_doc.chapters) == 5
    ), f"Full document should contain 5 chapters, got {len(full_doc.chapters)}"
    assert (
        full_doc.total_word_count > 200
    ), "Full document should have substantial word count"
    assert (
        full_doc.total_paragraph_count == 55
    ), f"Expected 55 total paragraphs (5*11), got {full_doc.total_paragraph_count}"

    # Test paragraph reading
    paragraph = read_paragraph_content(
        document_name=doc_name,
        chapter_name="01-chapter.md",
        paragraph_index_in_chapter=0,
    )
    assert paragraph is not None, "Should be able to read first paragraph"
    assert isinstance(paragraph, ParagraphDetail), "Expected ParagraphDetail"
    assert paragraph.paragraph_index_in_chapter == 0, "Paragraph index should be 0"
    assert len(paragraph.content) > 10, "Paragraph should have meaningful content"


def test_comprehensive_data_consistency(document_factory, test_docs_root: Path, validate_test_data):
    """Test data consistency across different operations and views."""
    # Create a document with 5 chapters, each with Lorem ipsum content and 10 paragraphs
    chapters = []
    for i in range(1, 6):
        content = f"# Chapter {i}\n\n"
        # Add 10 paragraphs per chapter
        for j in range(1, 11):
            content += f"Lorem ipsum dolor sit amet, consectetur adipiscing elit paragraph {j}.\n\n"
        chapters.append((f"{i:02d}-chapter.md", content))
    
    doc_name = document_factory(
        doc_type="simple",
        name="consistency_test_document",
        chapters=chapters
    )
    
    # Validate the document was created correctly
    validate_test_data.document_exists(test_docs_root, doc_name)

    # Get document info from different sources
    docs = list_documents()
    test_doc = next(doc for doc in docs if doc.document_name == doc_name)

    chapters = list_chapters(document_name=doc_name)
    doc_stats = get_document_statistics(document_name=doc_name)
    full_doc = read_full_document(document_name=doc_name)

    # Check chapter count consistency
    assert test_doc.total_chapters == len(
        chapters
    ), "Document info chapter count doesn't match chapters list"
    assert (
        test_doc.total_chapters == doc_stats.chapter_count
    ), "Document info doesn't match statistics"
    assert (
        len(chapters) == doc_stats.chapter_count
    ), "Chapters list doesn't match statistics"
    assert (
        len(full_doc.chapters) == doc_stats.chapter_count
    ), "Full document chapters don't match statistics"

    # Check word count consistency (allow some variance due to different counting methods)
    total_words_from_chapters = sum(chapter.word_count for chapter in chapters)
    assert (
        abs(total_words_from_chapters - doc_stats.word_count) <= 10
    ), f"Chapter word counts ({total_words_from_chapters}) don't reasonably match document stats ({doc_stats.word_count})"

    assert (
        abs(full_doc.total_word_count - doc_stats.word_count) <= 10
    ), f"Full document word count ({full_doc.total_word_count}) doesn't reasonably match statistics ({doc_stats.word_count})"

    # Check paragraph count consistency
    total_paragraphs_from_chapters = sum(
        chapter.paragraph_count for chapter in chapters
    )
    assert (
        total_paragraphs_from_chapters == doc_stats.paragraph_count
    ), f"Chapter paragraph counts ({total_paragraphs_from_chapters}) don't match document stats ({doc_stats.paragraph_count})"

    assert (
        full_doc.total_paragraph_count == doc_stats.paragraph_count
    ), f"Full document paragraph count ({full_doc.total_paragraph_count}) doesn't match statistics ({doc_stats.paragraph_count})"


def test_comprehensive_error_handling(document_factory, test_docs_root: Path):
    """Test error handling across different operations."""
    nonexistent_doc = "nonexistent_document_xyz"
    nonexistent_chapter = "nonexistent_chapter.md"

    # Test operations on nonexistent documents
    assert (
        list_chapters(document_name=nonexistent_doc) is None
    ), "Should return None for nonexistent document"
    assert (
        get_document_statistics(document_name=nonexistent_doc) is None
    ), "Should return None for nonexistent document"
    assert (
        read_full_document(document_name=nonexistent_doc) is None
    ), "Should return None for nonexistent document"

    # Test operations on nonexistent chapters
    # Create a valid document using document_factory
    test_doc_name = document_factory(doc_type="simple", name="test_doc_for_errors", chapter_count=0)

    assert (
        read_chapter_content(
            document_name=test_doc_name, chapter_name=nonexistent_chapter
        )
        is None
    ), "Should return None for nonexistent chapter"
    assert (
        get_chapter_statistics(
            document_name=test_doc_name, chapter_name=nonexistent_chapter
        )
        is None
    ), "Should return None for nonexistent chapter"

    # Test search in nonexistent document/chapter
    empty_results = find_text_in_document(
        document_name=nonexistent_doc, query="anything"
    )
    assert isinstance(
        empty_results, list
    ), "Search in nonexistent document should return a list"
    assert (
        len(empty_results) == 0
    ), "Search in nonexistent document should return empty list, not None or other value"

    empty_chapter_results = find_text_in_chapter(
        document_name=test_doc_name,
        chapter_name=nonexistent_chapter,
        query="anything",
    )
    assert isinstance(
        empty_chapter_results, list
    ), "Search in nonexistent chapter should return a list"
    assert (
        len(empty_chapter_results) == 0
    ), "Search in nonexistent chapter should return empty list, not None or other value"


# --- Test Cases ---


# Test Document Management Tools
def test_create_document_success(test_docs_root: Path):
    """Creates a document successfully."""
    doc_name = "my_test_document"
    status = create_document(document_name=doc_name)
    _assert_operation_success(status, "created successfully")
    assert (test_docs_root / doc_name).is_dir()
    assert status.details["document_name"] == doc_name


def test_create_document_duplicate(test_docs_root: Path):
    doc_name = "my_duplicate_doc"
    create_document(document_name=doc_name)
    status = create_document(document_name=doc_name)
    _assert_operation_failure(status, "already exists")


def test_list_documents_empty(test_docs_root: Path):
    docs_list = list_documents()
    assert isinstance(docs_list, list)
    assert len(docs_list) == 0


def test_list_documents_with_one_doc(document_factory, test_docs_root: Path, validate_test_data):
    """Lists a single created document."""
    doc_name = document_factory(
        doc_type="simple",
        name="listed_document",
        chapter_count=1,
        chapters=[("01-intro.md", "# Hello")]
    )
    
    # Validate the document was created correctly
    validate_test_data.document_exists(test_docs_root, doc_name)

    docs_list = list_documents()
    assert len(docs_list) == 1
    doc_info = docs_list[0]
    assert isinstance(doc_info, DocumentInfo)
    assert doc_info.document_name == doc_name
    assert doc_info.total_chapters == 1  # Because we added one chapter
    assert doc_info.has_summary is False # Default from factory


def test_list_documents_with_summary_file(document_factory, test_docs_root: Path, validate_test_data):
    """Lists a document with an actual summary file."""
    doc_name = document_factory(
        doc_type="simple",
        name="doc_with_actual_summary",
        chapter_count=1,
        chapters=[("01-intro.md", "# Hello")]
    )
    # Manually create a _SUMMARY.md file
    summary_content = "This is a test summary."
    summary_file_path = test_docs_root / doc_name / DOCUMENT_SUMMARY_FILE
    summary_file_path.write_text(summary_content, encoding="utf-8")

    validate_test_data.document_exists(test_docs_root, doc_name)
    assert summary_file_path.exists()

    docs_list = list_documents()
    assert len(docs_list) >= 1

    doc_info = next((d for d in docs_list if d.document_name == doc_name), None)
    assert doc_info is not None, f"Document {doc_name} not found in list_documents output."

    assert isinstance(doc_info, DocumentInfo)
    assert doc_info.document_name == doc_name
    assert doc_info.total_chapters == 1
    assert doc_info.has_summary is True


def test_delete_document_success(document_factory, test_docs_root: Path):
    """Deletes a document successfully."""
    doc_name = document_factory(
        doc_type="simple",
        name="to_be_deleted_doc",
        chapter_count=1,
        chapters=[("file.md", "content")]
    )

    status = delete_document(document_name=doc_name)
    _assert_operation_success(status, "deleted successfully")
    assert not (test_docs_root / doc_name).exists()


def test_delete_document_non_existent(test_docs_root: Path):
    """Fails to delete a non-existent document."""
    status = delete_document(document_name="non_existent_doc")
    _assert_operation_failure(status, "not found")


# Test Chapter Management Tools
def test_create_chapter_success(document_factory, test_docs_root: Path):
    """Creates a chapter successfully."""
    doc_name = document_factory(doc_type="simple", name="doc_for_chapters", chapter_count=0)
    chapter_name = "01-my_chapter.md"
    initial_content = "# Chapter Title"

    status = create_chapter(
        document_name=doc_name,
        chapter_name=chapter_name,
        initial_content=initial_content,
    )
    _assert_operation_success(status, "created successfully")
    chapter_path = test_docs_root / doc_name / chapter_name
    assert chapter_path.is_file()
    assert chapter_path.read_text() == initial_content
    assert status.details["document_name"] == doc_name
    assert status.details["chapter_name"] == chapter_name


def test_create_chapter_invalid_name(document_factory, test_docs_root: Path):
    doc_name = document_factory(doc_type="simple", name="doc_invalid_chapter_name", chapter_count=0)
    
    status = create_chapter(
        document_name=doc_name, chapter_name="chapter_no_md", initial_content=""
    )
    _assert_operation_failure(status, "must end with .md")

    status_manifest = create_chapter(
        document_name=doc_name, chapter_name="_manifest.json", initial_content=""
    )
    _assert_operation_failure(status_manifest, "cannot be reserved name")


def test_create_chapter_in_non_existent_document(test_docs_root: Path):
    """Fails to create a chapter in a non-existent document."""
    status = create_chapter(
        document_name="non_existent_doc_for_chapter", chapter_name="01-chap.md"
    )
    _assert_operation_failure(status, "not found")


def test_create_chapter_duplicate(document_factory, test_docs_root: Path):
    doc_name = document_factory(doc_type="simple", name="doc_for_duplicate_chapter", chapter_count=0)
    chapter_name = "01-dupe.md"
    
    create_chapter(document_name=doc_name, chapter_name=chapter_name)  # First one
    status = create_chapter(
        document_name=doc_name, chapter_name=chapter_name
    )  # Duplicate
    _assert_operation_failure(status, "already exists")


def test_list_chapters_empty(document_factory, test_docs_root: Path):
    """Returns an empty list for a document with no chapters."""
    doc_name = document_factory(doc_type="simple", name="doc_empty_chapters", chapter_count=0)
    
    chapters_list = list_chapters(document_name=doc_name)
    assert isinstance(
        chapters_list, list
    ), "list_chapters should return a list for existing document"
    assert (
        len(chapters_list) == 0
    ), "Newly created document should have zero chapters, not None or other value"


def test_list_chapters_non_existent_doc(test_docs_root: Path):
    """Returns None for a non-existent document."""
    chapters_list = list_chapters(document_name="non_existent_doc_for_list_chapters")
    assert (
        chapters_list is None
    ), "list_chapters should return None specifically for non-existent documents, not empty list or other value"


def test_list_chapters_with_multiple_chapters(document_factory, test_docs_root: Path, validate_test_data):
    """Lists multiple chapters in a document."""
    chapters = [
        ("00-zeroth.md", "Content 0"),  # To test ordering
        ("01-first.md", "Content 1"),
        ("02-second.md", "Content 2"),
        ("notes.txt", "ignore this")  # Non-md file, should be ignored
    ]
    
    doc_name = document_factory(
        doc_type="simple",
        name="doc_with_chapters",
        chapters=chapters
    )
    
    # Validate the document was created correctly
    validate_test_data.document_exists(test_docs_root, doc_name)

    chapters_list = list_chapters(document_name=doc_name)
    assert len(chapters_list) == 3  # Only .md files should be counted
    assert isinstance(chapters_list[0], ChapterMetadata)
    assert chapters_list[0].chapter_name == "00-zeroth.md"  # Should be sorted
    assert chapters_list[1].chapter_name == "01-first.md"
    assert chapters_list[2].chapter_name == "02-second.md"

    # Check some metadata
    assert chapters_list[0].word_count == 2  # "Content 0"
    assert chapters_list[0].paragraph_count == 1


def test_delete_chapter_success(document_factory, test_docs_root: Path):
    """Deletes a chapter successfully."""
    doc_name = document_factory(doc_type="simple", name="doc_for_deleting_chapter", chapter_count=0)
    chapter_name = "ch_to_delete.md"
    
    create_chapter(document_name=doc_name, chapter_name=chapter_name)
    assert (test_docs_root / doc_name / chapter_name).exists()

    status = delete_chapter(document_name=doc_name, chapter_name=chapter_name)
    _assert_operation_success(status, "deleted successfully")
    assert not (test_docs_root / doc_name / chapter_name).exists()


def test_delete_chapter_non_existent(document_factory, test_docs_root: Path):
    """Fails to delete a non-existent chapter."""
    doc_name = document_factory(doc_type="simple", name="doc_for_deleting_non_existent_chapter", chapter_count=0)
    
    status = delete_chapter(document_name=doc_name, chapter_name="ghost_chapter.md")
    _assert_operation_failure(status, "not found")


def test_delete_chapter_invalid_name(document_factory, test_docs_root: Path):
    doc_name = document_factory(doc_type="simple", name="doc_delete_invalid_chapter", chapter_count=0)
    
    status = delete_chapter(document_name=doc_name, chapter_name="not_a_md_file.txt")
    _assert_operation_failure(status, "not a valid chapter")


# --- Test Read/Write Content Tools ---


def test_read_chapter_content_success(document_factory, test_docs_root: Path, validate_test_data):
    """Reads chapter content successfully."""
    content = "# Title\nHello World\n\nThis is a paragraph."
    chapters = [("readable_chapter.md", content)]
    
    doc_name = document_factory(
        doc_type="simple",
        name="doc_read_content",
        chapters=chapters
    )
    
    # Validate the document was created correctly
    validate_test_data.document_exists(test_docs_root, doc_name)

    chapter_obj = read_chapter_content(
        document_name=doc_name, chapter_name="readable_chapter.md"
    )
    assert (
        chapter_obj is not None
    ), f"Should successfully read existing chapter readable_chapter.md"
    assert isinstance(
        chapter_obj, ChapterContent
    ), f"Expected ChapterContent object, got {type(chapter_obj)}"
    assert chapter_obj.document_name == doc_name
    assert chapter_obj.chapter_name == "readable_chapter.md"
    assert chapter_obj.content == content
    assert (
        chapter_obj.word_count == 8
    )  # Adjusted: "# Title Hello World This is a paragraph." (was 7)
    assert (
        chapter_obj.paragraph_count == 2
    )  # "# Title\nHello World" and "This is a paragraph."


def test_read_chapter_content_non_existent_chapter(document_factory, test_docs_root: Path):
    """Fails to read a non-existent chapter."""
    doc_name = document_factory(doc_type="simple", name="doc_read_non_existent_chap", chapter_count=0)
    
    chapter_obj = read_chapter_content(
        document_name=doc_name, chapter_name="no_such_chapter.md"
    )
    assert (
        chapter_obj is None
    ), "Reading non-existent chapter should return None specifically, not empty object or other value"


def test_write_chapter_content_overwrite(document_factory, test_docs_root: Path, validate_test_data):
    """Overwrites chapter content successfully."""
    initial_content = "Old content."
    new_content = "# New Content\nThis is fresh."
    chapters = [("writable_chapter.md", initial_content)]
    
    doc_name = document_factory(
        doc_type="simple",
        name="doc_write_content",
        chapters=chapters
    )
    
    # Validate the document was created correctly
    validate_test_data.document_exists(test_docs_root, doc_name)

    status = write_chapter_content(
        document_name=doc_name, chapter_name="writable_chapter.md", new_content=new_content
    )
    _assert_operation_success(status, "updated successfully")
    assert (test_docs_root / doc_name / "writable_chapter.md").read_text() == new_content


def test_write_chapter_content_create_new(document_factory, test_docs_root: Path):
    """Creates a new chapter with content."""
    doc_name = document_factory(doc_type="simple", name="doc_write_new_chap", chapter_count=0)
    chapter_name = "newly_written_chapter.md"
    new_content = "Content for a new chapter."

    status = write_chapter_content(
        document_name=doc_name, chapter_name=chapter_name, new_content=new_content
    )
    _assert_operation_success(
        status, "updated successfully"
    )  # Message might be generic
    assert (test_docs_root / doc_name / chapter_name).read_text() == new_content


def test_read_paragraph_content_success(document_factory, test_docs_root: Path, validate_test_data):
    """Reads paragraph content successfully."""
    paras = ["Paragraph 0.", "Paragraph 1.", "Paragraph 2."]
    content = "\n\n".join(paras)
    chapters = [("chapter_with_paras.md", content)]
    
    doc_name = document_factory(
        doc_type="simple",
        name="doc_read_para",
        chapters=chapters
    )
    
    # Validate the document was created correctly
    validate_test_data.document_exists(test_docs_root, doc_name)

    para_obj = read_paragraph_content(
        document_name=doc_name, chapter_name="chapter_with_paras.md", paragraph_index_in_chapter=1
    )
    assert (
        para_obj is not None
    ), "Should successfully read existing paragraph at valid index"
    assert isinstance(
        para_obj, ParagraphDetail
    ), f"Expected ParagraphDetail object, got {type(para_obj)}"
    assert para_obj.content == "Paragraph 1."
    assert para_obj.paragraph_index_in_chapter == 1
    assert para_obj.word_count == 2


def test_read_paragraph_content_out_of_bounds(document_factory, test_docs_root: Path, validate_test_data):
    """Reads paragraph content successfully."""
    content = "Para1\n\nPara2"
    chapters = [("chapter_few_paras.md", content)]
    
    doc_name = document_factory(
        doc_type="simple",
        name="doc_read_para_oob",
        chapters=chapters
    )
    
    # Validate the document was created correctly
    validate_test_data.document_exists(test_docs_root, doc_name)

    para_obj = read_paragraph_content(
        document_name=doc_name, chapter_name="chapter_few_paras.md", paragraph_index_in_chapter=5
    )
    assert (
        para_obj is None
    ), "Reading paragraph at out-of-bounds index should return None specifically, not empty object or error"


def test_append_paragraph_to_chapter_success(document_factory, test_docs_root: Path, validate_test_data):
    """Appends a paragraph to a chapter successfully."""
    initial_content = "First line.\n\nSecond line."
    appended_para = "Third line, appended."
    chapters = [("chap_append.md", initial_content)]
    
    doc_name = document_factory(
        doc_type="simple",
        name="doc_append_para",
        chapters=chapters
    )
    
    # Validate the document was created correctly
    validate_test_data.document_exists(test_docs_root, doc_name)

    status = append_paragraph_to_chapter(doc_name, "chap_append.md", appended_para)
    _assert_operation_success(status)
    expected_content = initial_content + "\n\n" + appended_para
    assert (test_docs_root / doc_name / "chap_append.md").read_text() == expected_content


def test_append_paragraph_to_empty_chapter(document_factory, test_docs_root: Path, validate_test_data):
    appended_para = "Only line."
    chapters = [("chap_append_empty.md", "")]  # Empty chapter
    
    doc_name = document_factory(
        doc_type="simple",
        name="doc_append_para_empty",
        chapters=chapters
    )
    
    # Validate the document was created correctly
    validate_test_data.document_exists(test_docs_root, doc_name)

    status = append_paragraph_to_chapter(doc_name, "chap_append_empty.md", appended_para)
    _assert_operation_success(status)
    assert (test_docs_root / doc_name / "chap_append_empty.md").read_text() == appended_para


def test_replace_text_in_chapter_success(document_factory, test_docs_root: Path, validate_test_data):
    """Replaces text in a chapter successfully."""
    content = "Old text is old. Another old occurrence."
    chapters = [("chap_replace.md", content)]
    
    doc_name = document_factory(
        doc_type="simple",
        name="doc_replace_text_chap",
        chapters=chapters
    )
    
    # Validate the document was created correctly
    validate_test_data.document_exists(test_docs_root, doc_name)

    status = replace_text_in_chapter(doc_name, "chap_replace.md", "old", "new")
    _assert_operation_success(status, "replaced")
    expected_content = "Old text is new. Another new occurrence."
    assert (test_docs_root / doc_name / "chap_replace.md").read_text() == expected_content


def test_replace_text_in_chapter_no_occurrence(document_factory, test_docs_root: Path, validate_test_data):
    """Fails to replace text in a chapter when no occurrence is found."""
    content = "Some text without the target."
    chapters = [("chap_replace_no_op.md", content)]
    
    doc_name = document_factory(
        doc_type="simple",
        name="doc_replace_text_chap_no_op",
        chapters=chapters
    )
    
    # Validate the document was created correctly
    validate_test_data.document_exists(test_docs_root, doc_name)

    status = replace_text_in_chapter(
        doc_name, "chap_replace_no_op.md", "missing_text", "replacement"
    )
    _assert_operation_success(status, "not found in chapter")  # Success, but no change
    assert (
        test_docs_root / doc_name / "chap_replace_no_op.md"
    ).read_text() == content  # Should be unchanged


# Test reading full document
def test_read_full_document_success(document_factory, test_docs_root: Path, validate_test_data):
    """Reads the full document successfully."""
    ch1_content = "# Chapter 1\nContent of chapter one."
    ch2_content = "## Chapter 2\nSome more text here."
    chapters = [
        ("01_ch1.md", ch1_content),
        ("02_ch2.md", ch2_content)
    ]
    
    doc_name = document_factory(
        doc_type="simple",
        name="full_doc_read",
        chapters=chapters
    )
    
    # Validate the document was created correctly
    validate_test_data.document_exists(test_docs_root, doc_name)

    full_doc_obj = read_full_document(document_name=doc_name)
    assert full_doc_obj is not None
    assert isinstance(full_doc_obj, FullDocumentContent)
    assert full_doc_obj.document_name == doc_name
    assert len(full_doc_obj.chapters) == 2
    assert full_doc_obj.chapters[0].content == ch1_content
    assert full_doc_obj.chapters[1].content == ch2_content
    assert full_doc_obj.total_word_count == 14  # Corrected: ch1 (7) + ch2 (7)
    assert (
        full_doc_obj.total_paragraph_count == 2
    )  # "# Chapter 1\nContent of chapter one." is 1 para. "## Chapter 2\nSome more text here." is 1 para.

def test_read_full_document_ignores_summary(document_factory, test_docs_root: Path, validate_test_data):
    """Reads the full document, ignoring the summary file."""
    ch1_content = "# Chapter 1\nContent of chapter one."
    chapters = [("01_ch1.md", ch1_content)]
    doc_name = document_factory(
        doc_type="simple",
        name="doc_ignores_summary",
        chapters=chapters
    )
    # Manually create a _SUMMARY.md file
    summary_content = "This summary should be ignored by read_full_document."
    summary_file_path = test_docs_root / doc_name / DOCUMENT_SUMMARY_FILE
    summary_file_path.write_text(summary_content, encoding="utf-8")

    validate_test_data.document_exists(test_docs_root, doc_name)
    assert summary_file_path.exists()

    full_doc_obj = read_full_document(document_name=doc_name)
    assert full_doc_obj is not None
    assert len(full_doc_obj.chapters) == 1 # Only the actual chapter, not the summary
    assert full_doc_obj.chapters[0].content == ch1_content
    assert summary_content not in full_doc_obj.chapters[0].content # Double check

    # Verify that total counts also ignore the summary file
    stats = get_document_statistics(document_name=doc_name)
    assert stats is not None
    assert stats.chapter_count == 1 # Only counts actual chapters


def test_read_full_document_empty_doc(document_factory, test_docs_root: Path):
    """Reads an empty document successfully."""
    doc_name = document_factory(doc_type="simple", name="empty_doc_for_full_read", chapter_count=0)
    
    full_doc_obj = read_full_document(document_name=doc_name)
    assert full_doc_obj is not None
    assert len(full_doc_obj.chapters) == 0
    assert full_doc_obj.total_word_count == 0


def test_read_full_document_non_existent(test_docs_root: Path):
    """Fails to read a non-existent document."""
    full_doc_obj = read_full_document(document_name="no_doc_here_for_full_read")
    assert full_doc_obj is None


# Test replacing text across document
def test_replace_text_in_document_success(document_factory, test_docs_root: Path, validate_test_data):
    """Replaces text across the document successfully."""
    ch1_content = "Global old term, chapter 1. Another old one."
    ch2_content = "Chapter 2, no target. But old is here!"
    ch3_content = "Only fresh new terms."
    chapters = [
        ("01_ch1.md", ch1_content),
        ("02_ch2.md", ch2_content),
        ("03_ch3.md", ch3_content)
    ]
    
    doc_name = document_factory(
        doc_type="simple",
        name="doc_replace_global",
        chapters=chapters
    )
    
    # Validate the document was created correctly
    validate_test_data.document_exists(test_docs_root, doc_name)

    status = replace_text_in_document(doc_name, "old", "new")
    _assert_operation_success(status, "replacement completed")
    assert status.details["chapters_modified_count"] == 2
    assert status.details["total_occurrences_replaced"] == 3  # 2 in ch1, 1 in ch2

    assert (
        test_docs_root / doc_name / "01_ch1.md"
    ).read_text() == "Global new term, chapter 1. Another new one."
    assert (
        test_docs_root / doc_name / "02_ch2.md"
    ).read_text() == "Chapter 2, no target. But new is here!"
    assert (
        test_docs_root / doc_name / "03_ch3.md"
    ).read_text() == ch3_content  # Unchanged


# --- Test Analyze and Retrieval Tools ---


def test_get_chapter_statistics_success(document_factory, test_docs_root: Path, validate_test_data):
    """Gets chapter statistics successfully."""
    content = "# Stats Test\nThis chapter has five words.\n\nAnd two paragraphs total."
    chapters = [("chap_for_stats.md", content)]
    
    doc_name = document_factory(
        doc_type="simple",
        name="doc_stats_chap",
        chapters=chapters
    )
    
    # Validate the document was created correctly
    validate_test_data.document_exists(test_docs_root, doc_name)

    stats = get_chapter_statistics(document_name=doc_name, chapter_name="chap_for_stats.md")
    assert stats is not None
    assert isinstance(stats, StatisticsReport)
    assert stats.scope == f"chapter: {doc_name}/chap_for_stats.md"
    # Content: "# Stats Test\nThis chapter has five words.\n\nAnd two paragraphs total."
    # "#", "Stats", "Test", "This", "chapter", "has", "five", "words", "And", "two", "paragraphs", "total" = 12 words
    assert stats.word_count == 12  # Adjusted (was 11)
    assert (
        stats.paragraph_count == 2
    )  # Correct: "# Stats Test\nThis chapter has five words." AND "And two paragraphs total."


def test_get_chapter_statistics_non_existent(document_factory, test_docs_root: Path):
    """Fails to get statistics for a non-existent chapter."""
    doc_name = document_factory(doc_type="simple", name="doc_stats_chap_ne", chapter_count=0)
    
    stats = get_chapter_statistics(document_name=doc_name, chapter_name="no_chap.md")
    assert stats is None


def test_get_document_statistics_success(document_factory, test_docs_root: Path, validate_test_data):
    """Gets document statistics successfully."""
    chapters = [
        ("01.md", "# Chapter 1\n\nFirst chapter content here."),
        ("02.md", "# Chapter 2\n\nSecond chapter content here.")
    ]
    
    doc_name = document_factory(
        doc_type="simple",
        name="doc_stats_success",
        chapters=chapters
    )
    
    # Validate the document was created correctly
    validate_test_data.document_exists(test_docs_root, doc_name)

    stats = get_document_statistics(document_name=doc_name)
    assert stats is not None
    assert isinstance(stats, StatisticsReport)
    assert stats.scope == f"document: {doc_name}"
    assert stats.chapter_count == 2
    assert stats.word_count > 10  # Should have reasonable content
    assert stats.paragraph_count == 4  # 2 chapters, each with title + content paragraph


def test_get_document_statistics_empty_doc(document_factory, test_docs_root: Path):
    """Gets statistics for an empty document."""
    doc_name = document_factory(doc_type="simple", name="empty_doc_stats", chapter_count=0)
    
    stats = get_document_statistics(document_name=doc_name)
    assert stats is not None
    assert stats.chapter_count == 0
    assert stats.word_count == 0
    assert stats.paragraph_count == 0


def test_get_document_statistics_non_existent_doc(test_docs_root: Path):
    """Fails to get statistics for a non-existent document."""
    stats = get_document_statistics(document_name="non_existent_doc_stats")
    assert stats is None


def test_find_text_in_chapter_success_case_insensitive(document_factory, test_docs_root: Path, validate_test_data):
    """Finds text in a chapter successfully (case-insensitive)."""
    content = "Hello World. This is a test chapter.\n\nAnother paragraph with HELLO again."
    chapters = [("search_chapter.md", content)]
    
    doc_name = document_factory(
        doc_type="simple",
        name="doc_search_chapter_case_insensitive",
        chapters=chapters
    )
    
    # Validate the document was created correctly
    validate_test_data.document_exists(test_docs_root, doc_name)

    # Case insensitive search (default)
    results = find_text_in_chapter(
        document_name=doc_name, chapter_name="search_chapter.md", query="hello"
    )
    assert len(results) == 2  # Should find both "Hello" and "HELLO"
    assert all("hello" in match.content.lower() for match in results)

    # Verify match details
    for match in results:
        assert match.chapter_name == "search_chapter.md"
        assert match.paragraph_index_in_chapter >= 0


def test_find_text_in_chapter_success_case_sensitive(document_factory, test_docs_root: Path, validate_test_data):
    """Finds text in a chapter successfully (case-sensitive)."""
    content = "Hello World. This is a test chapter.\n\nAnother paragraph with HELLO again."
    chapters = [("search_chapter_case.md", content)]
    
    doc_name = document_factory(
        doc_type="simple",
        name="doc_search_chapter_case_sensitive",
        chapters=chapters
    )
    
    # Validate the document was created correctly
    validate_test_data.document_exists(test_docs_root, doc_name)

    # Case sensitive search
    results = find_text_in_chapter(
        document_name=doc_name,
        chapter_name="search_chapter_case.md",
        query="Hello",
        case_sensitive=True,
    )
    assert len(results) == 1  # Should find only "Hello", not "HELLO"
    assert "Hello" in results[0].content
    assert results[0].chapter_name == "search_chapter_case.md"


def test_find_text_in_chapter_no_match(document_factory, test_docs_root: Path, validate_test_data):
    """Fails to find text in a chapter when no match is found."""
    content = "This chapter has no matching terms."
    chapters = [("no_match_chapter.md", content)]
    
    doc_name = document_factory(
        doc_type="simple",
        name="doc_search_no_match",
        chapters=chapters
    )
    
    # Validate the document was created correctly
    validate_test_data.document_exists(test_docs_root, doc_name)

    results = find_text_in_chapter(
        document_name=doc_name, chapter_name="no_match_chapter.md", query="nonexistent"
    )
    assert len(results) == 0  # Should find no matches


def test_find_text_in_document_success(document_factory, test_docs_root: Path, validate_test_data):
    """Finds text in a document successfully."""
    ch1_content = "First chapter with searchable content."
    ch2_content = "Second chapter also has searchable text."
    ch3_content = "Third chapter without the target word."
    chapters = [
        ("01_search.md", ch1_content),
        ("02_search.md", ch2_content),
        ("03_search.md", ch3_content)
    ]
    
    doc_name = document_factory(
        doc_type="simple",
        name="doc_search_multi_chapter",
        chapters=chapters
    )
    
    # Validate the document was created correctly
    validate_test_data.document_exists(test_docs_root, doc_name)

    # Search for term that appears in multiple chapters
    results = find_text_in_document(document_name=doc_name, query="searchable")
    assert len(results) == 2  # Should find in first two chapters

    # Verify results
    chapter_names = [match.chapter_name for match in results]
    assert "01_search.md" in chapter_names
    assert "02_search.md" in chapter_names
    assert "03_search.md" not in chapter_names

    for match in results:
        assert "searchable" in match.content.lower()
        assert match.paragraph_index_in_chapter >= 0

    # Search for term in specific chapter
    chapter_results = find_text_in_document(document_name=doc_name, query="First")
    assert len(chapter_results) == 1
    assert chapter_results[0].chapter_name == "01_search.md"


def test_find_text_in_document_no_match(document_factory, test_docs_root: Path, validate_test_data):
    """Fails to find text in a document when no match is found."""
    chapters = [
        ("01_no_match.md", "This chapter has some content."),
        ("02_no_match.md", "This chapter has different content.")
    ]
    
    doc_name = document_factory(
        doc_type="simple",
        name="doc_search_no_match_multi",
        chapters=chapters
    )
    
    # Validate the document was created correctly
    validate_test_data.document_exists(test_docs_root, doc_name)

    results = find_text_in_document(document_name=doc_name, query="nonexistent")
    assert len(results) == 0  # Should find no matches


# --- Test Document Summary Tool ---

def test_read_document_summary_success(document_factory, test_docs_root: Path, validate_test_data):
    """Reads the document summary successfully."""
    doc_name = document_factory(doc_type="simple", name="doc_for_summary_read", chapter_count=0)
    summary_content = "This is the official summary of the document."
    summary_file = test_docs_root / doc_name / DOCUMENT_SUMMARY_FILE
    summary_file.write_text(summary_content, encoding="utf-8")

    validate_test_data.document_exists(test_docs_root, doc_name) # Checks doc dir
    assert summary_file.exists()

    result = read_document_summary(document_name=doc_name)
    assert result == summary_content

def test_read_document_summary_no_summary_file(document_factory, test_docs_root: Path, validate_test_data):
    """Fails to read the document summary when no summary file exists."""
    doc_name = document_factory(doc_type="simple", name="doc_no_summary_file", chapter_count=1)

    validate_test_data.document_exists(test_docs_root, doc_name)
    summary_file = test_docs_root / doc_name / DOCUMENT_SUMMARY_FILE
    assert not summary_file.exists() # Ensure it really doesn't exist

    result = read_document_summary(document_name=doc_name)
    assert result is None

def test_read_document_summary_non_existent_document(test_docs_root: Path):
    """Fails to read the document summary for a non-existent document."""
    result = read_document_summary(document_name="non_existent_doc_for_summary")
    assert result is None

def test_read_document_summary_empty_summary_file(document_factory, test_docs_root: Path, validate_test_data):
    doc_name = document_factory(doc_type="simple", name="doc_empty_summary", chapter_count=0)
    summary_file = test_docs_root / doc_name / DOCUMENT_SUMMARY_FILE
    summary_file.write_text("", encoding="utf-8") # Empty summary

    validate_test_data.document_exists(test_docs_root, doc_name)
    assert summary_file.exists()

    result = read_document_summary(document_name=doc_name)
    assert result == ""


# Cleanup function to ensure all test artifacts are removed
def pytest_sessionfinish(session, exitstatus):
    """Clean up any remaining test artifacts after all tests complete."""
    try:
        # Ensure doc_tool_server path is restored to default
        if hasattr(doc_tool_server, "DOCS_ROOT_PATH"):
            default_path = Path.cwd() / ".documents_storage" # Default from server
            # Check if original_server_path was captured, if so restore it
            # This part might need more robust handling if tests modify DOCS_ROOT_PATH in complex ways
            # For now, assume it should revert to the standard default if not otherwise managed by fixtures.
            # The `test_docs_root` fixture in conftest.py should handle restoring its specific changes.
            current_server_path_obj = getattr(doc_tool_server, "DOCS_ROOT_PATH")
            if str(current_server_path_obj) != str(default_path) and not str(current_server_path_obj).startswith(tempfile.gettempdir()):
                         doc_tool_server.DOCS_ROOT_PATH = default_path


    except Exception:
        pass  # Ignore cleanup errors


# Will be added incrementally (This comment can be removed if all tests are done for this file)
