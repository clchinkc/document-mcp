"""
Test data generation utilities for Document MCP testing.

This module provides consistent test data creation patterns
used across all test types to ensure reproducible and
well-structured test scenarios.
"""

import shutil
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

# Common test content constants
SAMPLE_CHAPTER_CONTENT = "# Test Chapter\n\nThis is a test chapter with some content."
SAMPLE_SEARCH_TEXT = "Lorem ipsum dolor sit amet"

# Default test document structure
DEFAULT_DOC_CHAPTERS = [
    ("01-introduction.md", "# Introduction\n\nWelcome to the test document."),
    ("02-methods.md", "# Methods\n\nThis chapter describes the methodology."),
    ("03-results.md", "# Results\n\nHere are the findings and results."),
]


class TestDataType(Enum):
    """Enumeration of test data types for consistent categorization."""

    SIMPLE = "simple"
    LARGE = "large"
    SEARCHABLE = "searchable"
    MULTI_FORMAT = "multi_format"
    ERROR_SCENARIO = "error_scenario"
    PERFORMANCE = "performance"
    STATISTICAL = "statistical"


@dataclass
class TestDocumentSpec:
    """Specification for creating test documents with validation."""

    name: Optional[str] = None
    doc_type: TestDataType = TestDataType.SIMPLE
    chapter_count: int = 3
    chapters: Optional[List[Tuple[str, str]]] = None
    search_terms: Optional[List[str]] = None
    target_word_count: Optional[int] = None
    target_paragraph_count: Optional[int] = None
    paragraphs_per_chapter: int = 5
    cleanup_on_error: bool = True

    def __post_init__(self):
        """Validate test document specification."""
        if self.chapter_count < 0:
            raise ValueError("Chapter count must be non-negative")
        if self.target_word_count is not None and self.target_word_count < 0:
            raise ValueError("Target word count must be non-negative")
        if self.target_paragraph_count is not None and self.target_paragraph_count < 0:
            raise ValueError("Target paragraph count must be non-negative")


@dataclass
class TestDataRegistry:
    """Registry to track created test data for cleanup and validation."""

    created_documents: List[str] = field(default_factory=list)
    created_directories: List[Path] = field(default_factory=list)
    temp_files: List[Path] = field(default_factory=list)
    active_specs: List[TestDocumentSpec] = field(default_factory=list)

    def register_document(self, doc_name: str, spec: TestDocumentSpec) -> None:
        """Register a created document for tracking."""
        if doc_name not in self.created_documents:
            self.created_documents.append(doc_name)
            self.active_specs.append(spec)

    def register_directory(self, directory: Path) -> None:
        """Register a created directory for cleanup."""
        if directory not in self.created_directories:
            self.created_directories.append(directory)

    def register_temp_file(self, file_path: Path) -> None:
        """Register a temporary file for cleanup."""
        if file_path not in self.temp_files:
            self.temp_files.append(file_path)

    def cleanup_all(self, docs_root: Path) -> None:
        """Clean up all registered test data."""
        # Clean up documents
        cleanup_test_documents(docs_root, self.created_documents)

        # Clean up directories
        for directory in self.created_directories:
            if directory.exists():
                shutil.rmtree(directory, ignore_errors=True)

        # Clean up temp files
        for temp_file in self.temp_files:
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)

        # Clear registry
        self.created_documents.clear()
        self.created_directories.clear()
        self.temp_files.clear()
        self.active_specs.clear()

    def get_document_spec(self, doc_name: str) -> Optional[TestDocumentSpec]:
        """Get the specification for a registered document."""
        try:
            index = self.created_documents.index(doc_name)
            return self.active_specs[index]
        except (ValueError, IndexError):
            return None

    def validate_test_state(self, docs_root: Path) -> List[str]:
        """Validate that test data is in expected state."""
        issues = []

        for doc_name in self.created_documents:
            doc_path = docs_root / doc_name
            if not doc_path.exists():
                issues.append(f"Document {doc_name} is registered but doesn't exist")
            elif not doc_path.is_dir():
                issues.append(f"Document {doc_name} exists but is not a directory")

        return issues


# Global test data registry for tracking across test sessions
_global_test_registry = TestDataRegistry()


def get_test_registry() -> TestDataRegistry:
    """Get the global test data registry."""
    return _global_test_registry


def generate_unique_name(prefix: str = "test") -> str:
    """
    Generate a unique name for test resources.

    Args:
        prefix: Prefix for the generated name

    Returns:
        Unique name string
    """
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def create_test_document_from_spec(
    docs_root: Path,
    spec: TestDocumentSpec,
    registry: Optional[TestDataRegistry] = None,
) -> str:
    """
    Create a test document from a specification with full validation and tracking.

    Args:
        docs_root: Root directory for documents
        spec: Specification for the document to create
        registry: Optional registry to track created document

    Returns:
        The name of the created document

    Raises:
        ValueError: If specification is invalid
        FileExistsError: If document already exists
    """
    if registry is None:
        registry = get_test_registry()

    # Generate document name if not provided
    doc_name = spec.name or generate_unique_name(f"test_{spec.doc_type.value}")

    # Check if document already exists
    doc_path = docs_root / doc_name
    if doc_path.exists():
        if spec.cleanup_on_error:
            shutil.rmtree(doc_path, ignore_errors=True)
        else:
            raise FileExistsError(f"Document {doc_name} already exists")

    try:
        # Create document based on type
        if spec.doc_type == TestDataType.SIMPLE:
            actual_doc_name = create_test_document(
                docs_root=docs_root,
                doc_name=doc_name,
                chapter_count=spec.chapter_count,
                chapters=spec.chapters,
            )
        elif spec.doc_type == TestDataType.LARGE:
            actual_doc_name = create_large_test_document(
                docs_root=docs_root,
                doc_name=doc_name,
                chapter_count=spec.chapter_count,
                paragraphs_per_chapter=spec.paragraphs_per_chapter,
            )
        elif spec.doc_type == TestDataType.SEARCHABLE:
            actual_doc_name, _ = create_searchable_test_document(
                docs_root=docs_root,
                doc_name=doc_name,
                search_terms=spec.search_terms,
            )
        elif spec.doc_type == TestDataType.STATISTICAL:
            actual_doc_name, _, _ = create_test_document_with_statistics(
                docs_root=docs_root,
                doc_name=doc_name,
                target_word_count=spec.target_word_count or 500,
                target_paragraph_count=spec.target_paragraph_count or 20,
                chapter_count=spec.chapter_count,
            )
        elif spec.doc_type == TestDataType.MULTI_FORMAT:
            actual_doc_name = create_multi_format_test_document(
                docs_root=docs_root,
                doc_name=doc_name,
            )
        else:
            raise ValueError(f"Unsupported document type: {spec.doc_type}")

        # Register the created document
        registry.register_document(actual_doc_name, spec)

        return actual_doc_name

    except Exception:
        # Cleanup on error if requested
        if spec.cleanup_on_error and doc_path.exists():
            shutil.rmtree(doc_path, ignore_errors=True)
        raise


def create_test_document(
    docs_root: Path,
    doc_name: Optional[str] = None,
    chapter_count: int = 3,
    chapters: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """
    Create a test document with chapters for testing.

    Args:
        docs_root: Root directory for documents
        doc_name: Name of the document (auto-generated if None)
        chapter_count: Number of chapters to create (ignored if chapters provided)
        chapters: List of (chapter_name, content) tuples

    Returns:
        The name of the created document
    """
    if doc_name is None:
        doc_name = generate_unique_name("test_doc")

    doc_path = docs_root / doc_name
    doc_path.mkdir(parents=True, exist_ok=True)

    # Use provided chapters or generate default ones
    if chapters:
        chapter_data = chapters
    elif chapter_count <= len(DEFAULT_DOC_CHAPTERS):
        chapter_data = DEFAULT_DOC_CHAPTERS[:chapter_count]
    else:
        # Generate additional chapters if needed
        chapter_data = list(DEFAULT_DOC_CHAPTERS)
        for i in range(len(DEFAULT_DOC_CHAPTERS) + 1, chapter_count + 1):
            chapter_name = f"{i:02d}-chapter.md"
            content = f"# Chapter {i}\n\nThis is chapter {i} content."
            chapter_data.append((chapter_name, content))

    # Create chapter files
    for chapter_name, content in chapter_data:
        chapter_file = doc_path / chapter_name
        chapter_file.write_text(content)

    return doc_name


def create_large_test_document(
    docs_root: Path,
    doc_name: Optional[str] = None,
    chapter_count: int = 100,
    paragraphs_per_chapter: int = 50,
) -> str:
    """
    Create a large test document for performance testing.

    Args:
        docs_root: Root directory for documents
        doc_name: Name of the document (auto-generated if None)
        chapter_count: Number of chapters to create
        paragraphs_per_chapter: Number of paragraphs per chapter

    Returns:
        The name of the created document
    """
    if doc_name is None:
        doc_name = generate_unique_name("large_test_doc")

    doc_path = docs_root / doc_name
    doc_path.mkdir(parents=True, exist_ok=True)

    for i in range(1, chapter_count + 1):
        chapter_file = doc_path / f"{i:03d}-chapter.md"
        content = f"# Chapter {i}\n\n"

        for j in range(1, paragraphs_per_chapter + 1):
            content += (
                f"This is paragraph {j} of chapter {i}. "
                f"Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                f"Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
                f"Ut enim ad minim veniam, quis nostrud exercitation.\n\n"
            )

        chapter_file.write_text(content)

    return doc_name


def create_searchable_test_document(
    docs_root: Path,
    doc_name: Optional[str] = None,
    search_terms: Optional[List[str]] = None,
) -> Tuple[str, List[str]]:
    """
    Create a test document with specific searchable content.

    Args:
        docs_root: Root directory for documents
        doc_name: Name of the document (auto-generated if None)
        search_terms: List of terms to include in chapters

    Returns:
        Tuple of (document_name, search_terms_used)
    """
    if doc_name is None:
        doc_name = generate_unique_name("searchable_doc")

    if search_terms is None:
        search_terms = ["searchable", "findable", "discoverable", "locatable"]
    else:
        # Create a copy to avoid modifying the original list
        search_terms = search_terms.copy()

    # Ensure we have at least 4 terms for the template
    while len(search_terms) < 4:
        search_terms.extend(["placeholder", "example", "content", "text"])
    search_terms = search_terms[:4]  # Use only first 4 to avoid duplication

    doc_path = docs_root / doc_name
    doc_path.mkdir(parents=True, exist_ok=True)

    chapters = [
        (
            "01-intro.md",
            f"# Introduction\n\nThis chapter contains {search_terms[0]} content. "
            f"You can find {search_terms[1]} information here.",
        ),
        (
            "02-body.md",
            f"# Main Content\n\nMore {search_terms[2]} text here. "
            f"This section has {search_terms[3]} elements.",
        ),
        (
            "03-conclusion.md",
            f"# Conclusion\n\nSummary with {search_terms[0]} and {search_terms[2]} terms.",
        ),
    ]

    for chapter_name, content in chapters:
        chapter_file = doc_path / chapter_name
        chapter_file.write_text(content)

    return doc_name, search_terms


def create_test_document_with_statistics(
    docs_root: Path,
    doc_name: Optional[str] = None,
    target_word_count: int = 500,
    target_paragraph_count: int = 20,
    chapter_count: int = 3,
) -> Tuple[str, int, int]:
    """
    Create a test document with specific word and paragraph counts.

    Args:
        docs_root: Root directory for documents
        doc_name: Name of the document (auto-generated if None)
        target_word_count: Approximate target word count
        target_paragraph_count: Approximate target paragraph count
        chapter_count: Number of chapters to create

    Returns:
        Tuple of (document_name, actual_word_count, actual_paragraph_count)
    """
    if doc_name is None:
        doc_name = generate_unique_name("stats_doc")

    doc_path = docs_root / doc_name
    doc_path.mkdir(parents=True, exist_ok=True)

    # Calculate words and paragraphs per chapter
    chapters_count = chapter_count
    words_per_chapter = target_word_count // chapters_count
    paragraphs_per_chapter = target_paragraph_count // chapters_count
    words_per_paragraph = max(1, words_per_chapter // paragraphs_per_chapter)

    actual_word_count = 0
    actual_paragraph_count = 0

    for i in range(1, chapters_count + 1):
        chapter_file = doc_path / f"{i:02d}-chapter.md"
        content = f"# Chapter {i}\n\n"

        for j in range(paragraphs_per_chapter):
            # Generate paragraph with approximately the right word count
            paragraph = "Lorem ipsum " * (words_per_paragraph // 2)
            paragraph += f"Chapter {i} paragraph {j + 1}."
            content += paragraph + "\n\n"

            actual_word_count += len(paragraph.split())
            actual_paragraph_count += 1

        chapter_file.write_text(content)

    return doc_name, actual_word_count, actual_paragraph_count


def create_multi_format_test_document(
    docs_root: Path,
    doc_name: Optional[str] = None,
) -> str:
    """
    Create a test document with various markdown formatting.

    Args:
        docs_root: Root directory for documents
        doc_name: Name of the document (auto-generated if None)

    Returns:
        The name of the created document
    """
    if doc_name is None:
        doc_name = generate_unique_name("format_doc")

    doc_path = docs_root / doc_name
    doc_path.mkdir(parents=True, exist_ok=True)

    chapters = [
        (
            "01-headers.md",
            "# Main Header\n\n## Sub Header\n\n### Sub-sub Header\n\n"
            "Regular paragraph text with **bold** and *italic* formatting.",
        ),
        (
            "02-lists.md",
            "# Lists\n\n## Bullet List\n\n- Item 1\n- Item 2\n- Item 3\n\n"
            "## Numbered List\n\n1. First item\n2. Second item\n3. Third item",
        ),
        (
            "03-code.md",
            "# Code Examples\n\n```python\ndef hello():\n    print('Hello, World!')\n```\n\n"
            "Inline `code` example in paragraph.",
        ),
        (
            "04-mixed.md",
            "# Mixed Content\n\n> This is a blockquote.\n\n"
            "Regular paragraph with [link](http://example.com) and more text.\n\n"
            "| Column 1 | Column 2 |\n|----------|----------|\n| Data 1   | Data 2   |",
        ),
    ]

    for chapter_name, content in chapters:
        chapter_file = doc_path / chapter_name
        chapter_file.write_text(content)

    return doc_name


def create_error_test_scenarios(docs_root: Path) -> List[Tuple[str, str]]:
    """
    Create test scenarios for error handling.

    Args:
        docs_root: Root directory for documents

    Returns:
        List of (scenario_name, description) tuples
    """
    scenarios = []

    # Create document with invalid chapter name
    invalid_doc = generate_unique_name("invalid_doc")
    invalid_doc_path = docs_root / invalid_doc
    invalid_doc_path.mkdir(parents=True, exist_ok=True)

    # Create a file without .md extension
    invalid_file = invalid_doc_path / "invalid_chapter.txt"
    invalid_file.write_text("This is not a markdown file")
    scenarios.append((invalid_doc, "Document with non-markdown file"))

    # Create empty document
    empty_doc = generate_unique_name("empty_doc")
    empty_doc_path = docs_root / empty_doc
    empty_doc_path.mkdir(parents=True, exist_ok=True)
    scenarios.append((empty_doc, "Empty document directory"))

    # Create document with empty chapter
    empty_chapter_doc = generate_unique_name("empty_chapter_doc")
    empty_chapter_doc_path = docs_root / empty_chapter_doc
    empty_chapter_doc_path.mkdir(parents=True, exist_ok=True)
    empty_chapter_file = empty_chapter_doc_path / "01-empty.md"
    empty_chapter_file.write_text("")
    scenarios.append((empty_chapter_doc, "Document with empty chapter"))

    return scenarios


def cleanup_test_documents(docs_root: Path, doc_names: List[str]) -> None:
    """
    Clean up test documents after testing.

    Args:
        docs_root: Root directory for documents
        doc_names: List of document names to clean up
    """
    import shutil

    for doc_name in doc_names:
        doc_path = docs_root / doc_name
        if doc_path.exists():
            shutil.rmtree(doc_path, ignore_errors=True)


# Test data templates for common scenarios
TEMPLATE_BOOK_STRUCTURE = [
    ("00-preface.md", "# Preface\n\nWelcome to this book."),
    ("01-introduction.md", "# Introduction\n\nThis book covers important topics."),
    ("02-chapter1.md", "# Chapter 1: Getting Started\n\nLet's begin our journey."),
    (
        "03-chapter2.md",
        "# Chapter 2: Advanced Topics\n\nNow for more complex subjects.",
    ),
    ("04-conclusion.md", "# Conclusion\n\nThank you for reading."),
    ("05-appendix.md", "# Appendix\n\nAdditional resources and references."),
]

TEMPLATE_RESEARCH_PAPER = [
    ("00-abstract.md", "# Abstract\n\nThis paper presents research findings."),
    (
        "01-introduction.md",
        "# Introduction\n\nBackground and motivation for the research.",
    ),
    ("02-methodology.md", "# Methodology\n\nApproach and methods used in the study."),
    ("03-results.md", "# Results\n\nFindings and data analysis."),
    ("04-discussion.md", "# Discussion\n\nInterpretation and implications."),
    ("05-conclusion.md", "# Conclusion\n\nSummary and future work."),
    ("06-references.md", "# References\n\nBibliography and citations."),
]

TEMPLATE_TECHNICAL_MANUAL = [
    ("01-overview.md", "# Overview\n\nSystem overview and architecture."),
    ("02-installation.md", "# Installation\n\nSetup and configuration instructions."),
    ("03-usage.md", "# Usage\n\nHow to use the system effectively."),
    ("04-api.md", "# API Reference\n\nDetailed API documentation."),
    ("05-troubleshooting.md", "# Troubleshooting\n\nCommon issues and solutions."),
    ("06-faq.md", "# FAQ\n\nFrequently asked questions."),
]
