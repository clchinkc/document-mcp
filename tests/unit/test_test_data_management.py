"""
Tests for test data management system.

This module validates the test data management infrastructure,
including test data factories, registries, and validation utilities.
"""

from pathlib import Path

import pytest

from tests.shared.test_data import (
    TestDataRegistry,
    TestDataType,
    TestDocumentSpec,
    create_test_document_from_spec,
    get_test_registry,
)


class TestTestDataRegistry:
    """Test suite for TestDataRegistry functionality."""

    def test_registry_initialization(self):
        """Test registry initializes with empty state."""
        registry = TestDataRegistry()

        assert len(registry.created_documents) == 0
        assert len(registry.created_directories) == 0
        assert len(registry.temp_files) == 0
        assert len(registry.active_specs) == 0

    def test_document_registration(self):
        """Test document registration functionality."""
        registry = TestDataRegistry()
        spec = TestDocumentSpec(name="test_doc", doc_type=TestDataType.SIMPLE)

        registry.register_document("test_doc", spec)

        assert "test_doc" in registry.created_documents
        assert len(registry.active_specs) == 1
        assert registry.get_document_spec("test_doc") == spec

    def test_duplicate_document_registration(self):
        """Test that duplicate documents are not registered twice."""
        registry = TestDataRegistry()
        spec = TestDocumentSpec(name="test_doc", doc_type=TestDataType.SIMPLE)

        registry.register_document("test_doc", spec)
        registry.register_document("test_doc", spec)  # Try to register again

        assert registry.created_documents.count("test_doc") == 1
        assert len(registry.active_specs) == 1

    def test_directory_registration(self):
        """Test directory registration functionality."""
        registry = TestDataRegistry()
        test_dir = Path("/tmp/test_dir")

        registry.register_directory(test_dir)

        assert test_dir in registry.created_directories

    def test_temp_file_registration(self):
        """Test temporary file registration functionality."""
        registry = TestDataRegistry()
        temp_file = Path("/tmp/test_file.txt")

        registry.register_temp_file(temp_file)

        assert temp_file in registry.temp_files

    def test_get_nonexistent_document_spec(self):
        """Test getting spec for non-existent document returns None."""
        registry = TestDataRegistry()

        spec = registry.get_document_spec("nonexistent_doc")

        assert spec is None


class TestTestDocumentSpec:
    """Test suite for TestDocumentSpec validation."""

    def test_spec_default_values(self):
        """Test specification with default values."""
        spec = TestDocumentSpec()

        assert spec.name is None
        assert spec.doc_type == TestDataType.SIMPLE
        assert spec.chapter_count == 3
        assert spec.chapters is None
        assert spec.search_terms is None
        assert spec.target_word_count is None
        assert spec.target_paragraph_count is None
        assert spec.paragraphs_per_chapter == 5
        assert spec.cleanup_on_error is True

    def test_spec_custom_values(self):
        """Test specification with custom values."""
        chapters = [("01-intro.md", "# Introduction")]
        search_terms = ["test", "demo"]

        spec = TestDocumentSpec(
            name="custom_doc",
            doc_type=TestDataType.SEARCHABLE,
            chapter_count=5,
            chapters=chapters,
            search_terms=search_terms,
            target_word_count=1000,
            target_paragraph_count=50,
            paragraphs_per_chapter=10,
            cleanup_on_error=False,
        )

        assert spec.name == "custom_doc"
        assert spec.doc_type == TestDataType.SEARCHABLE
        assert spec.chapter_count == 5
        assert spec.chapters == chapters
        assert spec.search_terms == search_terms
        assert spec.target_word_count == 1000
        assert spec.target_paragraph_count == 50
        assert spec.paragraphs_per_chapter == 10
        assert spec.cleanup_on_error is False

    def test_spec_validation_negative_chapter_count(self):
        """Test specification validation rejects negative chapter count."""
        with pytest.raises(ValueError, match="Chapter count must be non-negative"):
            TestDocumentSpec(chapter_count=-1)

    def test_spec_validation_negative_word_count(self):
        """Test specification validation rejects negative word count."""
        with pytest.raises(ValueError, match="Target word count must be non-negative"):
            TestDocumentSpec(target_word_count=-100)

    def test_spec_validation_negative_paragraph_count(self):
        """Test specification validation rejects negative paragraph count."""
        with pytest.raises(
            ValueError, match="Target paragraph count must be non-negative"
        ):
            TestDocumentSpec(target_paragraph_count=-10)


class TestDocumentFactory:
    """Test suite for document factory functionality."""

    def test_create_simple_document(self, test_docs_root, test_data_registry):
        """Test creating a simple document through the factory system."""
        spec = TestDocumentSpec(
            name="simple_test_doc",
            doc_type=TestDataType.SIMPLE,
            chapter_count=2,
        )

        doc_name = create_test_document_from_spec(
            docs_root=test_docs_root,
            spec=spec,
            registry=test_data_registry,
        )

        assert doc_name == "simple_test_doc"
        assert doc_name in test_data_registry.created_documents

        # Verify document exists and has correct structure
        doc_path = test_docs_root / doc_name
        assert doc_path.exists()
        assert doc_path.is_dir()

        # Check chapters were created
        md_files = list(doc_path.glob("*.md"))
        assert len(md_files) == 2

    def test_create_large_document(self, test_docs_root, test_data_registry):
        """Test creating a large document through the factory system."""
        spec = TestDocumentSpec(
            name="large_test_doc",
            doc_type=TestDataType.LARGE,
            chapter_count=3,
            paragraphs_per_chapter=2,  # Small for test performance
        )

        doc_name = create_test_document_from_spec(
            docs_root=test_docs_root,
            spec=spec,
            registry=test_data_registry,
        )

        assert doc_name == "large_test_doc"
        assert doc_name in test_data_registry.created_documents

        # Verify document structure
        doc_path = test_docs_root / doc_name
        md_files = list(doc_path.glob("*.md"))
        assert len(md_files) == 3

        # Check that chapters have content (large documents have more content)
        first_chapter = md_files[0]
        content = first_chapter.read_text()
        assert len(content) > 50  # Should have substantial content

    def test_create_searchable_document(self, test_docs_root, test_data_registry):
        """Test creating a searchable document through the factory system."""
        search_terms = ["findme", "searchable"]
        spec = TestDocumentSpec(
            name="searchable_test_doc",
            doc_type=TestDataType.SEARCHABLE,
            search_terms=search_terms,
        )

        doc_name = create_test_document_from_spec(
            docs_root=test_docs_root,
            spec=spec,
            registry=test_data_registry,
        )

        assert doc_name == "searchable_test_doc"

        # Verify search terms are present in content
        doc_path = test_docs_root / doc_name
        md_files = list(doc_path.glob("*.md"))

        all_content = ""
        for md_file in md_files:
            all_content += md_file.read_text()

        for term in search_terms:
            assert term in all_content

    def test_create_document_with_existing_name_cleanup(
        self, test_docs_root, test_data_registry
    ):
        """Test document creation with existing name and cleanup enabled."""
        doc_name = "existing_doc"

        # Create a document directory manually
        existing_path = test_docs_root / doc_name
        existing_path.mkdir()
        (existing_path / "existing_file.txt").write_text("existing content")

        # Try to create document with same name (should clean up and succeed)
        spec = TestDocumentSpec(
            name=doc_name,
            doc_type=TestDataType.SIMPLE,
            cleanup_on_error=True,
        )

        result_name = create_test_document_from_spec(
            docs_root=test_docs_root,
            spec=spec,
            registry=test_data_registry,
        )

        assert result_name == doc_name

        # Verify old content was cleaned up and new content exists
        doc_path = test_docs_root / doc_name
        assert not (doc_path / "existing_file.txt").exists()
        assert len(list(doc_path.glob("*.md"))) > 0

    def test_create_document_with_existing_name_no_cleanup(
        self, test_docs_root, test_data_registry
    ):
        """Test document creation with existing name and cleanup disabled."""
        doc_name = "existing_doc_no_cleanup"

        # Create a document directory manually
        existing_path = test_docs_root / doc_name
        existing_path.mkdir()

        # Try to create document with same name (should fail)
        spec = TestDocumentSpec(
            name=doc_name,
            doc_type=TestDataType.SIMPLE,
            cleanup_on_error=False,
        )

        with pytest.raises(
            FileExistsError, match=f"Document {doc_name} already exists"
        ):
            create_test_document_from_spec(
                docs_root=test_docs_root,
                spec=spec,
                registry=test_data_registry,
            )

    def test_create_document_unsupported_type(self, test_docs_root, test_data_registry):
        """Test document creation with unsupported document type."""
        spec = TestDocumentSpec(
            name="unsupported_doc",
            doc_type=TestDataType.ERROR_SCENARIO,  # Not yet implemented
        )

        with pytest.raises(ValueError, match="Unsupported document type"):
            create_test_document_from_spec(
                docs_root=test_docs_root,
                spec=spec,
                registry=test_data_registry,
            )


class TestFixtureIntegration:
    """Test suite for fixture integration and usage patterns."""

    def test_document_factory_fixture(
        self, document_factory, test_docs_root, validate_test_data
    ):
        """Test the document factory fixture functionality."""
        # Create a document using the factory fixture
        doc_name = document_factory(
            doc_type="simple",
            name="factory_test_doc",
            chapter_count=2,
        )

        assert doc_name == "factory_test_doc"

        # Validate the document was created correctly
        validate_test_data.document_exists(test_docs_root, doc_name)

    def test_parametrized_document_fixture(
        self, parametrized_document, test_docs_root, validate_test_data
    ):
        """Test the parametrized document fixture."""
        # This test will run multiple times with different document types
        assert parametrized_document is not None

        # Validate the document was created correctly
        validate_test_data.document_exists(test_docs_root, parametrized_document)

    def test_test_data_registry_fixture(self, test_data_registry, test_docs_root):
        """Test the test data registry fixture functionality."""
        # Registry should start empty for each test
        assert len(test_data_registry.created_documents) == 0

        # Register a document manually
        from tests.shared.test_data import TestDataType, TestDocumentSpec

        spec = TestDocumentSpec(name="registry_test", doc_type=TestDataType.SIMPLE)
        test_data_registry.register_document("registry_test", spec)

        assert "registry_test" in test_data_registry.created_documents
        assert test_data_registry.get_document_spec("registry_test") == spec

    def test_validation_fixture(
        self, document_factory, test_docs_root, test_data_registry, validate_test_data
    ):
        """Test the validation fixture functionality."""
        # Create a document
        doc_name = document_factory(
            doc_type="simple",
            name="validation_test_doc",
            chapter_count=2,
        )

        # Test validation functions
        validate_test_data.document_exists(test_docs_root, doc_name)
        validate_test_data.registry_state(test_data_registry, test_docs_root)

        # Test content validation with expected chapters
        expected_chapters = [("01-introduction.md", "Introduction")]
        validate_test_data.document_content(test_docs_root, doc_name, expected_chapters)


class TestGlobalRegistry:
    """Test suite for global registry functionality."""

    def test_global_registry_access(self):
        """Test accessing the global test registry."""
        registry = get_test_registry()

        assert isinstance(registry, TestDataRegistry)

        # Multiple calls should return the same instance
        registry2 = get_test_registry()
        assert registry is registry2

    def test_global_registry_isolation(self):
        """Test that global registry maintains state across calls."""
        registry = get_test_registry()
        initial_count = len(registry.created_documents)

        # Register a test document
        spec = TestDocumentSpec(name="global_test", doc_type=TestDataType.SIMPLE)
        registry.register_document("global_test", spec)

        # Verify the change persists
        registry2 = get_test_registry()
        assert len(registry2.created_documents) == initial_count + 1
        assert "global_test" in registry2.created_documents
