"""
Integration tests for document summary functionality.

This module provides comprehensive testing for document summary features
including agent workflow integration with proper test isolation.
"""

import tempfile
import os
from pathlib import Path
import pytest

# Import after environment setup
def get_tools():
    """Import tools after environment is set up."""
    from document_mcp.doc_tool_server import (
        read_document_summary,
        list_documents,
        create_document,
        create_chapter,
        DOCUMENT_SUMMARY_FILE,
    )
    return (read_document_summary, list_documents, create_document, 
            create_chapter, DOCUMENT_SUMMARY_FILE)


class TestSummaryFunctionality:
    """Test summary functionality with proper isolation."""
    
    def setup_method(self):
        """Set up isolated test environment."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix='test_summary_'))
        os.environ['DOCUMENT_ROOT_DIR'] = str(self.temp_dir)
        
        # Clear module cache to ensure fresh imports
        import sys
        if 'document_mcp.doc_tool_server' in sys.modules:
            import importlib
            importlib.reload(sys.modules['document_mcp.doc_tool_server'])
        
        # Import tools after environment setup
        (self.read_document_summary, self.list_documents, 
         self.create_document, self.create_chapter, 
         self.DOCUMENT_SUMMARY_FILE) = get_tools()
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_basic_summary_workflow(self):
        """Test basic summary creation and reading."""
        doc_name = 'test_doc'
        
        # Create document
        result = self.create_document(doc_name)
        assert result.success is True
        
        # Initially no summary
        docs = self.list_documents()
        test_doc = next((d for d in docs if d.document_name == doc_name), None)
        assert test_doc is not None
        assert test_doc.has_summary is False
        
        summary_content = self.read_document_summary(doc_name)
        assert summary_content is None
        
        # Create summary file
        doc_dir = self.temp_dir / doc_name
        doc_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        summary_file = doc_dir / self.DOCUMENT_SUMMARY_FILE
        test_summary = "# Test Summary\n\nThis is a test summary."
        summary_file.write_text(test_summary)
        
        # Verify summary is detected
        docs = self.list_documents()
        test_doc = next((d for d in docs if d.document_name == doc_name), None)
        assert test_doc.has_summary is True
        
        # Verify summary content
        read_summary = self.read_document_summary(doc_name)
        assert read_summary == test_summary
    
    def test_summary_with_chapters(self):
        """Test summary functionality alongside chapters."""
        doc_name = 'doc_with_chapters'
        
        # Create document and summary
        self.create_document(doc_name)
        doc_dir = self.temp_dir / doc_name
        doc_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        summary_file = doc_dir / self.DOCUMENT_SUMMARY_FILE
        summary_content = "# Document Summary\n\nOverview of the document."
        summary_file.write_text(summary_content)
        
        # Add chapters
        self.create_chapter(doc_name, '01-intro.md', '# Introduction')
        self.create_chapter(doc_name, '02-content.md', '# Content')
        
        # Verify state
        docs = self.list_documents()
        doc_info = next((d for d in docs if d.document_name == doc_name), None)
        assert doc_info is not None
        assert doc_info.has_summary is True
        assert doc_info.total_chapters == 2  # Should not count summary
        
        # Summary should still be readable
        read_summary = self.read_document_summary(doc_name)
        assert read_summary == summary_content
    
    def test_multiple_documents_with_different_summary_states(self):
        # Document with summary
        doc1 = 'doc_with_summary'
        self.create_document(doc1)
        doc1_dir = self.temp_dir / doc1
        doc1_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        summary1_file = doc1_dir / self.DOCUMENT_SUMMARY_FILE
        summary1_file.write_text("# Summary 1")
        
        # Document without summary
        doc2 = 'doc_without_summary'
        self.create_document(doc2)
        
        # Verify both documents
        docs = self.list_documents()
        docs_by_name = {d.document_name: d for d in docs}
        
        assert docs_by_name[doc1].has_summary is True
        assert docs_by_name[doc2].has_summary is False
        
        assert self.read_document_summary(doc1) == "# Summary 1"
        assert self.read_document_summary(doc2) is None
    
    def test_summary_file_not_found_error_handling(self):
        """Test error handling when summary file doesn't exist."""
        doc_name = 'no_summary_doc'
        self.create_document(doc_name)
        
        # Should handle missing summary gracefully
        summary_content = self.read_document_summary(doc_name)
        assert summary_content is None
        
        docs = self.list_documents()
        doc_info = next((d for d in docs if d.document_name == doc_name), None)
        assert doc_info.has_summary is False
    
    def test_empty_summary_file(self):
        """Test handling of empty summary files."""
        doc_name = 'empty_summary_doc'
        self.create_document(doc_name)
        
        # Create empty summary file
        doc_dir = self.temp_dir / doc_name
        doc_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        summary_file = doc_dir / self.DOCUMENT_SUMMARY_FILE
        summary_file.write_text('')
        
        # Should still be detected as having summary
        docs = self.list_documents()
        doc_info = next((d for d in docs if d.document_name == doc_name), None)
        assert doc_info.has_summary is True
        
        # Should return empty content
        summary_content = self.read_document_summary(doc_name)
        assert summary_content == ''
    
    def test_nonexistent_document_summary_read(self):
        """Test reading summary for non-existent document."""
        summary_content = self.read_document_summary('nonexistent_doc')
        assert summary_content is None


class TestSummaryWorkflowEdgeCases:
    """Test edge cases and boundary conditions for summary workflows."""
    
    def setup_method(self):
        """Set up isolated test environment."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix='test_edge_summary_'))
        os.environ['DOCUMENT_ROOT_DIR'] = str(self.temp_dir)
        
        # Clear module cache to ensure fresh imports
        import sys
        if 'document_mcp.doc_tool_server' in sys.modules:
            import importlib
            importlib.reload(sys.modules['document_mcp.doc_tool_server'])
        
        # Import tools after environment setup
        (self.read_document_summary, self.list_documents, 
         self.create_document, self.create_chapter, 
         self.DOCUMENT_SUMMARY_FILE) = get_tools()
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_summary_with_special_characters(self):
        """Test summary handling with special characters and Unicode."""
        doc_name = 'unicode_test_doc'
        self.create_document(doc_name)
        
        doc_dir = self.temp_dir / doc_name
        doc_dir.mkdir(parents=True, exist_ok=True)
        summary_file = doc_dir / self.DOCUMENT_SUMMARY_FILE
        
        # Test with Unicode and special characters
        unicode_summary = """# 🚀 Project Summary 

## Overview
- Task: Implement AI-powered document system
- Status: ✅ Complete
- Languages: English, 中文, Français, Español

## Key Features
- Multi-language support
- Special chars: @#$%^&*()
- Emojis: 📝📊📈
"""
        summary_file.write_text(unicode_summary, encoding='utf-8')
        
        # Verify reading works correctly
        docs = self.list_documents()
        test_doc = next((d for d in docs if d.document_name == doc_name), None)
        assert test_doc.has_summary is True
        
        read_summary = self.read_document_summary(doc_name)
        assert read_summary == unicode_summary
    
    def test_summary_persistence_across_chapter_operations(self):
        """Test that summaries persist correctly during chapter operations."""
        doc_name = 'persistence_test_doc'
        self.create_document(doc_name)
        
        doc_dir = self.temp_dir / doc_name
        doc_dir.mkdir(parents=True, exist_ok=True)
        summary_file = doc_dir / self.DOCUMENT_SUMMARY_FILE
        original_summary = '# Persistent Summary\n\nThis should persist across operations.'
        summary_file.write_text(original_summary)
        
        # Perform chapter operations
        self.create_chapter(doc_name, '01-intro.md', '# Introduction')
        assert self.read_document_summary(doc_name) == original_summary
        
        self.create_chapter(doc_name, '02-methods.md', '# Methods')
        assert self.read_document_summary(doc_name) == original_summary
        
        # Verify final state
        docs = self.list_documents()
        doc_info = next((d for d in docs if d.document_name == doc_name), None)
        assert doc_info.has_summary is True
        assert doc_info.total_chapters == 2
    
    def test_summary_vs_chapter_independence(self):
        """Test that summary content is independent of chapter content."""
        doc_name = 'independence_test_doc'
        self.create_document(doc_name)
        
        doc_dir = self.temp_dir / doc_name
        doc_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summary with specific content
        summary_file = doc_dir / self.DOCUMENT_SUMMARY_FILE
        summary_content = """# Document Summary

Key points about this project:
- Summary information
- High-level overview
- Executive briefing
"""
        summary_file.write_text(summary_content)
        
        # Create chapter with different content
        self.create_chapter(doc_name, '01-chapter.md', """# Chapter 1

Detailed implementation notes:
- Technical specifications
- Low-level details
- Implementation notes
""")
        
        # Verify independence
        read_summary = self.read_document_summary(doc_name)
        assert read_summary == summary_content
        assert 'Summary information' in read_summary
        assert 'Technical specifications' not in read_summary


if __name__ == '__main__':
    pytest.main([__file__]) 