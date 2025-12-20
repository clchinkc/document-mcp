"""Content-related MCP tools for document management.

This module provides unified content access tools that work across different
scopes (document, chapter, paragraph) with a consistent interface.
"""

import os
import time

import google.generativeai as genai
import numpy as np

from ..helpers import _count_words
from ..helpers import _get_chapter_path
from ..helpers import _get_document_path
from ..helpers import _get_ordered_chapter_files
from ..helpers import _split_into_paragraphs
from ..logger_config import ErrorCategory
from ..logger_config import log_mcp_call
from ..logger_config import log_structured_error
from ..models import ChapterContent
from ..models import OperationStatus
from ..models import PaginatedContent
from ..models import PaginationInfo
from ..models import ParagraphDetail
from ..models import SemanticSearchResponse
from ..models import SemanticSearchResult
from ..models import StatisticsReport
from ..utils.decorators import auto_snapshot
from ..utils.embedding_cache import EmbeddingCache
from ..utils.validation import validate_chapter_name
from ..utils.validation import validate_content
from ..utils.validation import validate_document_name
from ..utils.validation import validate_paragraph_index
from ..utils.validation import validate_search_query


def register_content_tools(mcp_server):
    """Register all content-related tools with the MCP server."""

    @mcp_server.tool()
    @log_mcp_call
    def read_content(
        document_name: str,
        scope: str = "document",
        chapter_name: str | None = None,
        paragraph_index: int | None = None,
        page: int = 1,
        page_size: int = 50000,
    ) -> PaginatedContent | ChapterContent | ParagraphDetail | None:
        """Unified content reading with scope-based targeting and pagination.

        Parameters:
            document_name: Name of the document to read from
            scope: Reading scope - "document" (paginated), "chapter", or "paragraph"
            chapter_name: Required for "chapter" and "paragraph" scopes
            paragraph_index: Required for "paragraph" scope (0-indexed)
            page: Page number for document scope (1-indexed, default: 1)
            page_size: Characters per page for document scope (default: 50,000 ≈ 12K tokens)

        Returns:
            Content object matching the requested scope, or None if not found.
            - document scope: PaginatedContent with pagination metadata
            - chapter scope: ChapterContent with metadata
            - paragraph scope: ParagraphDetail with metadata

        """
        # Import models that aren't imported at module level
        from ..helpers import _get_chapter_metadata
        from ..models import ChapterContent
        from ..models import ParagraphDetail

        # Validate document name
        is_valid_doc, doc_error = validate_document_name(document_name)
        if not is_valid_doc:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Invalid document name: {doc_error}",
                context={"document_name": document_name, "scope": scope},
                operation="read_content",
            )
            return None

        # Validate scope-specific parameters
        if scope == "chapter":
            if not chapter_name:
                log_structured_error(
                    category=ErrorCategory.ERROR,
                    message="chapter_name required for chapter scope",
                    context={"document_name": document_name, "scope": scope},
                    operation="read_content",
                )
                return None
            is_valid_chapter, chapter_error = validate_chapter_name(chapter_name)
            if not is_valid_chapter:
                log_structured_error(
                    category=ErrorCategory.ERROR,
                    message=f"Invalid chapter name: {chapter_error}",
                    context={
                        "document_name": document_name,
                        "chapter_name": chapter_name,
                        "scope": scope,
                    },
                    operation="read_content",
                )
                return None

        elif scope == "paragraph":
            if not chapter_name or paragraph_index is None:
                log_structured_error(
                    category=ErrorCategory.ERROR,
                    message="chapter_name and paragraph_index required for paragraph scope",
                    context={"document_name": document_name, "scope": scope},
                    operation="read_content",
                )
                return None
            is_valid_chapter, chapter_error = validate_chapter_name(chapter_name)
            if not is_valid_chapter:
                log_structured_error(
                    category=ErrorCategory.ERROR,
                    message=f"Invalid chapter name: {chapter_error}",
                    context={
                        "document_name": document_name,
                        "chapter_name": chapter_name,
                        "scope": scope,
                    },
                    operation="read_content",
                )
                return None
            is_valid_index, index_error = validate_paragraph_index(paragraph_index)
            if not is_valid_index:
                log_structured_error(
                    category=ErrorCategory.ERROR,
                    message=f"Invalid paragraph index: {index_error}",
                    context={
                        "document_name": document_name,
                        "paragraph_index": paragraph_index,
                        "scope": scope,
                    },
                    operation="read_content",
                )
                return None

        elif scope != "document":
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Invalid scope: {scope}. Must be 'document', 'chapter', or 'paragraph'",
                context={"document_name": document_name, "scope": scope},
                operation="read_content",
            )
            return None

        # Validate pagination parameters for document scope
        if scope == "document":
            if page <= 0:
                log_structured_error(
                    category=ErrorCategory.ERROR,
                    message=f"Invalid page: {page}. Must be greater than 0",
                    context={"document_name": document_name, "page": page},
                    operation="read_content",
                )
                return None
            if page_size <= 0:
                log_structured_error(
                    category=ErrorCategory.ERROR,
                    message=f"Invalid page_size: {page_size}. Must be greater than 0",
                    context={"document_name": document_name, "page_size": page_size},
                    operation="read_content",
                )
                return None

        # Scope-based dispatch
        try:
            if scope == "document":
                doc_path = _get_document_path(document_name)
                if not doc_path.exists():
                    return None

                page_content, pagination_info = _paginate_document_efficiently(document_name, page, page_size)

                result = PaginatedContent(
                    content=page_content,
                    document_name=document_name,
                    scope=scope,
                    chapter_name=chapter_name,
                    paragraph_index=paragraph_index,
                    pagination=pagination_info,
                )
                return result

            elif scope == "chapter":
                doc_path = _get_document_path(document_name)
                chapter_path = doc_path / chapter_name
                if not chapter_path.exists():
                    return None

                content = chapter_path.read_text(encoding="utf-8")
                metadata = _get_chapter_metadata(document_name, chapter_path)
                result = ChapterContent(
                    document_name=document_name,
                    chapter_name=chapter_name,
                    content=content,
                    word_count=metadata.word_count,
                    paragraph_count=metadata.paragraph_count,
                    last_modified=metadata.last_modified,
                )
                return result

            elif scope == "paragraph":
                doc_path = _get_document_path(document_name)
                chapter_path = doc_path / chapter_name
                if not chapter_path.exists():
                    return None

                content = chapter_path.read_text(encoding="utf-8")
                paragraphs = _split_into_paragraphs(content)

                if paragraph_index >= len(paragraphs):
                    return None

                paragraph_content = paragraphs[paragraph_index]
                result = ParagraphDetail(
                    document_name=document_name,
                    chapter_name=chapter_name,
                    paragraph_index_in_chapter=paragraph_index,
                    content=paragraph_content,
                    word_count=len(paragraph_content.split()),
                )
                return result

        except Exception as e:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Error reading content with scope {scope}: {str(e)}",
                context={
                    "document_name": document_name,
                    "scope": scope,
                    "chapter_name": chapter_name,
                    "paragraph_index": paragraph_index,
                    "page": page,
                    "page_size": page_size,
                },
                operation="read_content",
            )
            return None

    @mcp_server.tool()
    @log_mcp_call
    def find_text(
        document_name: str,
        search_text: str,
        scope: str = "document",  # "document", "chapter"
        chapter_name: str | None = None,
        case_sensitive: bool = False,
        max_results: int = 100,
    ) -> list[ParagraphDetail] | None:
        """Unified text search with scope-based targeting.

        This tool consolidates document and chapter text search into a single interface,
        providing consistent search capabilities across different scopes with flexible
        case sensitivity options.

        Parameters:
            document_name (str): Name of the document to search within
            search_text (str): Text pattern to search for
            scope (str): Search scope determining where to search:
                - "document": Search across entire document (all chapters)
                - "chapter": Search within specific chapter only
            chapter_name (Optional[str]): Required for "chapter" scope.
                Must be valid .md filename (e.g., "01-introduction.md")
            case_sensitive (bool): Whether search should be case-sensitive (default: False)
            max_results (int): Maximum number of search results to return (default: 100).
                Prevents overwhelming responses for common search terms

        Returns:
            Optional[List[Dict[str, Any]]]: List of search results, None if error.
            Each result contains location and context information.

            For scope="document": Results from find_text_in_document
            For scope="chapter": Results from find_text_in_chapter

        Example Usage:
            ```json
            // Search entire document
            {
                "name": "find_text",
                "arguments": {
                    "document_name": "My Book",
                    "search_text": "important concept",
                    "scope": "document",
                    "case_sensitive": false
                }
            }

            // Search with limited results
            {
                "name": "find_text",
                "arguments": {
                    "document_name": "My Book",
                    "search_text": "common term",
                    "scope": "document",
                    "max_results": 50
                }
            }

            // Search specific chapter
            {
                "name": "find_text",
                "arguments": {
                    "document_name": "My Book",
                    "search_text": "introduction",
                    "scope": "chapter",
                    "chapter_name": "01-intro.md",
                    "case_sensitive": true
                }
            }
            ```
        """
        # Import here to avoid circular imports

        # Validate document name
        is_valid_doc, doc_error = validate_document_name(document_name)
        if not is_valid_doc:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Invalid document name: {doc_error}",
                context={"document_name": document_name, "scope": scope},
                operation="find_text",
            )
            return None

        # Validate search text
        is_valid_search, search_error = validate_search_query(search_text)
        if not is_valid_search:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Invalid search text: {search_error}",
                context={
                    "document_name": document_name,
                    "search_text": search_text,
                    "scope": scope,
                },
                operation="find_text",
            )
            return None

        # Validate scope-specific parameters
        if scope == "chapter":
            if not chapter_name:
                log_structured_error(
                    category=ErrorCategory.ERROR,
                    message="chapter_name required for chapter scope",
                    context={"document_name": document_name, "scope": scope},
                    operation="find_text",
                )
                return None
            is_valid_chapter, chapter_error = validate_chapter_name(chapter_name)
            if not is_valid_chapter:
                log_structured_error(
                    category=ErrorCategory.ERROR,
                    message=f"Invalid chapter name: {chapter_error}",
                    context={
                        "document_name": document_name,
                        "chapter_name": chapter_name,
                        "scope": scope,
                    },
                    operation="find_text",
                )
                return None
        elif scope != "document":
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Invalid scope: {scope}. Must be 'document' or 'chapter'",
                context={"document_name": document_name, "scope": scope},
                operation="find_text",
            )
            return None

        # Validate max_results parameter
        if max_results <= 0:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Invalid max_results: {max_results}. Must be greater than 0",
                context={
                    "document_name": document_name,
                    "max_results": max_results,
                    "scope": scope,
                },
                operation="find_text",
            )
            return None

        # Scope-based dispatch to helper functions
        try:
            if scope == "document":
                result = _find_text_in_document(document_name, search_text, case_sensitive, max_results)
                return result if result else []

            elif scope == "chapter":
                result = _find_text_in_chapter(
                    document_name, chapter_name, search_text, case_sensitive, max_results
                )
                return result if result else []

        except Exception as e:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Error searching text with scope {scope}: {str(e)}",
                context={
                    "document_name": document_name,
                    "scope": scope,
                    "search_text": search_text,
                    "chapter_name": chapter_name,
                },
                operation="find_text",
            )
            return None

    @mcp_server.tool()
    @log_mcp_call
    @auto_snapshot("replace_text")
    def replace_text(
        document_name: str,
        find_text: str,
        replace_text: str,
        scope: str = "document",  # "document", "chapter"
        chapter_name: str | None = None,
    ) -> OperationStatus | None:
        """Unified text replacement with scope-based targeting.

        This tool consolidates document and chapter text replacement into a single interface,
        providing consistent replacement capabilities across different scopes with atomic
        operation guarantees.

        Parameters:
            document_name (str): Name of the document to perform replacement in
            find_text (str): Text pattern to find and replace
            replace_text (str): Text to replace occurrences with
            scope (str): Replacement scope determining where to replace:
                - "document": Replace across entire document (all chapters)
                - "chapter": Replace within specific chapter only
            chapter_name (Optional[str]): Required for "chapter" scope.
                Must be valid .md filename (e.g., "01-introduction.md")

        Returns:
            Optional[Dict[str, Any]]: Replacement operation results, None if error.
            Contains success status and replacement statistics.

        Example Usage:
            ```json
            // Replace across entire document
            {
                "name": "replace_text",
                "arguments": {
                    "document_name": "My Book",
                    "find_text": "old term",
                    "replace_text": "new term",
                    "scope": "document"
                }
            }

            // Replace in specific chapter
            {
                "name": "replace_text",
                "arguments": {
                    "document_name": "My Book",
                    "find_text": "draft text",
                    "replace_text": "final text",
                    "scope": "chapter",
                    "chapter_name": "01-intro.md"
                }
            }
            ```
        """
        # Validate document name
        is_valid_doc, doc_error = validate_document_name(document_name)
        if not is_valid_doc:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Invalid document name: {doc_error}",
                context={"document_name": document_name, "scope": scope},
                operation="replace_text",
            )
            return None

        # Validate find and replace text
        is_valid_find, find_error = validate_search_query(find_text)
        if not is_valid_find:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Invalid find text: {find_error}",
                context={
                    "document_name": document_name,
                    "find_text": find_text,
                    "scope": scope,
                },
                operation="replace_text",
            )
            return None

        is_valid_replace, replace_error = validate_content(replace_text)
        if not is_valid_replace:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Invalid replace text: {replace_error}",
                context={
                    "document_name": document_name,
                    "replace_text": replace_text,
                    "scope": scope,
                },
                operation="replace_text",
            )
            return None

        # Validate scope-specific parameters
        if scope == "chapter":
            if not chapter_name:
                log_structured_error(
                    category=ErrorCategory.ERROR,
                    message="chapter_name required for chapter scope",
                    context={"document_name": document_name, "scope": scope},
                    operation="replace_text",
                )
                return None
            is_valid_chapter, chapter_error = validate_chapter_name(chapter_name)
            if not is_valid_chapter:
                log_structured_error(
                    category=ErrorCategory.ERROR,
                    message=f"Invalid chapter name: {chapter_error}",
                    context={
                        "document_name": document_name,
                        "chapter_name": chapter_name,
                        "scope": scope,
                    },
                    operation="replace_text",
                )
                return None
        elif scope != "document":
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Invalid scope: {scope}. Must be 'document' or 'chapter'",
                context={"document_name": document_name, "scope": scope},
                operation="replace_text",
            )
            return None

        # Scope-based dispatch to helper functions
        try:
            if scope == "document":
                result = _replace_text_in_document(document_name, find_text, replace_text)
                return result if result else None

            elif scope == "chapter":
                result = _replace_text_in_chapter(document_name, chapter_name, find_text, replace_text)
                return result if result else None

        except Exception as e:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Error replacing text with scope {scope}: {str(e)}",
                context={
                    "document_name": document_name,
                    "scope": scope,
                    "find_text": find_text,
                    "replace_text": replace_text,
                    "chapter_name": chapter_name,
                },
                operation="replace_text",
            )
            return None

    @mcp_server.tool()
    @log_mcp_call
    def get_statistics(
        document_name: str,
        scope: str = "document",  # "document", "chapter"
        chapter_name: str | None = None,
    ) -> StatisticsReport | None:
        """Unified statistics collection with scope-based targeting.

        This tool consolidates document and chapter statistics into a single interface,
        providing consistent analytics capabilities across different scopes with
        comprehensive word, paragraph, and chapter counts.

        Parameters:
            document_name (str): Name of the document to analyze
            scope (str): Statistics scope determining what to analyze:
                - "document": Analyze entire document (all chapters)
                - "chapter": Analyze specific chapter only
            chapter_name (Optional[str]): Required for "chapter" scope.
                Must be valid .md filename (e.g., "01-introduction.md")

        Returns:
            Optional[Dict[str, Any]]: Statistics report, None if error.
            Contains word counts, paragraph counts, and scope information.

            For scope="document": Results from get_document_statistics
            For scope="chapter": Results from get_chapter_statistics

        Example Usage:
            ```json
            // Get document statistics
            {
                "name": "get_statistics",
                "arguments": {
                    "document_name": "My Book",
                    "scope": "document"
                }
            }

            // Get chapter statistics
            {
                "name": "get_statistics",
                "arguments": {
                    "document_name": "My Book",
                    "scope": "chapter",
                    "chapter_name": "01-intro.md"
                }
            }
            ```
        """
        # Use local helper functions

        # Validate document name
        is_valid_doc, doc_error = validate_document_name(document_name)
        if not is_valid_doc:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Invalid document name: {doc_error}",
                context={"document_name": document_name, "scope": scope},
                operation="get_statistics",
            )
            return None

        # Validate scope-specific parameters
        if scope == "chapter":
            if not chapter_name:
                log_structured_error(
                    category=ErrorCategory.ERROR,
                    message="chapter_name required for chapter scope",
                    context={"document_name": document_name, "scope": scope},
                    operation="get_statistics",
                )
                return None
            is_valid_chapter, chapter_error = validate_chapter_name(chapter_name)
            if not is_valid_chapter:
                log_structured_error(
                    category=ErrorCategory.ERROR,
                    message=f"Invalid chapter name: {chapter_error}",
                    context={
                        "document_name": document_name,
                        "chapter_name": chapter_name,
                        "scope": scope,
                    },
                    operation="get_statistics",
                )
                return None
        elif scope != "document":
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Invalid scope: {scope}. Must be 'document' or 'chapter'",
                context={"document_name": document_name, "scope": scope},
                operation="get_statistics",
            )
            return None

        # Scope-based dispatch to local helper functions
        try:
            if scope == "document":
                result = _get_document_statistics(document_name)
                return result if result else None

            elif scope == "chapter":
                result = _get_chapter_statistics(document_name, chapter_name)
                if result:
                    # For chapter scope, create new StatisticsReport without chapter_count
                    from ..models import StatisticsReport

                    return StatisticsReport(
                        scope=result.scope,
                        word_count=result.word_count,
                        paragraph_count=result.paragraph_count,
                        chapter_count=None,  # Exclude chapter_count for chapter scope
                    )
                return None

        except Exception as e:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Error getting statistics with scope {scope}: {str(e)}",
                context={
                    "document_name": document_name,
                    "scope": scope,
                    "chapter_name": chapter_name,
                },
                operation="get_statistics",
            )
            return None

    @mcp_server.tool()
    @log_mcp_call
    def find_similar_text(
        document_name: str,
        query_text: str,
        scope: str = "document",  # "document", "chapter"
        chapter_name: str | None = None,
        similarity_threshold: float = 0.7,
        max_results: int = 10,
    ) -> SemanticSearchResponse | None:
        """Semantic text search with scope-based targeting and similarity scoring.

        This tool uses embedding-based semantic search to find content similar in meaning
        to the query text, rather than exact string matching. It provides configurable
        similarity thresholds and result limits for precise control over search results.

        Parameters:
            document_name (str): Name of the document to search within
            query_text (str): Text query to find semantically similar content for
            scope (str): Search scope determining where to search:
                - "document": Search across entire document (all chapters)
                - "chapter": Search within specific chapter only
            chapter_name (Optional[str]): Required for "chapter" scope.
                Must be valid .md filename (e.g., "01-introduction.md")
            similarity_threshold (float): Minimum similarity score (0.0-1.0) for results (default: 0.7)
            max_results (int): Maximum number of results to return (default: 10)

        Returns:
            Optional[Dict[str, Any]]: Semantic search response with scored results, None if error.
            Contains query metadata, execution time, and list of semantically similar content.

            Response structure:
            - document_name (str): Name of the searched document
            - scope (str): Search scope used ("document" or "chapter")
            - query_text (str): Original query text
            - results (List[SemanticSearchResult]): Sorted list of similar content
            - total_results (int): Number of results returned
            - execution_time_ms (float): Time taken to complete the search

            Each result in results contains:
            - document_name (str): Name of the parent document
            - chapter_name (str): Name of the chapter containing the match
            - paragraph_index (int): Zero-indexed position within the chapter
            - content (str): Full text content of the matching paragraph
            - similarity_score (float): Semantic similarity score (0.0-1.0)
            - context_snippet (Optional[str]): Surrounding text for context

        Example Usage:
            ```json
            // Search entire document for themes
            {
                "name": "find_similar_text",
                "arguments": {
                    "document_name": "My Book",
                    "query_text": "character development themes",
                    "scope": "document",
                    "similarity_threshold": 0.7,
                    "max_results": 5
                }
            }

            // Search specific chapter for concepts
            {
                "name": "find_similar_text",
                "arguments": {
                    "document_name": "My Book",
                    "query_text": "introduction to key concepts",
                    "scope": "chapter",
                    "chapter_name": "01-intro.md",
                    "similarity_threshold": 0.6
                }
            }
            ```
        """
        start_time = time.time()

        # Validate document name
        is_valid_doc, doc_error = validate_document_name(document_name)
        if not is_valid_doc:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Invalid document name: {doc_error}",
                context={"document_name": document_name, "scope": scope},
                operation="find_similar_text",
            )
            return None

        # Validate query text
        is_valid_query, query_error = validate_search_query(query_text)
        if not is_valid_query:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Invalid query text: {query_error}",
                context={
                    "document_name": document_name,
                    "query_text": query_text,
                    "scope": scope,
                },
                operation="find_similar_text",
            )
            return None

        # Validate similarity threshold
        if not (0.0 <= similarity_threshold <= 1.0):
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Invalid similarity threshold: {similarity_threshold}. Must be between 0.0 and 1.0",
                context={
                    "document_name": document_name,
                    "similarity_threshold": similarity_threshold,
                    "scope": scope,
                },
                operation="find_similar_text",
            )
            return None

        # Validate max_results
        if max_results <= 0:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Invalid max_results: {max_results}. Must be greater than 0",
                context={
                    "document_name": document_name,
                    "max_results": max_results,
                    "scope": scope,
                },
                operation="find_similar_text",
            )
            return None

        # Validate scope-specific parameters
        if scope == "chapter":
            if not chapter_name:
                log_structured_error(
                    category=ErrorCategory.ERROR,
                    message="chapter_name required for chapter scope",
                    context={"document_name": document_name, "scope": scope},
                    operation="find_similar_text",
                )
                return None
            is_valid_chapter, chapter_error = validate_chapter_name(chapter_name)
            if not is_valid_chapter:
                log_structured_error(
                    category=ErrorCategory.ERROR,
                    message=f"Invalid chapter name: {chapter_error}",
                    context={
                        "document_name": document_name,
                        "chapter_name": chapter_name,
                        "scope": scope,
                    },
                    operation="find_similar_text",
                )
                return None
        elif scope != "document":
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Invalid scope: {scope}. Must be 'document' or 'chapter'",
                context={"document_name": document_name, "scope": scope},
                operation="find_similar_text",
            )
            return None

        # Perform semantic search
        try:
            results = _perform_semantic_search(
                document_name=document_name,
                query_text=query_text,
                scope=scope,
                chapter_name=chapter_name,
                similarity_threshold=similarity_threshold,
                max_results=max_results,
            )

            execution_time_ms = (time.time() - start_time) * 1000

            response = SemanticSearchResponse(
                document_name=document_name,
                scope=scope,
                query_text=query_text,
                results=results or [],
                total_results=len(results or []),
                execution_time_ms=execution_time_ms,
            )

            return response

        except Exception as e:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Error performing semantic search: {str(e)}",
                context={
                    "document_name": document_name,
                    "scope": scope,
                    "query_text": query_text,
                    "chapter_name": chapter_name,
                },
                operation="find_similar_text",
            )
            return None


# Helper functions for content tools


def _paginate_content(content: str, page: int, page_size: int) -> tuple[str, PaginationInfo]:
    """Core pagination logic with boundary handling."""
    total_chars = len(content)
    total_pages = (total_chars + page_size - 1) // page_size

    if page < 1 or (total_pages > 0 and page > total_pages):
        raise ValueError(f"Page {page} out of range [1, {total_pages}]")

    if total_chars == 0:
        return "", PaginationInfo(
            page=1,
            page_size=page_size,
            total_characters=0,
            total_pages=1,
            has_more=False,
            has_previous=False,
            next_page=None,
            previous_page=None,
        )

    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_chars)

    page_content = content[start_idx:end_idx]

    pagination = PaginationInfo(
        page=page,
        page_size=page_size,
        total_characters=total_chars,
        total_pages=total_pages,
        has_more=page < total_pages,
        has_previous=page > 1,
        next_page=page + 1 if page < total_pages else None,
        previous_page=page - 1 if page > 1 else None,
    )

    return page_content, pagination


def _paginate_document_efficiently(
    document_name: str, page: int, page_size: int
) -> tuple[str, PaginationInfo]:
    """Document pagination with bounded memory usage.

    Loads full document and applies pagination. Provides predictable page sizes
    with memory usage bounded by document size (not unbounded like truncation).
    """
    chapter_files = _get_ordered_chapter_files(document_name)

    if not chapter_files:
        return "", PaginationInfo(
            page=1,
            page_size=page_size,
            total_characters=0,
            total_pages=1,
            has_more=False,
            has_previous=False,
            next_page=None,
            previous_page=None,
        )

    # Load document content
    full_content = ""
    for chapter_file in chapter_files:
        try:
            content = chapter_file.read_text(encoding="utf-8")
            if full_content:
                full_content += "\n\n" + content
            else:
                full_content = content
        except Exception as e:
            log_structured_error(
                category=ErrorCategory.WARNING,
                message=f"Error reading chapter {chapter_file.name}: {str(e)}",
                context={
                    "document_name": document_name,
                    "chapter_file": str(chapter_file),
                },
                operation="read_content",
            )
            continue

    return _paginate_content(full_content, page, page_size)


def _find_text_in_document(
    document_name: str, query: str, case_sensitive: bool = False, max_results: int = 100
) -> list[ParagraphDetail]:
    """Search for paragraphs containing specific text across all chapters."""
    results = []
    doc_path = _get_document_path(document_name)
    if not doc_path.exists():
        return results

    chapter_files = _get_ordered_chapter_files(document_name)
    for chapter_file in chapter_files:
        if len(results) >= max_results:
            break
        chapter_results = _find_text_in_chapter(
            document_name, chapter_file.name, query, case_sensitive, max_results - len(results)
        )
        results.extend(chapter_results)

    return results[:max_results]  # Ensure we don't exceed the limit


def _find_text_in_chapter(
    document_name: str, chapter_name: str, query: str, case_sensitive: bool = False, max_results: int = 100
) -> list[ParagraphDetail]:
    """Search for paragraphs containing specific text within a single chapter."""
    results = []
    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.exists():
        return results

    try:
        content = chapter_path.read_text(encoding="utf-8")
        paragraphs = _split_into_paragraphs(content)

        search_query = query if case_sensitive else query.lower()

        for i, paragraph in enumerate(paragraphs):
            if len(results) >= max_results:
                break
            search_text = paragraph if case_sensitive else paragraph.lower()
            if search_query in search_text:
                results.append(
                    ParagraphDetail(
                        document_name=document_name,
                        chapter_name=chapter_name,
                        paragraph_index_in_chapter=i,
                        content=paragraph,
                        word_count=_count_words(paragraph),
                    )
                )
    except Exception:
        pass

    return results


def _replace_text_in_document(
    document_name: str, text_to_find: str, replacement_text: str
) -> OperationStatus:
    """Replace all occurrences of text throughout all chapters of a document."""
    doc_path = _get_document_path(document_name)
    if not doc_path.exists():
        return OperationStatus(success=False, message=f"Document '{document_name}' not found.")

    chapter_files = _get_ordered_chapter_files(document_name)
    total_replacements = 0

    for chapter_file in chapter_files:
        result = _replace_text_in_chapter(document_name, chapter_file.name, text_to_find, replacement_text)
        if result.success and result.details:
            total_replacements += result.details.get("occurrences_replaced", 0)

    return OperationStatus(
        success=True,
        message=f"Replaced {total_replacements} occurrences of '{text_to_find}' with '{replacement_text}' in document '{document_name}'",
        details={"total_occurrences_replaced": total_replacements},
    )


def _replace_text_in_chapter(
    document_name: str, chapter_name: str, text_to_find: str, replacement_text: str
) -> OperationStatus:
    """Replace all occurrences of text within a specific chapter."""
    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.exists():
        return OperationStatus(
            success=False,
            message=f"Chapter '{chapter_name}' not found in document '{document_name}'",
        )

    try:
        content = chapter_path.read_text(encoding="utf-8")
        new_content = content.replace(text_to_find, replacement_text)
        replacements_made = content.count(text_to_find)

        chapter_path.write_text(new_content, encoding="utf-8")

        return OperationStatus(
            success=True,
            message=f"Replaced {replacements_made} occurrences of '{text_to_find}' in chapter '{chapter_name}'",
            details={"occurrences_replaced": replacements_made},
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error replacing text in chapter '{chapter_name}': {str(e)}",
        )


def _get_document_statistics(document_name: str) -> StatisticsReport | None:
    """Get comprehensive statistics for an entire document."""
    doc_path = _get_document_path(document_name)
    if not doc_path.exists():
        return None

    chapter_files = _get_ordered_chapter_files(document_name)
    total_words = 0
    total_paragraphs = 0

    for chapter_file in chapter_files:
        try:
            content = chapter_file.read_text(encoding="utf-8")
            paragraphs = _split_into_paragraphs(content)
            total_words += _count_words(content)
            total_paragraphs += len(paragraphs)
        except Exception:
            continue

    return StatisticsReport(
        scope=f"document:{document_name}",
        chapter_count=len(chapter_files),
        word_count=total_words,
        paragraph_count=total_paragraphs,
    )


def _get_chapter_statistics(document_name: str, chapter_name: str) -> StatisticsReport | None:
    """Get comprehensive statistics for a specific chapter."""
    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.exists():
        return None

    try:
        content = chapter_path.read_text(encoding="utf-8")
        paragraphs = _split_into_paragraphs(content)

        return StatisticsReport(
            scope=f"chapter:{document_name}/{chapter_name}",
            chapter_count=1,
            word_count=_count_words(content),
            paragraph_count=len(paragraphs),
        )
    except Exception:
        return None


def _perform_semantic_search(
    document_name: str,
    query_text: str,
    scope: str,
    chapter_name: str | None,
    similarity_threshold: float,
    max_results: int,
) -> list[SemanticSearchResult]:
    """Perform semantic search using Google Gemini embeddings with caching."""
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        log_structured_error(
            category=ErrorCategory.ERROR,
            message="GEMINI_API_KEY environment variable not set",
            context={"document_name": document_name, "scope": scope},
            operation="find_similar_text",
        )
        return []

    try:
        # Initialize cache and configure Gemini API
        model = "models/text-embedding-004"
        cache = EmbeddingCache(model_version=model)
        genai.configure(api_key=api_key)

        # Collect paragraphs and organize by chapter
        chapters_data = {}  # chapter_name -> {"paragraphs": [...], "content": {...}}
        all_paragraphs_data = []

        if scope == "document":
            # Search across all chapters
            doc_path = _get_document_path(document_name)
            if not doc_path.exists():
                return []

            chapter_files = _get_ordered_chapter_files(document_name)
            for chapter_file in chapter_files:
                try:
                    content = chapter_file.read_text(encoding="utf-8")
                    paragraphs = _split_into_paragraphs(content)
                    chapter_paragraphs = []

                    for i, paragraph in enumerate(paragraphs):
                        if paragraph.strip():  # Skip empty paragraphs
                            paragraph_data = {
                                "document_name": document_name,
                                "chapter_name": chapter_file.name,
                                "paragraph_index": i,
                                "content": paragraph,
                            }
                            chapter_paragraphs.append(paragraph_data)
                            all_paragraphs_data.append(paragraph_data)

                    if chapter_paragraphs:
                        chapters_data[chapter_file.name] = {
                            "paragraphs": chapter_paragraphs,
                            "content": {p["paragraph_index"]: p["content"] for p in chapter_paragraphs},
                        }

                except Exception:
                    continue

        elif scope == "chapter":
            # Search within specific chapter
            chapter_path = _get_chapter_path(document_name, chapter_name)
            if not chapter_path.exists():
                return []

            try:
                content = chapter_path.read_text(encoding="utf-8")
                paragraphs = _split_into_paragraphs(content)
                chapter_paragraphs = []

                for i, paragraph in enumerate(paragraphs):
                    if paragraph.strip():  # Skip empty paragraphs
                        paragraph_data = {
                            "document_name": document_name,
                            "chapter_name": chapter_name,
                            "paragraph_index": i,
                            "content": paragraph,
                        }
                        chapter_paragraphs.append(paragraph_data)
                        all_paragraphs_data.append(paragraph_data)

                if chapter_paragraphs:
                    chapters_data[chapter_name] = {
                        "paragraphs": chapter_paragraphs,
                        "content": {p["paragraph_index"]: p["content"] for p in chapter_paragraphs},
                    }

            except Exception:
                return []

        if not all_paragraphs_data:
            return []

        # Load cached embeddings and identify what needs to be embedded
        all_paragraph_embeddings = {}  # (chapter_name, paragraph_index) -> embedding
        paragraphs_to_embed = []  # Paragraphs that need new embeddings
        chapters_to_cache = {}  # chapter_name -> {paragraph_index: embedding}

        for chapter_name, chapter_data in chapters_data.items():
            # Try to load cached embeddings for this chapter
            cached_embeddings = cache.get_chapter_embeddings(document_name, chapter_name)

            chapter_needs_caching = {}
            for paragraph_data in chapter_data["paragraphs"]:
                paragraph_index = paragraph_data["paragraph_index"]

                if paragraph_index in cached_embeddings:
                    # Use cached embedding
                    all_paragraph_embeddings[(chapter_name, paragraph_index)] = cached_embeddings[
                        paragraph_index
                    ]
                else:
                    # Need to embed this paragraph
                    paragraphs_to_embed.append(paragraph_data)
                    chapter_needs_caching[paragraph_index] = paragraph_data["content"]

            if chapter_needs_caching:
                chapters_to_cache[chapter_name] = chapter_needs_caching

        # Prepare content for embedding (query + uncached paragraphs)
        texts_to_embed = [query_text] + [p["content"] for p in paragraphs_to_embed]

        # Get embeddings from Gemini (only for uncached content)
        if len(texts_to_embed) > 1:  # If there are paragraphs to embed
            response = genai.embed_content(
                model=model,
                content=texts_to_embed,
                task_type="retrieval_document",
            )

            embeddings = response["embedding"]
            query_embedding = np.array(embeddings[0])
            new_paragraph_embeddings = [np.array(emb) for emb in embeddings[1:]]

            # Store new embeddings and prepare for caching
            for i, paragraph_data in enumerate(paragraphs_to_embed):
                chapter_name = paragraph_data["chapter_name"]
                paragraph_index = paragraph_data["paragraph_index"]
                embedding = new_paragraph_embeddings[i]

                # Add to our working set
                all_paragraph_embeddings[(chapter_name, paragraph_index)] = embedding

                # Prepare for caching
                if chapter_name not in chapters_to_cache:
                    chapters_to_cache[chapter_name] = {}
                chapters_to_cache[chapter_name][paragraph_index] = embedding

        else:
            # Only need query embedding
            response = genai.embed_content(
                model=model,
                content=[query_text],
                task_type="retrieval_document",
            )
            query_embedding = np.array(response["embedding"][0])

        # Cache new embeddings by chapter
        for chapter_name, embeddings_to_cache in chapters_to_cache.items():
            if embeddings_to_cache and isinstance(list(embeddings_to_cache.values())[0], np.ndarray):
                # Convert embeddings dict to the format expected by cache
                paragraph_embeddings = {
                    idx: emb for idx, emb in embeddings_to_cache.items() if isinstance(emb, np.ndarray)
                }
                paragraph_contents = chapters_data[chapter_name]["content"]

                cache.store_chapter_embeddings(
                    document_name,
                    chapter_name,
                    paragraph_embeddings,
                    paragraph_contents,
                )

        # Calculate cosine similarities
        similarities = []
        for i, paragraph_data in enumerate(all_paragraphs_data):
            chapter_name = paragraph_data["chapter_name"]
            paragraph_index = paragraph_data["paragraph_index"]

            if (chapter_name, paragraph_index) in all_paragraph_embeddings:
                paragraph_embedding = all_paragraph_embeddings[(chapter_name, paragraph_index)]

                # Cosine similarity using numpy
                dot_product = np.dot(query_embedding, paragraph_embedding)
                query_norm = np.linalg.norm(query_embedding)
                paragraph_norm = np.linalg.norm(paragraph_embedding)
                similarity = dot_product / (query_norm * paragraph_norm)

                if similarity >= similarity_threshold:
                    similarities.append((i, similarity))

        # Sort by similarity score (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Limit results
        similarities = similarities[:max_results]

        # Build results
        results = []
        for idx, similarity_score in similarities:
            paragraph_data = all_paragraphs_data[idx]

            # Generate context snippet (surrounding paragraphs)
            context_snippet = _generate_context_snippet(paragraph_data["content"], all_paragraphs_data, idx)

            result = SemanticSearchResult(
                document_name=paragraph_data["document_name"],
                chapter_name=paragraph_data["chapter_name"],
                paragraph_index=paragraph_data["paragraph_index"],
                content=paragraph_data["content"],
                similarity_score=float(similarity_score),
                context_snippet=context_snippet,
            )
            results.append(result)

        return results

    except Exception as e:
        log_structured_error(
            category=ErrorCategory.ERROR,
            message=f"Error in semantic search implementation: {str(e)}",
            context={
                "document_name": document_name,
                "scope": scope,
                "query_text": query_text,
            },
            operation="find_similar_text",
        )
        return []


def _generate_context_snippet(
    target_content: str, all_paragraphs: list[dict], target_index: int
) -> str | None:
    """Generate context snippet with surrounding paragraphs."""
    try:
        # Get previous and next paragraphs from the same chapter
        target_chapter = all_paragraphs[target_index]["chapter_name"]
        chapter_paragraphs = [p for p in all_paragraphs if p["chapter_name"] == target_chapter]

        # Find the target paragraph within the chapter
        target_para_idx = None
        for i, p in enumerate(chapter_paragraphs):
            if p["content"] == target_content:
                target_para_idx = i
                break

        if target_para_idx is None:
            return None

        # Get surrounding context (previous and next paragraph if available)
        context_parts = []

        if target_para_idx > 0:
            context_parts.append(f"...{chapter_paragraphs[target_para_idx - 1]['content'][:100]}...")

        context_parts.append(f"[MATCH] {target_content}")

        if target_para_idx < len(chapter_paragraphs) - 1:
            context_parts.append(f"...{chapter_paragraphs[target_para_idx + 1]['content'][:100]}...")

        return " ".join(context_parts)
    except Exception:
        return None


# Export helper functions for use by other modules
find_text_in_document = _find_text_in_document
find_text_in_chapter = _find_text_in_chapter
replace_text_in_document = _replace_text_in_document
replace_text_in_chapter = _replace_text_in_chapter
get_document_statistics = _get_document_statistics
get_chapter_statistics = _get_chapter_statistics
