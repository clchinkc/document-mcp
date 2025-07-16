"""Content-related MCP tools for document management.

This module provides unified content access tools that work across different
scopes (document, chapter, paragraph) with a consistent interface.
"""

from typing import Any

from ..batch import register_batchable_operation
from ..logger_config import ErrorCategory
from ..logger_config import log_mcp_call
from ..logger_config import log_structured_error
from ..utils.validation import validate_chapter_name as _validate_chapter_name
from ..utils.validation import validate_content as _validate_content
from ..utils.validation import validate_document_name as _validate_document_name
from ..utils.validation import validate_paragraph_index as _validate_paragraph_index
from ..utils.validation import validate_search_query as _validate_search_query


def register_content_tools(mcp_server):
    """Register all content-related tools with the MCP server."""

    @mcp_server.tool()
    @register_batchable_operation("read_content")
    @log_mcp_call
    def read_content(
        document_name: str,
        scope: str = "document",  # "document", "chapter", "paragraph"
        chapter_name: str | None = None,
        paragraph_index: int | None = None,
    ) -> dict[str, Any] | None:
        """Unified content reading with scope-based targeting.

        This tool consolidates three separate reading operations into a single, scope-based
        interface. It can read complete documents, individual chapters, or specific paragraphs
        depending on the scope parameter, providing a consistent API for all content access.

        Parameters:
            document_name (str): Name of the document directory to read from
            scope (str): Reading scope determining what content to retrieve:
                - "document": Read complete document with all chapters (default)
                - "chapter": Read specific chapter content and metadata
                - "paragraph": Read specific paragraph content and metadata
            chapter_name (Optional[str]): Required for "chapter" and "paragraph" scopes.
                Must be valid .md filename (e.g., "01-introduction.md")
            paragraph_index (Optional[int]): Required for "paragraph" scope.
                Zero-indexed position of paragraph within the chapter (≥0)

        Returns:
            Optional[Dict[str, Any]]: Content object matching the requested scope, None if not found.

            For scope="document":
                FullDocumentContent with fields:
                - document_name (str): Name of the document
                - chapters (List[ChapterContent]): Ordered list of all chapter contents
                - total_word_count (int): Sum of words across all chapters
                - total_paragraph_count (int): Sum of paragraphs across all chapters

            For scope="chapter":
                ChapterContent with fields:
                - document_name (str): Name of the parent document
                - chapter_name (str): Filename of the chapter
                - content (str): Full raw text content of the chapter
                - word_count (int): Number of words in the chapter
                - paragraph_count (int): Number of paragraphs in the chapter
                - last_modified (datetime): Timestamp of last file modification

            For scope="paragraph":
                ParagraphDetail with fields:
                - document_name (str): Name of the parent document
                - chapter_name (str): Name of the chapter file
                - paragraph_index_in_chapter (int): Zero-indexed position within the chapter
                - content (str): Full text content of the paragraph
                - word_count (int): Number of words in the paragraph

        Example Usage:
            ```json
            // Read full document
            {
                "name": "read_content",
                "arguments": {
                    "document_name": "My Book",
                    "scope": "document"
                }
            }

            // Read specific chapter
            {
                "name": "read_content",
                "arguments": {
                    "document_name": "My Book",
                    "scope": "chapter",
                    "chapter_name": "01-introduction.md"
                }
            }

            // Read specific paragraph
            {
                "name": "read_content",
                "arguments": {
                    "document_name": "My Book",
                    "scope": "paragraph",
                    "chapter_name": "01-introduction.md",
                    "paragraph_index": 0
                }
            }
            ```
        """
        # Import here to avoid circular imports
        from ..doc_tool_server import read_chapter_content
        from ..doc_tool_server import read_full_document
        from ..doc_tool_server import read_paragraph_content

        # Validate document name
        is_valid_doc, doc_error = _validate_document_name(document_name)
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
            is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
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
            is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
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
            is_valid_index, index_error = _validate_paragraph_index(paragraph_index)
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

        # Scope-based dispatch to existing internal functions
        try:
            if scope == "document":
                result = read_full_document(document_name)
                return result.model_dump() if result else None

            elif scope == "chapter":
                result = read_chapter_content(document_name, chapter_name)
                return result.model_dump() if result else None

            elif scope == "paragraph":
                result = read_paragraph_content(
                    document_name, chapter_name, paragraph_index
                )
                return result.model_dump() if result else None

        except Exception as e:
            log_structured_error(
                category=ErrorCategory.SYSTEM,
                message=f"Error reading content with scope {scope}: {str(e)}",
                context={
                    "document_name": document_name,
                    "scope": scope,
                    "chapter_name": chapter_name,
                    "paragraph_index": paragraph_index,
                },
                operation="read_content",
            )
            return None

    @mcp_server.tool()
    @register_batchable_operation("find_text")
    @log_mcp_call
    def find_text(
        document_name: str,
        search_text: str,
        scope: str = "document",  # "document", "chapter"
        chapter_name: str | None = None,
        case_sensitive: bool = False,
    ) -> list[dict[str, Any]] | None:
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
        from ..doc_tool_server import find_text_in_chapter
        from ..doc_tool_server import find_text_in_document

        # Validate document name
        is_valid_doc, doc_error = _validate_document_name(document_name)
        if not is_valid_doc:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Invalid document name: {doc_error}",
                context={"document_name": document_name, "scope": scope},
                operation="find_text",
            )
            return None

        # Validate search text
        is_valid_search, search_error = _validate_search_query(search_text)
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
            is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
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

        # Scope-based dispatch to existing functions
        try:
            if scope == "document":
                result = find_text_in_document(
                    document_name, search_text, case_sensitive
                )
                return [r.model_dump() for r in result] if result else []

            elif scope == "chapter":
                result = find_text_in_chapter(
                    document_name, chapter_name, search_text, case_sensitive
                )
                return [r.model_dump() for r in result] if result else []

        except Exception as e:
            log_structured_error(
                category=ErrorCategory.SYSTEM,
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
    @register_batchable_operation("replace_text")
    @log_mcp_call
    def replace_text(
        document_name: str,
        find_text: str,
        replace_text: str,
        scope: str = "document",  # "document", "chapter"
        chapter_name: str | None = None,
    ) -> dict[str, Any] | None:
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

            For scope="document": Results from replace_text_in_document
            For scope="chapter": Results from replace_text_in_chapter

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
        # Import here to avoid circular imports
        from ..doc_tool_server import auto_snapshot
        from ..doc_tool_server import replace_text_in_chapter
        from ..doc_tool_server import replace_text_in_document

        # Apply auto snapshot decorator functionality
        @auto_snapshot("replace_text")
        def _replace_text_with_snapshot():
            # Validate document name
            is_valid_doc, doc_error = _validate_document_name(document_name)
            if not is_valid_doc:
                log_structured_error(
                    category=ErrorCategory.ERROR,
                    message=f"Invalid document name: {doc_error}",
                    context={"document_name": document_name, "scope": scope},
                    operation="replace_text",
                )
                return None

            # Validate find and replace text
            is_valid_find, find_error = _validate_search_query(find_text)
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

            is_valid_replace, replace_error = _validate_content(replace_text)
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
                is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
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

            # Scope-based dispatch to existing functions
            try:
                if scope == "document":
                    result = replace_text_in_document(
                        document_name, find_text, replace_text
                    )
                    return result.model_dump() if result else None

                elif scope == "chapter":
                    result = replace_text_in_chapter(
                        document_name, chapter_name, find_text, replace_text
                    )
                    return result.model_dump() if result else None

            except Exception as e:
                log_structured_error(
                    category=ErrorCategory.SYSTEM,
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

        return _replace_text_with_snapshot()

    @mcp_server.tool()
    @register_batchable_operation("get_statistics")
    @log_mcp_call
    def get_statistics(
        document_name: str,
        scope: str = "document",  # "document", "chapter"
        chapter_name: str | None = None,
    ) -> dict[str, Any] | None:
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
        # Import here to avoid circular imports
        from ..doc_tool_server import get_chapter_statistics
        from ..doc_tool_server import get_document_statistics

        # Validate document name
        is_valid_doc, doc_error = _validate_document_name(document_name)
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
            is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
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

        # Scope-based dispatch to existing functions
        try:
            if scope == "document":
                result = get_document_statistics(document_name)
                return result.model_dump() if result else None

            elif scope == "chapter":
                result = get_chapter_statistics(document_name, chapter_name)
                if result:
                    # For chapter scope, exclude chapter_count field
                    data = result.model_dump()
                    data.pop("chapter_count", None)  # Remove chapter_count if present
                    return data
                return None

        except Exception as e:
            log_structured_error(
                category=ErrorCategory.SYSTEM,
                message=f"Error getting statistics with scope {scope}: {str(e)}",
                context={
                    "document_name": document_name,
                    "scope": scope,
                    "chapter_name": chapter_name,
                },
                operation="get_statistics",
            )
            return None
