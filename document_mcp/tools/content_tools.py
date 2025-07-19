"""Content-related MCP tools for document management.

This module provides unified content access tools that work across different
scopes (document, chapter, paragraph) with a consistent interface.
"""

from typing import Any

from ..batch import register_batchable_operation
from ..helpers import _count_words
from ..helpers import _get_chapter_path
from ..helpers import _get_document_path
from ..helpers import _get_ordered_chapter_files
from ..helpers import _split_into_paragraphs
from ..helpers import validate_chapter_name
from ..helpers import validate_content
from ..helpers import validate_document_name
from ..helpers import validate_paragraph_index
from ..helpers import validate_search_query
from ..logger_config import ErrorCategory
from ..logger_config import log_mcp_call
from ..logger_config import log_structured_error
from ..models import OperationStatus
from ..models import ParagraphDetail
from ..models import StatisticsReport
from ..utils.decorators import auto_snapshot


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
        # Import models that aren't imported at module level
        from ..helpers import _get_chapter_metadata
        from ..models import ChapterContent
        from ..models import FullDocumentContent
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

        # Scope-based dispatch with inline implementations
        try:
            if scope == "document":
                # Read complete document with all chapters (inline implementation)
                doc_path = _get_document_path(document_name)
                if not doc_path.exists():
                    return None

                chapters = []
                total_word_count = 0
                total_paragraph_count = 0

                for chapter_file in _get_ordered_chapter_files(document_name):
                    try:
                        content = chapter_file.read_text(encoding="utf-8")
                        metadata = _get_chapter_metadata(document_name, chapter_file)
                        if metadata:
                            chapter_content = ChapterContent(
                                document_name=document_name,
                                chapter_name=chapter_file.name,
                                content=content,
                                word_count=metadata.word_count,
                                paragraph_count=metadata.paragraph_count,
                                last_modified=metadata.last_modified,
                            )
                            chapters.append(chapter_content)
                            total_word_count += metadata.word_count
                            total_paragraph_count += metadata.paragraph_count
                    except Exception as e:
                        # Log the exception to help debug
                        log_structured_error(
                            category=ErrorCategory.WARNING,
                            message=f"Error reading chapter {chapter_file.name}: {str(e)}",
                            context={"document_name": document_name, "chapter_file": str(chapter_file)},
                            operation="read_content",
                        )
                        continue

                result = FullDocumentContent(
                    document_name=document_name,
                    chapters=chapters,
                    total_word_count=total_word_count,
                    total_paragraph_count=total_paragraph_count,
                )
                return result

            elif scope == "chapter":
                # Read specific chapter content (inline implementation)
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
                # Read specific paragraph content (inline implementation)
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

        # Scope-based dispatch to helper functions
        try:
            if scope == "document":
                result = _find_text_in_document(
                    document_name, search_text, case_sensitive
                )
                return result if result else []

            elif scope == "chapter":
                result = _find_text_in_chapter(
                    document_name, chapter_name, search_text, case_sensitive
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
    @register_batchable_operation("replace_text")
    @log_mcp_call
    @auto_snapshot("replace_text")
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
                        chapter_count=None  # Exclude chapter_count for chapter scope
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


# Helper functions for content tools
def _find_text_in_document(
    document_name: str, query: str, case_sensitive: bool = False
) -> list[ParagraphDetail]:
    """Search for paragraphs containing specific text across all chapters."""
    results = []
    doc_path = _get_document_path(document_name)
    if not doc_path.exists():
        return results

    chapter_files = _get_ordered_chapter_files(document_name)
    for chapter_file in chapter_files:
        chapter_results = _find_text_in_chapter(document_name, chapter_file.name, query, case_sensitive)
        results.extend(chapter_results)

    return results


def _find_text_in_chapter(
    document_name: str, chapter_name: str, query: str, case_sensitive: bool = False
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
        return OperationStatus(
            success=False,
            message=f"Document '{document_name}' not found."
        )

    chapter_files = _get_ordered_chapter_files(document_name)
    total_replacements = 0

    for chapter_file in chapter_files:
        result = _replace_text_in_chapter(document_name, chapter_file.name, text_to_find, replacement_text)
        if result.success and result.details:
            total_replacements += result.details.get("occurrences_replaced", 0)

    return OperationStatus(
        success=True,
        message=f"Replaced {total_replacements} occurrences of '{text_to_find}' with '{replacement_text}' in document '{document_name}'",
        details={"total_occurrences_replaced": total_replacements}
    )


def _replace_text_in_chapter(
    document_name: str, chapter_name: str, text_to_find: str, replacement_text: str
) -> OperationStatus:
    """Replace all occurrences of text within a specific chapter."""
    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.exists():
        return OperationStatus(
            success=False,
            message=f"Chapter '{chapter_name}' not found in document '{document_name}'"
        )

    try:
        content = chapter_path.read_text(encoding="utf-8")
        new_content = content.replace(text_to_find, replacement_text)
        replacements_made = content.count(text_to_find)

        chapter_path.write_text(new_content, encoding="utf-8")

        return OperationStatus(
            success=True,
            message=f"Replaced {replacements_made} occurrences of '{text_to_find}' in chapter '{chapter_name}'",
            details={"occurrences_replaced": replacements_made}
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error replacing text in chapter '{chapter_name}': {str(e)}"
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


# Export helper functions for use by other modules
find_text_in_document = _find_text_in_document
find_text_in_chapter = _find_text_in_chapter
replace_text_in_document = _replace_text_in_document
replace_text_in_chapter = _replace_text_in_chapter
get_document_statistics = _get_document_statistics
get_chapter_statistics = _get_chapter_statistics
