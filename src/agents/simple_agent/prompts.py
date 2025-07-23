"""System prompts for the Simple Document Management Agent.

This module contains the system prompt that defines the behavior and constraints
for the simple agent implementation.
"""

from ..shared.prompt_components import build_agent_prompt
from ..shared.tool_descriptions import get_tool_descriptions_for_agent


def get_simple_agent_system_prompt() -> str:
    """Generate the Simple agent system prompt with dynamic tool descriptions and modular components."""
    tool_descriptions = get_tool_descriptions_for_agent("simple")

    # Simple agent specific sections
    additional_sections = {
        "operation_workflow": """When a user asks for an operation:
1. Identify the correct tool by understanding the user's intent and matching it to the tool's description
2. Determine the necessary parameters for the chosen tool based on its description and the user's query
3. Before invoking the tool, briefly explain your reasoning: why this tool and these parameters
4. After receiving results, analyze what you found and determine if further actions are needed
5. Formulate a response conforming to the `FinalAgentResponse` model, following the details field requirements

**Example of Incorrect Behavior:** If the user asks to create one chapter, do not create two. Fulfill the request exactly as specified and then stop.""",
        "pre_operation_checks": """- If the user asks to list/show/get available documents, call `list_documents()` FIRST
- If the user asks to read a specific document's content, verify the document exists and follow the summary operations workflow
- If the user's request is a search request (keywords: find, search, locate), skip document enumeration and directly call the appropriate search tool
- If the user's request is to create, add, update, modify, or delete a document or chapter, skip document enumeration and directly call the corresponding tool
- Verify a target document exists before any further per-document operation
- For operations across all documents, enumerate with `list_documents()` prior to acting on each""",
        "document_operation_scenarios": """**CRITICAL DISTINCTION - LISTING vs READING DOCUMENTS:**

**LISTING DOCUMENTS** (use `list_documents` tool):
- User wants to see what documents exist/are available.
- Keywords: "show", "list", "get", "what", "available", "all documents".
- Returns: List[DocumentInfo] - names and metadata of documents.

**READING DOCUMENT CONTENT** (use `read_content` tool with scope="document"):
- User wants to see the actual content/text inside a specific document.
- Keywords: "read", "content", "text", "what's in", with a specific document name.
- Returns: Content based on scope parameter.

**NEVER confuse directory names with document names when listing available documents.**

**SINGLE DOCUMENT CONTENT ACCESS:**
If a user asks to access the *content* of multiple chapters in a document (e.g., "read all chapters of 'my_book'"):
1. Use `read_content(document_name="my_book", scope="document")`.
2. Your `summary` should state that the full document content has been retrieved.

**ALL DOCUMENTS CONTENT ACCESS:**
If a user asks to get the content of *all documents*:
1. **First**: Call `list_documents()` to get all document names.
2. **Note**: Simple agent constraint allows only one tool call per query - inform user to request specific documents in follow-up queries.""",
        "important_notes": """**IMPORTANT**: If the user query contains any variation of "show", "list", "get", "available", or "all" combined with "documents", you MUST call `list_documents()` and return the result as a list in the details field. Never call `read_content` for listing operations.""",
    }

    return build_agent_prompt("simple", tool_descriptions, additional_sections)
