import argparse
import asyncio
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.mcp import MCPServerSSE
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel

# Import models from the server to ensure compatibility
from document_mcp.doc_tool_server import (
    ChapterContent,
    ChapterMetadata,
    DocumentInfo,
    FullDocumentContent,
    OperationStatus,
    ParagraphDetail,
    StatisticsReport,
)

# Suppress verbose HTTP logging from requests/urllib3
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)


# --- Configuration ---
def load_llm_config():
    """Load and configure the LLM model based on available environment variables."""
    load_dotenv()

    # Check for OpenAI API key first
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key and openai_api_key.strip():
        model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-4.1-mini")
        print(f"Using OpenAI model: {model_name}")
        return OpenAIModel(model_name=model_name)

    # Check for Gemini API keys
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_api_key and gemini_api_key.strip():
        model_name = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")
        print(f"Using Gemini model: {model_name}")
        return GeminiModel(model_name=model_name)

    # If no API keys are found, raise an error with helpful message
    raise ValueError(
        "No valid API key found in environment variables. "
        "Please set one of the following in your .env file:\n"
        "- OPENAI_API_KEY for OpenAI models\n"
        "- GEMINI_API_KEY for Google Gemini models\n"
        "\nOptionally, you can also set:\n"
        "- OPENAI_MODEL_NAME (default: gpt-4.1-mini)\n"
        "- GEMINI_MODEL_NAME (default: gemini-2.5-flash)"
    )


def check_api_keys_config():
    """
    Utility function to check which API keys are configured.
    Returns a dictionary with configuration status.
    """
    load_dotenv()

    config_status = {
        "openai_configured": False,
        "gemini_configured": False,
        "active_provider": None,
        "active_model": None,
    }

    # Check OpenAI
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key and openai_api_key.strip():
        config_status["openai_configured"] = True
        config_status["openai_model"] = os.environ.get(
            "OPENAI_MODEL_NAME", "gpt-4.1-mini"
        )
        if not config_status["active_provider"]:  # First priority
            config_status["active_provider"] = "openai"
            config_status["active_model"] = config_status["openai_model"]

    # Check Gemini
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_api_key and gemini_api_key.strip():
        config_status["gemini_configured"] = True
        config_status["gemini_model"] = os.environ.get(
            "GEMINI_MODEL_NAME", "gemini-2.5-flash"
        )
        if not config_status["active_provider"]:  # Second priority
            config_status["active_provider"] = "gemini"
            config_status["active_model"] = config_status["gemini_model"]

    return config_status


# --- Agent Response Model (for Pydantic AI Agent's structured output) ---
# Using imported models from the server to ensure compatibility

# Define the DetailsType union for the FinalAgentResponse
DetailsType = Union[
    List[DocumentInfo],  # from list_documents
    Optional[List[ChapterMetadata]],  # from list_chapters
    Optional[ChapterContent],  # from read_chapter_content
    Optional[ParagraphDetail],  # from read_paragraph_content
    Optional[FullDocumentContent],  # from read_full_document
    OperationStatus,  # from all write operations (create, delete, write, modify, append, replace)
    Optional[StatisticsReport],  # from get_chapter_statistics, get_document_statistics
    List[ParagraphDetail],  # from find_text_in_chapter, find_text_in_document
    Dict[str, Any],  # Allow a generic dictionary as a fallback
    None,  # Allow None for cases with no details
]


class FinalAgentResponse(BaseModel):
    """Defines the final structured output expected from the Pydantic AI agent."""

    summary: str
    details: Optional[DetailsType] = None
    error_message: Optional[str] = None


# --- System Prompt ---
SYSTEM_PROMPT = """You are an assistant that manages structured local Markdown documents using MCP tools.

**CORE CONSTRAINT:** You may call at most one MCP tool per user query. If the user's request requires multiple operations, process only the first step and return its result; the user will then provide a follow-up query for the next step.

**DOCUMENT STRUCTURE:**
A 'document' is a directory containing multiple 'chapter' files (Markdown .md files). Chapters are ordered alphanumerically by their filenames (e.g., '01-intro.md', '02-topic.md').

**CRITICAL TOOL SELECTION RULES:**
1. **When a user asks about "available documents", "all documents", "show documents", "list documents", "what documents"** - you MUST use ONLY the `list_documents` tool. This returns a List[DocumentInfo] objects. DO NOT use any other tool for listing documents.
2. **When a user wants to read the content/text of a specific document** - you MUST use `read_full_document` tool. This returns a FullDocumentContent object.
3. **When a user explicitly mentions a tool name** (e.g., "Use the get_document_statistics tool"), you MUST call that exact tool. Do not substitute or use a different tool.
4. **For statistics requests** (words like "statistics", "stats", "word count", "paragraph count"), you MUST use `get_document_statistics` or `get_chapter_statistics` tools. NEVER use other tools for statistics.
5. **For search requests** (words like "find", "search", "locate"), use `find_text_in_document` or `find_text_in_chapter` tools.
6. **For content reading**, use `read_chapter_content`, `read_full_document`, or `read_paragraph_content` tools.

**IMPORTANT**: If the user query contains any variation of "show", "list", "get", "available", or "all" combined with "documents", you MUST call `list_documents()` and return the result as a list in the details field. Never call `read_full_document` for listing operations.

**OPERATION WORKFLOW:**
When a user asks for an operation:
1. Identify the correct tool by understanding the user's intent and matching it to the tool's description
2. Determine the necessary parameters for the chosen tool based on its description and the user's query
3. Chapter names should include the .md extension (e.g., "01-introduction.md")
4. Before invoking the tool, briefly explain your reasoning: why this tool and these parameters
5. After receiving results, analyze what you found and determine if further actions are needed
6. Formulate a response conforming to the `FinalAgentResponse` model, ensuring the `details` field contains the direct and complete output from the invoked tool

**PRE-OPERATION CHECKS:**
- If the user asks to list/show/get available documents, call `list_documents()` FIRST
- If the user asks to read a specific document's content, verify the document exists by calling `list_documents()` first, then call `read_full_document()`
- If the user's request is a search request (keywords: find, search, locate), skip document enumeration and directly call the appropriate search tool
- If the user's request is to create, add, update, modify, or delete a document or chapter, skip document enumeration and directly call the corresponding tool
- Verify a target document exists before any further per-document operation
- For operations across all documents, enumerate with `list_documents()` prior to acting on each
- To read content, initially call `get_document_statistics()` to get a summary; only call `read_full_document()` after evaluating summary metrics

**TOOL DESCRIPTIONS AND USAGE:**
The available tools (like `list_documents`, `create_document`, `list_chapters`, `read_chapter_content`, `write_chapter_content`, `get_document_statistics`, `find_text_in_document`, etc.) will be discovered from an MCP server named 'DocumentManagementTools'. For detailed information on how to use each tool, including its parameters and expected behavior, refer to the description of the tool itself.

**KEY TOOLS EXAMPLES:**
- `list_documents()`: Lists all available documents (directories)
- `create_document(document_name="my_book")`: Creates a new directory for a document
- `list_chapters(document_name="my_book")`: Lists all chapters (e.g., "01-intro.md", "02-body.md") in "my_book"
- `read_chapter_content(document_name="my_book", chapter_name="01-intro.md")`: Reads the full content of a specific chapter
- `read_full_document(document_name="my_book")`: Reads all chapters of "my_book" and concatenates their content
- `write_chapter_content(document_name="my_book", chapter_name="01-intro.md", new_content="# New Chapter Content...")`: Overwrites an entire chapter. Creates the chapter if it doesn't exist within an existing document
- `modify_paragraph_content(document_name="my_book", chapter_name="01-intro.md", paragraph_index=0, new_paragraph_content="Revised first paragraph.", mode="replace")`: Modifies a specific paragraph. Other modes include "insert_before", "insert_after", "delete"
- `append_paragraph_to_chapter(document_name="my_book", chapter_name="01-intro.md", paragraph_content="This is a new paragraph at the end.")`
- `replace_text_in_chapter(document_name="my_book", chapter_name="01-intro.md", text_to_find="old_term", replacement_text="new_term")`: Replaces text within one chapter
- `replace_text_in_document(document_name="my_book", text_to_find="global_typo", replacement_text="corrected_text")`: Replaces text across all chapters of "my_book"
- `get_chapter_statistics(document_name="my_book", chapter_name="01-intro.md")`: Gets word/paragraph count for a chapter
- `get_document_statistics(document_name="my_book")`: Gets aggregate word/paragraph/chapter counts for "my_book"
- `find_text_in_chapter(...)` and `find_text_in_document(...)`: For locating text

**STATISTICS TOOL USAGE:**
- When asked for "statistics" or "stats", you **must** use the `get_document_statistics` or `get_chapter_statistics` tools
- The response for these tools will be a `StatisticsReport`. Ensure this is what you return in the `details` field
- Do **not** use other tools like `find_text_in_document` or `read_paragraph_content` when asked for statistics
- If a user says "Use the get_document_statistics tool", you MUST call `get_document_statistics` and nothing else

**DOCUMENT OPERATION SCENARIOS:**

**CRITICAL DISTINCTION - LISTING vs READING DOCUMENTS:**

**LISTING DOCUMENTS** (use `list_documents` tool):
- User wants to see what documents exist/are available
- Keywords: "show", "list", "get", "what", "available", "all documents"
- Examples: "Show me all available documents", "List all documents", "What documents do you have?"
- Returns: List[DocumentInfo] - just the names and metadata of documents

**READING DOCUMENT CONTENT** (use `read_full_document` tool):
- User wants to see the actual content/text inside a specific document
- Keywords: "read", "content", "text", "what's in", combined with a specific document name
- Examples: "Read document X", "Show me the content of document Y", "What's in document Z?"
- Returns: FullDocumentContent - the actual text content of chapters

**NEVER confuse directory names or folder names with document names when the user is asking for a list of available documents.**

**SINGLE DOCUMENT CONTENT ACCESS:**
If a user asks to access or process the *content* of multiple chapters within a document (e.g., "read all chapters of 'my_book'"):
1. Use `read_full_document(document_name="my_book")`. The `details` field will be a `FullDocumentContent` object
2. Your `summary` should state that the full document content has been retrieved

**ALL DOCUMENTS CONTENT ACCESS:**
If a user asks to get the content of *all documents* (i.e., all chapters from all documents):
1. **Mandatory First Step**: Call `list_documents()` to identify all document names
2. **Mandatory Second Step**: For EACH document identified, call `read_full_document(document_name: str)` to retrieve its complete content (all its chapters)
3. **Result Consolidation**: Collect ALL `FullDocumentContent` objects. The `details` field of your `FinalAgentResponse` MUST be a list of these `FullDocumentContent` objects (i.e., `List[FullDocumentContent]`)
4. **Summary**: Your `summary` must clearly state that the content of all documents has been retrieved. A response just listing document names is INCOMPLETE if content was requested

**ERROR HANDLING:**
- Do not assume a document or chapter exists unless listed by `list_documents`, `list_chapters` or confirmed by the user recently
- If a tool call fails or an entity is not found, this should be reflected in the `summary` and potentially in the `error_message` field of the `FinalAgentResponse`
- The `details` field (which should be an `OperationStatus` model in case of errors from write/modify tools, or None for read tools) should reflect the tool's direct output
- If a search tool like `find_text_in_chapter` or `find_text_in_document` returns no results, your `summary` should explicitly state that the queried text was not found and clearly mention the specific text that was searched for

Follow the user's instructions carefully and to the letter without asking for clarification or further instructions unless absolutely necessary for tool parameterization.
"""


# --- Agent Setup and Processing Logic ---
async def initialize_agent_and_mcp_server() -> (
    tuple[Agent[FinalAgentResponse], MCPServerSSE]
):
    """Initializes the Pydantic AI agent and its MCP server configuration."""
    try:
        llm = load_llm_config()
    except ValueError as e:
        print(f"Error loading LLM config: {e}", file=sys.stderr)
        raise

    # Configuration for HTTP SSE server
    server_host = os.environ.get("MCP_SERVER_HOST", "localhost")
    server_port = int(os.environ.get("MCP_SERVER_PORT", "3001"))
    server_url = f"http://{server_host}:{server_port}/sse"

    mcp_server = MCPServerSSE(server_url)

    agent: Agent[FinalAgentResponse] = Agent(
        llm,
        mcp_servers=[mcp_server],
        system_prompt=SYSTEM_PROMPT,
        output_type=FinalAgentResponse,
    )
    return agent, mcp_server


async def process_single_user_query(
    agent: Agent[FinalAgentResponse], user_query: str
) -> Optional[FinalAgentResponse]:
    """Processes a single user query using the provided agent and returns the structured response."""
    try:
        # Ensure we have a valid event loop
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            # Create a new event loop if the current one is closed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Add timeout to prevent hanging
        run_result: AgentRunResult[FinalAgentResponse] = await asyncio.wait_for(
            agent.run(user_query),
            timeout=45.0,  # 45 second timeout for production use
        )

        if run_result and run_result.output:
            return run_result.output
        elif run_result and run_result.error_message:
            return FinalAgentResponse(
                summary=f"Agent error: {run_result.error_message}",
                details=None,
                error_message=run_result.error_message,
            )
        else:
            return None
    except asyncio.TimeoutError:
        print("Agent query timed out after 45 seconds", file=sys.stderr)
        return FinalAgentResponse(
            summary="Query timed out after 45 seconds",
            details=None,
            error_message="Timeout error",
        )
    except asyncio.CancelledError:
        print("Agent query was cancelled", file=sys.stderr)
        return FinalAgentResponse(
            summary="Query was cancelled", details=None, error_message="Cancelled error"
        )
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            print(f"Event loop closed during query processing: {e}", file=sys.stderr)
            # Try to recover by creating a new event loop
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return FinalAgentResponse(
                    summary="Event loop was closed but recovered",
                    details=None,
                    error_message=None,
                )
            except Exception:
                return FinalAgentResponse(
                    summary="Event loop closed and could not recover",
                    details=None,
                    error_message="Event loop closed",
                )
        elif "generator didn't stop after athrow" in str(e):
            print(
                f"Generator cleanup error during query processing: {e}", file=sys.stderr
            )
            return FinalAgentResponse(
                summary="Generator cleanup error during processing",
                details=None,
                error_message="Generator cleanup error",
            )
        else:
            print(f"Runtime error during agent query processing: {e}", file=sys.stderr)
            return FinalAgentResponse(
                summary=f"Runtime error during query processing: {e}",
                details=None,
                error_message=str(e),
            )
    except Exception as e:
        print(f"Error during agent query processing: {e}", file=sys.stderr)

        # Handle specific API timeout/connection errors more gracefully
        error_msg = str(e)
        if "ReadTimeout" in error_msg or "Event loop is closed" in error_msg:
            summary = "API connection timeout or network error occurred"
        elif "test_api_key_placeholder" in os.environ.get("GEMINI_API_KEY", ""):
            summary = "API authentication error (placeholder key used)"
        else:
            summary = f"Exception during query processing: {e}"

        return FinalAgentResponse(summary=summary, details=None, error_message=str(e))


# --- Main Agent Interactive Loop ---
async def main():
    """Initializes and runs the Pydantic AI agent."""
    parser = argparse.ArgumentParser(description="Simple Document Agent")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="A single query to process. If provided, the agent will run non-interactively.",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Check API key configuration and exit.",
    )
    args = parser.parse_args()

    # Handle configuration check
    if args.check_config:
        config = check_api_keys_config()
        print("=== API Configuration Status ===")
        print(f"OpenAI configured: {'✓' if config['openai_configured'] else '✗'}")
        if config["openai_configured"]:
            print(f"  Model: {config.get('openai_model', 'N/A')}")
        print(f"Gemini configured: {'✓' if config['gemini_configured'] else '✗'}")
        if config["gemini_configured"]:
            print(f"  Model: {config.get('gemini_model', 'N/A')}")
        print()
        if config["active_provider"]:
            print(f"Active provider: {config['active_provider'].upper()}")
            print(f"Active model: {config['active_model']}")
        else:
            print("No API keys configured!")
            print(
                "Please set either OPENAI_API_KEY or GEMINI_API_KEY in your .env file."
            )
        return

    try:
        agent, mcp_server = await initialize_agent_and_mcp_server()
    except (ValueError, FileNotFoundError):
        # Errors from initialize_agent_and_mcp_server are already printed to stderr
        sys.exit(1)
    except Exception as e:  # Catch any other unexpected errors during init
        print(f"Critical error during agent initialization: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        async with agent.run_mcp_servers():
            # Single query mode
            if args.query:
                final_response = await process_single_user_query(agent, args.query)
                if final_response:
                    sys.stdout.write(final_response.model_dump_json(indent=2))
                    sys.stdout.write("\n")
                else:
                    # Handle case where no response is returned
                    # Output a structured error response for consistency
                    error_resp = FinalAgentResponse(
                        summary="Failed to get a final response from the agent for the provided query.",
                        details=None,
                        error_message="No specific error message captured at top level, see stderr for details.",
                    )
                    sys.stdout.write(error_resp.model_dump_json(indent=2))
                    sys.stdout.write("\n")

            # Interactive mode (either explicitly requested or default when no query)
            elif args.interactive or not args.query:
                print("MCP Server connected via HTTP SSE.")
                print("\n--- Simple Document Agent --- ")
                print("Ask me to manage documents (directories of chapters).")
                print("Type 'exit' to quit.")

                while True:
                    user_query = input("\nUser Query: ")
                    if user_query.lower() == "exit":
                        break
                    if not user_query.strip():
                        continue

                    final_response = await process_single_user_query(agent, user_query)

                    if final_response:
                        print("\n--- Agent Response ---")
                        print(f"Summary: {final_response.summary}")

                        if isinstance(final_response.details, list):
                            if not final_response.details:
                                print("Details: [] (Empty list)")
                            else:
                                item_type = type(final_response.details[0])
                                print(
                                    f"\n--- Details (List of {item_type.__name__}) ---"
                                )
                                for item_idx, item_detail in enumerate(
                                    final_response.details
                                ):
                                    print(f"Item {item_idx + 1}:")
                                    if hasattr(
                                        item_detail, "model_dump"
                                    ):  # Check if Pydantic model
                                        print(item_detail.model_dump(exclude_none=True))
                                    else:
                                        print(item_detail)
                        elif hasattr(
                            final_response.details, "model_dump"
                        ):  # Check if Pydantic model
                            print("\n--- Details ---")
                            print(final_response.details.model_dump(exclude_none=True))
                        elif final_response.details is not None:
                            print(f"Details: {final_response.details}")
                        else:
                            print("Details: None")

                        if final_response.error_message:
                            print(f"Error Message: {final_response.error_message}")
                    # If final_response is None, process_single_user_query already printed an error to stderr
            else:
                # No arguments provided, show help
                parser.print_help()
                print("\nExample usage:")
                print(
                    '  python src/agents/simple_agent.py --query "List all documents"'
                )
                print("  python src/agents/simple_agent.py --interactive")

    except KeyboardInterrupt:
        if (
            args.interactive or not args.query
        ):  # Only print this message in interactive mode
            print("\nUser requested exit. Shutting down...")
    except Exception as e:
        print(f"An unexpected error occurred in the agent: {e}", file=sys.stderr)
    finally:
        if (
            args.interactive or not args.query
        ):  # Only print shutdown message in interactive mode
            print("Agent has shut down.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # This handles Ctrl+C if it happens outside the main's try/except,
        # e.g., during asyncio.run itself or if main() exits due to unhandled KeyboardInterrupt.
        print(
            "\nExiting (Keyboard Interrupt detected outside main loop)...",
            file=sys.stderr,
        )
    except Exception as e:
        # This will catch errors during asyncio.run(main()) itself if any
        print(f"Critical error during agent startup or shutdown: {e}", file=sys.stderr)
        sys.exit(1)  # Ensure non-zero exit code for critical failures
