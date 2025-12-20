import asyncio
import logging
import sys
import time
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.mcp import MCPServerStdio

# Import models from the server to ensure compatibility
from src.agents.shared.error_handling import RetryManager
from src.agents.shared.performance_metrics import AgentPerformanceMetrics
from src.agents.shared.performance_metrics import PerformanceMetricsCollector

# Suppress verbose HTTP logging
for logger_name in ["httpx", "httpcore", "urllib3", "requests"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


from src.agents.shared.cli import handle_config_check
from src.agents.shared.cli import parse_simple_agent_args
from src.agents.shared.cli import setup_document_root

# --- Configuration ---
from src.agents.shared.config import MCP_SERVER_CMD
from src.agents.shared.config import load_llm_config
from src.agents.simple_agent.prompts import get_simple_agent_system_prompt

# --- MCP Tool Response Extraction ---


def extract_mcp_tool_responses(agent_result: AgentRunResult) -> dict[str, Any]:
    """Extract MCP tool responses from agent execution result.

    Uses the same working pattern as the ReAct Agent to ensure consistency.
    """
    tool_responses = {}

    if hasattr(agent_result, "all_messages"):
        messages = agent_result.all_messages()

        for message in messages:
            if hasattr(message, "parts"):
                for part in message.parts:
                    # Check for tool returns (ToolReturnPart) - same pattern as ReAct Agent
                    if (
                        hasattr(part, "tool_name")
                        and hasattr(part, "content")
                        and type(part).__name__ == "ToolReturnPart"
                    ):
                        tool_name = part.tool_name
                        tool_content = part.content

                        # Store the actual MCP tool response data
                        if isinstance(tool_content, list):
                            tool_responses[tool_name] = {"documents": tool_content}
                        elif isinstance(tool_content, dict):
                            tool_responses[tool_name] = tool_content
                        else:
                            tool_responses[tool_name] = {"content": tool_content}

    return tool_responses


# --- Agent Response Model (for Pydantic AI Agent's structured output) ---
# Using imported models from the server to ensure compatibility


class LLMOnlyResponse(BaseModel):
    """Defines what the LLM should generate - only the summary field."""

    summary: str
    error_message: str | None = None


class FinalAgentResponse(BaseModel):
    """Defines the final structured output returned to users."""

    summary: str
    details: str | None = None  # Programmatically populated with MCP tool responses
    error_message: str | None = None


# --- System Prompt ---
# System prompt is now imported from separate module


# --- Agent Setup and Processing Logic ---
_retry_manager = RetryManager()


async def initialize_agent_and_mcp_server() -> tuple[Agent[LLMOnlyResponse], MCPServerStdio]:
    """Initializes the Pydantic AI agent and its MCP server configuration."""
    try:
        llm = await _retry_manager.execute_with_retry(load_llm_config)
    except ValueError as e:
        print(f"Error loading LLM config: {e}", file=sys.stderr)
        raise

    from src.agents.shared.config import prepare_mcp_server_environment

    server_env = prepare_mcp_server_environment()

    try:
        mcp_server = MCPServerStdio(command=MCP_SERVER_CMD[0], args=MCP_SERVER_CMD[1:], env=server_env)
    except Exception as e:
        print(f"Error creating MCP server: {e}", file=sys.stderr)
        raise RuntimeError("Agent creation failed") from e

    try:
        agent: Agent[LLMOnlyResponse] = Agent(
            llm,
            mcp_servers=[mcp_server],
            system_prompt=get_simple_agent_system_prompt(),
            output_type=LLMOnlyResponse,
        )
    except Exception as e:
        print(f"Error creating agent: {e}", file=sys.stderr)
        raise RuntimeError("Agent creation failed") from e

    return agent, mcp_server


async def process_single_user_query(
    agent: Agent[LLMOnlyResponse], user_query: str, collect_metrics: bool = False
) -> FinalAgentResponse | None | tuple[FinalAgentResponse | None, AgentPerformanceMetrics]:
    """Processes a single user query using the provided agent and returns the structured response.

    Args:
        agent: The Pydantic AI agent to run
        user_query: The user's query string
        collect_metrics: If True, returns tuple with metrics; if False, returns just response

    Returns:
        If collect_metrics=False: FinalAgentResponse | None
        If collect_metrics=True: tuple[FinalAgentResponse | None, AgentPerformanceMetrics]
    """
    start_time = time.time() if collect_metrics else None

    try:

        async def _run_agent():
            return await agent.run(user_query)

        run_result: AgentRunResult[LLMOnlyResponse] = await _retry_manager.execute_with_retry(_run_agent)

        # Collect metrics if requested
        metrics = None
        if collect_metrics:
            metrics = PerformanceMetricsCollector.collect_from_agent_result(
                agent_result=run_result,
                agent_type="simple",
                execution_start_time=start_time,
            )

        if run_result and run_result.output:
            llm_response = run_result.output

            # Extract MCP tool responses programmatically
            mcp_tool_responses = extract_mcp_tool_responses(run_result)

            import json

            # Construct final response with LLM summary + programmatic details
            final_response = FinalAgentResponse(
                summary=llm_response.summary,
                details=json.dumps(mcp_tool_responses) if mcp_tool_responses else None,
                error_message=llm_response.error_message,
            )

            if collect_metrics:
                metrics.success = True
                metrics.response_data = final_response.model_dump()
                return final_response, metrics
            else:
                return final_response

        elif run_result and run_result.error_message:
            error_response = FinalAgentResponse(
                summary=f"Agent error: {run_result.error_message}",
                details=None,
                error_message=run_result.error_message,
            )

            if collect_metrics:
                metrics.success = False
                metrics.error_message = run_result.error_message
                metrics.response_data = error_response.model_dump()
                return error_response, metrics
            else:
                return error_response
        else:
            if collect_metrics:
                metrics.success = False
                metrics.error_message = "No response from agent"
                return None, metrics
            else:
                return None

    except Exception as e:
        print(f"Error during agent query processing: {e}", file=sys.stderr)
        error_response = FinalAgentResponse(
            summary=f"Agent processing failed: {e}",
            details=None,
            error_message=str(e),
        )

        if collect_metrics:
            error_metrics = PerformanceMetricsCollector.collect_from_timing_and_response(
                execution_start_time=start_time,
                agent_type="simple",
                response_data={"error": str(e)},
                success=False,
                error_message=str(e),
            )
            return error_response, error_metrics
        else:
            return error_response


# --- Main Agent Interactive Loop ---
async def main():
    """Initializes and runs the Pydantic AI agent."""
    args = parse_simple_agent_args()

    # Handle configuration check
    if args.check_config:
        handle_config_check()
        return

    # Set the document root directory for the server-side logic
    setup_document_root()

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
                                print(f"\n--- Details (List of {item_type.__name__}) ---")
                                for item_idx, item_detail in enumerate(final_response.details):
                                    print(f"Item {item_idx + 1}:")
                                    if hasattr(item_detail, "model_dump"):  # Check if Pydantic model
                                        print(item_detail.model_dump(exclude_none=True))
                                    else:
                                        print(item_detail)
                        elif hasattr(final_response.details, "model_dump"):  # Check if Pydantic model
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
                action_parser.print_help()
                print("\nExample usage:")
                print('  python src/agents/simple_agent.py --query "List all documents"')
                print("  python src/agents/simple_agent.py --interactive")

    except KeyboardInterrupt:
        if args.interactive or not args.query:  # Only print this message in interactive mode
            print("\nUser requested exit. Shutting down...")
    except Exception as e:
        print(f"An unexpected error occurred in the agent: {e}", file=sys.stderr)
    finally:
        if args.interactive or not args.query:  # Only print shutdown message in interactive mode
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
