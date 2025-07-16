import asyncio
import logging
import os
import sys
import time

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.mcp import MCPServerStdio

# Import models from the server to ensure compatibility
from src.agents.shared.error_handling import RetryManager
from src.agents.shared.performance_metrics import AgentPerformanceMetrics
from src.agents.shared.performance_metrics import PerformanceMetricsCollector

# Suppress verbose HTTP logging from requests/urllib3
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)


from src.agents.shared.cli import handle_config_check
from src.agents.shared.cli import parse_simple_agent_args
from src.agents.shared.cli import setup_document_root

# --- Configuration ---
from src.agents.shared.config import MCP_SERVER_CMD
from src.agents.shared.config import load_llm_config
from src.agents.simple_agent.prompts import get_simple_agent_system_prompt

# --- Agent Response Model (for Pydantic AI Agent's structured output) ---
# Using imported models from the server to ensure compatibility


class FinalAgentResponse(BaseModel):
    """Defines the final structured output expected from the Pydantic AI agent."""

    summary: str
    details: str | None = (
        None  # Use string instead of Dict to avoid Gemini additionalProperties limitation
    )
    error_message: str | None = None


# --- System Prompt ---
# System prompt is now imported from separate module


# --- Agent Setup and Processing Logic ---
_retry_manager = RetryManager()


async def initialize_agent_and_mcp_server() -> tuple[
    Agent[FinalAgentResponse], MCPServerStdio
]:
    """Initializes the Pydantic AI agent and its MCP server configuration."""
    try:
        llm = await _retry_manager.execute_with_retry(load_llm_config)
    except ValueError as e:
        print(f"Error loading LLM config: {e}", file=sys.stderr)
        raise

    # Configuration for stdio transport using shared constant

    # Prepare environment for MCP server subprocess
    server_env = None
    if "DOCUMENT_ROOT_DIR" in os.environ:
        server_env = {
            **os.environ,
            "PYTEST_CURRENT_TEST": "1",  # Signal to MCP server we're in test mode
        }

    try:
        mcp_server = MCPServerStdio(
            command=MCP_SERVER_CMD[0], args=MCP_SERVER_CMD[1:], env=server_env
        )
    except Exception as e:
        print(f"Error creating MCP server: {e}", file=sys.stderr)
        raise RuntimeError("Agent creation failed") from e

    try:
        agent: Agent[FinalAgentResponse] = Agent(
            llm,
            mcp_servers=[mcp_server],
            system_prompt=get_simple_agent_system_prompt(),
            output_type=FinalAgentResponse,
        )
    except Exception as e:
        print(f"Error creating agent: {e}", file=sys.stderr)
        raise RuntimeError("Agent creation failed") from e

    return agent, mcp_server


async def process_single_user_query(
    agent: Agent[FinalAgentResponse], user_query: str
) -> FinalAgentResponse | None:
    """Processes a single user query using the provided agent and returns the structured response."""
    try:

        async def _run_agent():
            return await agent.run(user_query)

        # Use RetryManager for robust error handling - let it manage its own timeouts
        run_result: AgentRunResult[
            FinalAgentResponse
        ] = await _retry_manager.execute_with_retry(_run_agent)

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
    except Exception as e:
        print(f"Error during agent query processing: {e}", file=sys.stderr)
        return FinalAgentResponse(
            summary=f"Agent processing failed: {e}",
            details=None,
            error_message=str(e),
        )


async def process_single_user_query_with_metrics(
    agent: Agent[FinalAgentResponse], user_query: str
) -> tuple[FinalAgentResponse | None, AgentPerformanceMetrics]:
    """Processes a single user query and returns both response and real performance metrics.

    This function captures actual performance data from the agent execution,
    replacing hardcoded mock values with real measurements.
    """
    start_time = time.time()

    try:

        async def _run_agent():
            return await agent.run(user_query)

        # Use RetryManager for robust error handling - let it manage its own timeouts
        run_result: AgentRunResult[
            FinalAgentResponse
        ] = await _retry_manager.execute_with_retry(_run_agent)

        # Collect real performance metrics from the agent result
        metrics = PerformanceMetricsCollector.collect_from_agent_result(
            agent_result=run_result,
            agent_type="simple",
            execution_start_time=start_time,
        )

        # Process the response
        if run_result and run_result.output:
            response = run_result.output
            metrics.success = True
            if hasattr(response, "model_dump"):
                metrics.response_data = response.model_dump()
            return response, metrics
        elif run_result and run_result.error_message:
            response = FinalAgentResponse(
                summary=f"Agent error: {run_result.error_message}",
                details=None,
                error_message=run_result.error_message,
            )
            metrics.success = False
            metrics.error_message = run_result.error_message
            metrics.response_data = response.model_dump()
            return response, metrics
        else:
            # No response case
            metrics.success = False
            metrics.error_message = "No response from agent"
            return None, metrics

    except Exception as e:
        # Handle execution exceptions
        metrics = PerformanceMetricsCollector.collect_from_timing_and_response(
            execution_start_time=start_time,
            agent_type="simple",
            response_data={"error": str(e)},
            success=False,
            error_message=str(e),
        )

        response = FinalAgentResponse(
            summary=f"Agent processing failed: {e}",
            details=None,
            error_message=str(e),
        )

        print(f"Error during agent query processing: {e}", file=sys.stderr)
        return response, metrics


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
