#!/usr/bin/env python3
"""ReAct Document Management Agent.

This module implements a ReAct (Reasoning and Acting) agent that can manage
structured markdown documents through systematic reasoning and tool execution.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from rich.console import Console
from rich.panel import Panel

from document_mcp import doc_tool_server

# Import the shared error handling module
from src.agents.react_agent.models import ReActStep

# Import from shared config
from src.agents.react_agent.parser import ActionParser
from src.agents.react_agent.prompts import get_react_system_prompt
from src.agents.shared.cli import handle_config_check
from src.agents.shared.cli import parse_react_agent_args
from src.agents.shared.cli import validate_api_configuration
from src.agents.shared.config import DEFAULT_TIMEOUT
from src.agents.shared.config import MAX_RETRIES
from src.agents.shared.config import MCP_SERVER_CMD
from src.agents.shared.config import load_llm_config
from src.agents.shared.performance_metrics import AgentPerformanceMetrics
from src.agents.shared.performance_metrics import MetricsCollectionContext
from src.agents.shared.performance_metrics import build_response_data

# --- ReAct Execution Loop ---




async def run_react_agent_with_metrics(
    user_query: str, max_steps: int = 10
) -> tuple[list[dict[str, Any]], AgentPerformanceMetrics]:
    """Run the React agent with real performance metrics collection.

    This function provides the same functionality as the main agent execution
    but captures actual performance data from LLM usage and agent execution.
    """
    with MetricsCollectionContext("react") as ctx:
        try:
            # Prepare environment for MCP server subprocess
            from src.agents.shared.config import prepare_mcp_server_environment

            server_env = prepare_mcp_server_environment()

            mcp_server = MCPServerStdio(
                command="python3",
                args=["-m", "document_mcp.doc_tool_server", "stdio"],
                env=server_env,
            )

            # Create agent directly
            llm = await load_llm_config()
            agent = Agent(
                llm,
                mcp_servers=[mcp_server],
                system_prompt=get_react_system_prompt(),
                output_type=ReActStep,
            )

            async with mcp_server:
                # Run the React loop and capture execution history with agent results
                history, agent_results = await run_react_loop(
                    agent, mcp_server, user_query, max_steps, capture_agent_results=True
                )

            # Add agent results for token tracking
            for result in agent_results:
                ctx.add_agent_result(result)

            # Extract and add tool calls from history
            tool_calls = ctx.extract_tool_calls_from_history(history)
            for tool in tool_calls:
                ctx.add_tool_call(tool)

            # Build response data
            response_data = build_response_data(
                "react",
                steps_executed=len(history),
                max_steps=max_steps,
                final_step=history[-1] if history else None,
                execution_summary=f"Completed in {len(history)} steps",
                success=history and history[-1].get("action") is None,
            )

            return history, ctx.create_metrics(response_data)

        except Exception as e:
            # Handle exceptions with error response data
            error_data = build_response_data("react", error=str(e), success=False)
            return [], ctx.create_metrics(error_data)


async def run_react_loop(
    agent: Agent,
    mcp_server: MCPServerStdio,
    user_query: str,
    max_steps: int = 10,
    conversation_history: list[dict[str, Any]] | None = None,
    capture_agent_results: bool = False,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Main execution loop for the ReAct agent."""
    console = Console()
    history = []
    action_action_parser = ActionParser()
    agent_results = []  # Capture AgentRunResult objects for token tracking

    if conversation_history is None:
        conversation_history = []

    console.print(
        Panel(
            f"Starting ReAct loop for query: [bold cyan]{user_query}[/bold cyan]\nMaximum steps: {max_steps}",
            title="ReAct Agent",
            expand=False,
        )
    )

    # Set the document root directory for the server-side logic
    # This ensures consistency if the agent is run from a different CWD
    if "DOCUMENT_ROOT_DIR" in os.environ:
        doc_tool_server.DOCS_ROOT_PATH = Path(os.environ["DOCUMENT_ROOT_DIR"])

    try:
        # Helper function to build context from history
        def build_context_from_history(history_list):
            """Build context string from history list."""
            if not history_list:
                return ""

            context_parts = []
            for i, step_data in enumerate(history_list, 1):
                context_parts.append(f"Step {i}:")
                if "thought" in step_data:
                    context_parts.append(f"  Thought: {step_data['thought']}")
                if "action" in step_data:
                    context_parts.append(f"  Action: {step_data['action']}")
                if "observation" in step_data:
                    context_parts.append(f"  Observation: {step_data['observation']}")

            return "\n".join(context_parts)

        # Initialize context and step counter
        step = len(conversation_history)

        # Create initial context with user query
        context_from_history = build_context_from_history(conversation_history)
        current_context = f"User Query: {user_query}\n\n{context_from_history}\n\nPlease provide your next thought and action."

        while step < max_steps:
            step += 1

            # Implement retry logic for timeout resilience
            result = None
            retry_count = 0
            last_error = None

            while retry_count <= MAX_RETRIES:
                try:
                    # Run the agent to get the next step
                    result = await asyncio.wait_for(agent.run(current_context), timeout=DEFAULT_TIMEOUT)
                    break  # Success - exit retry loop

                except asyncio.TimeoutError as e:
                    last_error = e
                    retry_count += 1
                    if retry_count <= MAX_RETRIES:
                        console.print(
                            f"[yellow]API timeout (attempt {retry_count}/{MAX_RETRIES + 1}), retrying...[/yellow]"
                        )
                        await asyncio.sleep(1.0 * retry_count)  # Exponential backoff
                        continue
                    # All retries exhausted - fall through to error handling

                except Exception as e:
                    # Non-timeout errors - fail immediately
                    last_error = e
                    break

            if result is not None:
                # Success path
                # Capture AgentRunResult for token tracking if requested
                if capture_agent_results:
                    agent_results.append(result)

                # Extract the ReActStep from the result
                react_step = result.output if hasattr(result, "output") else result
            else:
                # All retries failed - handle error
                error_msg = (
                    str(last_error) if str(last_error) else f"{type(last_error).__name__}: No error message"
                )
                error_details = f"Exception type: {type(last_error).__name__}, Message: {error_msg}"
                if retry_count > MAX_RETRIES:
                    error_details += f" (failed after {MAX_RETRIES + 1} attempts)"

                console.print(f"[red]Agent execution error: {error_details}[/red]")
                step_data = {
                    "step": step,
                    "thought": f"Error occurred during agent execution: {error_details}",
                    "action": None,
                    "observation": f"Error: {error_details}",
                }
                history.append(step_data)
                break

            # Prepare step data
            step_data = {
                "step": step,
                "thought": react_step.thought,
                "action": react_step.action,
                "observation": None,
            }

            # If there's an action, execute it by running the agent with a tool execution prompt
            if react_step.action and react_step.action.strip():
                try:
                    # Parse the action to extract tool name and parameters
                    tool_name, kwargs = action_action_parser.parse(react_step.action)

                    # Create a prompt that will cause the agent to call the MCP tool
                    tool_prompt = f"Call the {tool_name} tool with these exact parameters: {kwargs}"

                    # Run the agent to execute the tool
                    tool_result = await asyncio.wait_for(agent.run(tool_prompt), timeout=DEFAULT_TIMEOUT)

                    # Extract the actual tool response from the agent result
                    observation = "Tool execution completed"
                    tool_response_data = None
                    if hasattr(tool_result, "all_messages"):
                        messages = tool_result.all_messages()
                        # Look for tool execution results in the messages
                        for message in messages:
                            if hasattr(message, "parts"):
                                for part in message.parts:
                                    # Use the same pattern as Simple Agent for consistency
                                    if (
                                        hasattr(part, "tool_name")
                                        and hasattr(part, "content")
                                        and type(part).__name__ == "ToolReturnPart"
                                    ):
                                        tool_content = part.content

                                        # Store the actual MCP tool response data using Simple Agent pattern
                                        if isinstance(tool_content, list):
                                            tool_response_data = {"documents": tool_content}
                                        elif isinstance(tool_content, dict):
                                            tool_response_data = tool_content
                                        else:
                                            tool_response_data = {"content": tool_content}

                                        observation = str(tool_response_data)
                                        break

                    # Store the structured tool response for later extraction
                    step_data["tool_response_data"] = tool_response_data
                    step_data["tool_name"] = tool_name

                    step_data["observation"] = observation

                except Exception as e:
                    step_data["observation"] = f"Error executing action: {str(e)}"
            else:
                # No action means the agent has completed the task
                step_data["observation"] = "Task completed."
                history.append(step_data)
                break

            # Add step to history and context
            history.append(step_data)
            conversation_history.append(step_data)
            context_from_history = build_context_from_history(conversation_history)
            current_context = f"User Query: {user_query}\n\n{context_from_history}\n\nPlease provide your next thought and action."

            # Display step results
            console.print(f"[bold]Step {step}:[/bold]")
            console.print(Panel(f"[yellow]Thought:[/yellow]\n{step_data['thought']}"))
            if step_data.get("action"):
                console.print(
                    Panel(
                        f"[cyan]Action:[/cyan]\n{step_data['action']}",
                        border_style="cyan",
                    )
                )
            if step_data.get("observation") is not None:
                console.print(
                    Panel(
                        f"[magenta]Observation:[/magenta]\n{step_data['observation']}",
                        border_style="magenta",
                    )
                )
            console.print("-" * 80)

    except Exception as e:
        console.print(Panel(f"[bold red]Error during ReAct loop:[/bold red]\n{e}"))
        # Add error details to history for inspection
        if not history:  # Only add error if no history exists yet
            history.append({"step": 1, "error": str(e)})

    final_thought = history[-1].get("thought", "No final thought.") if history else "No steps completed."
    console.print(
        Panel(
            f"ReAct loop finished.\n[bold]Final Thought:[/bold] {final_thought}",
            title="ReAct Agent Complete",
            border_style="green",
        )
    )
    return history, agent_results


# --- Main Function and CLI ---
async def main():
    """Main function to run the ReAct agent."""
    args = parse_react_agent_args()

    # Handle configuration check
    if args.check_config:
        handle_config_check()
        return

    # Check API key configuration
    print("Checking API key configuration...")
    if not validate_api_configuration():
        return
    print()

    if args.interactive:
        # Interactive mode
        print("ReAct Agent - Interactive Mode")
        print("Type 'exit' or 'quit' to stop")
        print("=" * 60)

        # Prepare environment for MCP server subprocess
        from src.agents.shared.config import prepare_mcp_server_environment

        server_env = prepare_mcp_server_environment()

        mcp_server = MCPServerStdio(command=MCP_SERVER_CMD[0], args=MCP_SERVER_CMD[1:], env=server_env)
        # Create agent directly (no caching needed for E2E tests)
        llm = await load_llm_config()
        agent = Agent(
            llm,
            mcp_servers=[mcp_server],
            system_prompt=get_react_system_prompt(),
            output_type=ReActStep,
        )
        async with mcp_server:
            conversation_history = []
            while True:
                try:
                    user_query = input("\nEnter your query: ").strip()

                    if user_query.lower() in ["exit", "quit"]:
                        print("Goodbye!")
                        break

                    if not user_query:
                        print("Please enter a valid query.")
                        continue

                    # Run the ReAct loop
                    print()
                    history, _ = await run_react_loop(
                        agent,
                        mcp_server,
                        user_query,
                        args.max_steps,
                        conversation_history,
                    )

                    # Update conversation history for next iteration
                    conversation_history.extend(history)

                    # Display summary
                    print("\nExecution Summary:")
                    print(f"   Total steps: {len(history)}")
                    if history:
                        final_step = history[-1]
                        if final_step["action"] is None:
                            print("   Status: Completed successfully")
                        else:
                            print("   Status: Incomplete (max steps reached)")
                    print()

                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    print()

    elif args.query:
        # Single query mode
        print(f"Processing query: {args.query}")
        print()

        try:
            # Prepare environment for MCP server subprocess
            from src.agents.shared.config import prepare_mcp_server_environment

            server_env = prepare_mcp_server_environment()

            mcp_server = MCPServerStdio(
                command="python3",
                args=["-m", "document_mcp.doc_tool_server", "stdio"],
                env=server_env,
            )
            # Create agent directly
            llm = await load_llm_config()
            agent = Agent(
                llm,
                mcp_servers=[mcp_server],
                system_prompt=get_react_system_prompt(),
                output_type=ReActStep,
            )
            async with mcp_server:
                history, _ = await run_react_loop(agent, mcp_server, args.query, args.max_steps)

            # Display summary
            print("\nExecution Summary:")
            print(f"   Query: {args.query}")
            print(f"   Total steps: {len(history)}")
            if history:
                final_step = history[-1]
                if final_step["action"] is None:
                    print("   Status: Completed successfully")
                else:
                    print("   Status: Incomplete (max steps reached)")

            # Output JSON for programmatic use
            from src.agents.shared.output_formatter import AgentResponseFormatter

            # Generate execution summary
            execution_summary = f"Completed in {len(history)} steps"
            if history:
                final_step = history[-1]
                if final_step["action"] is None:
                    execution_summary = f"Successfully completed task in {len(history)} steps"
                else:
                    execution_summary = f"Execution incomplete after {len(history)} steps (max steps reached)"

            # Format execution log
            execution_log = ""
            if history:
                log_entries = []
                for h in history:
                    log_entries.append(f"Step {h['step']}: {h['thought']}")
                execution_log = "\n".join(log_entries)

            # Format structured details - collect MCP tool responses
            mcp_tool_responses = {}
            execution_steps = []
            if history:
                execution_steps = [
                    {
                        "step": h["step"],
                        "thought": h["thought"],
                        "action": h["action"],
                        "observation": h["observation"],
                    }
                    for h in history
                ]

                # Extract MCP tool responses for details field from structured data
                for h in history:
                    # Use the structured tool response data we stored during execution
                    if h.get("tool_response_data") and h.get("tool_name"):
                        tool_name = h["tool_name"]
                        tool_response = h["tool_response_data"]
                        response_key = f"{tool_name}_response"
                        mcp_tool_responses[response_key] = tool_response

            json_output = AgentResponseFormatter.format_react_agent_response(
                summary=execution_summary,
                mcp_tool_responses=mcp_tool_responses,
                steps_executed=execution_steps,
                execution_log=execution_log,
                max_steps=args.max_steps,
            )

            # Print separator and JSON output
            print("\n" + "=" * 50)
            print("JSON OUTPUT:")
            print(json_output)

        except Exception as e:
            print(f"Error: {e}")

            # Output error in JSON format as well
            from src.agents.shared.output_formatter import AgentResponseFormatter

            json_output = AgentResponseFormatter.format_react_agent_response(
                summary=f"Error during execution: {e}",
                mcp_tool_responses={},
                steps_executed=[],
                execution_log="",
                max_steps=args.max_steps,
                error_message=str(e),
            )

            print("\n" + "=" * 50)
            print("JSON OUTPUT:")
            print(json_output)

    else:
        # No query provided, show help
        import argparse

        parser = argparse.ArgumentParser(description="ReAct Agent")
        parser.print_help()
        print("\nExample usage:")
        print('  python src/agents/react_agent/main.py --query "Create a document called My Story"')
        print("  python src/agents/react_agent/main.py --interactive")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)
