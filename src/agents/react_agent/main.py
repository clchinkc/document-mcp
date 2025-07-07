#!/usr/bin/env python3
"""
ReAct Document Management Agent

This module implements a ReAct (Reasoning and Acting) agent that can manage
structured markdown documents through systematic reasoning and tool execution.
"""

import argparse
import asyncio
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from pathlib import Path

# Remove invalid imports
# from .react_agent import ReactAgent
# from .react_prompts import REACT_SYSTEM_PROMPT

# Import from shared config
from src.agents.react_agent.parser import ActionParser
from src.agents.react_agent.prompts import REACT_SYSTEM_PROMPT
from src.agents.shared.config import load_llm_config, check_api_keys_config, MCP_SERVER_CMD, DEFAULT_TIMEOUT
from src.agents.shared.cli import parse_react_agent_args, handle_config_check, setup_document_root, validate_api_configuration
# Import the shared error handling module
from src.agents.react_agent.models import ReActStep

try:
    from document_mcp import doc_tool_server
except ImportError:
    # Fallback for development/testing without installed package
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from document_mcp import doc_tool_server



# --- ReAct Execution Loop ---

async def execute_mcp_tool_directly(agent: Agent, tool_name: str, kwargs: Dict[str, Any]) -> str:
    """Execute an MCP tool directly and return the result as a string."""
    try:
        # Use the agent's built-in tool execution mechanism
        tool_call = f"{tool_name}({', '.join(f'{k}={repr(v)}' for k, v in kwargs.items())})"
        result = await agent.run(f"Use the {tool_name} tool with these parameters: {kwargs}")
        
        # Extract the result from the agent response
        if hasattr(result, 'output'):
            return str(result.output)
        else:
            return str(result)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


async def run_react_loop(
    agent: Agent, 
    mcp_server: MCPServerStdio, 
    user_query: str, 
    max_steps: int = 10,
    conversation_history: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """Main execution loop for the ReAct agent."""
    console = Console()
    history = []
    action_parser = ActionParser()

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
                if 'thought' in step_data:
                    context_parts.append(f"  Thought: {step_data['thought']}")
                if 'action' in step_data:
                    context_parts.append(f"  Action: {step_data['action']}")
                if 'observation' in step_data:
                    context_parts.append(f"  Observation: {step_data['observation']}")
            
            return "\n".join(context_parts)

        # Initialize context and step counter
        step = len(conversation_history)
        
        # Create initial context with user query
        context_from_history = build_context_from_history(conversation_history)
        current_context = f"User Query: {user_query}\n\n{context_from_history}\n\nPlease provide your next thought and action."
        
        while step < max_steps:
            step += 1
            
            try:
                # Run the agent to get the next step
                result = await asyncio.wait_for(agent.run(current_context), timeout=DEFAULT_TIMEOUT)
                
                # Extract the ReActStep from the result
                if hasattr(result, 'output'):
                    react_step = result.output
                else:
                    react_step = result
            except Exception as e:
                console.print(f"[red]Agent execution error: {e}[/red]")
                step_data = {
                    "step": step,
                    "thought": f"Error occurred during agent execution: {str(e)}",
                    "action": None,
                    "observation": f"Error: {str(e)}"
                }
                history.append(step_data)
                break
            
            # Prepare step data
            step_data = {
                "step": step,
                "thought": react_step.thought,
                "action": react_step.action,
                "observation": None
            }
            
            # If there's an action, execute it
            if react_step.action and react_step.action.strip():
                try:
                    # Parse and execute the action
                    tool_name, kwargs = action_parser.parse(react_step.action)
                    observation = await execute_mcp_tool_directly(agent, tool_name, kwargs)
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
    return history


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
        server_env = None
        if "DOCUMENT_ROOT_DIR" in os.environ:
            server_env = {
                **os.environ,
                "PYTEST_CURRENT_TEST": "1"  # Signal to MCP server we're in test mode
            }
        
        mcp_server = MCPServerStdio(
            command=MCP_SERVER_CMD[0], args=MCP_SERVER_CMD[1:], env=server_env
        )
        # Create agent directly (no caching needed for E2E tests)
        llm = await load_llm_config()
        agent = Agent(
            llm,
            mcp_servers=[mcp_server],
            system_prompt=REACT_SYSTEM_PROMPT,
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
                    history = await run_react_loop(
                        agent, mcp_server, user_query, args.max_steps, conversation_history
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
            server_env = None
            if "DOCUMENT_ROOT_DIR" in os.environ:
                server_env = {
                    **os.environ,
                    "PYTEST_CURRENT_TEST": "1"  # Signal to MCP server we're in test mode
                }
            
            mcp_server = MCPServerStdio(
                command="python3", args=["-m", "document_mcp.doc_tool_server", "stdio"], env=server_env
            )
            # Create agent directly
            llm = await load_llm_config()
            agent = Agent(
                llm,
                mcp_servers=[mcp_server],
                system_prompt=REACT_SYSTEM_PROMPT,
                output_type=ReActStep,
            )
            async with mcp_server:
                history = await run_react_loop(agent, mcp_server, args.query, args.max_steps)

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

        except Exception as e:
            print(f"Error: {e}")

    else:
        # No query provided, show help
        parser.print_help()
        print("\nExample usage:")
        print(
            '  python src/agents/react_agent/main.py --query "Create a document called My Story"'
        )
        print("  python src/agents/react_agent/main.py --interactive")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)
