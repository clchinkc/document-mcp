"""
Planner Agent - Plan-and-Execute approach for complex multi-step tasks.

This agent handles complex, multi-step user requests by first generating a complete
plan of tool calls, then executing them sequentially without further LLM intervention.
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, List, Tuple

from dotenv import load_dotenv
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

# Import planner agent specific components
from src.agents.planner_agent.models import (
    ExecutionPlan,
    PlannerAgentResponse,
    PlanStep,
    StepResult,
)
from src.agents.planner_agent.prompts import get_planner_system_prompt
from src.agents.shared.cli import (
    handle_config_check,
    parse_planner_agent_args,
    setup_document_root,
)

# Import shared configuration and utilities
from src.agents.shared.config import MCP_SERVER_CMD, load_llm_config
from src.agents.shared.error_handling import RetryManager
from src.agents.shared.performance_metrics import (
    AgentPerformanceMetrics, 
    PerformanceMetricsCollector,
    MetricsCollectionContext,
    build_response_data
)

# Suppress verbose HTTP logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)


# --- Core Agent Logic ---
_retry_manager = RetryManager()


async def initialize_planning_agent() -> Agent[str]:
    """Initialize the planning agent for generating execution plans."""
    try:
        llm = await _retry_manager.execute_with_retry(load_llm_config)
    except ValueError as e:
        print(f"Error loading LLM config: {e}", file=sys.stderr)
        raise

    # Create a simple agent for planning (no MCP server needed for planning phase)
    try:
        agent = Agent(
            llm,
            system_prompt=get_planner_system_prompt(),
            output_type=str,  # We expect JSON string response
        )
    except Exception as e:
        print(f"Error creating planning agent: {e}", file=sys.stderr)
        raise RuntimeError("Planning agent creation failed") from e

    return agent


async def initialize_mcp_server() -> MCPServerStdio:
    """Initialize the MCP server for plan execution."""
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
        raise RuntimeError("MCP server creation failed") from e

    return mcp_server


async def generate_execution_plan(
    planning_agent: Agent[str], user_query: str
) -> Tuple[ExecutionPlan, Any]:
    """Generate an execution plan using the planning agent."""
    try:
        # Get the plan from the LLM
        result = await _retry_manager.execute_with_retry(
            lambda: planning_agent.run(user_query)
        )

        # Parse the JSON response
        try:
            raw_output = result.output.strip()

            # Extract JSON from markdown code blocks if present
            if raw_output.startswith("```json") and raw_output.endswith("```"):
                # Remove the markdown code block wrapper
                json_content = raw_output[7:-3].strip()  # Remove ```json and ```
            elif raw_output.startswith("```") and raw_output.endswith("```"):
                # Remove generic code block wrapper
                json_content = raw_output[3:-3].strip()  # Remove ``` and ```
            else:
                json_content = raw_output

            plan_data = json.loads(json_content)
            # Validate each step in the plan
            execution_plan = [PlanStep(**step) for step in plan_data]
            return execution_plan, result
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Invalid plan format from LLM: {e}")

    except Exception as e:
        print(f"Error generating execution plan: {e}", file=sys.stderr)
        raise


async def execute_plan_step(
    mcp_server: MCPServerStdio, step: PlanStep, step_index: int
) -> StepResult:
    """Execute a single step in the plan."""
    try:
        # Call the MCP tool using the client
        result = await mcp_server._client.call_tool(step.tool_name, step.arguments)

        # Parse the result - it's a list of TextContent objects with JSON text
        result_text = result.content[0].text
        result_data = json.loads(result_text)

        return StepResult(
            step_index=step_index,
            tool_name=step.tool_name,
            arguments=step.arguments,
            success=True,
            result=result_data,
            error=None,
        )
    except Exception as e:
        return StepResult(
            step_index=step_index,
            tool_name=step.tool_name,
            arguments=step.arguments,
            success=False,
            result=None,
            error=str(e),
        )


async def execute_plan(
    mcp_server: MCPServerStdio, plan: ExecutionPlan
) -> List[StepResult]:
    """Execute the complete plan step by step."""
    results = []

    for i, step in enumerate(plan):
        print(f"Executing step {i+1}/{len(plan)}: {step.tool_name}")

        step_result = await execute_plan_step(mcp_server, step, i)
        results.append(step_result)

        # Stop execution if a step fails (either MCP call fails or logical operation fails)
        if not step_result.success:
            print(f"Step {i+1} failed: {step_result.error}")
            break

        # Also check if the logical operation itself failed
        if step_result.result and not step_result.result.get("success", True):
            print(
                f"Step {i+1} operation failed: {step_result.result.get('error', 'Unknown error')}"
            )
            break

    return results


async def generate_final_summary(results: List[StepResult]) -> str:
    """Generate a human-readable summary of the executed plan."""
    try:
        # Create a summary of what was executed
        successful_steps = [r for r in results if r.success]
        failed_steps = [r for r in results if not r.success]

        if not successful_steps and not failed_steps:
            return "No steps were executed."

        summary_parts = []

        if successful_steps:
            summary_parts.append(
                f"Successfully executed {len(successful_steps)} step(s):"
            )
            for result in successful_steps:
                summary_parts.append(f"- {result.tool_name}")

        if failed_steps:
            summary_parts.append(f"Failed to execute {len(failed_steps)} step(s):")
            for result in failed_steps:
                summary_parts.append(f"- {result.tool_name}: {result.error}")

        return "\\n".join(summary_parts)

    except Exception as e:
        return f"Error generating summary: {e}"


async def run_planner_agent(user_query: str) -> PlannerAgentResponse:
    """Main function to run the planner agent with a user query."""
    print(f"Processing query: {user_query}")

    # Phase 1: Planning
    print("Phase 1: Generating execution plan...")
    planning_agent = await initialize_planning_agent()

    try:
        execution_plan, _ = await generate_execution_plan(planning_agent, user_query)
        print(f"Generated plan with {len(execution_plan)} steps")
        plan_generated = True
    except Exception as e:
        return PlannerAgentResponse(
            query=user_query,
            plan_generated=False,
            execution_completed=False,
            steps_executed=[],
            summary=f"Failed to generate execution plan: {e}",
            error=str(e),
        )

    # Phase 2: Execution
    print("Phase 2: Executing plan...")
    mcp_server = await initialize_mcp_server()

    try:
        async with mcp_server:
            step_results = await execute_plan(mcp_server, execution_plan)
            execution_completed = all(result.success for result in step_results)

            # Phase 3: Summary
            print("Phase 3: Generating summary...")
            summary = await generate_final_summary(step_results)

            return PlannerAgentResponse(
                query=user_query,
                plan_generated=plan_generated,
                execution_completed=execution_completed,
                steps_executed=step_results,
                summary=summary,
                error=(
                    None
                    if execution_completed
                    else "Some steps failed during execution"
                ),
            )

    except Exception as e:
        return PlannerAgentResponse(
            query=user_query,
            plan_generated=plan_generated,
            execution_completed=False,
            steps_executed=[],
            summary=f"Failed to execute plan: {e}",
            error=str(e),
        )


async def run_planner_agent_with_metrics(user_query: str) -> Tuple[PlannerAgentResponse, AgentPerformanceMetrics]:
    """
    Run the Planner agent with real performance metrics collection.
    
    This function provides the same functionality as run_planner_agent
    but captures actual performance data from LLM usage and agent execution.
    """
    with MetricsCollectionContext("planner") as ctx:
        try:
            print(f"Processing query: {user_query}")

            # Phase 1: Planning with metrics collection
            print("Phase 1: Generating execution plan...")
            planning_agent = await initialize_planning_agent()
            
            try:
                execution_plan, planning_result = await generate_execution_plan(planning_agent, user_query)
                print(f"Generated plan with {len(execution_plan)} steps")
                plan_generated = True
                
                # Add planning result for token tracking
                if planning_result:
                    ctx.add_agent_result(planning_result)
                    
            except Exception as e:
                response = PlannerAgentResponse(
                    query=user_query,
                    plan_generated=False,
                    execution_completed=False,
                    steps_executed=[],
                    summary=f"Failed to generate execution plan: {e}",
                    error=str(e),
                )
                
                error_data = build_response_data("planner", error=str(e), success=False)
                return response, ctx.create_metrics(error_data)

            # Phase 2: Execution
            print("Phase 2: Executing plan...")
            mcp_server = await initialize_mcp_server()

            try:
                async with mcp_server:
                    step_results = await execute_plan(mcp_server, execution_plan)
                    execution_completed = all(result.success for result in step_results)

                    # Phase 3: Summary
                    print("Phase 3: Generating summary...")
                    summary = await generate_final_summary(step_results)

                    response = PlannerAgentResponse(
                        query=user_query,
                        plan_generated=plan_generated,
                        execution_completed=execution_completed,
                        steps_executed=step_results,
                        summary=summary,
                        error=(
                            None
                            if execution_completed
                            else "Some steps failed during execution"
                        ),
                    )

            except Exception as e:
                response = PlannerAgentResponse(
                    query=user_query,
                    plan_generated=plan_generated,
                    execution_completed=False,
                    steps_executed=[],
                    summary=f"Failed to execute plan: {e}",
                    error=str(e),
                )
            
            # Extract tool names from executed steps
            for step_result in response.steps_executed:
                if hasattr(step_result, 'tool_name'):
                    ctx.add_tool_call(step_result.tool_name)
            
            # Build response data
            response_data = build_response_data(
                "planner",
                plan_generated=response.plan_generated,
                execution_completed=response.execution_completed,
                steps_executed=len(response.steps_executed),
                total_steps_planned=len(response.steps_executed),
                success=response.execution_completed and not response.error,
                summary=response.summary
            )
            
            return response, ctx.create_metrics(response_data)
            
        except Exception as e:
            # Handle any exceptions with error response data
            error_response = PlannerAgentResponse(
                query=user_query,
                plan_generated=False,
                execution_completed=False,
                steps_executed=[],
                summary=f"Agent execution failed: {e}",
                error=str(e),
            )
            
            error_data = build_response_data("planner", error=str(e), success=False)
            return error_response, ctx.create_metrics(error_data)


# --- CLI Interface ---
async def main():
    """Main entry point for the planner agent."""
    args = parse_planner_agent_args()

    # Load environment variables
    load_dotenv()

    # Handle configuration check
    if args.check_config:
        handle_config_check()
        return

    # Setup document root
    setup_document_root()

    # Interactive mode
    if args.interactive:
        print("Planner Agent - Interactive Mode")
        print("Type 'exit' to quit")

        while True:
            try:
                user_query = input("\\nEnter your query: ").strip()
                if user_query.lower() in ["exit", "quit"]:
                    break

                if not user_query:
                    continue

                response = await run_planner_agent(user_query)
                print(f"\\n{response.summary}")

                if response.error:
                    print(f"Error: {response.error}")

            except KeyboardInterrupt:
                print("\\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

    # Single query mode
    elif args.query:
        response = await run_planner_agent(args.query)
        print(response.summary)

        # Output JSON for programmatic use
        from src.agents.shared.output_formatter import AgentResponseFormatter

        # Convert step results to dict format for JSON serialization
        steps_dict = [step.dict() for step in response.steps_executed]

        json_output = AgentResponseFormatter.format_planner_agent_response(
            summary=response.summary,
            plan_generated=response.plan_generated,
            execution_completed=response.execution_completed,
            steps_executed=steps_dict,
            error_message=response.error,
        )

        # Print separator and JSON output
        print("\n" + "=" * 50)
        print("JSON OUTPUT:")
        print(json_output)

        if response.error:
            print(f"Error: {response.error}", file=sys.stderr)
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
