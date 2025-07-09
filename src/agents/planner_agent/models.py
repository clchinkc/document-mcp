"""
Pydantic models for the Planner Agent's plan structure and responses.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


class PlanStep(BaseModel):
    """Represents a single step in the execution plan."""

    tool_name: str = Field(..., description="Name of the MCP tool to call")
    arguments: Dict[str, Any] = Field(..., description="Arguments to pass to the tool")

    model_config = ConfigDict(extra="forbid")


# ExecutionPlan is now just a type alias for List[PlanStep]
ExecutionPlan = List[PlanStep]


class StepResult(BaseModel):
    """Represents the result of executing a single step."""

    step_index: int = Field(..., description="Index of the step in the plan")
    tool_name: str = Field(..., description="Name of the tool that was called")
    arguments: Dict[str, Any] = Field(..., description="Arguments passed to the tool")
    success: bool = Field(..., description="Whether the step executed successfully")
    result: Optional[Dict[str, Any]] = Field(
        None, description="Result from the tool if successful"
    )
    error: Optional[str] = Field(None, description="Error message if step failed")

    model_config = ConfigDict(extra="forbid")


class PlannerAgentResponse(BaseModel):
    """Final response from the Planner Agent."""

    query: str = Field(..., description="Original user query")
    plan_generated: bool = Field(
        ..., description="Whether plan generation was successful"
    )
    execution_completed: bool = Field(
        ..., description="Whether all steps were executed successfully"
    )
    steps_executed: List[StepResult] = Field(
        ..., description="Results of each executed step"
    )
    summary: str = Field(
        ..., description="Human-readable summary of the completed work"
    )
    error: Optional[str] = Field(
        None, description="Error message if planning or execution failed"
    )

    model_config = ConfigDict(extra="forbid")