"""Pydantic models for the ReAct agent."""

from pydantic import BaseModel
from pydantic import Field


class ReActStep(BaseModel):
    """A single step in the ReAct process, containing thought and action."""

    thought: str = Field(
        min_length=1, description="The agent's reasoning and plan for the next action."
    )
    action: str | None = Field(
        default=None,
        description="The tool call to execute, or null if reasoning is complete.",
    )


class FinalAgentResponse(BaseModel):
    """Defines the final structured output returned to users."""

    summary: str
    details: str | None = Field(
        default=None, description="Programmatically populated with MCP tool responses"
    )
    execution_steps: list[dict] | None = Field(
        default=None, description="ReAct reasoning steps for debugging"
    )
    error_message: str | None = None
