"""
Pydantic models for the ReAct agent.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ReActStep(BaseModel):
    """A single step in the ReAct process, containing thought and action."""

    thought: str = Field(
        min_length=1, description="The agent's reasoning and plan for the next action."
    )
    action: Optional[str] = Field(
        default=None,
        description="The tool call to execute, or null if reasoning is complete.",
    )
