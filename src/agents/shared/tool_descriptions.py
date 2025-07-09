"""
Shared tool descriptions for all document management agents.

This module provides structured tool descriptions that can be dynamically formatted
for different agent architectures. Instead of hardcoding tool descriptions in prompts,
agents request format-specific descriptions that match their operational patterns.
"""

from typing import Dict, List
from dataclasses import dataclass
from enum import Enum


class ToolFormat(Enum):
    """Format styles for tool descriptions based on agent architecture."""
    FULL = "full"          # Detailed examples with full parameter syntax
    COMPACT = "compact"    # Brief descriptions with simplified syntax  
    PLANNER = "planner"    # Type-annotated signatures for planning
    MINIMAL = "minimal"    # Tool names only for reference


@dataclass
class ToolDescription:
    """Structured representation of a tool description."""
    name: str
    description: str
    parameters: Dict[str, str]  # parameter_name: type_hint
    example: str
    category: str
    planner_signature: str  # Type signature for planner agent


class ToolDescriptionManager:
    """Manages tool descriptions for dynamic prompt generation across all agents."""
    
    def __init__(self):
        self._tools = self._initialize_tools()
    
    def _initialize_tools(self) -> List[ToolDescription]:
        """Initialize all tool descriptions with unified format."""
        return [
            # Document Management (4 tools)
            ToolDescription(
                name="list_documents",
                description="Lists all available documents",
                parameters={},
                example="list_documents()",
                category="Document Management",
                planner_signature="list_documents()"
            ),
            ToolDescription(
                name="create_document",
                description="Creates a new document directory",
                parameters={"document_name": "str"},
                example='create_document(document_name="My Book")',
                category="Document Management",
                planner_signature="create_document(document_name: str)"
            ),
            ToolDescription(
                name="delete_document",
                description="Deletes an entire document",
                parameters={"document_name": "str"},
                example='delete_document(document_name="My Book")',
                category="Document Management",
                planner_signature="delete_document(document_name: str)"
            ),
            ToolDescription(
                name="read_document_summary",
                description="Reads the _SUMMARY.md file for a document. **Use this first before reading content**",
                parameters={"document_name": "str"},
                example='read_document_summary(document_name="My Book")',
                category="Document Management",
                planner_signature="read_document_summary(document_name: str)"
            ),
            
            # Chapter Management (6 tools)
            ToolDescription(
                name="list_chapters",
                description="Lists all chapters in a document",
                parameters={"document_name": "str"},
                example='list_chapters(document_name="My Book")',
                category="Chapter Management",
                planner_signature="list_chapters(document_name: str)"
            ),
            ToolDescription(
                name="create_chapter",
                description="Creates a new chapter",
                parameters={"document_name": "str", "chapter_name": "str", "initial_content": "str"},
                example='create_chapter(document_name="My Book", chapter_name="01-introduction.md", initial_content="# Introduction")',
                category="Chapter Management",
                planner_signature="create_chapter(document_name: str, chapter_name: str, initial_content: str)"
            ),
            ToolDescription(
                name="delete_chapter",
                description="Deletes a chapter",
                parameters={"document_name": "str", "chapter_name": "str"},
                example='delete_chapter(document_name="My Book", chapter_name="01-introduction.md")',
                category="Chapter Management",
                planner_signature="delete_chapter(document_name: str, chapter_name: str)"
            ),
            ToolDescription(
                name="read_chapter_content",
                description="Reads a specific chapter",
                parameters={"document_name": "str", "chapter_name": "str"},
                example='read_chapter_content(document_name="My Book", chapter_name="01-intro.md")',
                category="Chapter Management",
                planner_signature="read_chapter_content(document_name: str, chapter_name: str)"
            ),
            ToolDescription(
                name="write_chapter_content",
                description="Overwrites chapter content",
                parameters={"document_name": "str", "chapter_name": "str", "new_content": "str"},
                example='write_chapter_content(document_name="My Book", chapter_name="01-intro.md", new_content="# New Content")',
                category="Chapter Management",
                planner_signature="write_chapter_content(document_name: str, chapter_name: str, new_content: str)"
            ),
            ToolDescription(
                name="append_to_chapter_content",
                description="Appends content to end of chapter",
                parameters={"document_name": "str", "chapter_name": "str", "additional_content": "str"},
                example='append_to_chapter_content(document_name="My Book", chapter_name="01-intro.md", additional_content="More content")',
                category="Chapter Management",
                planner_signature="append_to_chapter_content(document_name: str, chapter_name: str, additional_content: str)"
            ),
            
            # Reading Operations (2 tools)
            ToolDescription(
                name="read_full_document",
                description="Reads all chapters of a document",
                parameters={"document_name": "str"},
                example='read_full_document(document_name="My Book")',
                category="Reading Operations",
                planner_signature="read_full_document(document_name: str)"
            ),
            ToolDescription(
                name="read_paragraph_content",
                description="Reads a specific paragraph",
                parameters={"document_name": "str", "chapter_name": "str", "paragraph_index": "int"},
                example='read_paragraph_content(document_name="My Book", chapter_name="01-intro.md", paragraph_index_in_chapter=0)',
                category="Reading Operations",
                planner_signature="read_paragraph_content(document_name: str, chapter_name: str, paragraph_index: int)"
            ),
            
            # Paragraph Operations (8 tools)
            ToolDescription(
                name="append_paragraph_to_chapter",
                description="Adds content to end of chapter",
                parameters={"document_name": "str", "chapter_name": "str", "paragraph_content": "str"},
                example='append_paragraph_to_chapter(document_name="My Book", chapter_name="01-intro.md", paragraph_content="New paragraph.")',
                category="Paragraph Operations",
                planner_signature="append_paragraph_to_chapter(document_name: str, chapter_name: str, paragraph_content: str)"
            ),
            ToolDescription(
                name="replace_paragraph",
                description="Replaces a specific paragraph",
                parameters={"document_name": "str", "chapter_name": "str", "paragraph_index": "int", "new_content": "str"},
                example='replace_paragraph(document_name="My Book", chapter_name="01-intro.md", paragraph_index=0, new_content="Updated text.")',
                category="Paragraph Operations",
                planner_signature="replace_paragraph(document_name: str, chapter_name: str, paragraph_index: int, new_content: str)"
            ),
            ToolDescription(
                name="insert_paragraph_before",
                description="Inserts paragraph before specified index",
                parameters={"document_name": "str", "chapter_name": "str", "paragraph_index": "int", "new_content": "str"},
                example='insert_paragraph_before(document_name="My Book", chapter_name="01-intro.md", paragraph_index=1, new_content="New paragraph.")',
                category="Paragraph Operations",
                planner_signature="insert_paragraph_before(document_name: str, chapter_name: str, paragraph_index: int, new_content: str)"
            ),
            ToolDescription(
                name="insert_paragraph_after",
                description="Inserts paragraph after specified index",
                parameters={"document_name": "str", "chapter_name": "str", "paragraph_index": "int", "new_content": "str"},
                example='insert_paragraph_after(document_name="My Book", chapter_name="01-intro.md", paragraph_index=0, new_content="New paragraph.")',
                category="Paragraph Operations",
                planner_signature="insert_paragraph_after(document_name: str, chapter_name: str, paragraph_index: int, new_content: str)"
            ),
            ToolDescription(
                name="delete_paragraph",
                description="Deletes a specific paragraph",
                parameters={"document_name": "str", "chapter_name": "str", "paragraph_index": "int"},
                example='delete_paragraph(document_name="My Book", chapter_name="01-intro.md", paragraph_index=2)',
                category="Paragraph Operations",
                planner_signature="delete_paragraph(document_name: str, chapter_name: str, paragraph_index: int)"
            ),
            ToolDescription(
                name="move_paragraph_before",
                description="Moves paragraph to new position",
                parameters={"document_name": "str", "chapter_name": "str", "move_paragraph_index": "int", "target_paragraph_index": "int"},
                example='move_paragraph_before(document_name="My Book", chapter_name="01-intro.md", paragraph_to_move_index=3, target_paragraph_index=1)',
                category="Paragraph Operations",
                planner_signature="move_paragraph_before(document_name: str, chapter_name: str, move_paragraph_index: int, target_paragraph_index: int)"
            ),
            ToolDescription(
                name="move_paragraph_to_end",
                description="Moves paragraph to end",
                parameters={"document_name": "str", "chapter_name": "str", "paragraph_index": "int"},
                example='move_paragraph_to_end(document_name="My Book", chapter_name="01-intro.md", paragraph_to_move_index=0)',
                category="Paragraph Operations",
                planner_signature="move_paragraph_to_end(document_name: str, chapter_name: str, paragraph_index: int)"
            ),
            
            # Text Processing (3 tools)
            ToolDescription(
                name="replace_text_in_chapter",
                description="Find and replace in chapter",
                parameters={"document_name": "str", "chapter_name": "str", "find_text": "str", "replace_text": "str"},
                example='replace_text_in_chapter(document_name="My Book", chapter_name="01-intro.md", text_to_find="old", replacement_text="new")',
                category="Text Processing",
                planner_signature="replace_text_in_chapter(document_name: str, chapter_name: str, find_text: str, replace_text: str)"
            ),
            ToolDescription(
                name="replace_text_in_document",
                description="Find and replace across document",
                parameters={"document_name": "str", "find_text": "str", "replace_text": "str"},
                example='replace_text_in_document(document_name="My Book", text_to_find="old_term", replacement_text="new_term")',
                category="Text Processing",
                planner_signature="replace_text_in_document(document_name: str, find_text: str, replace_text: str)"
            ),
            
            # Content Analysis (4 tools)
            ToolDescription(
                name="get_chapter_statistics",
                description="Gets word/paragraph counts for a chapter",
                parameters={"document_name": "str", "chapter_name": "str"},
                example='get_chapter_statistics(document_name="My Book", chapter_name="01-intro.md")',
                category="Content Analysis",
                planner_signature="get_chapter_statistics(document_name: str, chapter_name: str)"
            ),
            ToolDescription(
                name="get_document_statistics",
                description="Gets aggregate statistics for entire document",
                parameters={"document_name": "str"},
                example='get_document_statistics(document_name="My Book")',
                category="Content Analysis",
                planner_signature="get_document_statistics(document_name: str)"
            ),
            ToolDescription(
                name="find_text_in_chapter",
                description="Searches within a chapter",
                parameters={"document_name": "str", "chapter_name": "str", "search_text": "str"},
                example='find_text_in_chapter(document_name="My Book", chapter_name="01-intro.md", query="search term", case_sensitive=false)',
                category="Content Analysis",
                planner_signature="find_text_in_chapter(document_name: str, chapter_name: str, search_text: str)"
            ),
            ToolDescription(
                name="find_text_in_document",
                description="Searches across entire document",
                parameters={"document_name": "str", "search_text": "str"},
                example='find_text_in_document(document_name="My Book", query="search term", case_sensitive=false)',
                category="Content Analysis",
                planner_signature="find_text_in_document(document_name: str, search_text: str)"
            ),
        ]
    
    def get_tools_by_category(self) -> Dict[str, List[ToolDescription]]:
        """Get tools organized by category."""
        categories = {}
        for tool in self._tools:
            if tool.category not in categories:
                categories[tool.category] = []
            categories[tool.category].append(tool)
        return categories
    
    def get_tool_descriptions_text(self, format_type: ToolFormat = ToolFormat.FULL) -> str:
        """Generate tool descriptions text for prompt inclusion."""
        if format_type == ToolFormat.COMPACT:
            return self._generate_compact_format()
        elif format_type == ToolFormat.MINIMAL:
            return self._generate_minimal_format()
        elif format_type == ToolFormat.PLANNER:
            return self._generate_planner_format()
        else:
            return self._generate_full_format()
    
    def _generate_full_format(self) -> str:
        """Generate full format tool descriptions (ReAct agent format)."""
        categories = self.get_tools_by_category()
        sections = []
        
        for category, tools in categories.items():
            sections.append(f"**{category}:**")
            for tool in tools:
                sections.append(f"- `{tool.example}` - {tool.description}")
        
        return "\n".join(sections)
    
    def _generate_compact_format(self) -> str:
        """Generate compact format tool descriptions (Simple agent format)."""
        lines = []
        for tool in self._tools:
            lines.append(f"- `{tool.example}`: {tool.description}")
        return "\n".join(lines)
    
    def _generate_minimal_format(self) -> str:
        """Generate minimal format tool descriptions."""
        tool_names = [tool.name for tool in self._tools]
        return f"Available tools: {', '.join(tool_names)}"
    
    def _generate_planner_format(self) -> str:
        """Generate planner format tool descriptions with type hints."""
        categories = self.get_tools_by_category()
        sections = []
        
        for category, tools in categories.items():
            sections.append(f"**{category} ({len(tools)} tools):**")
            for tool in tools:
                sections.append(f"- `{tool.planner_signature}` - {tool.description}")
        
        return "\n".join(sections)
    
    def get_tool_count(self) -> int:
        """Get total number of tools."""
        return len(self._tools)
    
    def get_category_count(self) -> int:
        """Get number of tool categories."""
        return len(set(tool.category for tool in self._tools))
    
    def get_token_estimate(self, format_type: ToolFormat = ToolFormat.FULL) -> int:
        """Estimate token count for tool descriptions (rough approximation)."""
        text = self.get_tool_descriptions_text(format_type)
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4
    
    def compare_formats(self) -> Dict[str, tuple[int, int]]:
        """Compare character and token usage across different formats."""
        formats = {}
        for format_type in ToolFormat:
            text = self.get_tool_descriptions_text(format_type)
            formats[format_type.value] = (len(text), self.get_token_estimate(format_type))
        return formats


# Global instance
tool_manager = ToolDescriptionManager()


def get_tool_descriptions_for_agent(agent_type: str) -> str:
    """
    Get format-appropriate tool descriptions for the specified agent architecture.
    
    Each agent type uses a different format optimized for its operational pattern:
    - simple: Compact format for single-operation workflows
    - react: Full format with examples for multi-step reasoning  
    - planner: Type-annotated format for plan generation
    """
    format_map = {
        "simple": ToolFormat.COMPACT,
        "react": ToolFormat.FULL, 
        "planner": ToolFormat.PLANNER
    }
    
    format_type = format_map.get(agent_type.lower(), ToolFormat.FULL)
    return tool_manager.get_tool_descriptions_text(format_type)