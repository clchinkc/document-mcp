from ..shared.prompt_components import build_agent_prompt
from ..shared.tool_descriptions import get_tool_descriptions_for_agent


# --- ReAct System Prompt ---
def get_react_system_prompt() -> str:
    """Generate the ReAct system prompt with dynamic tool descriptions and modular components."""
    tool_descriptions = get_tool_descriptions_for_agent("react")

    # ReAct agent specific sections
    additional_sections = {
        "react_json_format": """You must respond with a JSON object that matches this structure:
{{
    "thought": "Your reasoning about what to do next",
    "action": "tool_name(param1=\"value1\", param2=\"value2\")"
}}

When you have completed the user's request, provide a final thought and set the `action` to `null` or omit it entirely.""",
        "critical_instructions": """- **You must issue exactly one tool call in the `action` field per turn.**
- Do not chain commands.
- Do not ask for user confirmation.
- Do not explain your work or plan; just execute the next logical step.""",
        "react_example": """**Example ReAct Sequence**

User Query: "Create a document named 'Project Alpha' and add a chapter to it called 'introduction'."

**Step 1:**
{{
    "thought": "The user wants to create a document and then add a chapter. I need to do this in two steps. First, I will create the document.",
    "action": "create_document(document_name=\\"Project Alpha\\")"
}}

**Observation:** {{"success": true, "message": "Document 'Project Alpha' created successfully."}}

**Step 2:**
{{
    "thought": "The document 'Project Alpha' has been created successfully. Now I need to add the 'introduction' chapter to it. The chapter name should be a valid filename with .md extension.",
    "action": "create_chapter(document_name=\\"Project Alpha\\", chapter_name=\\"01-introduction.md\\", initial_content=\\"# Introduction\\")"
}}

**Observation:** {{"success": true, "message": "Chapter '01-introduction.md' created successfully in document 'Project Alpha'.", "details": {{"document_name": "Project Alpha", "chapter_name": "01-introduction.md"}}}}

**Step 3:**
{{
    "thought": "I have successfully created the document 'Project Alpha' and added the 'introduction' chapter with basic markdown content. The task is now complete.",
    "action": null
}}""",
        "important_guidelines": """1. **One Action Per Step**: Execute only one tool call per response.
2. **Valid JSON**: Always respond with valid JSON matching the ReActStep structure.
3. **Parameter Formatting**: Use proper string quoting and escaping in action calls.
4. **Error Handling**: If a tool returns an error, analyze the error message and adjust your approach in the next thought.
5. **Completion**: When the user's request is fully satisfied, set `action` to `null` to indicate completion.
6. **Step-by-Step**: Break complex requests into logical, sequential steps.""",
    }

    return build_agent_prompt("react", tool_descriptions, additional_sections)
