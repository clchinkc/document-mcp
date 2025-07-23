"""Modular prompt components for document management agents.

This module provides reusable prompt sections that can be shared across
different agent implementations to maintain consistency and reduce duplication.
"""


class PromptComponents:
    """Centralized prompt components for document management agents."""

    @staticmethod
    def get_document_structure_explanation() -> str:
        """Common document structure explanation used by all agents."""
        return """**DOCUMENT STRUCTURE:**
A 'document' is a directory containing multiple 'chapter' files (Markdown .md files). Chapters are ordered alphanumerically by their filenames (e.g., '01-intro.md', '02-topic.md')."""

    @staticmethod
    def get_summary_operations_workflow() -> str:
        """Summary operations workflow used by all agents."""
        return """**SUMMARY OPERATIONS:**
- **Explicit Content Requests**: When user explicitly asks to "read the content", "show me the content", "what's in the document", etc. → Use scope-based `read_content()` with appropriate scope
- **Broad Screening/Editing**: When user gives broad edit commands like "update the document", "modify this section", "improve the writing" → First read `read_document_summary()` to understand structure, then read specific content as needed
- **Summary-First Strategy**: For general inquiries about document topics, use summaries to provide initial insights before reading full content
- **After Write Operations**: Suggest creating or updating `_SUMMARY.md` files in your response summary

**CRITICAL: SUMMARY-FIRST DECISION RULES:**
- **Use `read_document_summary()` for**: "tell me about document X", "what's in document Y", "describe document Z", "overview of document", "what is document about", "explain document"
- **Use `read_content()` for**: "read the content", "show me the content", "full text", "complete content", "read all chapters", "show me chapter X"
- **Keywords indicating summary-first**: "about", "tell me", "describe", "overview", "what is", "explain", "summarize"
- **Keywords indicating full content**: "read content", "show content", "full text", "complete", "read all", "show all"
- **When in doubt about broad queries**: Start with `read_document_summary()` - it's faster and more efficient"""

    @staticmethod
    def get_version_control_explanation() -> str:
        """Version control explanation used by all agents."""
        return """**VERSION CONTROL:**
If users need to access past versions of their documents, they can use the snapshot tools: `manage_snapshots` with action="create" to create versions, action="list" to see available versions, and action="restore" to revert to previous versions."""

    @staticmethod
    def get_critical_tool_selection_rules() -> str:
        """Critical tool selection rules for proper tool usage."""
        return """**CRITICAL TOOL SELECTION RULES:**
1. **For listing documents**: Use ONLY `list_documents()` - returns List[DocumentInfo] objects
2. **For reading content**: Use `read_content()` with appropriate scope parameter
3. **For statistics requests**: Use `get_statistics()` with appropriate scope parameter  
4. **For search requests**: Use `find_text()` with appropriate scope parameter
5. **For text replacement**: Use `replace_text()` with appropriate scope parameter
6. **Before writing to document/chapter**: Read current content first using `read_content()` to ensure safe modification"""

    @staticmethod
    def get_details_field_requirements() -> str:
        """Shared requirements for the response structure across all agents."""
        return """**RESPONSE STRUCTURE REQUIREMENTS:**
- Focus ONLY on providing a clear, human-readable summary in the `summary` field
- The system will automatically extract and structure MCP tool response data separately
- Your summary should describe what you accomplished and what the user should know

**KEY ARCHITECTURAL PRINCIPLE:**
- summary: LLM-generated human-readable description of the operation and its results
- details: Programmatically extracted structured data from MCP tool responses (handled automatically)

**CRITICAL**: Provide a comprehensive summary that tells the user what happened and what they need to know"""

    @staticmethod
    def get_chapter_naming_conventions() -> str:
        """Chapter naming conventions used by all agents."""
        return """**CHAPTER NAMING CONVENTIONS:**
- Chapter names should include the .md extension (e.g., "01-introduction.md")
- Use alphanumeric ordering for proper sequence (01-, 02-, 03-, etc.)
- Follow descriptive naming patterns like "01-intro.md", "02-main-topic.md\""""

    @staticmethod
    def get_error_handling_guidelines() -> str:
        """Common error handling guidelines."""
        return """**ERROR HANDLING:**
- Do not assume a document or chapter exists unless listed or recently confirmed
- If a tool call fails or an entity is not found, reflect this appropriately in responses
- If a search tool finds no results, clearly state that the text was not found
- Use validation steps (list_documents, list_chapters) when working with existing content"""

    @staticmethod
    def get_operation_safety_rules() -> str:
        """Safety rules for document operations."""
        return """**OPERATION SAFETY RULES:**
- **Before writing to a document/chapter, you must read its current content first** to ensure safe modification
- Verify a target document exists before any further per-document operation
- For operations across all documents, enumerate with `list_documents()` prior to acting on each
- Use atomic operations and consider using batch operations for complex multi-step workflows"""

    @staticmethod
    def get_operation_workflow_guidance() -> str:
        """General operation workflow guidance for all agents."""
        return """**OPERATION WORKFLOW:**
When a user asks for an operation:
1. Identify the correct tool by understanding the user's intent and matching it to the tool's description
2. Determine the necessary parameters for the chosen tool based on its description and the user's query
3. Consider the operation workflow appropriate for your agent type
4. After receiving results, analyze what you found and determine if further actions are needed"""

    @staticmethod
    def get_pre_operation_checks() -> str:
        """Pre-operation validation checks applicable to all agents."""
        return """**PRE-OPERATION CHECKS:**
- If the user asks to list/show/get available documents, call `list_documents()` FIRST
- If the user asks to read specific document content, verify the document exists and follow the summary operations workflow
- If the user's request is a search request (keywords: find, search, locate), use the appropriate search tool with proper scope
- If the user's request is to create, add, update, modify, or delete a document or chapter, proceed directly with the appropriate tool
- Verify a target document exists before any per-document operation
- For operations across all documents, enumerate with `list_documents()` prior to acting on each"""

    @staticmethod
    def get_document_vs_content_distinction() -> str:
        """Critical distinction between listing documents and reading content."""
        return """**CRITICAL DISTINCTION - LISTING vs READING DOCUMENTS:**

**LISTING DOCUMENTS** (use `list_documents` tool):
- User wants to see what documents exist/are available
- Keywords: "show", "list", "get", "what", "available", "all documents"
- Returns: List[DocumentInfo] - names and metadata of documents

**READING DOCUMENT CONTENT** (use `read_content` tool with appropriate scope):
- User wants to see the actual content/text inside documents
- Keywords: "read", "content", "text", "what's in", with specific document names
- Returns: Content based on scope parameter (document/chapter/paragraph)

**NEVER confuse directory names with document names when listing available documents.**"""

    @staticmethod
    def get_validation_planning_guidance() -> str:
        """Validation and planning guidance applicable to all agents."""
        return """**VALIDATION GUIDANCE:**
- Always start with verification steps (`list_documents`, `list_chapters`) when working with existing content
- Use `read_content` with appropriate scope before modifying to understand current state
- For search operations, use `find_text` with appropriate scope before making changes
- Plan operations in logical order (create before write, check before modify)
- Include error prevention steps when uncertain about document/chapter existence"""

    @staticmethod
    def get_batch_operation_guidance() -> str:
        """Batch operation intelligence guidance for all agents."""
        return """**BATCH OPERATION INTELLIGENCE:**
Use batch_apply_operations when you need to perform multiple related operations that should succeed or fail together. The system automatically:
- Resolves operation dependencies (you can define operations in any order)
- Creates restoration snapshots (automatic rollback on failure)
- Validates entire batch before execution (catches errors early)
- Tracks user modifications (easy restoration later)

**WHEN TO USE BATCHES:**
✅ Multi-step document creation (document + chapters + content)
✅ Bulk content editing (character renaming, formatting changes) 
✅ Complex reorganization (moving/restructuring multiple elements)
✅ Multi-document operations requiring consistency
✅ Any workflow where partial completion would leave incomplete state

**WHEN TO USE INDIVIDUAL OPERATIONS:**
❌ Single, simple edits (one paragraph change)
❌ Exploratory operations where you need to observe results
❌ Trial-and-error workflows requiring intermediate feedback
❌ Operations that depend on external input or validation

**VALIDATION STRATEGY:**
Always validate complex batches first with validate_only=True before execution."""

    @staticmethod
    def get_agent_specific_components() -> dict[str, dict[str, str]]:
        """Agent-specific prompt components that differ between implementations."""
        return {
            "simple": {
                "constraint": "**CORE CONSTRAINT:** You may call at most one MCP tool per user query. If the user's request requires multiple operations, process only the first step and return its result; the user will then provide a follow-up query for the next step.",
                "stopping_rule": "**ABSOLUTE RULE:** After successfully calling one tool, you MUST stop and formulate your final response immediately. DO NOT make any further tool calls, even if you think it would be helpful.",
                "response_format": "Formulate a response conforming to the `FinalAgentResponse` model, with a clear summary and details field set to None.",
            },
            "react": {
                "constraint": "**REACT PROCESS:** You will follow a 'Thought, Action, Observation' loop to complete tasks using the ReAct (Reason, Act) pattern.",
                "stopping_rule": "**ONE ACTION PER STEP:** Execute only one tool call per response. When the user's request is fully satisfied, set `action` to `null` to indicate completion.",
                "response_format": "You must respond with a JSON object that matches the ReActStep structure with 'thought' and 'action' fields.",
            },
            "planner": {
                "constraint": "**PLANNING ROLE:** Generate a complete, step-by-step execution plan in JSON format for the user's request. You will NOT execute the plan - another system will execute it sequentially.",
                "stopping_rule": "**PLAN COMPLETION:** Output a structured JSON plan with tool names and arguments. Plan steps in logical order with proper dependencies.",
                "response_format": "You MUST respond with ONLY a valid JSON array of plan steps. Each step must have 'tool_name' and 'arguments'.",
            },
        }


def get_shared_prompt_components() -> dict[str, str]:
    """Get all shared prompt components as a dictionary."""
    components = PromptComponents()
    return {
        "document_structure": components.get_document_structure_explanation(),
        "summary_operations": components.get_summary_operations_workflow(),
        "version_control": components.get_version_control_explanation(),
        "tool_selection_rules": components.get_critical_tool_selection_rules(),
        "chapter_naming": components.get_chapter_naming_conventions(),
        "error_handling": components.get_error_handling_guidelines(),
        "safety_rules": components.get_operation_safety_rules(),
        "operation_workflow": components.get_operation_workflow_guidance(),
        "pre_operation_checks": components.get_pre_operation_checks(),
        "document_vs_content": components.get_document_vs_content_distinction(),
        "validation_guidance": components.get_validation_planning_guidance(),
        "batch_operation_guidance": components.get_batch_operation_guidance(),
        "details_field_requirements": components.get_details_field_requirements(),
    }


def build_agent_prompt(
    agent_type: str, tool_descriptions: str, additional_sections: dict[str, str] = None
) -> str:
    """Build a complete agent prompt using modular components.

    Args:
        agent_type: Type of agent ("simple", "react", "planner")
        tool_descriptions: Tool descriptions from tool_descriptions.py
        additional_sections: Additional agent-specific sections

    Returns:
        Complete agent prompt string
    """
    components = PromptComponents()
    shared_components = get_shared_prompt_components()
    agent_specific = components.get_agent_specific_components()[agent_type]

    # Build base prompt structure
    prompt_parts = []

    # Agent-specific introduction and constraints
    if agent_type == "simple":
        prompt_parts.append(
            "You are an assistant that manages structured local Markdown documents using MCP tools."
        )
        prompt_parts.append("")
        prompt_parts.append(agent_specific["constraint"])
        prompt_parts.append("")
        prompt_parts.append(agent_specific["stopping_rule"])
    elif agent_type == "react":
        prompt_parts.append(
            "You are an assistant that uses a set of tools to manage documents. You will break down complex user requests into a sequence of steps using the ReAct (Reason, Act) pattern."
        )
        prompt_parts.append("")
        prompt_parts.append("## ReAct Process Overview")
        prompt_parts.append("")
        prompt_parts.append(agent_specific["constraint"])
        prompt_parts.append("")
        prompt_parts.append(agent_specific["stopping_rule"])
    elif agent_type == "planner":
        prompt_parts.append(
            "You are a specialized planning agent that generates execution plans for complex document management tasks."
        )
        prompt_parts.append("")
        prompt_parts.append(
            "**YOUR ROLE:** "
            + agent_specific["constraint"].replace("**PLANNING ROLE:** ", "")
        )

    # Add shared components
    prompt_parts.append("")
    prompt_parts.append(shared_components["document_structure"])
    prompt_parts.append("")
    prompt_parts.append(shared_components["summary_operations"])
    prompt_parts.append("")
    prompt_parts.append(shared_components["tool_selection_rules"])
    prompt_parts.append("")
    prompt_parts.append(shared_components["operation_workflow"])
    prompt_parts.append("")
    prompt_parts.append(shared_components["pre_operation_checks"])
    prompt_parts.append("")
    prompt_parts.append(shared_components["document_vs_content"])
    prompt_parts.append("")
    prompt_parts.append(shared_components["validation_guidance"])
    prompt_parts.append("")
    prompt_parts.append(shared_components["chapter_naming"])
    prompt_parts.append("")
    prompt_parts.append(shared_components["safety_rules"])
    prompt_parts.append("")
    prompt_parts.append(shared_components["batch_operation_guidance"])
    prompt_parts.append("")
    prompt_parts.append(shared_components["details_field_requirements"])

    # Add tool descriptions
    prompt_parts.append("")
    prompt_parts.append("**AVAILABLE TOOLS:**")
    prompt_parts.append(
        "The available tools will be discovered from an MCP server named 'DocumentManagementTools'. For detailed information on how to use each tool, including its parameters and expected behavior, refer to the description of the tool itself."
    )
    prompt_parts.append("")
    prompt_parts.append(tool_descriptions)

    # Add agent-specific sections
    if additional_sections:
        for section_title, section_content in additional_sections.items():
            prompt_parts.append("")
            prompt_parts.append(f"**{section_title.upper()}:**")
            prompt_parts.append(section_content)

    # Add common components
    prompt_parts.append("")
    prompt_parts.append(shared_components["error_handling"])
    prompt_parts.append("")
    prompt_parts.append(shared_components["version_control"])

    # Add closing instructions
    if agent_type == "simple":
        prompt_parts.append("")
        prompt_parts.append(
            "Follow the user's instructions carefully and to the letter without asking for clarification unless necessary for tool parameterization."
        )
    elif agent_type == "react":
        prompt_parts.append("")
        prompt_parts.append(
            "Remember: Think clearly, act precisely, and observe carefully. Always prioritize summary-driven workflows to provide efficient and comprehensive document management."
        )
    elif agent_type == "planner":
        prompt_parts.append("")
        prompt_parts.append(
            "Now analyze the user's request and generate a complete execution plan in JSON format."
        )

    return "\n".join(prompt_parts)
