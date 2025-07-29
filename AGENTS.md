# Agent Usage Guide

The Document MCP system includes two AI agents that demonstrate how to use the MCP tools effectively. These agents are **separate from the core MCP package** and run directly from source code.

## Prerequisites

1. **Install the MCP package** (for the core tools):
   ```bash
   pip install document-mcp
   ```

2. **Clone the repository** (for the agents):
   ```bash
   git clone https://github.com/your-org/document-mcp.git
   cd document-mcp
   ```

3. **Install agent dependencies**:
   ```bash
   uv sync --dev
   # or
   pip install -r requirements-agents.txt
   ```

4. **Set up API keys** (choose one):
   ```bash
   # For OpenAI
   export OPENAI_API_KEY="your-openai-key"
   
   # For Google Gemini
   export GEMINI_API_KEY="your-gemini-key"
   ```

## Agent Types

### Simple Agent
**Best for**: Quick operations, single-step tasks, structured JSON output

```bash
# Basic usage
uv run python src/agents/simple_agent/main.py --query "list all documents"

# With specific model
uv run python src/agents/simple_agent/main.py --query "create document 'Meeting Notes'" --model "gpt-4o"

# Interactive mode
uv run python src/agents/simple_agent/main.py --interactive

# Check configuration
uv run python src/agents/simple_agent/main.py --check-config
```

### ReAct Agent  
**Best for**: Complex workflows, multi-step planning, production environments

```bash
# Basic usage
uv run python src/agents/react_agent/main.py --query "create a book with chapters on AI basics"

# With planning output
uv run python src/agents/react_agent/main.py --query "organize my research notes" --verbose

# Interactive mode with rich console
uv run python src/agents/react_agent/main.py --interactive
```

## Common Usage Patterns

### Document Management
```bash
# Create and populate documents
uv run python src/agents/simple_agent/main.py --query "create document 'Project Plan' with chapters: Overview, Timeline, Resources"

# Search content
uv run python src/agents/simple_agent/main.py --query "find content similar to 'machine learning' in all documents"

# Generate summaries
uv run python src/agents/react_agent/main.py --query "create summary for document 'Research Notes'"
```

### Content Operations  
```bash
# Batch content updates
uv run python src/agents/react_agent/main.py --query "update all chapters in 'User Guide' to include warning sections"

# Complex restructuring
uv run python src/agents/react_agent/main.py --query "reorganize 'Meeting Notes' by date and create index"
```

### Development and Testing
```bash
# Test semantic search
uv run python src/agents/simple_agent/main.py --query "find similar content to 'API documentation' in document 'Dev Guide'"

# Performance testing
uv run python src/agents/simple_agent/main.py --query "list all documents" --verbose --timing
```

## Configuration Options

### Environment Variables
```bash
# LLM Configuration
OPENAI_API_KEY=your-key
GEMINI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key

# Agent Settings
AGENT_TIMEOUT=300                # Request timeout in seconds
AGENT_MAX_RETRIES=3             # Retry attempts for failed operations
AGENT_LOG_LEVEL=INFO            # Logging level

# MCP Server Settings  
MCP_SERVER_TIMEOUT=30           # MCP tool timeout
DOCUMENTS_STORAGE_PATH=./docs   # Custom storage location
```

### Command Line Options
```bash
# Model selection
--model gpt-4o                  # Use specific OpenAI model
--model gemini-1.5-pro         # Use specific Gemini model

# Output control
--verbose                       # Detailed output
--json                         # JSON-formatted output (Simple Agent)
--quiet                        # Minimal output

# Operational settings
--timeout 300                   # Custom timeout
--interactive                  # Interactive mode
--check-config                 # Validate configuration
```

## Integration Examples

### Python Integration
```python
from src.agents.simple_agent.agent import SimpleAgent
from src.agents.shared.config import AgentConfig

# Initialize agent
config = AgentConfig(model="gpt-4o", timeout=120)
agent = SimpleAgent(config)

# Execute query
result = await agent.execute("list all documents")
print(result.summary)  # Human-readable summary
print(result.details)  # Structured data
```

### Shell Scripts
```bash
#!/bin/bash
# Automated document processing script

# Create daily report
uv run python src/agents/simple_agent/main.py \
  --query "create document 'Daily Report $(date +%Y%m%d)'" \
  --quiet

# Add meeting notes
uv run python src/agents/react_agent/main.py \
  --query "add meeting notes from today to Daily Report" \
  --verbose
```

### CI/CD Integration
```yaml
# .github/workflows/docs.yml
- name: Update Documentation
  run: |
    uv run python src/agents/react_agent/main.py \
      --query "update API documentation with latest changes" \
      --timeout 600
```

## Troubleshooting

### Common Issues
```bash
# Check agent configuration
uv run python src/agents/simple_agent/main.py --check-config

# Test MCP server connection
uv run python -m document_mcp.doc_tool_server stdio

# Debug with verbose output
uv run python src/agents/react_agent/main.py --query "test" --verbose
```

### Performance Optimization
```bash
# Use Simple Agent for quick operations
uv run python src/agents/simple_agent/main.py --query "list documents"

# Use ReAct Agent for complex workflows
uv run python src/agents/react_agent/main.py --query "complex multi-step task"

# Enable caching for repeated operations
export AGENT_CACHE_ENABLED=true
```

### Error Recovery
```bash
# Check logs
tail -f document_mcp/doc_operations.log

# Test with minimal query
uv run python src/agents/simple_agent/main.py --query "help"

# Verify API keys
uv run python src/agents/simple_agent/main.py --check-config
```

## Development

### Running Tests
```bash
# Test agents specifically
uv run pytest tests/unit/test_simple_agent.py
uv run pytest tests/integration/test_agents_mcp_integration.py

# Full E2E testing (requires API keys)
uv run pytest tests/e2e/ --timeout=600
```

### Adding Custom Agents
```python
# Create new agent in src/agents/custom_agent/
from src.agents.shared.agent_base import AgentBase

class CustomAgent(AgentBase):
    async def execute(self, query: str) -> AgentResponse:
        # Your implementation
        pass
```

---

**Note**: The agents are development tools and examples. For production use, consider implementing your own agent logic tailored to your specific requirements while using the MCP tools as the foundation.