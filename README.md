[![codecov](https://codecov.io/gh/clchinkc/document-mcp/graph/badge.svg?token=TEGUTD2DIF)](https://codecov.io/gh/clchinkc/document-mcp)
[![Python Tests with Coverage](https://github.com/clchinkc/document-mcp/actions/workflows/python-test.yml/badge.svg)](https://github.com/clchinkc/document-mcp/actions/workflows/python-test.yml)
# Document MCP

[![PyPI version](https://badge.fury.io/py/document-mcp.svg)](https://badge.fury.io/py/document-mcp)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Model Context Protocol (MCP) server and agent implementations for managing structured Markdown documents.

## 🚀 Quick Start


### Step-by-Step Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install the package with development dependencies
pip install -e ".[dev]"

# 3. Set up environment variables
#    Create a `.env` file with your API key according to `.env.example`, and fill in the required values.

# 4. Verify your setup
python src/agents/simple_agent.py --check-config
```

### Environment Configuration

Create a `.env` file with your API key according to `.env.example`, and fill in the required values.

### Running the System

## 📖 What is Document MCP?

Document MCP provides a structured way to manage large documents composed of multiple chapters. Think of it as a file system specifically designed for books, research papers, documentation, or any content that benefits from being split into manageable sections.

### Key Features

- **📁 Document Structure**: Organize content as directories with chapter files
- **🔧 MCP Integration**: Full HTTP SSE Model Context Protocol support for AI agents
- **🤖 Dual AI Agents**: Simple single-step agent and advanced ReAct multi-step agent
- **📊 Analytics**: Built-in statistics and search capabilities
- **🔄 Version Control Friendly**: Plain Markdown files work great with Git

### Document Organization

```
.documents_storage/
├── my_novel/                    # A document
│   ├── 01-prologue.md          # Chapters ordered by filename
│   ├── 02-chapter-one.md
│   └── 03-chapter-two.md
└── research_paper/             # Another document
    ├── 00-abstract.md
    ├── 01-introduction.md
    └── 02-methodology.md
```

## 🏗️ Project Structure

```
.
├── document_mcp/               # 📦 Main MCP server package
│   ├── doc_tool_server.py      # MCP server implementation
│   ├── __init__.py             # Package initializer
│   └── README.md               # Package documentation (PyPI)
├── src/                        # 💡 Agent implementations
│   └── agents/
│       ├── simple_agent.py     # Simple single-step agent
│       └── react_agent/
│           └── main.py         # Advanced ReAct multi-step agent
├── tests/                      # 🧪 Comprehensive test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── fixtures/               # Test fixtures and demos
├── docs/                       # 📚 Documentation
│   ├── README.md               # Agent architecture guide
│   └── examples/               # Usage examples
├── scripts/                    # 🛠️ Development utilities
│   ├── run_pytest.py           # Pytest test runner
│   └── quality.py              # Code quality management
├── pyproject.toml              # Package configuration
└── README.md                   # This file (GitHub)
```

## 🤖 Agent Examples and Tutorials

### Agent Architecture Overview

This project provides two distinct agent implementations for document management using the Model Context Protocol (MCP). Both agents interact with the same document management tools but use different architectural approaches and execution patterns.

#### Quick Reference

| Feature | Simple Agent | ReAct Agent |
|---------|-------------|-------------|
| **Architecture** | Single-step execution | Multi-step reasoning loop with intelligent termination |
| **Chat History** | No memory between queries | Full history maintained across steps |
| **Context Retention** | Each query is independent | Builds context from all previous steps |
| **Best For** | Simple queries, direct operations | Complex multi-step tasks |
| **Output** | Structured JSON response | Step-by-step execution log |
| **Error Handling** | Basic timeout handling | Advanced retry & circuit breaker |
| **Performance** | Fast for simple tasks | Optimized for complex workflows |
| **Complexity** | Simple, straightforward | Advanced with caching & optimization |
| **Termination Logic** | N/A (single step) | **Intelligent completion detection**: task completion, step limits, error recovery, timeout management |
| **Conversational Flow** | **Independent query processing** | **Multi-round conversation support** |
| **State Management** | **Clean isolation between queries** | **Persistent context with proper cleanup** |

### Getting Started with the Agents

This project provides two agent implementations with unified command patterns. Choose based on your needs:

- **Simple Agent** (`src/agents/simple_agent.py`): For straightforward single-step operations
- **ReAct Agent** (`src/agents/react_agent/main.py`): For complex multi-step workflows with reasoning



### When to Choose Each Agent

**Choose Simple Agent when**:
- You need simple, predictable single-step operations
- Each query should be processed independently without previous context
- Your application requires structured JSON output for parsing
- You're building integrations that expect specific response formats
- Performance is critical for simple operations
- You're prototyping or testing basic functionality

**Choose ReAct Agent when**:
- You have complex, multi-step workflows
- You need the agent to remember previous steps and build upon context
- You need transparency in the reasoning process
- You're working in production environments requiring robust error handling
- Your tasks benefit from adaptive problem-solving and cumulative learning
- You need detailed execution logs for debugging or auditing

### Practical Examples - Step by Step

Once you have both the MCP server and an agent running, try these structured examples:

#### Example 1: Create Your First Document
```bash
👤 User: Create a new document called 'Getting Started Guide'
🤖 Agent: ✅ Successfully created the new document named 'Getting Started Guide'.
```

#### Example 2: Add Content with Chapters
```bash
👤 User: Add a chapter named '01-introduction.md' to 'Getting Started Guide' with content '# Introduction\nWelcome to Document MCP!'
🤖 Agent: ✅ Chapter '01-introduction.md' created successfully in document 'Getting Started Guide'.
```

#### Example 3: View Your Documents
```bash
👤 User: List all my documents
🤖 Agent: ✅ Found 1 document: 'Getting Started Guide' with 1 chapter, 4 words total.
```

#### Example 4: Analyze Document Statistics
```bash
👤 User: Get statistics for 'Getting Started Guide'
🤖 Agent: ✅ Document 'Getting Started Guide' contains 1 chapter, 4 words, and 1 paragraph.
```

#### Example 5: Read Full Content
```bash
👤 User: Read the full document 'Getting Started Guide'
🤖 Agent: ✅ Retrieved the full content of document 'Getting Started Guide'.
   Content: # Introduction
   Welcome to Document MCP!
```

#### Advanced Examples

**Search and Replace Operations:**
```bash
👤 User: Find "Welcome" in document 'Getting Started Guide'
🤖 Agent: ✅ Found 1 paragraph containing "Welcome" in chapter '01-introduction.md'.

👤 User: Replace "Welcome" with "Hello" in document 'Getting Started Guide'
🤖 Agent: ✅ Replaced 1 occurrence across 1 chapter in document 'Getting Started Guide'.
```

**Complex Multi-Step Workflows (ReAct Agent):**
```bash
👤 User: Create a research paper structure with an abstract, introduction, methodology, and conclusion
🤖 ReAct Agent: 
   Thought: I need to create a document and then add multiple chapters for a research paper structure.
   Action: create_document with name 'Research Paper'
   Observation: Document created successfully
   Thought: Now I need to add the abstract chapter...
   Action: add_chapter with name '00-abstract.md'...
   [Continues with step-by-step execution]
```

#### Try It Now - Interactive Walkthrough

Once you have the MCP server running, you can immediately test all features:

**Quick Configuration Check:**
```bash
# Verify your setup is working
python src/agents/simple_agent.py --check-config
```

**Test the Complete Workflow:**
```bash
# Start with simple operations
python src/agents/simple_agent.py --query "Create a new document called 'Test Document'"
python src/agents/simple_agent.py --query "Add a chapter named '01-intro.md' with content 'Hello World!'"
python src/agents/simple_agent.py --query "List all my documents"
python src/agents/simple_agent.py --query "Read the full document 'Test Document'"

# Try complex multi-step workflows
python src/agents/react_agent/main.py --query "Create a book outline with 3 chapters"
```

**Interactive Mode for Extended Testing:**
```bash
# Simple agent for straightforward tasks
python src/agents/simple_agent.py --interactive

# ReAct agent for complex reasoning
python src/agents/react_agent/main.py --interactive
```

This immediate hands-on approach lets you:
- ✅ Verify your configuration works correctly
- 🚀 See real responses from both agent types
- 🧠 Experience the ReAct agent's reasoning process
- 📋 Build confidence before diving deeper

### Automatic LLM Detection

The system automatically detects which LLM to use based on your `.env` configuration:

1. **OpenAI** (Priority 1): If `OPENAI_API_KEY` is set, uses OpenAI models (default: `gpt-4.1-mini`)
2. **Gemini** (Priority 2): If `GEMINI_API_KEY` is set, uses Gemini models (default: `gemini-2.5-flash`)

When an agent starts, it will display which model it's using:
```
Using OpenAI model: gpt-4.1-mini
```
or
```
Using Gemini model: gemini-2.5-flash
```

### Configuration

Both agents share the same configuration system and support the same command-line interface:

```bash
# Check configuration (both agents)
python src/agents/simple_agent.py --check-config
python src/agents/react_agent/main.py --check-config

# Single query mode (both agents)
python src/agents/simple_agent.py --query "list all documents"
python src/agents/react_agent/main.py --query "create a book with multiple chapters"

# Interactive mode (both agents)
python src/agents/simple_agent.py --interactive
python src/agents/react_agent/main.py --interactive
```

## 🔧 Troubleshooting

### Setup Verification Checklist

Run through this checklist if you're having issues:

- [ ] ✅ `.env` file exists with valid API key
- [ ] ✅ Virtual environment activated (`source .venv/bin/activate`)
- [ ] ✅ Package and dependencies installed (`pip install -e ".[dev]"`)
- [ ] ✅ MCP server running on localhost:3001
- [ ] ✅ Configuration check passes (`--check-config`)

### Getting Help

If you're still having issues:

1. **Run the configuration check**: `python src/agents/simple_agent.py --check-config`
2. **Test basic functionality**: `python src/agents/simple_agent.py --query "list documents"`
3. **Review the test suite**: `python scripts/run_pytest.py`
4. **Check the documentation**: See the Agent Examples section above for detailed agent architecture

## 🧪 Testing

The project is tested using a three-tier strategy: unit, integration, and end-to-end (E2E) tests. We use `pytest` for test execution and `pytest-mock` for mocking dependencies.

### Running Tests

To run all tests, use the following command:
```bash
pytest
```

To run specific test categories:
```bash
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
pytest tests/e2e/           # E2E tests only (requires API keys)
```

To run tests with coverage:
```bash
pytest --cov=document_mcp --cov-report=html
```

To skip slow tests:
```bash
pytest -m "not slow"
```

### Test Structure

- **Unit Tests (`tests/unit/`)**: Test individual functions and classes in isolation. These tests use heavy mocking and do not have external dependencies.
- **Integration Tests (`tests/integration/`)**: Test component interactions and MCP server integration. These tests may use a real MCP server via stdio transport but use mocked LLMs.
- **E2E Tests (`tests/e2e/`)**: Test complete workflows with real AI models. These tests require real API keys and are skipped if they are not available.

### Key Fixtures
- **`test_docs_root`**: Creates an isolated temporary directory for each test, ensuring test isolation.
- **`mock_environment`**: Sets up mock API keys and environment variables.
- **`document_factory`**: A factory for creating various types of test documents.
- **`mocker`**: The standard `pytest-mock` fixture for mocking objects and functions.

For more details on the testing strategy, see the [Testing Guidelines](tests/testing_guidelines.md).

## 🛠️ Development Setup

### Prerequisites

- Python 3.8+
- Git

### Local Development

```bash
# Clone the repository
git clone https://github.com/document-mcp/document-mcp.git
cd document-mcp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in editable mode with development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Use the pytest runner
python scripts/run_pytest.py

# Or run pytest directly
python -m pytest tests/ -v
```
### Code Quality Management

This project maintains high code quality standards through automated tools and scripts. The quality system is managed through a dedicated script:

```bash
# Quick quality check (linting and type checking only)
python scripts/quality.py check

# Apply automatic fixes and format code
python scripts/quality.py fix        # Remove unused imports, fix issues
python scripts/quality.py format     # Black formatting + isort

# Run specific quality tools
python scripts/quality.py lint       # flake8 linting
python scripts/quality.py typecheck  # mypy type checking

# Complete quality pass (recommended before commits)
python scripts/quality.py full       # fix + format + check

# Get detailed output for debugging
python scripts/quality.py check --verbose
```

#### Quality Tools Configured

- **Black**: Code formatting (88 character line length)
- **isort**: Import sorting and organization  
- **flake8**: Linting and style checking (configured in `.flake8`)
- **mypy**: Static type checking (configured in `pyproject.toml`)
- **autoflake**: Automated cleanup of unused imports/variables
- **pytest**: Comprehensive test suite execution

#### Quality Standards

- **Line Length**: 88 characters (Black standard)
- **Import Style**: Black-compatible with isort
- **Type Hints**: Encouraged for public APIs
- **Complexity**: Maximum cyclomatic complexity of 10
- **Test Coverage**: Comprehensive unit and integration tests (445+ tests)

#### Recommended Workflow

```bash
# During development
python scripts/quality.py format    # Format as you code

# Before committing  
python scripts/quality.py full      # Complete quality pass
python scripts/run_pytest.py        # Run tests

# Quick check
python scripts/quality.py check     # Verify quality standards
```

The quality management system provides comprehensive automation for maintaining code standards throughout development.

#### Test Coverage

The system provides enterprise-grade reliability with **445+ comprehensive tests** covering:

**Core Testing Areas:**
- **Document Operations**: Full CRUD operations and management
- **Agent Architecture**: Complete testing of both Simple and React agent implementations  
- **MCP Protocol**: End-to-end server-client communication validation
- **Multi-round Conversations**: Complex workflows with state management and error recovery
- **Performance & Reliability**: Timeout handling, resource management, and cleanup verification

The test suite spans unit, integration, and end-to-end categories, ensuring production-ready reliability with proper resource management and state isolation.

## 📚 Documentation

- **[Package Documentation](document_mcp/README.md)**: MCP server API reference
- **[API Reference](document_mcp/doc_tool_server.py)**: Complete MCP tools documentation

## 🤝 Contributing

I welcome any contribution!

## 🔗 Related Resources

- **[Pydantic AI Documentation](https://ai.pydantic.dev/)**: Learn more about Pydantic AI
- **[MCP Specification](https://spec.modelcontextprotocol.io/)**: Model Context Protocol details
- **[Model Context Protocol (MCP)](https://github.com/modelcontextprotocol)**: Official MCP repository

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol)
- Powered by [Pydantic AI](https://github.com/pydantic/pydantic-ai)
- Agents support both [OpenAI](https://openai.com/) and [Google Gemini](https://ai.google.dev/)

---

⭐ **Star this repo** if you find it useful!
