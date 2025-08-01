[![codecov](https://codecov.io/gh/clchinkc/document-mcp/graph/badge.svg?token=TEGUTD2DIF)](https://codecov.io/gh/clchinkc/document-mcp)
[![Python Tests with Coverage](https://github.com/clchinkc/document-mcp/actions/workflows/python-test.yml/badge.svg)](https://github.com/clchinkc/document-mcp/actions/workflows/python-test.yml)
# Document MCP

[![PyPI version](https://badge.fury.io/py/document-mcp.svg)](https://badge.fury.io/py/document-mcp)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Document MCP exists to **complement and diversify the predominantly STEM-oriented toolsets (e.g. Claude Code, bash/grep agents)** by giving writers, researchers, and knowledge-managers first-class, local-first control over large-scale Markdown documents with **built-in safety features** that prevent content loss.

## 🚀 Quick Start

### Installation

**PyPI Installation (Recommended):**
```bash
# Simple one-command install
pip install document-mcp

# Verify installation
document-mcp --version
```

**Development Setup:**
```bash
# Install uv package manager (if not already installed)
# macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Clone and install with development dependencies
git clone https://github.com/clchinkc/document-mcp.git
cd document-mcp
uv sync

# Install with development dependencies (includes testing tools)
uv pip install ".[dev]"
```

**Dependencies Information:**
- **Core Runtime**: The package includes all necessary dependencies for normal operation
- **Development Dependencies**: Additional tools for testing and development are available via `pip install ".[dev]"`:
  - `pytest`, `pytest-asyncio`, `pytest-cov` - Testing framework
  - `ruff`, `mypy` - Code quality and type checking  
  - `memory-profiler`, `psutil` - Performance monitoring
  - `twine` - Package validation and publishing

# Alternative: Traditional virtual environment setup
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### Configuration

```bash
# Create .env file with your API key
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
# OR for Google Gemini:
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env

# Verify your setup
python3 src/agents/simple_agent/main.py --check-config
```

### Quick Test

```bash
# Start MCP server (in one terminal)
# If installed via PyPI:
document-mcp stdio

# If using development setup:  
python3 -m document_mcp.doc_tool_server stdio

# Test basic functionality (in another terminal)  
python3 -c "from document_mcp import __version__; print(f'Document MCP v{__version__} ready')"
```

**📖 [Complete Manual Testing Guide](docs/manual_testing.md)** - Step-by-step workflows for creative writing, editing, and document management

## 🛠️ Development

### Modern Toolchain
This project uses modern Python development tools for enhanced performance and developer experience:

- **`uv`**: Ultra-fast Python package manager and dependency resolver (10-100x faster than pip)
- **`ruff`**: Lightning-fast Python linter and formatter (replaces black, isort, flake8, pydocstyle, autoflake)
- **`mypy`**: Static type checking for enhanced code quality
- **`pytest`**: Comprehensive testing framework with async support

### Testing Strategy
```bash
# Run all tests with uv (recommended)
uv run pytest

# Run with traditional python3 (for compatibility)
python3 -m pytest

# Run by test tier
uv run pytest tests/unit/              # Unit tests (fastest, no external deps)
uv run pytest tests/integration/       # Integration tests (real MCP, mocked LLM)
uv run pytest tests/e2e/               # E2E tests (requires API keys, 600s timeout)

# Run with coverage
uv run pytest --cov=document_mcp --cov-report=html

# Code quality checks
uv run ruff check                       # Lint code
uv run ruff check --fix                 # Auto-fix linting issues  
uv run ruff format                      # Format code
uv run mypy document_mcp/               # Type checking

# Quality checks script (uses uv and ruff internally)
python3 scripts/quality.py full
```

### Running the System
```bash
# Start MCP server (stdio transport)
uv run python -m document_mcp.doc_tool_server stdio
# Alternative: python3 -m document_mcp.doc_tool_server stdio

# Test agents
uv run python src/agents/simple_agent/main.py --query "list all documents"
uv run python src/agents/react_agent/main.py --query "create a book with multiple chapters"

# Interactive mode
uv run python src/agents/simple_agent/main.py --interactive
uv run python src/agents/react_agent/main.py --interactive

# Optimize agent prompts
uv run python -m prompt_optimizer simple     # Optimize specific agent
uv run python -m prompt_optimizer all        # Optimize all agents
uv run python -m prompt_optimizer simple  # Development use within repo only

# Development infrastructure testing
python3 scripts/development/metrics/test_production.py           # Test production metrics
python3 scripts/development/telemetry/scripts/test.py           # Test telemetry infrastructure
scripts/development/telemetry/scripts/start.sh                  # Start development telemetry
```

### Environment Configuration

Create a `.env` file with your API key according to `.env.example`, and fill in the required values.

### Running the System

## 📖 What is Document MCP?

Document MCP provides a structured way to manage large documents composed of multiple chapters. Think of it as a file system specifically designed for books, research papers, documentation, or any content that benefits from being split into manageable sections.

### Key Features

- **🛡️ Built-in Safety Features**: Write-safety system, automatic micro-snapshots, and comprehensive version control prevent content loss.
- **📁 Document Structure**: Organize content as directories with chapter files.
- **🔧 25 MCP Tools**: Comprehensive document manipulation API organized in 6 functional categories with tools for atomic paragraph operations, content analysis, semantic search, fine-grain summaries, and more.
- **🤖 AI Agents**: 
    - **Simple Agent**: Stateless, single-turn execution for discrete operations.
    - **ReAct Agent**: Stateful, multi-turn agent for complex workflows.
    - **Planner Agent**: Strategic planning with execution for complex task decomposition.
- **🚀 Prompt Optimizer**: Automated prompt optimization with performance benchmarking and real LLM evaluation.
- **📊 Observability**: Structured logging with OpenTelemetry and Prometheus metrics.
- **✅ Robust Testing**: 3-tier testing strategy (unit, integration, E2E, evaluation).
- **🔄 Version Control Friendly**: Plain Markdown files work great with Git.

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

## 🛡️ Safety Features

Document MCP includes comprehensive safety features designed to prevent content loss and enable safe writing workflows:

### Write-Safety System
- **Content Freshness Checking**: Automatically validates content modification status before writes
- **Conflict Detection**: Warns about potential overwrites if content has been modified externally
- **Safe Write Operations**: All write operations include safety information and warnings

### Automatic Version Control
- **Micro-Snapshots**: Automatic snapshots created before destructive operations
- **Named Snapshots**: Create checkpoints with messages using `snapshot_document`
- **Version Restoration**: Restore to any previous version using `restore_snapshot`
- **Intelligent Diffs**: Compare versions with prose-aware diff algorithms

### Modification History
- **Complete Tracking**: All document changes tracked with timestamps and operation details
- **Audit Trail**: Full history accessible via `get_modification_history`
- **Pattern Analysis**: Understand content change patterns over time

## 🏗️ Project Structure

```
document-mcp/
├── document_mcp/           # Core MCP server package
│   ├── doc_tool_server.py  # Main server with modular tool registrations
│   ├── tools/              # Modular tool architecture (26 tools)
│   │   ├── __init__.py     # Tool registration system
│   │   ├── document_tools.py    # Document management and summaries (6 tools)
│   │   ├── chapter_tools.py     # Chapter operations (5 tools)
│   │   ├── paragraph_tools.py   # Paragraph editing (7 tools)
│   │   ├── content_tools.py     # Unified content access (5 tools)
│   │   └── safety_tools.py      # Version control (3 tools)
│   ├── logger_config.py    # Structured logging with OpenTelemetry
│   └── metrics_config.py   # Prometheus metrics and monitoring
├── src/agents/             # AI agent implementations
│   ├── simple_agent/       # Stateless single-turn agent package
│   │   ├── main.py         # Agent execution logic
│   │   └── prompts.py      # System prompts
│   ├── react_agent/        # Stateful multi-turn ReAct agent
│   │   └── main.py
│   └── shared/             # Shared agent utilities
│       ├── cli.py          # Common CLI functionality
│       ├── config.py       # Enhanced Pydantic Settings
│       └── error_handling.py
├── prompt_optimizer/       # Automated prompt optimization tool
│   ├── core.py            # Main PromptOptimizer class
│   ├── evaluation.py      # Performance evaluation system
│   └── cli.py             # Command-line interface
└── tests/                  # 3-tier testing strategy
    ├── unit/              # Isolated component tests (mocked)
    ├── integration/       # Agent-server tests (real MCP, mocked LLM)
    ├── e2e/               # Full system tests (real APIs)
    └── evaluation/        # Performance benchmarking and prompt evaluation
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

Choose between three agent implementations:

- **Simple Agent**: Single-step operations, structured JSON output, fast performance
- **ReAct Agent**: Multi-step workflows, contextual reasoning, production reliability  
- **Planner Agent**: Strategic planning with execution, complex task decomposition

**Agent Selection:**
- Use Simple Agent for: direct operations, JSON output, prototyping
- Use ReAct Agent for: complex workflows, multi-step planning, production environments, reasoning transparency
- Use Planner Agent for: strategic planning, complex task decomposition, hierarchical execution

### 🚀 Prompt Optimization

The system includes an automated prompt optimizer that uses real performance benchmarks to improve agent efficiency:

```bash
# Optimize specific agent
python3 -m prompt_optimizer simple
python3 -m prompt_optimizer react
python3 -m prompt_optimizer planner

# Optimize all agents  
python3 -m prompt_optimizer all

```

**Key Features:**
- **Safe Optimization**: Conservative changes that preserve all existing functionality
- **Performance-Based**: Uses real execution metrics to evaluate improvements
- **Comprehensive Testing**: Validates changes against 296 tests (unit + integration + E2E + evaluation)
- **Automatic Backup**: Safe rollback if optimization fails or breaks functionality
- **Multi-Agent Support**: Works with Simple, ReAct, and Planner agents

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
python src/agents/simple_agent/main.py --check-config
```

**Test the Complete Workflow:**
```bash
# Start with simple operations
python src/agents/simple_agent/main.py --query "Create a new document called 'Test Document'"
python src/agents/simple_agent/main.py --query "Add a chapter named '01-intro.md' with content 'Hello World!'"
python src/agents/simple_agent/main.py --query "List all my documents"
python src/agents/simple_agent/main.py --query "Read the full document 'Test Document'"

# Try complex multi-step workflows
python src/agents/react_agent/main.py --query "Create a book outline with 3 chapters"
```

**Interactive Mode for Extended Testing:**
```bash
# Simple agent for straightforward tasks
python src/agents/simple_agent/main.py --interactive

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
python src/agents/simple_agent/main.py --check-config
python src/agents/react_agent/main.py --check-config

# Single query mode (both agents)
python src/agents/simple_agent/main.py --query "list all documents"
python src/agents/react_agent/main.py --query "create a book with multiple chapters"

# Interactive mode (both agents)
python src/agents/simple_agent/main.py --interactive
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

If you're experiencing issues, follow this support hierarchy:

1. **📖 [Check the Manual Testing Guide](docs/manual_testing.md#troubleshooting-guide)** - Comprehensive troubleshooting for common issues
2. **🔧 Run diagnostics**: `python src/agents/simple_agent/main.py --check-config`
3. **🧪 Test basic functionality**: `python src/agents/simple_agent/main.py --query "list documents"`
4. **🔍 Search existing issues**: [GitHub Issues](https://github.com/clchinkc/document-mcp/issues)
5. **💬 Open a new issue**: Include output from `--check-config` and system details

**Common Solutions:**
- API key issues → See [Configuration](#configuration) 
- MCP server connection → Check if `document-mcp stdio` is running
- Model loading failures → Try different model in `.env`
- Timeout issues → See [Performance troubleshooting](docs/manual_testing.md#performance-and-load-testing)

### Testing Strategy
```bash
# Run all tests
pytest

# Run by test tier
python3 -m pytest tests/unit/          # Unit tests (fastest, no external deps)
python3 -m pytest tests/integration/   # Integration tests (real MCP, mocked LLM)
python3 -m pytest tests/e2e/           # E2E tests (requires API keys)

# Run with coverage
python3 -m pytest --cov=document_mcp --cov-report=html

# Quality checks
python3 scripts/quality.py full
```

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
- **Test Coverage**: Comprehensive 3-tier testing (296 tests with 100% pass rate)

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

The system provides enterprise-grade reliability with **300 comprehensive tests** covering:

**Core Testing Areas:**
- **Document Operations**: Full CRUD operations and management (131 unit tests)
- **Agent Architecture**: Complete testing of both Simple and React agent implementations (153 integration tests)
- **MCP Protocol**: End-to-end server-client communication validation (6 E2E tests)
- **Performance Benchmarking**: Real API testing and prompt optimization (4 evaluation tests)
- **Monitoring & Metrics**: OpenTelemetry and Prometheus integration (6 metrics tests)
- **Quality Assurance**: Centralized fixtures and comprehensive cleanup validation

**Test Results:** 100% success rate (296/296 tests passing) with execution time under 3.5 minutes

The test suite spans unit, integration, and end-to-end categories, ensuring production-ready reliability with proper resource management and state isolation.

## 📚 Documentation

### User Guides
- **📖 [Manual Testing & User Workflows](docs/manual_testing.md)** - Complete guide for creative writing, editing workflows, and troubleshooting
- **🚀 [Quick Start Examples](#-quick-start)** - Get up and running in minutes
- **🤖 [Agent Architecture Guide](#-agent-examples-and-tutorials)** - Choose the right agent for your workflow

### Technical Reference  
- **[Package Documentation](document_mcp/README.md)** - MCP server API reference
- **[API Reference](document_mcp/doc_tool_server.py)** - Complete MCP tools documentation
- **[Testing Strategy](tests/README.md)** - 3-tier testing architecture and best practices

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
