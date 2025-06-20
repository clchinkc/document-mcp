[![codecov](https://codecov.io/gh/clchinkc/document-mcp/graph/badge.svg?token=TEGUTD2DIF)](https://codecov.io/gh/clchinkc/document-mcp)
# Document MCP

[![PyPI version](https://badge.fury.io/py/document-mcp.svg)](https://badge.fury.io/py/document-mcp)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Model Context Protocol (MCP) server and example agent for managing structured Markdown documents.

## 🚀 Quick Start

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in development mode
pip install -e .

# Install dependencies
pip install -r requirements.txt

# Set .env file with your API key (see below)

# Start the MCP server in one terminal
source .venv/bin/activate && python -m document_mcp.doc_tool_server sse --host localhost --port 3001

# Run the example agent (in another terminal, also activate virtual environment)
source .venv/bin/activate && python example_agent/agent.py
```

### Environment Configuration
Create a `.env` file with your API key (one of the following):
```bash
# For OpenAI (priority 1):
OPENAI_API_KEY="sk-your-openai-api-key-here"
OPENAI_MODEL_NAME="gpt-4.1-mini"  # optional

# For Gemini (priority 2):
GEMINI_API_KEY="your-google-api-key-here"
GEMINI_MODEL_NAME="gemini-2.5-flash"  # optional

# Optional settings:
DOCUMENT_ROOT_DIR=sample_doc/
```

## 📖 What is Document MCP?

Document MCP provides a structured way to manage large documents composed of multiple chapters. Think of it as a file system specifically designed for books, research papers, documentation, or any content that benefits from being split into manageable sections.

### Key Features

- **📁 Document Structure**: Organize content as directories with chapter files
- **🔧 MCP Integration**: Full HTTP SSE Model Context Protocol support for AI agents
- **🤖 AI Agent Example**: Ready-to-use Pydantic AI agent with natural language interface
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
│   ├── test_doc_tool_server.py # Test suite for MCP server
│   ├── __init__.py             # Package initializer
│   ├── conftest.py             # Pytest configuration for server tests
│   └── README.md               # Package documentation (PyPI)
├── example_agent/              # 💡 Usage examples and tutorials
│   ├── agent.py                # Example Pydantic AI agent
│   ├── test_agent.py           # Comprehensive test suite for the agent
│   └── conftest.py             # Pytest configuration for agent tests
├── pyproject.toml              # Package configuration
└── README.md                   # This file (GitHub)
```

## 🤖 Agent Examples and Tutorials

### Getting Started with the Example Agent

#### Prerequisites

1. **Set up virtual environment:**
   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install the package in development mode
   pip install -e .
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   Create a `.env` file in your project directory. The system automatically detects and uses the first available API key:
   ```env
   # === LLM Configuration (choose one) ===
   # The system will use the first available API key in this priority order:
   # 1. OpenAI (if OPENAI_API_KEY is set)
   # 2. Gemini (if GEMINI_API_KEY is set)
   
   # Option 1: OpenAI Configuration
   OPENAI_API_KEY=sk-your-openai-api-key-here
   OPENAI_MODEL_NAME=gpt-4.1-mini  # Optional: default is gpt-4.1-mini
   
   # Option 2: Gemini Configuration  
   GEMINI_API_KEY=your-google-api-key-here
   GEMINI_MODEL_NAME=gemini-2.5-flash  # Optional: default is gemini-2.5-flash
   
   # === Optional Settings ===
   # Custom document storage directory
   DOCUMENT_ROOT_DIR=sample_doc/
   
   # Optional: MCP server connection (defaults shown)
   MCP_SERVER_HOST=localhost
   MCP_SERVER_PORT=3001
   ```

3. **Start the MCP server:**
   ```bash
   source .venv/bin/activate && python -m document_mcp.doc_tool_server sse --host localhost --port 3001
   ```

4. **Run the example agent:**
   ```bash
   # Check your configuration first (optional)
   source .venv/bin/activate && python example_agent/agent.py --check-config
   
   # Run the interactive agent
   source .venv/bin/activate && python example_agent/agent.py
   ```

### What You'll Learn

The example demonstrates how to:
- Connect to the Document MCP Server
- Use Pydantic AI with Google Gemini for natural language processing
- Translate user requests into appropriate MCP tool calls
- Provide structured responses about document operations
- Handle errors gracefully
- Build extensible AI agents

### Architecture Overview

```
User Query → Pydantic AI Agent → MCP Client → Document MCP Server → File System
                ↓
         Structured Response ← Tool Results ← Tool Execution ← MCP Tools
```

The example uses:
- **Pydantic AI**: For LLM integration and structured outputs
- **Flexible LLM Support**: Automatically detects and uses OpenAI or Gemini based on available API keys
- **MCP Client**: To communicate with the document server
- **Structured Responses**: Using Pydantic models for consistent output

### Automatic LLM Detection

The system automatically detects which LLM to use based on your `.env` configuration:

1. **OpenAI** (Priority 1): If `OPENAI_API_KEY` is set, uses OpenAI models (default: `gpt-4.1-mini`)
2. **Gemini** (Priority 2): If `GEMINI_API_KEY` is set, uses Gemini models (default: `gemini-2.5-flash`)

When the agent starts, it will display which model it's using:
```
Using OpenAI model: gpt-4.1-mini
```
or
```
Using Gemini model: gemini-2.5-flash
```

### Example Interactions

#### Creating and Managing Documents

```
👤 User: Create a new document called 'My Novel'
🤖 Agent: Successfully created the new document named 'My Novel'.

👤 User: Add a chapter named '01-prologue.md' to 'My Novel' with content "# Prologue\nIt was a dark and stormy night..."
🤖 Agent: Chapter '01-prologue.md' created successfully in document 'My Novel'.

👤 User: List all my documents
🤖 Agent: Found 1 document: 'My Novel' with 1 chapter, 8 words total.
```

#### Reading and Analyzing Content

```
👤 User: Read the full document 'My Novel'
🤖 Agent: Retrieved the full content of document 'My Novel'.
Details: Complete document with all chapters and their content.

👤 User: Get statistics for 'My Novel'
🤖 Agent: Document 'My Novel' contains 1 chapter, 8 words, and 1 paragraph.

👤 User: Find "stormy" in document 'My Novel'
🤖 Agent: Found 1 paragraph containing "stormy" in chapter '01-prologue.md'.
```

#### Advanced Operations

```
👤 User: Replace "dark and stormy" with "bright and sunny" in document 'My Novel'
🤖 Agent: Replaced 1 occurrence across 1 chapter in document 'My Novel'.

👤 User: Add a new paragraph to chapter '01-prologue.md' in 'My Novel': "The rain poured down relentlessly."
🤖 Agent: Successfully appended paragraph to chapter '01-prologue.md'.
```

## 🧪 Testing

Run the test suite to verify everything is working correctly:

```bash
# Run all tests
source .venv/bin/activate && python -m pytest

# Run specific test suites
source .venv/bin/activate && python -m pytest document_mcp/     # Test the MCP server
source .venv/bin/activate && python -m pytest example_agent/    # Test the example agent

# Run with coverage
source .venv/bin/activate && python -m pytest --cov=document_mcp --cov=example_agent
```

The test suite includes:
- **MCP Server Tests**: Comprehensive testing of all document operations
- **Agent Tests**: Testing the AI agent integration and responses
- **Error Handling**: Testing edge cases and error conditions
- **Data Consistency**: Ensuring document integrity across operations

### Agent Features

#### Natural Language Processing
The agent understands various ways to express document operations:
- "Create a document" / "Make a new book" / "Start a project called..."
- "Add a chapter" / "Create a new section" / "Write a chapter named..."
- "Read the document" / "Show me the content" / "What's in my book?"

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

# Install development dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Running Tests

```bash
# Test the MCP server
cd document_mcp
pytest test_doc_tool_server.py -v

# Test the example agent
cd ../example_agent
pytest test_agent.py -v

# Run all tests
cd ..
python run_tests.py
```

#### Agent Test Coverage

The agent test suite covers:
- Document creation and management
- Chapter operations (create, read, update, delete)
- Content reading and writing
- Search and analysis features
- Error handling scenarios
- Agent response formatting
- Natural language understanding

## 📚 Documentation

- **[Package Documentation](document_mcp/README.md)**: MCP server API reference and installation
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
- Example agent uses [Google Gemini](https://ai.google.dev/)

---

⭐ **Star this repo** if you find it useful!