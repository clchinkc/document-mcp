# Prompt Optimizer

Automated prompt optimization tool that safely improves agent prompts using performance benchmarking and comprehensive testing.

## ✨ Key Features

- **Safe Optimization**: Conservative changes that preserve all existing functionality
- **Performance-Based**: Uses real execution metrics to evaluate improvements
- **Comprehensive Testing**: Validates changes against 300 tests (unit + integration + E2E + evaluation + metrics)
- **Automatic Backup**: Safe rollback if optimization fails or breaks functionality
- **Multi-Agent Support**: Works with Simple, ReAct, and Planner agents

## 🚀 Quick Start

### Basic Usage

```bash
# Optimize all agents with uv (recommended)
uv run python -m prompt_optimizer all

# Optimize specific agent
uv run python -m prompt_optimizer simple
uv run python -m prompt_optimizer react

# Traditional Python (alternative)
python3 -m prompt_optimizer all
python3 -m prompt_optimizer simple
python3 -m prompt_optimizer react

# Development use within repo only
uv run python -m prompt_optimizer simple
```

### As Python Package

```python
from prompt_optimizer import PromptOptimizer
import asyncio

async def optimize():
    optimizer = PromptOptimizer()
    result = await optimizer.optimize_agent("simple")
    print(f"Improved: {result.keep_improvement}")

asyncio.run(optimize())
```

## 📁 Clean Architecture

```
prompt_optimizer/
├── __init__.py          # Package exports
├── __main__.py          # CLI entry point  
├── core.py              # Main PromptOptimizer class
├── evaluation.py        # Performance evaluation system
├── cli.py               # Command-line interface
└── README.md           # This documentation
```

## 🔧 How It Works

1. **Baseline Measurement**: Measures current prompt performance across all tests
2. **Conservative Optimization**: LLM generates minimal, safe improvements  
3. **Comprehensive Validation**: Runs 105 tests plus performance benchmarks
4. **Decision Logic**: Accepts only if tests pass AND performance improves
5. **Safety First**: Automatic backup and restore if anything breaks

## ⚙️ Requirements

- Python 3.9+
- Valid LLM API keys (OpenAI or Gemini)
- Working test suite (105 tests total)
- All dependencies from `pyproject.toml`

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Run from project root directory
2. **Test Failures**: Changes that break tests are automatically rejected
3. **API Errors**: Check LLM API keys in environment variables
4. **Permission Errors**: Ensure write access to prompt files and backup directory

### File Locations

- **Agent Prompts**: `src/agents/{agent_type}/prompts.py`
- **Backups**: `prompt_backups/{agent}_prompt_backup_{timestamp}.py`
- **Tests**: `tests/unit/`, `tests/integration/`, `tests/e2e/`
