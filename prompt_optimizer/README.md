# Prompt Optimizer

Automated prompt optimization tool that safely improves agent prompts using performance benchmarking and comprehensive testing.

## Key Features

- **Safe Optimization**: Conservative changes that preserve all existing functionality
- **Performance-Based**: Uses real execution metrics to evaluate improvements
- **Comprehensive Testing**: Validates changes against 547+ tests (unit + integration + E2E + evaluation + metrics)
- **Automatic Backup**: Safe rollback if optimization fails or breaks functionality
- **Multi-Agent Support**: Works with Simple and ReAct agents

## Quick Start

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

## Architecture

```
prompt_optimizer/
├── __init__.py          # Package exports
├── __main__.py          # CLI entry point
├── core.py              # Main PromptOptimizer class
├── evaluation.py        # Performance evaluation system
├── cli.py               # Command-line interface
└── README.md            # This documentation
```

## How It Works

1. **Baseline Measurement**: Measures current prompt performance across evaluation benchmarks
2. **Conservative Optimization**: LLM generates minimal, safe improvements
3. **Comprehensive Validation**: Runs 547 tests plus performance benchmarks
4. **Decision Logic**: Accepts only if tests pass AND performance improves
5. **Safety First**: Automatic backup and restore if anything breaks

## Integration with Benchmarking Infrastructure

The optimizer integrates with the evaluation infrastructure:

- **Tool Selection Benchmarks**: `tests/evaluation/test_tool_selection_benchmark.py`
  - 37 scenarios across 7 categories
  - Measures tool selection accuracy

- **A/B Description Testing**: `tests/evaluation/test_ab_descriptions.py`
  - Compare description variants (Current, Improved, Minimal)
  - Multi-model testing (GPT-5, Gemini-3, Claude-4.5)

- **Performance Metrics**: `tests/evaluation/config.py`
  - Token usage tracking
  - Execution time thresholds
  - Tool call counting

## Requirements

- Python 3.9+
- Valid LLM API keys (OpenAI, Gemini, or OpenRouter)
- Working test suite (547 tests total)
- All dependencies from `pyproject.toml`

## Troubleshooting

### Common Issues

1. **Import Errors**: Run from project root directory
2. **Test Failures**: Changes that break tests are automatically rejected
3. **API Errors**: Check LLM API keys in environment variables
4. **Permission Errors**: Ensure write access to prompt files and backup directory

### File Locations

- **Agent Prompts**: `src/agents/{agent_type}/prompts.py`
- **Backups**: `prompt_backups/{agent}_prompt_backup_{timestamp}.py`
- **Tests**: `tests/unit/`, `tests/integration/`, `tests/e2e/`
- **Benchmarks**: `tests/evaluation/`
