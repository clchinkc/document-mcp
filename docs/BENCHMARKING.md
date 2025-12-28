# Benchmarking Infrastructure

**Status**: Active | **Last Updated**: December 29, 2024

This document covers the benchmarking and optimization infrastructure for testing LLM tool selection accuracy across multiple prompt variants with specialized metrics.

## Quick Links
- [Variant Comparison Results](VARIANT_RESULTS.md) - Architecture and December 28 baseline results
- [Variant Architecture](VARIANT_ARCHITECTURE.md) - System design for variant-specific metrics
- Scenario details: 185 total (Levels 1-6: simple, sequential, complex, ambiguous, edge cases, adversarial)
- Models tested: Gemini 3 Flash Preview, Claude Haiku 4.5, GPT-5 Mini

## Quick Start

### DSPy Optimization with Variant-Specific Metrics

```bash
# Optimize COST_OPTIMIZED variant (50/25/25 weighted score)
python -m dspy_optimizer --variant=cost_optimized --mode light

# Optimize ACCURACY_BALANCED variant (ratio metric)
python -m dspy_optimizer --variant=accuracy_balanced --mode light

# Optimize MAXIMUM_ACCURACY variant (pure accuracy)
python -m dspy_optimizer --variant=maximum_accuracy --mode light

# Compare all three variants with their respective metrics
python -m dspy_optimizer --all-variants --mode light

# Use medium or heavy optimization intensity for better results
python -m dspy_optimizer --all-variants --mode medium
python -m dspy_optimizer --all-variants --mode heavy
```

### Benchmark Infrastructure (Tool Sets and Scenarios)

```bash
# List available configurations
python -m benchmarks.runner --list

# Run benchmark with specific tool set
python -m benchmarks.runner --tool-set 4-tool
python -m benchmarks.runner --tool-set 8-tool

# Run with specific model
python -m benchmarks.runner --tool-set 4-tool --model gemini-3-flash

# Show scenario statistics
python -m benchmarks.runner --stats
```

## File-Based Configuration

```
benchmarks/
├── tool_sets/              # Tool set definitions (auto-detected)
│   ├── 2-tool.yaml         # Consolidated (modify_paragraph, move_paragraph)
│   ├── 4-tool.yaml         # Default (add, replace, delete, move)
│   └── 8-tool.yaml         # Atomic (insert_before, insert_after, append, etc.)
├── descriptions/           # Description style variants
│   ├── default.yaml
│   ├── minimal.yaml
│   └── verbose.yaml
├── models/                 # Model configurations
│   ├── gpt-5-mini.yaml
│   ├── claude-4.5-haiku.yaml
│   └── gemini-3-flash.yaml
└── config_loader.py        # Dynamic YAML config loading
```

### Adding New Configurations

Create a new YAML file in the appropriate directory. The benchmark will auto-detect it by filename.

**Tool Set Example** (`benchmarks/tool_sets/my-tools.yaml`):
```yaml
name: "My Custom Tools"
tool_count: 3

tools:
  - name: tool_a
    what: "Description of what it does"
    when: "When to use it"
  - name: tool_b
    what: "..."
    when: "..."

tool_mapping:
  insert_before: tool_a
  append: tool_b
```

## Scenario Complexity Levels

| Level | Type | Description | Count |
|-------|------|-------------|-------|
| L1 | Simple | Single tool selection | 45 |
| L2 | Sequential | 2-3 tool chain | 25 |
| L3 | Complex | Reasoning required | 20 |
| L4 | Ambiguous | Multiple valid tools | 25 |
| L5 | Edge Case | Empty/unicode/boundary | 28 |
| L6 | Adversarial | Confusing/contradictory | 42 |

**Total**: 185 scenarios (comprehensive coverage of all 28 MCP tools)

## Benchmark Results

**Latest Run**: December 29, 2024 (Comprehensive optimization - ALL scenarios)
**Model**: Gemini 3 Flash Preview (via OpenRouter)
**Scenarios**: 55 comprehensive scenarios covering ALL 28 MCP tools
**Optimizer**: COPRO (light mode, 2 iterations)
**Training/Validation**: 36 training, 19 validation
**Duration**: ~30 minutes total (all three variants)

### Variant Comparison Summary (December 29, 2024 - Final Results)

| Variant | Format | Metric | Baseline | Optimized | Status | Tokens |
|---------|--------|--------|----------|-----------|--------|--------|
| **COST_OPTIMIZED** | Minimal | 50/25/25 | 98.2% (0.853) | 98.2% (0.853) | 0.853 | 704/81 |
| **ACCURACY_BALANCED** | Compact | Ratio | 100.0% (1.000) | 100.0% (1.000) | 1.000 | 1065/80 |
| **MAXIMUM_ACCURACY** | Full | Pure | 100.0% (1.000) | 100.0% (1.000) | 1.000 | 3394/98 |

**Testing Results**:
- All three variants tested across all 28 MCP tools
- ACCURACY_BALANCED: 100% accuracy, 1065 tokens
- MAXIMUM_ACCURACY: 100% accuracy, 3394 tokens
- COST_OPTIMIZED: 98.2% accuracy, 704 tokens
- See [VARIANT_RESULTS.md](VARIANT_RESULTS.md) for detailed analysis

See [Variant Comparison Results](VARIANT_RESULTS.md) for detailed analysis and [Variant Architecture](VARIANT_ARCHITECTURE.md) for metric specifications.

## Programmatic Usage

```python
# Load tool set from YAML file
from benchmarks.config_loader import load_tool_set, load_model_config
tool_set = load_tool_set("4-tool")

# Get scenarios mapped to tool set
from benchmarks.scenarios import get_benchmark_scenarios
scenarios = get_benchmark_scenarios(
    include_edge_cases=True,
    include_adversarial=True,
    tool_set_name="8-tool",
)

# Run benchmark
from benchmarks.runner import run_benchmark
report = await run_benchmark(tool_set_name="8-tool", model_name="gemini-3-flash")
```

## Metrics

### Variant-Specific Scoring (December 29, 2024+)

Each variant optimizes for its own metric:

| Variant | Metric Formula | Focus |
|---------|---|---|
| **COST_OPTIMIZED** | `0.5 × accuracy + 0.25 × input_eff + 0.25 × output_eff` | Minimize token cost |
| **ACCURACY_BALANCED** | `accuracy / (1 + excess_tokens)` | Maximize accuracy per token |
| **MAXIMUM_ACCURACY** | `accuracy` | Maximize correctness |

Where efficiency = `1 / (1 + tokens / scale_factor)` (inverse scaling, capped at efficiency=1.0)

### Legacy Metrics (December 28 and earlier)

The benchmark previously used a unified composite score:

| Metric | Weight | Source |
|--------|--------|--------|
| Accuracy | 60% | Tool name comparison |
| Input tokens | 25% | API response metadata |
| Output tokens | 15% | API response metadata |

Results are saved to `benchmarks/results/` for comparison and historical tracking.

## Comparing Configurations

Run the benchmark twice with different settings and compare the JSON results:

```bash
python -m benchmarks.runner --tool-set 4-tool
python -m benchmarks.runner --tool-set 8-tool
# Compare: benchmarks/results/4-tool_*.json vs 8-tool_*.json
```
