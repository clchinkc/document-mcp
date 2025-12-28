"""MCP Tool Selection Benchmarking Infrastructure.

This package evaluates LLM accuracy in selecting the correct MCP tool
for document operations. Two benchmark modes are supported:

1. PARAGRAPH BENCHMARKS (Legacy):
   Tests paragraph manipulation tools (add, replace, delete, move)
   across 2-tool, 4-tool, and 8-tool configurations.

2. COMPREHENSIVE BENCHMARKS (New):
   Tests ALL 28 MCP tools covering document, chapter, paragraph,
   content, metadata, safety, overview, and discovery operations.

Configuration files:
- Tool set definitions (benchmarks/tool_sets/*.yaml)
- Description styles (benchmarks/descriptions/*.yaml)
- Model configurations (benchmarks/models/*.yaml)

Usage:
    # Run paragraph benchmarks (legacy)
    python -m benchmarks.runner --tool-set 4-tool

    # Get comprehensive training set for all 28 tools
    from benchmarks import get_all_tools_trainset, get_all_tools_stats
    trainset = get_all_tools_trainset()
    stats = get_all_tools_stats()

    # Load configs programmatically
    from benchmarks.config_loader import load_tool_set, load_model_config
    tool_set = load_tool_set("4-tool")
    model = load_model_config("gpt-5-mini")
"""

from .config import BENCHMARK_MODELS
from .config import DEFAULT_MODEL
from .config import BenchmarkConfig
from .config import ComparisonConfig
from .config import ModelConfig
from .config import ScoringWeights
from .config import get_available_models
from .config_loader import get_model_from_config
from .config_loader import get_tool_set_from_config
from .config_loader import list_available_configs
from .config_loader import load_description_style
from .config_loader import load_model_config
from .config_loader import load_tool_set
from .config_loader import print_available_configs
from .metrics import BenchmarkMetrics
from .metrics import ComparisonResult
from .metrics import CompositeScore
from .scenarios import ADVERSARIAL_SCENARIOS
from .scenarios import ALL_TOOL_SCENARIOS
from .scenarios import AMBIGUOUS_SCENARIOS
from .scenarios import COMPLEX_SCENARIOS
from .scenarios import EDGE_CASE_SCENARIOS
from .scenarios import PARAGRAPH_SCENARIOS
from .scenarios import SEQUENTIAL_SCENARIOS
from .scenarios import AdversarialScenario
from .scenarios import AmbiguousScenario
from .scenarios import ComplexityLevel
from .scenarios import EdgeCaseScenario
from .scenarios import Scenario
from .scenarios import SequentialScenario
from .scenarios import ToolScenario
from .scenarios import get_adversarial_scenarios
from .scenarios import get_all_scenarios_by_level
from .scenarios import get_all_tools_stats
from .scenarios import get_all_tools_trainset
from .scenarios import get_ambiguous_scenarios
from .scenarios import get_benchmark_scenarios
from .scenarios import get_benchmark_scenarios_for_tool_comparison
from .scenarios import get_dspy_trainset
from .scenarios import get_edge_case_scenarios
from .scenarios import get_scenario_stats
from .scenarios import get_scenarios
from .scenarios import get_sequential_scenarios
from .tools import ATOMIC_8_TOOLS
from .tools import ATOMIC_PARAGRAPH_TOOLS
from .tools import CONSOLIDATED_2_TOOLS
from .tools import CONSOLIDATED_PARAGRAPH_TOOLS
from .tools import DEFAULT_4_TOOLS
from .tools import ToolSet
from .tools import get_description_style_tool_sets
from .tools import get_implementation_tool_sets
from .tools import get_tool_set

__all__ = [
    # Config
    "BenchmarkConfig",
    "ComparisonConfig",
    "ModelConfig",
    "ScoringWeights",
    "BENCHMARK_MODELS",
    "DEFAULT_MODEL",
    "get_available_models",
    # Config Loader (file-based)
    "list_available_configs",
    "load_tool_set",
    "load_description_style",
    "load_model_config",
    "get_tool_set_from_config",
    "get_model_from_config",
    "print_available_configs",
    # Metrics
    "BenchmarkMetrics",
    "CompositeScore",
    "ComparisonResult",
    # Tools
    "ATOMIC_PARAGRAPH_TOOLS",
    "CONSOLIDATED_PARAGRAPH_TOOLS",
    "DEFAULT_4_TOOLS",
    "ATOMIC_8_TOOLS",
    "CONSOLIDATED_2_TOOLS",
    "ToolSet",
    "get_tool_set",
    "get_implementation_tool_sets",
    "get_description_style_tool_sets",
    # Scenarios (Paragraph - Legacy)
    "Scenario",
    "SequentialScenario",
    "AmbiguousScenario",
    "EdgeCaseScenario",
    "AdversarialScenario",
    "ComplexityLevel",
    "PARAGRAPH_SCENARIOS",
    "SEQUENTIAL_SCENARIOS",
    "COMPLEX_SCENARIOS",
    "AMBIGUOUS_SCENARIOS",
    "EDGE_CASE_SCENARIOS",
    "ADVERSARIAL_SCENARIOS",
    "get_scenarios",
    "get_sequential_scenarios",
    "get_ambiguous_scenarios",
    "get_edge_case_scenarios",
    "get_adversarial_scenarios",
    "get_all_scenarios_by_level",
    "get_scenario_stats",
    "get_benchmark_scenarios",
    "get_benchmark_scenarios_for_tool_comparison",
    "get_dspy_trainset",
    # Scenarios (Comprehensive - All 28 Tools)
    "ToolScenario",
    "ALL_TOOL_SCENARIOS",
    "get_all_tools_trainset",
    "get_all_tools_stats",
]
