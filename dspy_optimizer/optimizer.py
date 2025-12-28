"""DSPy MCP Tool Selection Optimizer.

Optimizes LLM prompts for selecting the correct MCP tool using DSPy.
Uses the SAME prompt format as production agents to ensure optimization results
transfer directly to deployment.

Composite Scoring:
    - 60% accuracy (correct tool selection)
    - 25% input token efficiency (inverse scaling, no baselines)
    - 15% output token efficiency (inverse scaling, no baselines)

Token Efficiency Formula:
    efficiency = 1 / (1 + tokens / scale_factor)
    - Lower tokens → efficiency approaches 1.0
    - Higher tokens → efficiency approaches 0.0
    - Scale factors (1000 input, 500 output) control sensitivity

Optimizers Available:
    - COPRO: Coordinate ascent for instruction optimization (recommended, default)
    - MIPROv2: 0-shot instruction optimization
    - BootstrapFewShot: Few-shot example bootstrapping
    - SIMBA: Sequential instruction modification
    - GEPA: Gradient-free prompt optimization

Usage:
    python -m dspy_optimizer.optimizer
    python -m dspy_optimizer.optimizer --variant full
    python -m dspy_optimizer.optimizer --optimizer copro
    python -m dspy_optimizer.optimizer --write-back
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import dspy

from benchmarks import BENCHMARK_MODELS
from benchmarks import BenchmarkMetrics
from benchmarks import get_benchmark_scenarios
from benchmarks.config import DEFAULT_MODEL
from src.agents.react_agent.prompts import get_react_system_prompt
from src.agents.shared.tool_descriptions import ToolFormat
from src.agents.shared.tool_descriptions import get_manager
from src.agents.simple_agent.prompts import get_simple_agent_system_prompt

# =============================================================================
# AGENT TYPES FOR BENCHMARKING
# =============================================================================


class AgentType(Enum):
    """Agent types for benchmarking."""

    SIMPLE = "simple"
    REACT = "react"


class PromptVariant(Enum):
    """Prompt variants optimized for different metrics.

    - COST_OPTIMIZED: 50% accuracy, 25% input, 25% output (minimize tokens)
    - ACCURACY_BALANCED: Accuracy / (1 + tokens/baseline) (best accuracy per token)
    - MAXIMUM_ACCURACY: 100% accuracy (maximize accuracy regardless of cost)
    """

    COST_OPTIMIZED = "cost_optimized"  # Minimal tool descriptions
    ACCURACY_BALANCED = "accuracy_balanced"  # Compact tool descriptions
    MAXIMUM_ACCURACY = "maximum_accuracy"  # Full WHAT/WHEN/RETURNS/AUTO format


class OptimizerType(Enum):
    """Available DSPy optimizers.

    - MIPRO: Best for large datasets (200+ examples), instruction optimization
    - BOOTSTRAP: Best for small datasets (<50 examples), simple few-shot
    - COPRO: Coordinate ascent for instruction optimization
    - SIMBA: Sequential instruction modification with bootstrapped answers
    - GEPA: Gradient-free prompt optimization
    """

    MIPRO = "mipro"  # MIPROv2
    BOOTSTRAP = "bootstrap"  # BootstrapFewShot
    COPRO = "copro"  # COPRO (Coordinate Ascent)
    SIMBA = "simba"  # SIMBA
    GEPA = "gepa"  # GEPA


def get_prompt_for_variant(variant: PromptVariant) -> str:
    """Get tool descriptions in the specified variant format."""
    mgr = get_manager()
    if variant == PromptVariant.COST_OPTIMIZED:
        return mgr.get_formatted_tools(ToolFormat.MINIMAL)
    elif variant == PromptVariant.ACCURACY_BALANCED:
        return mgr.get_formatted_tools(ToolFormat.COMPACT)
    else:  # MAXIMUM_ACCURACY
        return mgr.get_formatted_tools(ToolFormat.FULL)


def get_agent_prompt(agent_type: AgentType) -> str:
    """Get the full system prompt for a specific agent type."""
    if agent_type == AgentType.SIMPLE:
        return get_simple_agent_system_prompt()
    else:
        return get_react_system_prompt()


# =============================================================================
# DSPY SIGNATURES AND MODULES
# =============================================================================


@dataclass
class OptimizationResult:
    """Result of DSPy optimization."""

    baseline_metrics: BenchmarkMetrics
    optimized_metrics: BenchmarkMetrics
    improvement: float
    duration_seconds: float
    model_used: str
    variant_used: str
    optimized_instructions: dict | None = None  # Extracted optimized instructions
    write_back_version: str | None = None  # Version saved (e.g., "v001")
    kept_baseline: bool = False  # True if optimization was worse and baseline was kept

    @property
    def accuracy_improvement(self) -> float:
        return self.optimized_metrics.accuracy - self.baseline_metrics.accuracy

    def _get_variant_score(self, metrics: BenchmarkMetrics) -> float:
        """Get score for this result's variant."""
        if self.variant_used == PromptVariant.COST_OPTIMIZED.value:
            return metrics.cost_optimized_score()
        elif self.variant_used == PromptVariant.ACCURACY_BALANCED.value:
            return metrics.accuracy_balanced_score()
        else:  # MAXIMUM_ACCURACY
            return metrics.maximum_accuracy_score()

    @property
    def baseline_score(self) -> float:
        """Get baseline score using variant-specific metric."""
        return self._get_variant_score(self.baseline_metrics)

    @property
    def optimized_score(self) -> float:
        """Get optimized score using variant-specific metric."""
        return self._get_variant_score(self.optimized_metrics)


class MCPToolSelector(dspy.Signature):
    """Select the correct MCP tool for a document operation query.

    TOOL CATEGORIES:
    - Document operations: create_document, delete_document, list_documents
    - Chapter operations: create_chapter, delete_chapter, list_chapters, rename_chapter
    - Content access: read_content, get_document_outline, get_statistics
    - Search: find_text, find_similar_text, find_entity
    - Modification: replace_text, replace_paragraph, insert_paragraph, delete_paragraph
    - Metadata: read_metadata, update_metadata, list_metadata
    - Version control: manage_snapshots, compare_versions

    DISAMBIGUATION RULES:
    - For listing documents vs reading: "list/show/available documents" → list_documents, "read/content/text" → read_content
    - For search vs similar: exact text search → find_text, semantic/conceptual → find_similar_text
    - For outline vs content: structure/overview → get_document_outline, full text → read_content
    - For delete operations: verify intent carefully, suggest manage_snapshots for safety

    ERROR HANDLING:
    - If document not found, suggest list_documents first
    - If chapter not found, suggest list_chapters first
    - For ambiguous requests, prefer read operations over modify operations
    """

    system_prompt: str = dspy.InputField(desc="Agent system prompt with tool descriptions")
    query: str = dspy.InputField(desc="User query for document operation")
    tool_name: str = dspy.OutputField(desc="Exact MCP tool name to use")


class AgentToolSelector(dspy.Signature):
    """Agent-specific tool selection that matches production behavior.

    Apply category-specific rules and error handling based on query intent.
    For document listing, use list_documents.
    For content reading, use read_content with appropriate scope.
    For search, distinguish between exact (find_text) and semantic (find_similar_text).
    """

    agent_type: str = dspy.InputField(desc="Agent type: simple or react")
    system_prompt: str = dspy.InputField(desc="Full agent system prompt")
    query: str = dspy.InputField(desc="User query for document operation")
    tool_name: str = dspy.OutputField(desc="Exact MCP tool name to use")


def _get_optimized_section_for_variant(variant: PromptVariant) -> str:
    """Get optimized content for a variant (for cumulative optimization)."""
    try:
        from src.agents.shared.optimized_demos import get_optimized_examples_section

        return get_optimized_examples_section(variant.value)
    except Exception:
        return ""


class ToolSelectionModule(dspy.Module):
    """DSPy module for MCP tool selection using production prompts.

    Includes previously optimized content for cumulative optimization effect.
    """

    def __init__(self, variant: PromptVariant = PromptVariant.MAXIMUM_ACCURACY):
        super().__init__()
        self.selector = dspy.ChainOfThought(MCPToolSelector)
        self.variant = variant
        # Build prompt with tool descriptions + previously optimized content
        tool_descriptions = get_prompt_for_variant(variant)
        optimized_section = _get_optimized_section_for_variant(variant)
        if optimized_section:
            self.system_prompt = f"{tool_descriptions}\n\n{optimized_section}"
        else:
            self.system_prompt = tool_descriptions

    def forward(self, query: str) -> dspy.Prediction:
        return self.selector(system_prompt=self.system_prompt, query=query)


class AgentToolSelectionModule(dspy.Module):
    """DSPy module that tests the FULL agent prompt (not just tool descriptions)."""

    def __init__(self, agent_type: AgentType = AgentType.SIMPLE):
        super().__init__()
        self.selector = dspy.ChainOfThought(AgentToolSelector)
        self.agent_type = agent_type
        self.system_prompt = get_agent_prompt(agent_type)

    def forward(self, query: str) -> dspy.Prediction:
        return self.selector(
            agent_type=self.agent_type.value,
            system_prompt=self.system_prompt,
            query=query,
        )


def normalize_tool_name(name: str) -> str:
    r"""Normalize tool name by extracting just the function name.

    Handles cases where LLM returns:
    - Just the name: "create_document"
    - Full call: "create_document(document_name=\"my_story\")"
    - With backticks: "`create_document`"
    """
    name = name.strip()
    name = name.strip("`")
    if "(" in name:
        name = name.split("(")[0]
    return name.lower().strip()


# =============================================================================
# TOKEN CACHING AND METRICS
# =============================================================================

# Token cache: maps prompt hash to (input_tokens, output_tokens)
# This allows us to track tokens even when DSPy returns cached responses
_token_cache: dict[int, tuple[int, int]] = {}


def _get_prompt_hash(query: str) -> int:
    """Create a hash key for token caching."""
    return hash(query)


def _clear_token_cache() -> None:
    """Clear the token cache. Call at start of optimization."""
    _token_cache.clear()


def _get_tokens_from_history() -> tuple[int, int]:
    """Get token counts from the last LM call.

    Returns (input_tokens, output_tokens). Returns (0, 0) if unavailable.
    """
    lm = dspy.settings.lm
    if not lm or not hasattr(lm, "history") or not lm.history:
        return 0, 0

    last_call = lm.history[-1]
    if not isinstance(last_call, dict) or "usage" not in last_call:
        return 0, 0

    usage = last_call["usage"]
    return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def _get_tokens_with_cache(query: str) -> tuple[int, int]:
    """Get token counts, using cache for previously seen prompts.

    Returns (input_tokens, output_tokens).
    """
    input_tokens, output_tokens = _get_tokens_from_history()
    prompt_hash = _get_prompt_hash(query)

    # If we got real tokens, cache them
    if input_tokens > 0 or output_tokens > 0:
        _token_cache[prompt_hash] = (input_tokens, output_tokens)
        return input_tokens, output_tokens

    # If tokens are 0 (cached response), look up from our cache
    if prompt_hash in _token_cache:
        return _token_cache[prompt_hash]

    return 0, 0


def composite_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """Composite metric for DSPy optimization: accuracy + token efficiency.

    Scoring:
    - Accuracy: 1.0 if correct, 0.0 if wrong (60% weight in final score)
    - Token efficiency: inverse scaling (25% input, 15% output weight)

    Returns composite score between 0 and 1.
    """
    expected = example.expected_tool
    predicted = getattr(prediction, "tool_name", "") if prediction else ""

    # Get accuracy score
    accuracy = 1.0 if normalize_tool_name(expected) == normalize_tool_name(predicted) else 0.0

    # Get token counts
    input_tokens, output_tokens = _get_tokens_with_cache(example.query)

    # Calculate token efficiency using inverse scaling (no baselines)
    # efficiency = 1 / (1 + tokens / scale_factor)
    input_eff = 1.0 / (1.0 + input_tokens / 1000) if input_tokens > 0 else 1.0
    output_eff = 1.0 / (1.0 + output_tokens / 500) if output_tokens > 0 else 1.0

    # Composite: 60% accuracy, 25% input efficiency, 15% output efficiency
    return 0.6 * accuracy + 0.25 * input_eff + 0.15 * output_eff


# =============================================================================
# WRITE-BACK MECHANISM
# =============================================================================


def parse_rules_from_instruction(instruction: str) -> dict:
    """Parse structured rules from an optimized instruction.

    Extracts TOOL CATEGORIES, DISAMBIGUATION RULES, and ERROR HANDLING sections.

    Returns:
        Dict with 'categories', 'disambiguation', and 'error_handling' keys.
    """
    rules = {
        "categories": [],
        "disambiguation": [],
        "error_handling": [],
        "raw": instruction,
    }

    lines = instruction.split("\n")
    current_section = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith('"""'):
            continue

        # Detect section headers
        if "TOOL CATEGORIES" in line.upper():
            current_section = "categories"
        elif "DISAMBIGUATION" in line.upper():
            current_section = "disambiguation"
        elif "ERROR HANDLING" in line.upper():
            current_section = "error_handling"
        elif line.startswith("-") and current_section:
            # Extract rule content
            rule = line.lstrip("- ").strip()
            if rule:
                rules[current_section].append(rule)

    return rules


def extract_optimization_from_module(optimized_module: dspy.Module) -> dict:
    """Extract all optimized content from a DSPy module.

    DSPy optimizers produce different outputs:
    - COPRO/MIPROv2: Optimize 'instructions' in signature (includes rules)
    - BootstrapFewShot: Create 'demos' (few-shot examples with reasoning)

    Returns:
        Dict with 'instructions', 'rules', and 'demos' keys.
    """
    result = {"instructions": {}, "rules": {}, "demos": []}
    try:
        state = optimized_module.dump_state()
        for key, value in state.items():
            if not isinstance(value, dict):
                continue

            # Extract instructions from signature
            if "signature" in value:
                sig = value["signature"]
                if isinstance(sig, dict) and "instructions" in sig:
                    instruction = sig["instructions"]
                    result["instructions"][key] = instruction
                    # Parse rules from the instruction
                    parsed_rules = parse_rules_from_instruction(instruction)
                    if any(parsed_rules[k] for k in ["categories", "disambiguation", "error_handling"]):
                        result["rules"][key] = parsed_rules

            # Extract demos (few-shot examples from BootstrapFewShot)
            if "demos" in value and value["demos"]:
                for demo in value["demos"]:
                    if isinstance(demo, dict):
                        # Capture reasoning if available
                        demo_entry = {
                            "query": demo.get("query", demo.get("input", "")),
                            "tool_name": demo.get("tool_name", demo.get("output", "")),
                        }
                        # Include reasoning/rationale if present
                        if "reasoning" in demo:
                            demo_entry["reasoning"] = demo["reasoning"]
                        elif "rationale" in demo:
                            demo_entry["reasoning"] = demo["rationale"]
                        result["demos"].append(demo_entry)
    except Exception as e:
        print(f"[Extract] Warning: Could not extract optimization: {e}")
    return result


def extract_instructions_from_module(optimized_module: dspy.Module) -> dict:
    """Extract optimized instructions from a DSPy module (legacy compatibility).

    Returns:
        Dict with predictor names as keys and their optimized instructions as values.
    """
    return extract_optimization_from_module(optimized_module).get("instructions", {})


def write_back_optimized_prompt(
    optimized_module: dspy.Module,
    variant: PromptVariant,
    baseline_metrics: BenchmarkMetrics,
    optimized_metrics: BenchmarkMetrics,
    optimizer_type: str = "copro",
    model: str = "unknown",
) -> str | None:
    """Write optimized content to versioned storage.

    Extracts instructions, rules, and demos from the DSPy module and saves them
    with full version history support.

    Returns:
        Version string (e.g., "v001") or None if failed.
    """
    from src.agents.shared.optimization_store import get_store

    # Extract all optimized content from the module
    optimization = extract_optimization_from_module(optimized_module)
    instructions = optimization.get("instructions", {})
    rules = optimization.get("rules", {})
    demos = optimization.get("demos", [])

    if not instructions and not demos:
        print("[Write-back] No optimization content extracted, skipping save")
        return None

    # Calculate variant-specific score
    def get_variant_score(metrics: BenchmarkMetrics) -> float:
        if variant == PromptVariant.COST_OPTIMIZED:
            return metrics.cost_optimized_score()
        elif variant == PromptVariant.ACCURACY_BALANCED:
            return metrics.accuracy_balanced_score()
        else:  # MAXIMUM_ACCURACY
            return metrics.maximum_accuracy_score()

    # Prepare metrics dicts
    baseline_dict = {
        "accuracy": baseline_metrics.accuracy,
        "composite": get_variant_score(baseline_metrics),
        "avg_input_tokens": baseline_metrics.avg_input_tokens,
        "avg_output_tokens": baseline_metrics.avg_output_tokens,
    }
    optimized_dict = {
        "accuracy": optimized_metrics.accuracy,
        "composite": get_variant_score(optimized_metrics),
        "avg_input_tokens": optimized_metrics.avg_input_tokens,
        "avg_output_tokens": optimized_metrics.avg_output_tokens,
    }

    # Save to versioned store
    store = get_store()
    version = store.save(
        variant=variant.value,
        instructions=instructions,
        rules=rules,
        demos=demos,
        baseline_metrics=baseline_dict,
        optimized_metrics=optimized_dict,
        optimizer=optimizer_type,
        model=model,
    )

    print(f"[Write-back] Saved optimization {version} for variant '{variant.value}'")
    if instructions:
        print(f"[Write-back] Extracted {len(instructions)} optimized instruction(s)")
        for key, instr in instructions.items():
            preview = instr[:80] + "..." if len(instr) > 80 else instr
            print(f"  {key}: {preview}")
    if rules:
        print(f"[Write-back] Extracted rules from {len(rules)} predictor(s)")
        for key, rule_data in rules.items():
            cat_count = len(rule_data.get("categories", []))
            dis_count = len(rule_data.get("disambiguation", []))
            err_count = len(rule_data.get("error_handling", []))
            print(f"  {key}: {cat_count} categories, {dis_count} disambiguation, {err_count} error handling")
    if demos:
        print(f"[Write-back] Extracted {len(demos)} few-shot demo(s)")

    return version


# =============================================================================
# OPTIMIZER
# =============================================================================


class ToolSelectionOptimizer:
    """Optimizer using DSPy MIPROv2 with production prompt formats."""

    def __init__(
        self,
        model: str | None = None,
        variant: PromptVariant = PromptVariant.MAXIMUM_ACCURACY,
        agent_type: AgentType | None = None,
    ):
        """Initialize with specified model and prompt variant.

        Args:
            model: Model ID (OpenRouter format). Defaults to Gemini 3 Flash.
            variant: Prompt variant to optimize.
            agent_type: If set, tests full agent prompt instead of just tools.
        """
        self.model = model or DEFAULT_MODEL.model_id
        self.variant = variant
        self.agent_type = agent_type
        self._configure_dspy()

    def _configure_dspy(self):
        """Configure DSPy with OpenRouter."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY required")

        lm = dspy.LM(model=f"openrouter/{self.model}", api_key=api_key)
        dspy.configure(lm=lm)
        print(f"[DSPy] Configured: openrouter/{self.model}")

    def load_trainset(self) -> list[dspy.Example]:
        """Load training set from comprehensive tool scenarios.

        Uses all 55+ comprehensive scenarios covering all MCP tools
        for maximum coverage during optimization.
        """
        from src.agents.shared.tool_descriptions import get_all_scenarios

        raw_data = get_all_scenarios()
        return [
            dspy.Example(
                query=item["query"],
                expected_tool=item["expected_tool"],
            ).with_inputs("query")
            for item in raw_data
        ]

    def evaluate(
        self,
        module: dspy.Module,
        examples: list[dspy.Example],
        prime_cache: bool = False,
    ) -> BenchmarkMetrics:
        """Evaluate module with accuracy and token tracking for composite score.

        Args:
            module: The DSPy module to evaluate
            examples: List of examples to evaluate on
            prime_cache: If True, disable DSPy cache to get fresh token counts
        """
        metrics = BenchmarkMetrics(num_examples=len(examples))
        correct = 0
        confusion: dict[str, dict[str, int]] = {}

        # Optionally disable DSPy cache to prime our token cache
        lm = dspy.settings.lm
        original_cache = None
        if prime_cache and lm and hasattr(lm, "cache"):
            original_cache = lm.cache
            lm.cache = False

        for example in examples:
            try:
                prediction = module(query=example.query)
                raw_predicted = getattr(prediction, "tool_name", "")
                predicted = normalize_tool_name(raw_predicted)
                expected = normalize_tool_name(example.expected_tool)

                if predicted == expected:
                    correct += 1
                else:
                    confusion.setdefault(expected, {})
                    confusion[expected][predicted] = confusion[expected].get(predicted, 0) + 1

                # Track tokens using cache (handles both fresh and cached responses)
                input_tokens, output_tokens = _get_tokens_with_cache(example.query)
                metrics.input_tokens += input_tokens
                metrics.output_tokens += output_tokens

            except Exception as e:
                metrics.error_count += 1
                print(f"  Error: {e}")

        # Restore original cache setting
        if original_cache is not None and lm:
            lm.cache = original_cache

        metrics.accuracy = correct / len(examples) if examples else 0
        metrics.confusion_matrix = confusion
        return metrics

    def _get_variant_specific_score(self, metrics: BenchmarkMetrics) -> float:
        """Get score for this optimizer's variant."""
        if self.variant == PromptVariant.COST_OPTIMIZED:
            return metrics.cost_optimized_score()
        elif self.variant == PromptVariant.ACCURACY_BALANCED:
            return metrics.accuracy_balanced_score()
        else:  # MAXIMUM_ACCURACY
            return metrics.maximum_accuracy_score()

    def _create_optimizer(
        self,
        optimizer_type: OptimizerType,
        auto_mode: str,
    ) -> tuple[dspy.Module, str]:
        """Create the appropriate DSPy optimizer.

        Returns (optimizer_instance, optimizer_description).
        """
        if optimizer_type == OptimizerType.MIPRO:
            # 0-shot optimization: focus on instruction optimization only
            # No few-shot examples added to prompts
            optimizer = dspy.MIPROv2(
                metric=composite_metric,
                auto=auto_mode,
                max_bootstrapped_demos=0,
                max_labeled_demos=0,
            )
            return optimizer, f"MIPROv2 0-shot ({auto_mode} mode)"

        elif optimizer_type == OptimizerType.BOOTSTRAP:
            optimizer = dspy.BootstrapFewShot(
                metric=composite_metric,
                max_bootstrapped_demos=4,
                max_labeled_demos=4,
            )
            return optimizer, "BootstrapFewShot"

        elif optimizer_type == OptimizerType.COPRO:
            optimizer = dspy.COPRO(
                metric=composite_metric,
                breadth=5,
                depth=2,
            )
            return optimizer, "COPRO"

        elif optimizer_type == OptimizerType.SIMBA:
            optimizer = dspy.SIMBA(
                metric=composite_metric,
                max_bootstrapped_demos=4,
                max_labeled_demos=4,
            )
            return optimizer, "SIMBA"

        elif optimizer_type == OptimizerType.GEPA:
            optimizer = dspy.GEPA(
                metric=composite_metric,
                num_candidates=8,
            )
            return optimizer, "GEPA"

        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def optimize(
        self,
        optimizer_type: OptimizerType = OptimizerType.COPRO,
        auto_mode: str = "light",
        write_back: bool = False,
    ) -> OptimizationResult:
        """Run optimization with the specified optimizer.

        Args:
            optimizer_type: DSPy optimizer (mipro, bootstrap, copro, simba, gepa)
            auto_mode: MIPROv2 optimization intensity ("light", "medium", "heavy")
            write_back: Whether to write optimized prompts back to source files
        """
        start_time = time.time()
        _clear_token_cache()  # Start fresh for accurate token tracking
        examples = self.load_trainset()

        # Get unique tools in trainset
        tools_in_trainset = {ex.expected_tool for ex in examples}

        # Create optimizer
        optimizer, optimizer_desc = self._create_optimizer(optimizer_type, auto_mode)

        print(f"\n{'=' * 60}")
        print("DSPy MCP TOOL SELECTION OPTIMIZATION")
        print(f"{'=' * 60}")
        print(f"Model: {self.model}")
        print(f"Variant: {self.variant.value}")
        if self.agent_type:
            print(f"Agent: {self.agent_type.value}")
        print(f"Scenarios: {len(examples)} covering {len(tools_in_trainset)} tools")
        print(f"Optimizer: {optimizer_desc}")

        # Create module based on whether we're testing agents or just tools
        if self.agent_type:
            module = AgentToolSelectionModule(agent_type=self.agent_type)
        else:
            module = ToolSelectionModule(variant=self.variant)

        # Baseline evaluation (prime_cache=True to get fresh token counts)
        print("\n[Baseline] Evaluating (priming token cache)...")
        baseline_metrics = self.evaluate(module, examples, prime_cache=True)
        print(
            f"  Accuracy: {baseline_metrics.accuracy:.1%} ({int(baseline_metrics.accuracy * len(examples))}/{len(examples)})"
        )
        print(
            f"  Tokens: {baseline_metrics.avg_input_tokens:.0f} in / {baseline_metrics.avg_output_tokens:.0f} out (avg)"
        )
        # Get variant-specific baseline score
        baseline_score = self._get_variant_specific_score(baseline_metrics)
        print(f"  Composite: {baseline_score:.3f}")

        if baseline_metrics.confusion_matrix:
            print("  Errors:")
            for expected, predictions in baseline_metrics.confusion_matrix.items():
                for predicted, count in predictions.items():
                    print(f"    {expected} -> {predicted}: {count}")

        # Split for training/validation
        split_idx = max(1, len(examples) * 2 // 3)
        trainset = examples[:split_idx]
        valset = examples[split_idx:]

        # Optimize for composite score (accuracy + token efficiency)
        # IMPORTANT: Disable DSPy cache so optimizer sees real token costs
        print(f"\n[Optimize] {optimizer_desc} on {len(trainset)} train, {len(valset)} val...")
        print("  Optimizing for: composite (60% accuracy, 25% input eff, 15% output eff)")
        print("  Cache disabled for accurate token tracking")

        lm = dspy.settings.lm
        original_cache = getattr(lm, "cache", None) if lm else None
        if lm and hasattr(lm, "cache"):
            lm.cache = False

        optimized_module = None
        try:
            # Different optimizers have different compile signatures
            if optimizer_type == OptimizerType.MIPRO:
                # MIPROv2 uses both trainset and valset
                optimized_module = optimizer.compile(module, trainset=trainset, valset=valset)
            elif optimizer_type == OptimizerType.GEPA:
                # GEPA uses both trainset and valset
                optimized_module = optimizer.compile(module, trainset=trainset, valset=valset)
            elif optimizer_type == OptimizerType.COPRO:
                # COPRO requires eval_kwargs for evaluation settings
                optimized_module = optimizer.compile(
                    module,
                    trainset=trainset,
                    eval_kwargs={"num_threads": 1},
                )
            else:
                # BootstrapFewShot, SIMBA use trainset only
                optimized_module = optimizer.compile(module, trainset=trainset)
        except Exception as e:
            print(f"[Optimize] {optimizer_desc} error: {e}")
            # Restore cache before returning
            if original_cache is not None and lm:
                lm.cache = original_cache
            duration = time.time() - start_time
            return OptimizationResult(
                baseline_metrics=baseline_metrics,
                optimized_metrics=baseline_metrics,
                improvement=0.0,
                duration_seconds=duration,
                model_used=self.model,
                variant_used=self.variant.value,
            )

        # Restore cache after optimization
        if original_cache is not None and lm:
            lm.cache = original_cache

        # Evaluate optimized (prime_cache=True because prompts changed with few-shot examples)
        print("\n[Optimized] Evaluating...")
        _clear_token_cache()  # Clear since optimized module has different prompts
        optimized_metrics = self.evaluate(optimized_module, examples, prime_cache=True)
        print(
            f"  Accuracy: {optimized_metrics.accuracy:.1%} ({int(optimized_metrics.accuracy * len(examples))}/{len(examples)})"
        )
        print(
            f"  Tokens: {optimized_metrics.avg_input_tokens:.0f} in / {optimized_metrics.avg_output_tokens:.0f} out (avg)"
        )
        # Get variant-specific optimized score
        optimized_score = self._get_variant_specific_score(optimized_metrics)
        print(f"  Composite: {optimized_score:.3f}")

        if optimized_metrics.confusion_matrix:
            print("  Remaining errors:")
            for expected, predictions in optimized_metrics.confusion_matrix.items():
                for predicted, count in predictions.items():
                    print(f"    {expected} -> {predicted}: {count}")

        # Calculate variant-specific scores
        baseline_composite = self._get_variant_specific_score(baseline_metrics)
        optimized_composite = self._get_variant_specific_score(optimized_metrics)

        # Decide whether to keep optimized or revert to baseline
        kept_baseline = False
        if optimized_composite < baseline_composite:
            print(
                f"\n  ⚠️  Optimization decreased score ({optimized_composite:.3f} < {baseline_composite:.3f})"
            )
            print("      Keeping baseline (optimization discarded)")
            kept_baseline = True
            # Use baseline metrics as final result
            final_metrics = baseline_metrics
            final_composite = baseline_composite
        else:
            final_metrics = optimized_metrics
            final_composite = optimized_composite

        # Extract optimized instructions (even if keeping baseline, for analysis)
        optimized_instructions = None
        if optimized_module:
            optimized_instructions = extract_instructions_from_module(optimized_module)

        # Write-back only if optimization improved (not equal - must be better)
        write_back_version = None
        if write_back and optimized_module and not kept_baseline and optimized_composite > baseline_composite:
            print("\n[Write-back] Writing optimized instructions to versioned storage...")
            write_back_version = write_back_optimized_prompt(
                optimized_module,
                self.variant,
                baseline_metrics,
                optimized_metrics,
                optimizer_type=optimizer_type.value,
                model=self.model,
            )

        duration = time.time() - start_time
        improvement = final_composite - baseline_composite

        # Display results
        baseline_breakdown = baseline_metrics.composite_breakdown()
        final_breakdown = final_metrics.composite_breakdown()

        print(f"\n{'=' * 60}")
        print("RESULTS")
        print(f"{'=' * 60}")
        print(f"Variant:             {self.variant.value}")
        print(f"Baseline composite:  {baseline_composite:.3f} (acc={baseline_metrics.accuracy:.1%})")
        print(
            f"  Token efficiency:  {baseline_breakdown.input_token_efficiency:.2f} in / {baseline_breakdown.output_token_efficiency:.2f} out"
        )
        if kept_baseline:
            print("Final:               Kept baseline (optimization was worse)")
        else:
            print(f"Optimized composite: {optimized_composite:.3f} (acc={optimized_metrics.accuracy:.1%})")
            print(
                f"  Token efficiency:  {final_breakdown.input_token_efficiency:.2f} in / {final_breakdown.output_token_efficiency:.2f} out"
            )
        print(f"Improvement:         {improvement:+.3f}")
        print(f"Duration:            {duration:.1f}s")
        if write_back_version:
            print(f"Saved version:       {write_back_version}")
        print(f"{'=' * 60}")

        return OptimizationResult(
            baseline_metrics=baseline_metrics,
            optimized_metrics=final_metrics,
            improvement=improvement,
            duration_seconds=duration,
            model_used=self.model,
            variant_used=self.variant.value,
            optimized_instructions=optimized_instructions,
            write_back_version=write_back_version,
            kept_baseline=kept_baseline,
        )


# =============================================================================
# PUBLIC API
# =============================================================================


def run_optimization(
    model: str | None = None,
    variant: str = "accuracy_balanced",
    optimizer_type: str = "copro",
    auto_mode: str = "light",
    write_back: bool = False,
    agent_type: str | None = None,
) -> OptimizationResult:
    """Run DSPy optimization for MCP tool selection.

    Always uses all scenarios (Levels 1-6) including edge cases and adversarial
    scenarios for comprehensive optimization.

    Args:
        model: Model ID (default: Gemini 3 Flash via DEFAULT_MODEL)
        variant: Prompt variant with specific optimization target:
            - "cost_optimized": 50% accuracy, 25% input, 25% output (minimize tokens)
            - "accuracy_balanced": Accuracy / (1 + tokens/baseline) (best per token)
            - "maximum_accuracy": 100% accuracy (ignore token cost)
        optimizer_type: DSPy optimizer ("mipro", "bootstrap", "copro", "simba", "gepa")
        auto_mode: MIPROv2 intensity ("light", "medium", "heavy")
        write_back: Whether to write optimized prompts back to source files
        agent_type: If set ("simple" or "react"), tests full agent prompt
    """
    prompt_variant = PromptVariant(variant)
    opt_type = OptimizerType(optimizer_type)
    agent = AgentType(agent_type) if agent_type else None
    optimizer = ToolSelectionOptimizer(
        model=model,
        variant=prompt_variant,
        agent_type=agent,
    )
    return optimizer.optimize(optimizer_type=opt_type, auto_mode=auto_mode, write_back=write_back)


def run_variant_comparison(
    model: str | None = None, auto_mode: str = "light"
) -> dict[str, OptimizationResult]:
    """Run optimization across all prompt variants for comparison.

    Each variant optimizes for its own metric:
    - COST_OPTIMIZED: 50% accuracy + 25% input + 25% output
    - ACCURACY_BALANCED: Accuracy / (1 + tokens/baseline)
    - MAXIMUM_ACCURACY: Pure accuracy
    """
    results = {}

    for variant in PromptVariant:
        print(f"\n{'#' * 60}")
        print(f"# Variant: {variant.value}")
        print(f"{'#' * 60}")

        try:
            optimizer = ToolSelectionOptimizer(model=model, variant=variant)
            results[variant.value] = optimizer.optimize(auto_mode=auto_mode)
        except Exception as e:
            print(f"[ERROR] {variant.value}: {e}")

    if results:
        print(f"\n{'=' * 60}")
        print("VARIANT COMPARISON (each variant uses its own metric)")
        print(f"{'=' * 60}")
        for name, result in sorted(results.items(), key=lambda x: -x[1].optimized_score):
            print(f"{name}: {result.baseline_score:.3f} -> {result.optimized_score:.3f} ({result.improvement:+.3f})")

    return results


def run_agent_comparison(model: str | None = None, auto_mode: str = "light") -> dict[str, OptimizationResult]:
    """Run optimization across both agent types for comparison."""
    results = {}

    for agent_type in AgentType:
        print(f"\n{'#' * 60}")
        print(f"# Agent: {agent_type.value}")
        print(f"{'#' * 60}")

        try:
            optimizer = ToolSelectionOptimizer(
                model=model,
                variant=PromptVariant.MAXIMUM_ACCURACY,
                agent_type=agent_type,
            )
            results[agent_type.value] = optimizer.optimize(auto_mode=auto_mode)
        except Exception as e:
            print(f"[ERROR] {agent_type.value}: {e}")

    if results:
        print(f"\n{'=' * 60}")
        print("AGENT COMPARISON (by variant metric)")
        print(f"{'=' * 60}")
        for name, result in sorted(results.items(), key=lambda x: -x[1].optimized_score):
            print(f"{name}: {result.baseline_score:.3f} -> {result.optimized_score:.3f} ({result.improvement:+.3f})")

    return results


def run_multi_model_comparison(auto_mode: str = "light") -> dict[str, OptimizationResult]:
    """Run optimization across multiple models for comparison."""
    results = {}

    for model_config in BENCHMARK_MODELS:
        print(f"\n{'#' * 60}")
        print(f"# {model_config.name}")
        print(f"{'#' * 60}")

        try:
            optimizer = ToolSelectionOptimizer(model=model_config.model_id)
            results[model_config.name] = optimizer.optimize(auto_mode=auto_mode)
        except Exception as e:
            print(f"[ERROR] {model_config.name}: {e}")

    if results:
        print(f"\n{'=' * 60}")
        print("MULTI-MODEL COMPARISON (by variant metric)")
        print(f"{'=' * 60}")
        for name, result in sorted(results.items(), key=lambda x: -x[1].optimized_score):
            print(f"{name}: {result.baseline_score:.3f} -> {result.optimized_score:.3f} ({result.improvement:+.3f})")

    return results


def run_optimizer_comparison(
    model: str | None = None, auto_mode: str = "light"
) -> dict[str, OptimizationResult]:
    """Run optimization across all DSPy optimizer types for comparison.

    Compares: MIPROv2, BootstrapFewShot, COPRO, SIMBA, GEPA
    """
    results = {}

    for opt_type in OptimizerType:
        print(f"\n{'#' * 60}")
        print(f"# Optimizer: {opt_type.value}")
        print(f"{'#' * 60}")

        try:
            optimizer = ToolSelectionOptimizer(model=model)
            results[opt_type.value] = optimizer.optimize(
                optimizer_type=opt_type,
                auto_mode=auto_mode,
            )
        except Exception as e:
            print(f"[ERROR] {opt_type.value}: {e}")

    if results:
        print(f"\n{'=' * 60}")
        print("OPTIMIZER COMPARISON (by variant metric)")
        print(f"{'=' * 60}")
        for name, result in sorted(
            results.items(),
            key=lambda x: -x[1].optimized_score,
        ):
            print(f"{name}: {result.baseline_score:.3f} -> {result.optimized_score:.3f} (acc={result.optimized_metrics.accuracy:.1%})")

    return results


if __name__ == "__main__":
    import sys

    # Parse arguments
    model = None
    auto_mode = "light"
    variant = "maximum_accuracy"
    optimizer_type = "copro"
    write_back = False
    run_all_models = False
    run_all_variants = False
    run_all_agents = False
    run_all_optimizers = False
    agent_type = None

    for arg in sys.argv[1:]:
        if arg.startswith("--mode="):
            auto_mode = arg.split("=")[1]
        elif arg.startswith("--variant="):
            variant = arg.split("=")[1]
        elif arg.startswith("--optimizer="):
            optimizer_type = arg.split("=")[1]
        elif arg.startswith("--agent="):
            agent_type = arg.split("=")[1]
        elif arg == "--write-back":
            write_back = True
        elif arg == "--all-models":
            run_all_models = True
        elif arg == "--all-variants":
            run_all_variants = True
        elif arg == "--all-agents":
            run_all_agents = True
        elif arg == "--compare-optimizers":
            run_all_optimizers = True
        elif not arg.startswith("--"):
            model = arg

    # Run appropriate optimization
    if run_all_models:
        run_multi_model_comparison(auto_mode=auto_mode)
    elif run_all_variants:
        run_variant_comparison(model=model, auto_mode=auto_mode)
    elif run_all_agents:
        run_agent_comparison(model=model, auto_mode=auto_mode)
    elif run_all_optimizers:
        run_optimizer_comparison(model=model, auto_mode=auto_mode)
    else:
        result = run_optimization(
            model=model,
            variant=variant,
            optimizer_type=optimizer_type,
            auto_mode=auto_mode,
            write_back=write_back,
            agent_type=agent_type,
        )
        print(f"\nFinal score: {result.baseline_score:.3f} -> {result.optimized_score:.3f}")
