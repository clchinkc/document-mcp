"""Dynamic configuration loader for benchmarks.

Loads tool sets, descriptions, and model configs from YAML files,
enabling on-the-fly testing without code changes.
"""

import os
from pathlib import Path
from typing import Any

import yaml

from .tools import ToolDescription
from .tools import ToolSet

# Get the benchmarks directory
BENCHMARKS_DIR = Path(__file__).parent
TOOL_SETS_DIR = BENCHMARKS_DIR / "tool_sets"
DESCRIPTIONS_DIR = BENCHMARKS_DIR / "descriptions"
MODELS_DIR = BENCHMARKS_DIR / "models"


def list_available_configs(config_type: str) -> list[str]:
    """List available config files for a given type.

    Args:
        config_type: One of 'tool_sets', 'descriptions', 'models'

    Returns:
        List of config names (without .yaml extension)
    """
    dirs = {
        "tool_sets": TOOL_SETS_DIR,
        "descriptions": DESCRIPTIONS_DIR,
        "models": MODELS_DIR,
    }

    config_dir = dirs.get(config_type)
    if not config_dir or not config_dir.exists():
        return []

    return sorted(f.stem for f in config_dir.glob("*.yaml") if f.is_file())


def load_yaml_config(config_type: str, name: str) -> dict[str, Any]:
    """Load a YAML config file.

    Args:
        config_type: One of 'tool_sets', 'descriptions', 'models'
        name: Config name (without .yaml extension)

    Returns:
        Parsed YAML content as dict

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML is invalid
    """
    dirs = {
        "tool_sets": TOOL_SETS_DIR,
        "descriptions": DESCRIPTIONS_DIR,
        "models": MODELS_DIR,
    }

    config_dir = dirs.get(config_type)
    if not config_dir:
        raise ValueError(f"Unknown config type: {config_type}")

    config_path = config_dir / f"{name}.yaml"
    if not config_path.exists():
        available = list_available_configs(config_type)
        raise FileNotFoundError(f"Config '{name}' not found in {config_type}/. Available: {available}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def load_tool_set(name: str) -> ToolSet:
    """Load a tool set from a YAML file.

    Args:
        name: Tool set name (e.g., '4-tool', '8-tool')

    Returns:
        ToolSet instance ready for benchmarking
    """
    config = load_yaml_config("tool_sets", name)

    tools = [
        ToolDescription(
            name=t["name"],
            what=t["what"],
            when=t["when"],
        )
        for t in config["tools"]
    ]

    return ToolSet(
        name=config["name"],
        tool_count=config["tool_count"],
        tools=tools,
        tool_mapping=config.get("tool_mapping", {}),
    )


def load_description_style(name: str) -> dict[str, Any]:
    """Load a description style from a YAML file.

    Args:
        name: Style name (e.g., 'default', 'minimal', 'verbose')

    Returns:
        Description style config dict
    """
    return load_yaml_config("descriptions", name)


def load_model_config(name: str) -> dict[str, Any]:
    """Load a model configuration from a YAML file.

    Args:
        name: Model name (e.g., 'gpt-5-mini', 'gemini-3-flash')

    Returns:
        Model config dict with provider, model_id, etc.
    """
    config = load_yaml_config("models", name)

    # Resolve API key from environment if specified
    api_key_env = config.get("api_key_env")
    if api_key_env:
        config["api_key"] = os.environ.get(api_key_env)

    return config


def get_tool_set_from_config(name: str) -> ToolSet:
    """Get a tool set, trying file-based config first, then fallback to code.

    Args:
        name: Tool set name

    Returns:
        ToolSet instance
    """
    # Try file-based first
    try:
        return load_tool_set(name)
    except FileNotFoundError:
        pass

    # Fallback to code-based (for backwards compatibility)
    from .tools import get_tool_set

    return get_tool_set(name)


def get_model_from_config(name: str) -> dict[str, Any]:
    """Get model config, trying file-based first.

    Args:
        name: Model name

    Returns:
        Model config dict
    """
    # Try file-based first
    try:
        return load_model_config(name)
    except FileNotFoundError:
        pass

    # Fallback to code-based
    from .config import BENCHMARK_MODELS

    if name in BENCHMARK_MODELS:
        return {
            "name": name,
            "provider": BENCHMARK_MODELS[name].provider,
            "model_id": BENCHMARK_MODELS[name].model_id,
            "temperature": BENCHMARK_MODELS[name].temperature,
        }

    raise ValueError(f"Unknown model: {name}")


def print_available_configs() -> None:
    """Print all available configurations."""
    print("\n=== Available Benchmark Configurations ===\n")

    print("Tool Sets:")
    for name in list_available_configs("tool_sets"):
        try:
            ts = load_tool_set(name)
            print(f"  - {name}: {ts.name} ({ts.tool_count} tools)")
        except Exception as e:
            print(f"  - {name}: (error: {e})")

    print("\nDescription Styles:")
    for name in list_available_configs("descriptions"):
        try:
            desc = load_description_style(name)
            print(f"  - {name}: {desc.get('name', name)}")
        except Exception as e:
            print(f"  - {name}: (error: {e})")

    print("\nModels:")
    for name in list_available_configs("models"):
        try:
            model = load_model_config(name)
            print(f"  - {name}: {model.get('name', name)} ({model.get('provider', 'unknown')})")
        except Exception as e:
            print(f"  - {name}: (error: {e})")

    print()
