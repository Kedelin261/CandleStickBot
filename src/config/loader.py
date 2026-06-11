"""
M15 — Config System: ConfigLoader
Loads and merges YAML configuration with environment variable overrides.
Version: 3.1 (Phase 0)
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

from .models import BotConfig


# Default config file path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "default_config.yaml"
LOCAL_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "local_config.yaml"


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge override dict into base dict.
    Override values take precedence over base values.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(config_dict: Dict, prefix: str = "CSBOT__") -> Dict:
    """
    Apply environment variable overrides to config dict.
    Format: CSBOT__SECTION__KEY=value (double underscore as separator)
    Example: CSBOT__EXECUTION__MODE=paper
    Example: CSBOT__RISK__RISK_PER_TRADE_PCT=1.5
    """
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        # Strip prefix and split by double underscore
        path = env_key[len(prefix):].lower().split("__")
        if len(path) < 2:
            continue
        # Navigate to the correct nested dict
        target = config_dict
        for part in path[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        # Try to parse as appropriate Python type
        leaf_key = path[-1]
        target[leaf_key] = _parse_env_value(env_value)

    return config_dict


def _parse_env_value(value: str) -> Any:
    """Parse string environment variable into appropriate Python type."""
    # Boolean
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False
    # Integer
    try:
        return int(value)
    except ValueError:
        pass
    # Float
    try:
        return float(value)
    except ValueError:
        pass
    # String
    return value


def load_config(
    config_path: Optional[Path] = None,
    override_path: Optional[Path] = None,
    apply_env: bool = True,
) -> BotConfig:
    """
    Load and validate bot configuration.

    Priority (highest to lowest):
    1. Environment variables (CSBOT__ prefix)
    2. local_config.yaml (or override_path)
    3. default_config.yaml (or config_path)

    Args:
        config_path: Path to base config file (default: config/default_config.yaml)
        override_path: Path to local overrides (default: config/local_config.yaml)
        apply_env: Whether to apply environment variable overrides

    Returns:
        Validated BotConfig instance

    Raises:
        FileNotFoundError: If base config file not found
        ValidationError: If configuration values fail pydantic validation
    """
    base_path = config_path or DEFAULT_CONFIG_PATH
    local_path = override_path or LOCAL_CONFIG_PATH

    # Load base config
    if not base_path.exists():
        raise FileNotFoundError(
            f"Base configuration file not found: {base_path}. "
            f"Expected at: {DEFAULT_CONFIG_PATH}"
        )

    with open(base_path, "r") as f:
        config_dict = yaml.safe_load(f) or {}

    # Merge local override config if it exists
    if local_path.exists():
        with open(local_path, "r") as f:
            local_dict = yaml.safe_load(f) or {}
        config_dict = _deep_merge(config_dict, local_dict)

    # Apply environment variable overrides
    if apply_env:
        config_dict = _apply_env_overrides(config_dict)

    # Validate and return
    return BotConfig.model_validate(config_dict)


def get_config(
    config_path: Optional[Path] = None,
    override_path: Optional[Path] = None,
) -> BotConfig:
    """
    Convenience function to load config with defaults.
    Singleton-style: returns same instance if called multiple times with same args.
    """
    return load_config(config_path=config_path, override_path=override_path)


def save_config(config: BotConfig, path: Path) -> None:
    """
    Serialize a BotConfig to YAML file.
    Useful for saving modified configuration after parameter changes.

    Args:
        config: Validated BotConfig instance
        path: Destination file path
    """
    # Use mode='json' to convert enums to their string values
    config_dict = config.model_dump(mode='json')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)
