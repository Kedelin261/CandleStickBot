"""
Tests for M15 — Config System: Loader
Validates config loading, merging, and environment variable overrides.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config.loader import load_config, save_config, _deep_merge, _parse_env_value
from src.config.models import BotConfig, ExecutionMode


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def default_config_path():
    """Return path to the default config file."""
    return Path(__file__).parent.parent.parent.parent / "config" / "default_config.yaml"


@pytest.fixture
def minimal_config_yaml(tmp_path):
    """Create a minimal valid YAML config for testing."""
    config = {
        "system": {"phase": 1, "log_level": "INFO"},
        "execution": {"mode": "backtest"},
        "symbols": ["EURUSD"],
    }
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    return config_file


# ===========================================================================
# CONFIG LOADING TESTS
# ===========================================================================

class TestConfigLoading:
    def test_load_default_config(self, default_config_path):
        """Default config file loads and validates without errors."""
        config = load_config(config_path=default_config_path)
        assert isinstance(config, BotConfig)

    def test_default_mode_is_backtest(self, default_config_path):
        """Default execution mode must be backtest (safest)."""
        config = load_config(config_path=default_config_path)
        assert config.execution.mode == ExecutionMode.BACKTEST

    def test_default_symbols_eurusd_only(self, default_config_path):
        """Phase 1 default must only include EURUSD."""
        config = load_config(config_path=default_config_path)
        assert config.symbols == ["EURUSD"]

    def test_missing_config_raises_error(self, tmp_path):
        """FileNotFoundError raised when config file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_config(config_path=tmp_path / "nonexistent.yaml")

    def test_empty_yaml_loads_defaults(self, tmp_path):
        """Empty YAML file results in all-defaults config."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")
        config = load_config(config_path=empty_file)
        assert isinstance(config, BotConfig)
        assert config.execution.mode == ExecutionMode.BACKTEST

    def test_override_config_merged(self, default_config_path, tmp_path):
        """Local override config merges correctly into base config."""
        override = tmp_path / "override.yaml"
        override.write_text("execution:\n  mode: paper\n")
        config = load_config(config_path=default_config_path, override_path=override)
        assert config.execution.mode == ExecutionMode.PAPER

    def test_nonexistent_override_ignored(self, default_config_path, tmp_path):
        """Missing local override file is silently ignored."""
        config = load_config(
            config_path=default_config_path,
            override_path=tmp_path / "nonexistent_override.yaml",
        )
        assert isinstance(config, BotConfig)


# ===========================================================================
# DEEP MERGE TESTS
# ===========================================================================

class TestDeepMerge:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 99}

    def test_nested_override(self):
        base = {"execution": {"mode": "backtest", "slippage": 1.0}}
        override = {"execution": {"mode": "paper"}}
        result = _deep_merge(base, override)
        assert result["execution"]["mode"] == "paper"
        assert result["execution"]["slippage"] == 1.0  # Base value preserved

    def test_new_key_added(self):
        base = {"a": 1}
        override = {"b": 2}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_base_not_mutated(self):
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        _deep_merge(base, override)
        assert base["a"]["b"] == 1  # Original not modified


# ===========================================================================
# ENVIRONMENT VARIABLE TESTS
# ===========================================================================

class TestEnvOverrides:
    def test_env_mode_override(self, default_config_path, monkeypatch):
        """CSBOT__EXECUTION__MODE env var overrides execution mode."""
        monkeypatch.setenv("CSBOT__EXECUTION__MODE", "paper")
        config = load_config(config_path=default_config_path)
        assert config.execution.mode == ExecutionMode.PAPER

    def test_env_non_csbot_prefix_ignored(self, default_config_path, monkeypatch):
        """Env vars without CSBOT__ prefix are ignored."""
        monkeypatch.setenv("EXECUTION__MODE", "live")
        config = load_config(config_path=default_config_path)
        assert config.execution.mode == ExecutionMode.BACKTEST

    def test_parse_env_bool_true(self):
        for val in ("true", "True", "TRUE", "yes", "1"):
            assert _parse_env_value(val) is True

    def test_parse_env_bool_false(self):
        for val in ("false", "False", "FALSE", "no", "0"):
            assert _parse_env_value(val) is False

    def test_parse_env_int(self):
        assert _parse_env_value("42") == 42

    def test_parse_env_float(self):
        assert _parse_env_value("3.14") == pytest.approx(3.14)

    def test_parse_env_string(self):
        assert _parse_env_value("paper") == "paper"


# ===========================================================================
# CONFIG SAVE TESTS
# ===========================================================================

class TestConfigSave:
    def test_save_and_reload(self, default_config_path, tmp_path):
        """Config saves to YAML and reloads identically."""
        config = load_config(config_path=default_config_path)
        save_path = tmp_path / "saved_config.yaml"
        save_config(config, save_path)
        assert save_path.exists()
        # Reload and verify key fields
        reloaded = load_config(config_path=save_path)
        assert reloaded.execution.mode == config.execution.mode
        assert reloaded.symbols == config.symbols

    def test_save_creates_parent_dir(self, default_config_path, tmp_path):
        """save_config creates parent directories if they don't exist."""
        config = load_config(config_path=default_config_path)
        nested_path = tmp_path / "nested" / "dir" / "config.yaml"
        save_config(config, nested_path)
        assert nested_path.exists()
