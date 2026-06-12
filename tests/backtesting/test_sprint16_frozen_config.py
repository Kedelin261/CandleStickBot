"""
Sprint 16 — Stage 2: Frozen Config Drift-Alarm Tests
=====================================================
Asserts that config/baseline_phase1_frozen.yaml contains exactly the values
registered during Sprint 14/16. Any deviation in the file triggers a test
failure, which is the INTENDED behaviour — these tests are a drift alarm.

If you are seeing a failure here, it means:
  (a) the frozen config file was modified (forbidden without pre-registration), OR
  (b) you are intentionally updating the spec (requires: formal pre-registration,
      train/validation split, owner sign-off, update these constants too).

Reference: config/baseline_phase1_frozen.yaml (provenance block)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# FROZEN CONSTANTS — single source of truth for drift alarm
# These values MUST NOT change without the process described in the module
# docstring above. Keep in sync with config/baseline_phase1_frozen.yaml.
# ─────────────────────────────────────────────────────────────────────────────

_FROZEN_OFFICIAL = {
    "lookback_window": 200,
    "minimum_tqs": 0.0,
    "minimum_rr": 2.0,
    "slippage_pips": 1.0,
    "initial_balance": 10000,
}

_FROZEN_APPENDIX_TQS60 = {
    "lookback_window": 200,
    "minimum_tqs": 60.0,
    "minimum_rr": 2.0,
    "slippage_pips": 1.0,
    "initial_balance": 10000,
}

_FROZEN_VERDICT = {
    "profit_factor_min": 1.10,
    "expectancy_r_min": 0.0,
    "max_drawdown_pct_max": 25.0,
    "min_trades": 30,
}

_CONFIG_PATH = Path(__file__).parents[2] / "config" / "baseline_phase1_frozen.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def frozen_config() -> dict:
    """Load the frozen config YAML file once per module."""
    assert _CONFIG_PATH.exists(), (
        f"Frozen config file not found: {_CONFIG_PATH}\n"
        "This file must exist. If this is a fresh environment, ensure "
        "config/baseline_phase1_frozen.yaml was committed and is present."
    )
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Test Class 1 — File existence and structure
# ─────────────────────────────────────────────────────────────────────────────

class TestFrozenConfigFileExists:
    def test_config_file_exists(self):
        """The frozen config file must exist at the expected path."""
        assert _CONFIG_PATH.exists(), f"Missing: {_CONFIG_PATH}"

    def test_config_has_official_section(self, frozen_config):
        """File must contain an 'official' section."""
        assert "official" in frozen_config, (
            "Frozen config missing 'official' section"
        )

    def test_config_has_appendix_tqs60_section(self, frozen_config):
        """File must contain an 'appendix_tqs60' section."""
        assert "appendix_tqs60" in frozen_config, (
            "Frozen config missing 'appendix_tqs60' section"
        )

    def test_config_has_verdict_criteria_section(self, frozen_config):
        """File must contain a 'verdict_criteria' section."""
        assert "verdict_criteria" in frozen_config, (
            "Frozen config missing 'verdict_criteria' section"
        )

    def test_config_has_provenance_section(self, frozen_config):
        """File must contain a 'provenance' section."""
        assert "provenance" in frozen_config, (
            "Frozen config missing 'provenance' section"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test Class 2 — Official variant drift alarm
# ─────────────────────────────────────────────────────────────────────────────

class TestOfficialVariantFrozen:
    """Drift-alarm: official variant must match _FROZEN_OFFICIAL exactly."""

    def test_official_lookback_window(self, frozen_config):
        assert frozen_config["official"]["lookback_window"] == _FROZEN_OFFICIAL["lookback_window"], \
            "DRIFT DETECTED: official.lookback_window changed from frozen value 200"

    def test_official_minimum_tqs(self, frozen_config):
        assert frozen_config["official"]["minimum_tqs"] == _FROZEN_OFFICIAL["minimum_tqs"], \
            "DRIFT DETECTED: official.minimum_tqs changed from frozen value 0.0"

    def test_official_minimum_rr(self, frozen_config):
        assert frozen_config["official"]["minimum_rr"] == _FROZEN_OFFICIAL["minimum_rr"], \
            "DRIFT DETECTED: official.minimum_rr changed from frozen value 2.0"

    def test_official_slippage_pips(self, frozen_config):
        assert frozen_config["official"]["slippage_pips"] == _FROZEN_OFFICIAL["slippage_pips"], \
            "DRIFT DETECTED: official.slippage_pips changed from frozen value 1.0"

    def test_official_initial_balance(self, frozen_config):
        assert frozen_config["official"]["initial_balance"] == _FROZEN_OFFICIAL["initial_balance"], \
            "DRIFT DETECTED: official.initial_balance changed from frozen value 10000"

    def test_official_no_extra_keys(self, frozen_config):
        """Official section must not gain unexpected new keys (parameter proliferation guard)."""
        file_keys = set(frozen_config["official"].keys())
        expected_keys = set(_FROZEN_OFFICIAL.keys())
        extra = file_keys - expected_keys
        assert not extra, (
            f"DRIFT DETECTED: unexpected keys in official section: {extra}\n"
            "Adding parameters to the frozen config requires pre-registration."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test Class 3 — Appendix tqs60 variant drift alarm
# ─────────────────────────────────────────────────────────────────────────────

class TestAppendixTqs60Frozen:
    """Drift-alarm: appendix_tqs60 variant must match _FROZEN_APPENDIX_TQS60 exactly."""

    def test_appendix_lookback_window(self, frozen_config):
        assert frozen_config["appendix_tqs60"]["lookback_window"] == _FROZEN_APPENDIX_TQS60["lookback_window"], \
            "DRIFT DETECTED: appendix_tqs60.lookback_window changed"

    def test_appendix_minimum_tqs(self, frozen_config):
        assert frozen_config["appendix_tqs60"]["minimum_tqs"] == _FROZEN_APPENDIX_TQS60["minimum_tqs"], \
            "DRIFT DETECTED: appendix_tqs60.minimum_tqs changed from frozen value 60.0"

    def test_appendix_minimum_rr(self, frozen_config):
        assert frozen_config["appendix_tqs60"]["minimum_rr"] == _FROZEN_APPENDIX_TQS60["minimum_rr"], \
            "DRIFT DETECTED: appendix_tqs60.minimum_rr changed from frozen value 2.0"

    def test_appendix_slippage_pips(self, frozen_config):
        assert frozen_config["appendix_tqs60"]["slippage_pips"] == _FROZEN_APPENDIX_TQS60["slippage_pips"], \
            "DRIFT DETECTED: appendix_tqs60.slippage_pips changed from frozen value 1.0"

    def test_appendix_initial_balance(self, frozen_config):
        assert frozen_config["appendix_tqs60"]["initial_balance"] == _FROZEN_APPENDIX_TQS60["initial_balance"], \
            "DRIFT DETECTED: appendix_tqs60.initial_balance changed from frozen value 10000"

    def test_appendix_differs_from_official_only_in_tqs(self, frozen_config):
        """The ONLY difference between official and appendix_tqs60 must be minimum_tqs."""
        off = frozen_config["official"]
        app = frozen_config["appendix_tqs60"]
        for key in _FROZEN_OFFICIAL:
            if key == "minimum_tqs":
                continue  # permitted to differ
            assert off[key] == app[key], (
                f"DRIFT DETECTED: official.{key} != appendix_tqs60.{key} — "
                "appendix variant must only differ in minimum_tqs"
            )

    def test_appendix_tqs_strictly_greater_than_official(self, frozen_config):
        """appendix_tqs60.minimum_tqs must be > official.minimum_tqs."""
        assert frozen_config["appendix_tqs60"]["minimum_tqs"] > frozen_config["official"]["minimum_tqs"], \
            "DRIFT DETECTED: appendix_tqs60.minimum_tqs must be greater than official.minimum_tqs"


# ─────────────────────────────────────────────────────────────────────────────
# Test Class 4 — Verdict criteria drift alarm
# ─────────────────────────────────────────────────────────────────────────────

class TestVerdictCriteriaFrozen:
    """Drift-alarm: verdict_criteria must match _FROZEN_VERDICT exactly."""

    def test_verdict_profit_factor_min(self, frozen_config):
        assert frozen_config["verdict_criteria"]["profit_factor_min"] == _FROZEN_VERDICT["profit_factor_min"], \
            "DRIFT DETECTED: verdict_criteria.profit_factor_min changed from 1.10"

    def test_verdict_expectancy_r_min(self, frozen_config):
        assert frozen_config["verdict_criteria"]["expectancy_r_min"] == _FROZEN_VERDICT["expectancy_r_min"], \
            "DRIFT DETECTED: verdict_criteria.expectancy_r_min changed from 0.0"

    def test_verdict_max_drawdown_pct_max(self, frozen_config):
        assert frozen_config["verdict_criteria"]["max_drawdown_pct_max"] == _FROZEN_VERDICT["max_drawdown_pct_max"], \
            "DRIFT DETECTED: verdict_criteria.max_drawdown_pct_max changed from 25.0"

    def test_verdict_min_trades(self, frozen_config):
        assert frozen_config["verdict_criteria"]["min_trades"] == _FROZEN_VERDICT["min_trades"], \
            "DRIFT DETECTED: verdict_criteria.min_trades changed from 30"


# ─────────────────────────────────────────────────────────────────────────────
# Test Class 5 — Consistency: frozen constants match BacktestResult.passes_baseline
# ─────────────────────────────────────────────────────────────────────────────

class TestFrozenConstantsMatchImplementation:
    """
    Verify that the frozen verdict criteria in the YAML match the live
    implementation in BacktestResult.passes_baseline.
    This is the cross-check between the spec (YAML) and the code.
    """

    # Shared kwargs for BacktestResult construction (required positional fields)
    _BR_KWARGS = dict(
        symbol="EURUSD",
        timeframe="D1",
        strategy_mode="combined",
        trades_generated=50,
        wins=0,
        losses=0,
        win_rate=0.0,
        initial_balance=10_000.0,
        final_balance=10_000.0,
    )

    def _make_result(self, **overrides):
        from src.backtesting.backtest_runner import BacktestResult
        kwargs = {**self._BR_KWARGS, **overrides}
        r = BacktestResult(**kwargs)
        # Set metric attributes that are not __init__ params
        r.profit_factor = overrides.pop("profit_factor", 1.50)
        r.expectancy_r = overrides.pop("expectancy_r", 0.20)
        r.max_drawdown_pct = overrides.pop("max_drawdown_pct", 10.0)
        return r

    def test_passes_baseline_consistent_with_frozen_min_trades(self, frozen_config):
        """BacktestResult with exactly N=min_trades-1 must fail passes_baseline."""
        from src.backtesting.backtest_runner import BacktestResult
        min_trades = frozen_config["verdict_criteria"]["min_trades"]
        r = BacktestResult(**{**self._BR_KWARGS, "trades_executed": min_trades - 1})
        r.profit_factor = 1.50
        r.expectancy_r = 0.20
        r.max_drawdown_pct = 10.0
        assert not r.passes_baseline, (
            f"BacktestResult with trades={min_trades-1} should FAIL passes_baseline "
            f"(frozen min_trades={min_trades})"
        )

    def test_passes_baseline_consistent_with_frozen_min_trades_exact(self, frozen_config):
        """BacktestResult with exactly N=min_trades and all other criteria met must pass."""
        from src.backtesting.backtest_runner import BacktestResult
        min_trades = frozen_config["verdict_criteria"]["min_trades"]
        r = BacktestResult(**{**self._BR_KWARGS, "trades_executed": min_trades})
        r.profit_factor = 1.50
        r.expectancy_r = 0.20
        r.max_drawdown_pct = 10.0
        assert r.passes_baseline, (
            f"BacktestResult with trades={min_trades} and good metrics should PASS passes_baseline"
        )

    def test_passes_baseline_consistent_with_frozen_pf(self, frozen_config):
        """PF at exactly the threshold (not strictly above) must fail."""
        from src.backtesting.backtest_runner import BacktestResult
        pf_min = frozen_config["verdict_criteria"]["profit_factor_min"]
        r = BacktestResult(**{**self._BR_KWARGS, "trades_executed": 30})
        r.profit_factor = pf_min   # exactly at threshold, not strictly above
        r.expectancy_r = 0.20
        r.max_drawdown_pct = 10.0
        assert not r.passes_baseline, (
            f"PF exactly at threshold {pf_min} should FAIL (criterion is strictly >)"
        )

    def test_passes_baseline_consistent_with_frozen_expectancy(self, frozen_config):
        """Expectancy_r at exactly 0.0 (not strictly above) must fail."""
        from src.backtesting.backtest_runner import BacktestResult
        r = BacktestResult(**{**self._BR_KWARGS, "trades_executed": 30})
        r.profit_factor = 1.50
        r.expectancy_r = 0.0   # at boundary, not strictly above
        r.max_drawdown_pct = 10.0
        assert not r.passes_baseline, \
            "Expectancy_r=0.0 should FAIL (criterion is strictly > 0.0)"

    def test_passes_baseline_consistent_with_frozen_drawdown(self, frozen_config):
        """DD at exactly the max threshold must fail."""
        from src.backtesting.backtest_runner import BacktestResult
        dd_max = frozen_config["verdict_criteria"]["max_drawdown_pct_max"]
        r = BacktestResult(**{**self._BR_KWARGS, "trades_executed": 30})
        r.profit_factor = 1.50
        r.expectancy_r = 0.20
        r.max_drawdown_pct = dd_max   # exactly at boundary
        assert not r.passes_baseline, (
            f"DD={dd_max}% exactly at threshold should FAIL (criterion is strictly <)"
        )
