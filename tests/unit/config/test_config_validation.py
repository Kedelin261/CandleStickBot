"""
Tests for M15 — Config System: Pydantic validation models
Validates all constraint enforcement including hard caps and phase boundaries.
"""

import pytest
from pydantic import ValidationError

from src.config.models import (
    BotConfig,
    RiskConfig,
    TQSConfig,
    ExecutionConfig,
    ExecutionMode,
    StrategiesConfig,
    LevelsConfig,
    FibonacciConfig,
    SupplyDemandConfig,
    InsideBarStrategyConfig,
    FalseBreakoutStrategyConfig,
    PortfolioConfig,
    OptimizationConfig,
    SystemConfig,
    PinBarStrategyConfig,
)


# ===========================================================================
# RISK CONFIG VALIDATION
# ===========================================================================

class TestRiskConfigValidation:

    def test_valid_risk_config(self):
        """Standard 1% risk config is valid."""
        rc = RiskConfig(risk_per_trade_pct=1.0, max_risk_per_trade_pct=2.0)
        assert rc.risk_per_trade_pct == 1.0

    def test_hard_cap_2pct_enforced(self):
        """max_risk_per_trade_pct > 2.0 raises ValidationError (hard cap)."""
        with pytest.raises(ValidationError):
            RiskConfig(max_risk_per_trade_pct=2.5)

    def test_risk_per_trade_cannot_exceed_max(self):
        """risk_per_trade_pct > max_risk_per_trade_pct raises ValidationError."""
        with pytest.raises(ValidationError):
            RiskConfig(risk_per_trade_pct=1.8, max_risk_per_trade_pct=1.5)

    def test_min_rr_ratio_cannot_be_below_2(self):
        """min_rr_ratio < 2.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            RiskConfig(min_rr_ratio=1.5)

    def test_min_rr_ratio_exactly_2_valid(self):
        """min_rr_ratio = 2.0 is valid (minimum allowed)."""
        rc = RiskConfig(min_rr_ratio=2.0)
        assert rc.min_rr_ratio == 2.0

    def test_kill_switch_drawdown_must_be_positive(self):
        """kill_switch_drawdown_pct must be >= 2.0."""
        with pytest.raises(ValidationError):
            RiskConfig(kill_switch_drawdown_pct=1.0)

    def test_daily_loss_limit_range(self):
        """daily_loss_limit_pct must be in valid range."""
        with pytest.raises(ValidationError):
            RiskConfig(daily_loss_limit_pct=0.1)  # Too low (min 0.5)


# ===========================================================================
# TQS CONFIG VALIDATION
# ===========================================================================

class TestTQSConfigValidation:

    def test_premium_risk_default_is_1_pct(self):
        """Premium risk default is 1.0% (same as standard — disabled by default)."""
        tqs = TQSConfig()
        assert tqs.premium_risk_pct == 1.0

    def test_premium_risk_cannot_exceed_hard_cap(self):
        """premium_risk_pct > 2.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            TQSConfig(premium_risk_pct=2.5)

    def test_premium_risk_1_5_valid_explicit(self):
        """Setting premium_risk_pct to 1.5 is valid when explicit."""
        tqs = TQSConfig(premium_risk_pct=1.5)
        assert tqs.premium_risk_pct == 1.5

    def test_min_score_threshold_valid(self):
        """min_score_to_trade must be 0-100."""
        with pytest.raises(ValidationError):
            TQSConfig(min_score_to_trade=101)

    def test_premium_threshold_above_min_score(self):
        """premium_threshold must be >= min_score_to_trade."""
        tqs = TQSConfig(min_score_to_trade=60, premium_threshold=80)
        assert tqs.premium_threshold == 80


# ===========================================================================
# PHASE 1 SCOPE ENFORCEMENT
# ===========================================================================

class TestPhase1ScopeEnforcement:

    def _make_phase1_config(self, **overrides):
        """Helper: create minimal phase 1 config with optional overrides."""
        defaults = {
            "system": {"phase": 1},
            "symbols": ["EURUSD"],
            "timeframes": {"primary": "D1", "context": "W1"},
        }
        defaults.update(overrides)
        return defaults

    def test_phase1_eurusd_only_valid(self):
        """Phase 1 with only EURUSD is valid."""
        config = BotConfig(
            system=SystemConfig(phase=1),
            symbols=["EURUSD"],
        )
        assert config.symbols == ["EURUSD"]

    def test_phase1_multi_pair_rejected(self):
        """Phase 1 rejects multiple symbols."""
        with pytest.raises(ValidationError) as exc_info:
            BotConfig(
                system=SystemConfig(phase=1),
                symbols=["EURUSD", "GBPUSD"],
            )
        assert "Phase 1" in str(exc_info.value) or "multi-pair" in str(exc_info.value).lower()

    def test_phase1_inside_bar_disabled_required(self):
        """Phase 1: inside_bar.enabled=True raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BotConfig(
                system=SystemConfig(phase=1),
                symbols=["EURUSD"],
                strategies=StrategiesConfig(
                    inside_bar=InsideBarStrategyConfig(enabled=True)
                ),
            )
        assert "Phase 2" in str(exc_info.value) or "inside_bar" in str(exc_info.value).lower()

    def test_phase1_false_breakout_disabled_required(self):
        """Phase 1: false_breakout.enabled=True raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BotConfig(
                system=SystemConfig(phase=1),
                symbols=["EURUSD"],
                strategies=StrategiesConfig(
                    inside_bar_false_breakout=FalseBreakoutStrategyConfig(enabled=True)
                ),
            )
        assert "Phase 2" in str(exc_info.value)

    def test_phase1_fibonacci_disabled_required(self):
        """Phase 1: fibonacci.enabled=True raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BotConfig(
                system=SystemConfig(phase=1),
                symbols=["EURUSD"],
                levels=LevelsConfig(fibonacci=FibonacciConfig(enabled=True)),
            )
        assert "Phase 2" in str(exc_info.value) or "fibonacci" in str(exc_info.value).lower()

    def test_phase1_supply_demand_disabled_required(self):
        """Phase 1: supply_demand_zones.enabled=True raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BotConfig(
                system=SystemConfig(phase=1),
                symbols=["EURUSD"],
                levels=LevelsConfig(supply_demand_zones=SupplyDemandConfig(enabled=True)),
            )
        assert "Phase 2" in str(exc_info.value) or "supply" in str(exc_info.value).lower()

    def test_phase1_portfolio_disabled_required(self):
        """Phase 1: portfolio.enabled=True raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BotConfig(
                system=SystemConfig(phase=1),
                symbols=["EURUSD"],
                portfolio=PortfolioConfig(enabled=True),
            )
        assert "Phase 2" in str(exc_info.value) or "portfolio" in str(exc_info.value).lower()

    def test_phase1_optimization_disabled_required(self):
        """Phase 1: optimization.enabled=True raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BotConfig(
                system=SystemConfig(phase=1),
                symbols=["EURUSD"],
                optimization=OptimizationConfig(enabled=True),
            )
        assert "Phase 2" in str(exc_info.value) or "optimization" in str(exc_info.value).lower()

    def test_phase2_allows_multiple_symbols(self):
        """Phase 2 config with multiple symbols is valid."""
        config = BotConfig(
            system=SystemConfig(phase=2),
            symbols=["EURUSD", "GBPUSD"],
        )
        assert len(config.symbols) == 2

    def test_phase2_allows_fibonacci(self):
        """Phase 2 config with fibonacci enabled is valid."""
        config = BotConfig(
            system=SystemConfig(phase=2),
            symbols=["EURUSD", "GBPUSD"],
            levels=LevelsConfig(fibonacci=FibonacciConfig(enabled=True)),
        )
        assert config.levels.fibonacci.enabled is True


# ===========================================================================
# PIN BAR CONFIG VALIDATION
# ===========================================================================

class TestPinBarConfig:

    def test_tail_ratio_minimum(self):
        """min_tail_ratio must be >= 1.0."""
        with pytest.raises(ValidationError):
            PinBarStrategyConfig(min_tail_ratio=0.5)

    def test_tail_ratio_default(self):
        """Default tail ratio is 2.0."""
        pb = PinBarStrategyConfig()
        assert pb.min_tail_ratio == 2.0

    def test_quality_score_range(self):
        """quality_min_score must be 1-10."""
        with pytest.raises(ValidationError):
            PinBarStrategyConfig(quality_min_score=0)
        with pytest.raises(ValidationError):
            PinBarStrategyConfig(quality_min_score=11)

    def test_buffer_pips_minimum(self):
        """buffer_pips must be >= 1.0."""
        with pytest.raises(ValidationError):
            PinBarStrategyConfig(buffer_pips=0.5)


# ===========================================================================
# COMPLETE BOT CONFIG VALIDATION
# ===========================================================================

class TestBotConfigComplete:

    def test_full_default_config_valid(self):
        """BotConfig() with all defaults is valid."""
        config = BotConfig()
        assert isinstance(config, BotConfig)

    def test_all_11_strategy_risk_params_accessible(self):
        """All 11 key strategy/risk parameters are accessible (Section 16 requirement)."""
        config = BotConfig()
        # Strategy params
        assert config.strategies.pin_bar.min_tail_ratio == 2.0
        assert config.strategies.pin_bar.quality_min_score == 5
        assert config.strategies.engulfing_bar.strict_mode is False
        # Risk params
        assert config.risk.risk_per_trade_pct == 1.0
        assert config.risk.max_risk_per_trade_pct == 2.0
        assert config.risk.min_rr_ratio == 2.0
        assert config.risk.daily_loss_limit_pct == 3.0
        assert config.risk.weekly_loss_limit_pct == 6.0
        assert config.risk.kill_switch_drawdown_pct == 10.0
        assert config.risk.max_open_trades == 3
        # TQS
        assert config.tqs.min_score_to_trade == 60
        assert config.tqs.premium_threshold == 80
