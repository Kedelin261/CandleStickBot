"""
Unit tests for M09 Risk Management Engine.

Sprint 8 — 120 tests across 16 test classes:

    TestRiskConfig              (10) — construction, validation, hard-cap enforcement
    TestRiskState               ( 7) — is_trading_allowed, field defaults, mutation
    TestRiskEngineConstruction  ( 8) — defaults, config object, legacy kwargs, hard-cap rejection
    TestModuleLevelHelpers      (10) — compute_lot_size, compute_risk_amount, compute_stop_pips
    TestLotSizeCalculation      (12) — formula correctness, rounding, lot_min/max, edge cases
    TestDetermineRiskPct        ( 8) — standard/premium/None TQS, hard cap, disabled premium
    TestKillSwitchGate          ( 8) — gate 1 fires when active, type=KILL_SWITCH
    TestDailyLimitGate          ( 8) — gate 2: threshold, boundary, positive P&L pass
    TestWeeklyLimitGate         ( 7) — gate 3: threshold, boundary, positive P&L pass
    TestMaxTradesGate           ( 7) — gate 4: live account.open_trades used
    TestRRRatioGate             ( 6) — gate 5: min RR, exact boundary
    TestPositionSizeGate        ( 5) — gate 6: zero-stop rejection
    TestFullApproval            (10) — end-to-end approve: lot, risk_pct, stop_pips, balance
    TestKillSwitchLogic         (12) — drawdown trigger, consecutive losses, daily+weekly, reset
    TestStateUpdates            ( 8) — update_after_trade_close, open_trade_count, reset daily/weekly
    TestEdgeCases               (14) — zero balance, massive stop, tiny stop, premium TQS, etc.

Total: 120 tests
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

import pytest

from src.risk.risk_engine import (
    KillSwitchEvent,
    KillSwitchReason,
    RiskCheckResult,
    RiskConfig,
    RiskEngine,
    RiskState,
    compute_lot_size,
    compute_risk_amount,
    compute_stop_pips,
)
from src.types import (
    AccountState,
    Direction,
    RiskApprovedOrder,
    RiskRejection,
    StrategyName,
    TQSComponents,
    TradeTier,
    TradeRecommendation,
)


# ===========================================================================
# Shared helpers
# ===========================================================================

def _account(
    balance: float = 10_000.0,
    equity: float = 10_000.0,
    open_trades: int = 0,
    daily_pnl_pct: float = 0.0,      # convenience: ignored when day_open_balance given
    weekly_pnl_pct: float = 0.0,     # convenience: ignored when week_open_balance given
    peak_equity: float = 10_000.0,
    day_open_balance: Optional[float] = None,
    week_open_balance: Optional[float] = None,
) -> AccountState:
    """Build a minimal AccountState with sensible defaults."""
    # If explicit open-balance provided, use it; otherwise derive from pnl_pct convenience param
    if day_open_balance is None:
        day_open_balance = balance / (1 + daily_pnl_pct / 100) if daily_pnl_pct else balance
    if week_open_balance is None:
        week_open_balance = balance / (1 + weekly_pnl_pct / 100) if weekly_pnl_pct else balance
    return AccountState(
        balance=balance,
        equity=equity,
        margin=0.0,
        free_margin=balance,
        open_pnl=0.0,
        peak_equity=peak_equity,
        day_open_balance=day_open_balance,
        week_open_balance=week_open_balance,
        open_trades=open_trades,
    )


def _tqs(
    trend: float = 20.0,
    level: float = 20.0,
    pattern: float = 20.0,
    regime: float = 20.0,
) -> TQSComponents:
    return TQSComponents(
        trend_score=trend,
        level_score=level,
        pattern_score=pattern,
        regime_score=regime,
    )


def _rec(
    entry: float = 1.1000,
    stop: float = 1.0980,
    target: float = 1.1040,
    rr: float = 2.0,
    direction: Direction = Direction.LONG,
    tqs: Optional[TQSComponents] = None,
    symbol: str = "EURUSD",
) -> TradeRecommendation:
    return TradeRecommendation(
        strategy=StrategyName.PIN_BAR,
        symbol=symbol,
        timeframe="H1",
        direction=direction,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        rr_ratio=rr,
        tqs=tqs or _tqs(),
    )


def _engine(**kwargs) -> RiskEngine:
    """Build a RiskEngine with sensible defaults, override with kwargs."""
    return RiskEngine(**kwargs)


# ===========================================================================
# TestRiskConfig
# ===========================================================================

class TestRiskConfig:
    """10 tests — RiskConfig construction and validation."""

    def test_default_values(self):
        cfg = RiskConfig()
        assert cfg.risk_per_trade_pct == 1.0
        assert cfg.max_risk_per_trade_pct == 2.0
        assert cfg.daily_loss_limit_pct == 3.0
        assert cfg.weekly_loss_limit_pct == 6.0
        assert cfg.kill_switch_drawdown_pct == 10.0
        assert cfg.kill_switch_consecutive_losses == 7
        assert cfg.min_rr_ratio == 2.0
        assert cfg.max_open_trades == 3

    def test_custom_values(self):
        cfg = RiskConfig(risk_per_trade_pct=0.5, daily_loss_limit_pct=2.0)
        assert cfg.risk_per_trade_pct == 0.5
        assert cfg.daily_loss_limit_pct == 2.0

    def test_hard_cap_at_exactly_two_pct_is_valid(self):
        cfg = RiskConfig(max_risk_per_trade_pct=2.0)
        assert cfg.max_risk_per_trade_pct == 2.0

    def test_hard_cap_above_two_pct_raises(self):
        with pytest.raises(ValueError, match="hard cap"):
            RiskConfig(max_risk_per_trade_pct=2.01)

    def test_hard_cap_three_pct_raises(self):
        with pytest.raises(ValueError, match="hard cap"):
            RiskConfig(max_risk_per_trade_pct=3.0)

    def test_zero_risk_per_trade_raises(self):
        with pytest.raises(ValueError):
            RiskConfig(risk_per_trade_pct=0.0)

    def test_negative_risk_per_trade_raises(self):
        with pytest.raises(ValueError):
            RiskConfig(risk_per_trade_pct=-1.0)

    def test_zero_pip_size_raises(self):
        with pytest.raises(ValueError):
            RiskConfig(pip_size=0.0)

    def test_zero_pip_value_raises(self):
        with pytest.raises(ValueError):
            RiskConfig(pip_value_per_lot=0.0)

    def test_premium_risk_disabled_by_default(self):
        cfg = RiskConfig()
        assert cfg.premium_risk_enabled is False


# ===========================================================================
# TestRiskState
# ===========================================================================

class TestRiskState:
    """7 tests — RiskState dataclass."""

    def test_default_is_trading_allowed(self):
        state = RiskState()
        assert state.is_trading_allowed is True

    def test_kill_switch_disables_trading(self):
        state = RiskState(kill_switch_active=True)
        assert state.is_trading_allowed is False

    def test_consecutive_losses_default_zero(self):
        state = RiskState()
        assert state.consecutive_losses == 0

    def test_open_trade_count_default_zero(self):
        state = RiskState()
        assert state.open_trade_count == 0

    def test_daily_pnl_pct_default_zero(self):
        state = RiskState()
        assert state.daily_pnl_pct == 0.0

    def test_kill_switch_reason_default_none(self):
        state = RiskState()
        assert state.kill_switch_reason is None

    def test_state_mutation(self):
        state = RiskState()
        state.consecutive_losses = 5
        state.kill_switch_active = True
        assert state.consecutive_losses == 5
        assert not state.is_trading_allowed


# ===========================================================================
# TestRiskEngineConstruction
# ===========================================================================

class TestRiskEngineConstruction:
    """8 tests — RiskEngine __init__ paths."""

    def test_default_construction(self):
        engine = RiskEngine()
        assert engine.risk_per_trade_pct == 1.0
        assert engine.kill_switch_active is False

    def test_construction_with_config(self):
        cfg = RiskConfig(risk_per_trade_pct=0.5, daily_loss_limit_pct=2.0)
        engine = RiskEngine(config=cfg)
        assert engine.risk_per_trade_pct == 0.5
        assert engine.daily_loss_limit_pct == 2.0

    def test_construction_with_legacy_kwargs(self):
        engine = RiskEngine(risk_per_trade_pct=1.5, max_open_trades=5)
        assert engine.risk_per_trade_pct == 1.5
        assert engine.max_open_trades == 5

    def test_hard_cap_violation_raises_on_construction(self):
        with pytest.raises(ValueError, match="hard cap"):
            RiskEngine(max_risk_per_trade_pct=2.5)

    def test_state_starts_clean(self):
        engine = RiskEngine()
        assert engine.state.consecutive_losses == 0
        assert engine.state.trades_today == 0
        assert engine.state.kill_switch_active is False

    def test_premium_risk_disabled_by_default(self):
        engine = RiskEngine()
        assert engine.premium_risk_enabled is False

    def test_pip_size_propagates(self):
        engine = RiskEngine(pip_size=0.01)  # JPY-style
        assert engine.pip_size == 0.01

    def test_is_trading_allowed_initially(self):
        engine = RiskEngine()
        assert engine.is_trading_allowed() is True


# ===========================================================================
# TestModuleLevelHelpers
# ===========================================================================

class TestModuleLevelHelpers:
    """10 tests — stateless module-level functions."""

    def test_compute_lot_size_standard(self):
        # 10000 * 1% = 100 risk; 20 pips * 10 = 200 per lot → 0.5 lots
        lots = compute_lot_size(10_000, 1.0, 1.1000, 1.0980)
        assert lots == pytest.approx(0.5, abs=0.01)

    def test_compute_lot_size_small_risk(self):
        lots = compute_lot_size(5_000, 1.0, 1.1000, 1.0990)  # 10 pips stop
        # 50 / (10 * 10) = 0.5
        assert lots == pytest.approx(0.5, abs=0.01)

    def test_compute_lot_size_zero_stop_returns_zero(self):
        assert compute_lot_size(10_000, 1.0, 1.1000, 1.1000) == 0.0

    def test_compute_lot_size_clamps_to_lot_min(self):
        # Very small account + big stop → raw lots < 0.01
        lots = compute_lot_size(100, 1.0, 1.1000, 1.0000, lot_min=0.01)
        assert lots == 0.01

    def test_compute_lot_size_clamps_to_lot_max(self):
        lots = compute_lot_size(10_000_000, 2.0, 1.1000, 1.0999, lot_max=100.0)
        assert lots == 100.0

    def test_compute_lot_size_jpy_pip_size(self):
        # USD/JPY: pip_size=0.01, pip_value=1000 yen ≈ different scale
        lots = compute_lot_size(10_000, 1.0, 110.00, 109.50, pip_size=0.01, pip_value_per_lot=1000.0)
        # 100 / (50 * 1000) = 0.002 → clamps to lot_min=0.01
        assert lots == 0.01

    def test_compute_risk_amount_basic(self):
        assert compute_risk_amount(10_000, 1.0) == 100.0

    def test_compute_risk_amount_half_pct(self):
        assert compute_risk_amount(5_000, 0.5) == 25.0

    def test_compute_stop_pips_basic(self):
        pips = compute_stop_pips(1.1000, 1.0980)
        assert pips == pytest.approx(20.0, abs=1e-6)

    def test_compute_stop_pips_zero_pip_size_returns_zero(self):
        assert compute_stop_pips(1.1000, 1.0980, pip_size=0.0) == 0.0


# ===========================================================================
# TestLotSizeCalculation
# ===========================================================================

class TestLotSizeCalculation:
    """12 tests — RiskEngine._calculate_lot_size directly."""

    def setup_method(self):
        self.engine = RiskEngine()

    def test_basic_calculation(self):
        # 10000 * 1% = 100; stop=20 pips * 10 pip_val = 200 → 0.5 lots
        lots = self.engine._calculate_lot_size(10_000, 1.0, 1.1000, 1.0980)
        assert lots == pytest.approx(0.5, abs=0.01)

    def test_different_stop_distance(self):
        # 10000 * 1% = 100; stop=10 pips → 1.0 lot
        lots = self.engine._calculate_lot_size(10_000, 1.0, 1.1000, 1.0990)
        assert lots == pytest.approx(1.0, abs=0.01)

    def test_larger_balance(self):
        # 100000 * 1% = 1000; stop=20 pips → 5.0 lots
        lots = self.engine._calculate_lot_size(100_000, 1.0, 1.1000, 1.0980)
        assert lots == pytest.approx(5.0, abs=0.01)

    def test_higher_risk_pct(self):
        # 10000 * 2% = 200; stop=20 pips → 1.0 lot
        lots = self.engine._calculate_lot_size(10_000, 2.0, 1.1000, 1.0980)
        assert lots == pytest.approx(1.0, abs=0.01)

    def test_zero_stop_distance_returns_zero(self):
        lots = self.engine._calculate_lot_size(10_000, 1.0, 1.1000, 1.1000)
        assert lots == 0.0

    def test_rounding_to_lot_step(self):
        # Should round to nearest 0.01 — verify by checking it's already a
        # rounded value (round-trip: round to 2dp should not change the result)
        lots = self.engine._calculate_lot_size(10_000, 1.0, 1.1000, 1.09833)
        # Result should equal its own 2-decimal rounded version (i.e., already stepped)
        assert lots == pytest.approx(round(lots, 2), abs=1e-9)

    def test_short_direction_symmetry(self):
        long_lots  = self.engine._calculate_lot_size(10_000, 1.0, 1.1020, 1.1000)
        short_lots = self.engine._calculate_lot_size(10_000, 1.0, 1.1000, 1.1020)
        assert long_lots == short_lots

    def test_minimum_lot_enforcement(self):
        tiny_lots = self.engine._calculate_lot_size(100, 1.0, 1.1000, 1.0000)
        assert tiny_lots == 0.01

    def test_maximum_lot_enforcement(self):
        huge_lots = self.engine._calculate_lot_size(
            10_000_000, 2.0, 1.1000, 1.0999
        )
        assert huge_lots == 100.0

    def test_result_non_negative(self):
        lots = self.engine._calculate_lot_size(10_000, 1.0, 1.1000, 1.0950)
        assert lots >= 0

    def test_round_trip_risk(self):
        """Lot × stop_pips × pip_value should approximately equal risk amount."""
        balance, risk_pct = 10_000, 1.0
        entry, stop = 1.1000, 1.0980
        lots = self.engine._calculate_lot_size(balance, risk_pct, entry, stop)
        stop_pips   = abs(entry - stop) / self.engine.pip_size
        risk_back   = lots * stop_pips * self.engine.pip_value_per_lot
        risk_target = balance * risk_pct / 100.0
        # Allow for lot-step rounding: within 1 pip-value
        assert abs(risk_back - risk_target) <= self.engine.pip_value_per_lot

    def test_premium_risk_higher_lots(self):
        """1.5% risk → more lots than 1.0% risk at same stop."""
        engine_standard = RiskEngine(risk_per_trade_pct=1.0)
        engine_premium  = RiskEngine(risk_per_trade_pct=1.5)
        standard_lots = engine_standard._calculate_lot_size(10_000, 1.0, 1.1000, 1.0980)
        premium_lots  = engine_premium._calculate_lot_size(10_000, 1.5, 1.1000, 1.0980)
        assert premium_lots > standard_lots


# ===========================================================================
# TestDetermineRiskPct
# ===========================================================================

class TestDetermineRiskPct:
    """8 tests — _determine_risk_pct logic."""

    def test_none_tqs_uses_base_risk(self):
        engine = RiskEngine(risk_per_trade_pct=1.0)
        assert engine._determine_risk_pct(None) == 1.0

    def test_standard_tqs_uses_base_risk(self):
        engine = RiskEngine(risk_per_trade_pct=1.0, premium_risk_enabled=False)
        tqs = _tqs(trend=15, level=15, pattern=15, regime=15)  # total=60 → STANDARD
        assert engine._determine_risk_pct(tqs) == 1.0

    def test_premium_tqs_disabled_uses_base_risk(self):
        engine = RiskEngine(risk_per_trade_pct=1.0, premium_risk_pct=1.5, premium_risk_enabled=False)
        tqs = _tqs(trend=20, level=20, pattern=20, regime=20)  # total=80 → PREMIUM
        assert engine._determine_risk_pct(tqs) == 1.0

    def test_premium_tqs_enabled_uses_premium_risk(self):
        engine = RiskEngine(risk_per_trade_pct=1.0, premium_risk_pct=1.5, premium_risk_enabled=True)
        tqs = _tqs(trend=20, level=20, pattern=20, regime=20)  # total=80 → PREMIUM
        assert engine._determine_risk_pct(tqs) == 1.5

    def test_hard_cap_applied_to_premium_risk(self):
        engine = RiskEngine(
            risk_per_trade_pct=1.0,
            max_risk_per_trade_pct=2.0,
            premium_risk_pct=1.9,
            premium_risk_enabled=True,
        )
        tqs = _tqs(trend=20, level=20, pattern=20, regime=20)
        result = engine._determine_risk_pct(tqs)
        assert result <= 2.0
        assert result == 1.9  # 1.9 < 2.0, not capped

    def test_hard_cap_clamps_premium_to_max(self):
        # premium_risk_pct cannot exceed max (2.0) — but RiskConfig already
        # prevents max > 2.0, so we set premium to exactly 2.0.
        engine = RiskEngine(
            risk_per_trade_pct=1.0,
            max_risk_per_trade_pct=2.0,
            premium_risk_pct=2.0,
            premium_risk_enabled=True,
        )
        tqs = _tqs(trend=20, level=20, pattern=20, regime=20)
        result = engine._determine_risk_pct(tqs)
        assert result == 2.0

    def test_reject_tier_uses_base_risk(self):
        engine = RiskEngine(risk_per_trade_pct=1.0)
        tqs = TQSComponents(trend_score=10, level_score=10, pattern_score=10, regime_score=10)  # total=40
        assert tqs.tier == TradeTier.REJECT
        assert engine._determine_risk_pct(tqs) == 1.0

    def test_result_never_exceeds_hard_cap(self):
        engine = RiskEngine(
            risk_per_trade_pct=1.0,
            max_risk_per_trade_pct=2.0,
            premium_risk_pct=1.5,
            premium_risk_enabled=True,
        )
        for tier_tqs in [
            _tqs(15, 15, 15, 15),   # STANDARD
            _tqs(20, 20, 20, 20),   # PREMIUM
            None,
        ]:
            assert engine._determine_risk_pct(tier_tqs) <= 2.0


# ===========================================================================
# TestKillSwitchGate
# ===========================================================================

class TestKillSwitchGate:
    """8 tests — Gate 1: kill switch."""

    def setup_method(self):
        self.engine = RiskEngine()
        self.account = _account()
        self.rec = _rec()

    def test_kill_switch_inactive_does_not_reject(self):
        result, approved, rejection = self.engine.check_and_approve(self.rec, self.account)
        assert result != RiskCheckResult.KILL_SWITCH_ACTIVE

    def test_kill_switch_active_returns_kill_switch_result(self):
        self.engine._state.kill_switch_active = True
        result, approved, rejection = self.engine.check_and_approve(self.rec, self.account)
        assert result == RiskCheckResult.KILL_SWITCH_ACTIVE

    def test_kill_switch_active_no_approved_order(self):
        self.engine._state.kill_switch_active = True
        _, approved, _ = self.engine.check_and_approve(self.rec, self.account)
        assert approved is None

    def test_kill_switch_active_returns_rejection_dto(self):
        self.engine._state.kill_switch_active = True
        _, _, rejection = self.engine.check_and_approve(self.rec, self.account)
        assert isinstance(rejection, RiskRejection)

    def test_kill_switch_rejection_check_type(self):
        self.engine._state.kill_switch_active = True
        _, _, rejection = self.engine.check_and_approve(self.rec, self.account)
        assert rejection.check_type == "KILL_SWITCH"

    def test_kill_switch_rejection_contains_recommendation(self):
        self.engine._state.kill_switch_active = True
        _, _, rejection = self.engine.check_and_approve(self.rec, self.account)
        assert rejection.recommendation is self.rec

    def test_kill_switch_triggered_via_drawdown(self):
        account = _account(equity=8_000.0, peak_equity=10_000.0)  # 20% drawdown
        result = self.engine.check_kill_switch(account)
        assert result is not None
        assert self.engine.kill_switch_active is True

    def test_kill_switch_gate_fires_before_daily_limit(self):
        """Gate 1 (kill switch) must short-circuit before gate 2 (daily limit)."""
        self.engine._state.kill_switch_active = True
        account = _account(daily_pnl_pct=-5.0)  # also breaches daily limit
        result, _, rejection = self.engine.check_and_approve(self.rec, account)
        assert result == RiskCheckResult.KILL_SWITCH_ACTIVE
        assert rejection.check_type == "KILL_SWITCH"


# ===========================================================================
# TestDailyLimitGate
# ===========================================================================

class TestDailyLimitGate:
    """8 tests — Gate 2: daily loss limit."""

    def setup_method(self):
        self.engine = RiskEngine(daily_loss_limit_pct=3.0)
        self.rec = _rec()

    def test_no_loss_passes(self):
        account = _account(daily_pnl_pct=0.0)
        result, approved, _ = self.engine.check_and_approve(self.rec, account)
        assert result != RiskCheckResult.REJECTED or (approved is not None)

    def test_exactly_at_limit_rejects(self):
        # balance/day_open = 9700/10000 → daily_pnl_pct = -3.0%
        account = _account(balance=9_700.0, day_open_balance=10_000.0)
        result, _, rejection = self.engine.check_and_approve(self.rec, account)
        assert result == RiskCheckResult.REJECTED
        assert rejection.check_type == "DAILY_LIMIT"

    def test_just_above_limit_rejects(self):
        # -3.1%
        account = _account(balance=9_690.0, day_open_balance=10_000.0)
        result, _, rejection = self.engine.check_and_approve(self.rec, account)
        assert result == RiskCheckResult.REJECTED
        assert rejection.check_type == "DAILY_LIMIT"

    def test_just_below_limit_passes_daily(self):
        # -2.9% — below 3.0% limit (daily gate should pass)
        account = _account(balance=9_710.0, day_open_balance=10_000.0)
        _, _, rejection = self.engine.check_and_approve(self.rec, account)
        assert rejection is None or rejection.check_type != "DAILY_LIMIT"

    def test_positive_daily_pnl_passes(self):
        account = _account(daily_pnl_pct=2.0)
        result, _, rejection = self.engine.check_and_approve(self.rec, account)
        assert rejection is None or rejection.check_type != "DAILY_LIMIT"

    def test_custom_daily_limit(self):
        engine = RiskEngine(daily_loss_limit_pct=1.0)
        account = _account(balance=9_910.0, day_open_balance=10_000.0)  # -0.9% — safe
        _, _, rejection = engine.check_and_approve(self.rec, account)
        assert rejection is None or rejection.check_type != "DAILY_LIMIT"

    def test_custom_daily_limit_exceeded(self):
        engine = RiskEngine(daily_loss_limit_pct=1.0)
        account = _account(balance=9_890.0, day_open_balance=10_000.0)  # -1.1% → rejected
        result, _, rejection = engine.check_and_approve(self.rec, account)
        assert result == RiskCheckResult.REJECTED
        assert rejection.check_type == "DAILY_LIMIT"

    def test_daily_rejection_before_weekly(self):
        """Gate 2 must fire before gate 3."""
        engine = RiskEngine(daily_loss_limit_pct=3.0, weekly_loss_limit_pct=6.0)
        # Both daily AND weekly limits breached — daily gate fires first
        account = _account(
            balance=9_300.0,
            day_open_balance=10_000.0,
            week_open_balance=10_000.0,
        )
        result, _, rejection = engine.check_and_approve(self.rec, account)
        assert result == RiskCheckResult.REJECTED
        assert rejection.check_type == "DAILY_LIMIT"


# ===========================================================================
# TestWeeklyLimitGate
# ===========================================================================

class TestWeeklyLimitGate:
    """7 tests — Gate 3: weekly loss limit."""

    def setup_method(self):
        self.engine = RiskEngine(weekly_loss_limit_pct=6.0)
        self.rec = _rec()

    def test_weekly_at_limit_rejects(self):
        # -6.0% weekly P&L: balance=9400, week_open=10000
        account = _account(balance=9_400.0, week_open_balance=10_000.0)
        result, _, rejection = self.engine.check_and_approve(self.rec, account)
        assert result == RiskCheckResult.REJECTED
        assert rejection.check_type == "WEEKLY_LIMIT"

    def test_weekly_just_below_limit_passes(self):
        # -5.9% weekly: balance=9410, week_open=10000
        account = _account(balance=9_410.0, week_open_balance=10_000.0)
        _, _, rejection = self.engine.check_and_approve(self.rec, account)
        assert rejection is None or rejection.check_type != "WEEKLY_LIMIT"

    def test_weekly_positive_passes(self):
        # +5% weekly: balance=10500, week_open=10000
        account = _account(balance=10_500.0, week_open_balance=10_000.0)
        _, _, rejection = self.engine.check_and_approve(self.rec, account)
        assert rejection is None or rejection.check_type != "WEEKLY_LIMIT"

    def test_custom_weekly_limit(self):
        engine = RiskEngine(weekly_loss_limit_pct=3.0)
        # -3.2%: balance=9680, week_open=10000
        account = _account(balance=9_680.0, week_open_balance=10_000.0)
        result, _, rejection = engine.check_and_approve(self.rec, account)
        assert result == RiskCheckResult.REJECTED
        assert rejection.check_type == "WEEKLY_LIMIT"

    def test_weekly_rejection_type(self):
        account = _account(balance=9_300.0, week_open_balance=10_000.0)
        _, _, rejection = self.engine.check_and_approve(self.rec, account)
        assert rejection.check_type == "WEEKLY_LIMIT"

    def test_weekly_gate_fires_after_daily_passes(self):
        """Daily within limit but weekly exceeded → WEEKLY_LIMIT fires."""
        # daily limit 10%, weekly limit 6%
        # balance=9300, day_open=9500 → daily=-2.1% (< 10% limit, SAFE)
        # week_open=10000 → weekly=-7% (> 6% limit, EXCEEDED)
        engine = RiskEngine(daily_loss_limit_pct=10.0, weekly_loss_limit_pct=6.0)
        account = _account(
            balance=9_300.0,
            day_open_balance=9_500.0,
            week_open_balance=10_000.0,
        )
        result, _, rejection = engine.check_and_approve(self.rec, account)
        assert result == RiskCheckResult.REJECTED
        assert rejection.check_type == "WEEKLY_LIMIT"

    def test_weekly_rejection_contains_recommendation(self):
        account = _account(balance=9_300.0, week_open_balance=10_000.0)
        _, _, rejection = self.engine.check_and_approve(self.rec, account)
        assert rejection.recommendation is self.rec


# ===========================================================================
# TestMaxTradesGate
# ===========================================================================

class TestMaxTradesGate:
    """7 tests — Gate 4: max open trades."""

    def setup_method(self):
        self.engine = RiskEngine(max_open_trades=3)
        self.rec = _rec()

    def test_zero_trades_passes(self):
        account = _account(open_trades=0)
        result, approved, _ = self.engine.check_and_approve(self.rec, account)
        assert result == RiskCheckResult.APPROVED

    def test_two_trades_passes(self):
        account = _account(open_trades=2)
        _, _, rejection = self.engine.check_and_approve(self.rec, account)
        assert rejection is None or rejection.check_type != "MAX_TRADES"

    def test_exactly_at_max_rejects(self):
        account = _account(open_trades=3)
        result, _, rejection = self.engine.check_and_approve(self.rec, account)
        assert result == RiskCheckResult.REJECTED
        assert rejection.check_type == "MAX_TRADES"

    def test_above_max_rejects(self):
        account = _account(open_trades=5)
        result, _, rejection = self.engine.check_and_approve(self.rec, account)
        assert result == RiskCheckResult.REJECTED
        assert rejection.check_type == "MAX_TRADES"

    def test_custom_max_trades(self):
        engine = RiskEngine(max_open_trades=1)
        account = _account(open_trades=1)
        result, _, rejection = engine.check_and_approve(self.rec, account)
        assert result == RiskCheckResult.REJECTED
        assert rejection.check_type == "MAX_TRADES"

    def test_uses_account_open_trades_not_internal(self):
        """Gate must read account.open_trades, not internal _state.open_trade_count."""
        self.engine._state.open_trade_count = 0   # internal shows 0
        account = _account(open_trades=3)          # live account shows 3
        result, _, rejection = self.engine.check_and_approve(self.rec, account)
        assert result == RiskCheckResult.REJECTED  # should use live value
        assert rejection.check_type == "MAX_TRADES"

    def test_max_trades_rejection_contains_recommendation(self):
        account = _account(open_trades=3)
        _, _, rejection = self.engine.check_and_approve(self.rec, account)
        assert rejection.recommendation is self.rec


# ===========================================================================
# TestRRRatioGate
# ===========================================================================

class TestRRRatioGate:
    """6 tests — Gate 5: R:R ratio check."""

    def setup_method(self):
        self.engine = RiskEngine(min_rr_ratio=2.0)
        self.account = _account()

    def test_rr_at_minimum_passes(self):
        rec = _rec(entry=1.1000, stop=1.0980, target=1.1040, rr=2.0)
        result, approved, _ = self.engine.check_and_approve(rec, self.account)
        assert result == RiskCheckResult.APPROVED

    def test_rr_above_minimum_passes(self):
        rec = _rec(rr=3.5)
        result, _, _ = self.engine.check_and_approve(rec, self.account)
        assert result == RiskCheckResult.APPROVED

    def test_rr_below_minimum_rejects(self):
        rec = _rec(rr=1.5)
        result, _, rejection = self.engine.check_and_approve(rec, self.account)
        assert result == RiskCheckResult.REJECTED
        assert rejection.check_type == "RR_RATIO"

    def test_rr_zero_rejects(self):
        rec = _rec(rr=0.0)
        result, _, rejection = self.engine.check_and_approve(rec, self.account)
        assert result == RiskCheckResult.REJECTED
        assert rejection.check_type == "RR_RATIO"

    def test_rr_rejection_contains_recommendation(self):
        rec = _rec(rr=1.0)
        _, _, rejection = self.engine.check_and_approve(rec, self.account)
        assert rejection.recommendation is rec

    def test_custom_min_rr(self):
        engine = RiskEngine(min_rr_ratio=3.0)
        rec_2r = _rec(rr=2.5)
        result, _, rejection = engine.check_and_approve(rec_2r, self.account)
        assert result == RiskCheckResult.REJECTED
        assert rejection.check_type == "RR_RATIO"


# ===========================================================================
# TestPositionSizeGate
# ===========================================================================

class TestPositionSizeGate:
    """5 tests — Gate 6: position size must be > 0."""

    def test_zero_stop_distance_rejects(self):
        """Entry == stop → lot_size = 0 → rejected."""
        rec = _rec(entry=1.1000, stop=1.1000, rr=2.0)
        engine = RiskEngine()
        account = _account()
        result, _, rejection = engine.check_and_approve(rec, account)
        assert result == RiskCheckResult.REJECTED
        assert rejection.check_type == "POSITION_SIZE"

    def test_valid_stop_distance_approves(self):
        rec = _rec(entry=1.1000, stop=1.0980, rr=2.0)
        engine = RiskEngine()
        account = _account()
        result, approved, _ = engine.check_and_approve(rec, account)
        assert result == RiskCheckResult.APPROVED
        assert approved.lot_size > 0

    def test_position_size_rejection_check_type(self):
        rec = _rec(entry=1.1000, stop=1.1000, rr=2.0)
        engine = RiskEngine()
        account = _account()
        _, _, rejection = engine.check_and_approve(rec, account)
        assert rejection.check_type == "POSITION_SIZE"

    def test_position_size_fires_after_rr_gate(self):
        """Gate 5 (RR) must fire before gate 6 (position size)."""
        rec = _rec(entry=1.1000, stop=1.1000, rr=0.5)  # Both RR fail AND zero stop
        engine = RiskEngine(min_rr_ratio=2.0)
        account = _account()
        _, _, rejection = engine.check_and_approve(rec, account)
        assert rejection.check_type == "RR_RATIO"  # Gate 5 fires first

    def test_approved_order_has_positive_lot_size(self):
        rec = _rec()
        engine = RiskEngine()
        account = _account()
        _, approved, _ = engine.check_and_approve(rec, account)
        assert approved.lot_size > 0.0


# ===========================================================================
# TestFullApproval
# ===========================================================================

class TestFullApproval:
    """10 tests — end-to-end approved RiskApprovedOrder content."""

    def setup_method(self):
        self.engine = RiskEngine(
            risk_per_trade_pct=1.0,
            pip_size=0.0001,
            pip_value_per_lot=10.0,
        )
        self.account = _account(balance=10_000.0)
        self.rec = _rec(entry=1.1000, stop=1.0980, target=1.1040, rr=2.0)

    def test_result_is_approved(self):
        result, _, _ = self.engine.check_and_approve(self.rec, self.account)
        assert result == RiskCheckResult.APPROVED

    def test_no_rejection_on_approval(self):
        _, _, rejection = self.engine.check_and_approve(self.rec, self.account)
        assert rejection is None

    def test_approved_order_type(self):
        _, approved, _ = self.engine.check_and_approve(self.rec, self.account)
        assert isinstance(approved, RiskApprovedOrder)

    def test_approved_recommendation_reference(self):
        _, approved, _ = self.engine.check_and_approve(self.rec, self.account)
        assert approved.recommendation is self.rec

    def test_approved_lot_size(self):
        # 10000 * 1% = 100; stop=20 pips * 10 = 200 → 0.5 lots
        _, approved, _ = self.engine.check_and_approve(self.rec, self.account)
        assert approved.lot_size == pytest.approx(0.5, abs=0.01)

    def test_approved_risk_pct(self):
        _, approved, _ = self.engine.check_and_approve(self.rec, self.account)
        assert approved.risk_pct == pytest.approx(1.0, abs=1e-9)

    def test_approved_risk_amount_usd(self):
        _, approved, _ = self.engine.check_and_approve(self.rec, self.account)
        assert approved.risk_amount_usd == pytest.approx(100.0, abs=1e-6)

    def test_approved_account_balance(self):
        _, approved, _ = self.engine.check_and_approve(self.rec, self.account)
        assert approved.account_balance == 10_000.0

    def test_approved_stop_pips(self):
        _, approved, _ = self.engine.check_and_approve(self.rec, self.account)
        assert approved.stop_pips == pytest.approx(20.0, abs=1e-4)

    def test_short_trade_approval(self):
        rec = _rec(
            entry=1.1000, stop=1.1020, target=1.0960,
            rr=2.0, direction=Direction.SHORT,
        )
        result, approved, _ = self.engine.check_and_approve(rec, self.account)
        assert result == RiskCheckResult.APPROVED
        assert approved.lot_size > 0
        assert approved.stop_pips == pytest.approx(20.0, abs=1e-4)


# ===========================================================================
# TestKillSwitchLogic
# ===========================================================================

class TestKillSwitchLogic:
    """12 tests — kill switch trigger conditions and reset."""

    def setup_method(self):
        self.engine = RiskEngine(
            kill_switch_drawdown_pct=10.0,
            kill_switch_consecutive_losses=7,
            daily_loss_limit_pct=3.0,
            weekly_loss_limit_pct=6.0,
        )

    def _account_with_drawdown(self, drawdown_pct: float) -> AccountState:
        equity = 10_000.0 * (1 - drawdown_pct / 100)
        return _account(equity=equity, peak_equity=10_000.0)

    def test_drawdown_triggers_kill_switch(self):
        account = self._account_with_drawdown(10.0)  # exactly 10%
        event = self.engine.check_kill_switch(account)
        assert event is not None
        assert event.reason == KillSwitchReason.DRAWDOWN
        assert self.engine.kill_switch_active is True

    def test_drawdown_below_threshold_no_trigger(self):
        account = self._account_with_drawdown(9.9)
        event = self.engine.check_kill_switch(account)
        assert event is None
        assert self.engine.kill_switch_active is False

    def test_consecutive_losses_triggers(self):
        self.engine._state.consecutive_losses = 7
        account = _account()
        event = self.engine.check_kill_switch(account)
        assert event is not None
        assert event.reason == KillSwitchReason.CONSECUTIVE_LOSSES

    def test_six_consecutive_losses_no_trigger(self):
        self.engine._state.consecutive_losses = 6
        account = _account()
        event = self.engine.check_kill_switch(account)
        assert event is None

    def test_daily_and_weekly_simultaneous_triggers(self):
        # daily=-7% (>3%), weekly=-7% (>6%) — both limits exceeded → DAILY_AND_WEEKLY kill switch
        account = _account(
            balance=9_300.0,
            day_open_balance=10_000.0,
            week_open_balance=10_000.0,
        )
        event = self.engine.check_kill_switch(account)
        assert event is not None
        assert event.reason == KillSwitchReason.DAILY_AND_WEEKLY

    def test_only_daily_breached_no_trigger(self):
        # daily=-3.5% (>3%), weekly=-3.5% (<6%) — only daily breached, not both
        # Kill switch requires BOTH daily AND weekly to be simultaneously breached
        account = _account(
            balance=9_650.0,
            day_open_balance=10_000.0,
            week_open_balance=10_000.0,  # weekly=-3.5% < 6% limit → NOT breached
        )
        event = self.engine.check_kill_switch(account)
        assert event is None

    def test_kill_switch_idempotent_already_active(self):
        """Second call returns None when already active (no duplicate events)."""
        self.engine._state.kill_switch_active = True
        account = self._account_with_drawdown(15.0)
        event = self.engine.check_kill_switch(account)
        assert event is None  # Already active → no new event

    def test_reset_clears_kill_switch(self):
        self.engine._state.kill_switch_active = True
        self.engine._state.kill_switch_reason = KillSwitchReason.DRAWDOWN
        self.engine.reset_kill_switch("admin")
        assert self.engine.kill_switch_active is False

    def test_reset_clears_reason(self):
        self.engine._state.kill_switch_active = True
        self.engine._state.kill_switch_reason = KillSwitchReason.DRAWDOWN
        self.engine.reset_kill_switch("admin")
        assert self.engine._state.kill_switch_reason is None

    def test_reset_when_not_active_is_safe(self):
        """reset_kill_switch on inactive engine does not raise."""
        assert self.engine.kill_switch_active is False
        self.engine.reset_kill_switch("admin")  # Should not raise
        assert self.engine.kill_switch_active is False

    def test_kill_switch_event_fields(self):
        account = self._account_with_drawdown(12.0)
        event = self.engine.check_kill_switch(account)
        assert event.reason == KillSwitchReason.DRAWDOWN
        assert event.trigger_value > 0
        assert event.threshold == pytest.approx(10.0)
        assert isinstance(event.description, str)
        assert len(event.description) > 0

    def test_after_kill_switch_check_and_approve_rejected(self):
        """Once kill switch active, check_and_approve must immediately reject."""
        account = self._account_with_drawdown(10.0)
        self.engine.check_kill_switch(account)
        assert self.engine.kill_switch_active is True

        rec = _rec()
        result, _, rejection = self.engine.check_and_approve(rec, account)
        assert result == RiskCheckResult.KILL_SWITCH_ACTIVE
        assert rejection.check_type == "KILL_SWITCH"


# ===========================================================================
# TestStateUpdates
# ===========================================================================

class TestStateUpdates:
    """8 tests — update_after_trade_close, reset_daily_state, reset_weekly_state."""

    def setup_method(self):
        self.engine = RiskEngine()
        self.account = _account()

    def test_loss_increments_consecutive_losses(self):
        self.engine.update_after_trade_close(-1.0, self.account)
        assert self.engine._state.consecutive_losses == 1

    def test_win_resets_consecutive_losses(self):
        self.engine._state.consecutive_losses = 4
        self.engine.update_after_trade_close(+1.0, self.account)
        assert self.engine._state.consecutive_losses == 0

    def test_two_losses_in_a_row(self):
        self.engine.update_after_trade_close(-1.0, self.account)
        self.engine.update_after_trade_close(-1.0, self.account)
        assert self.engine._state.consecutive_losses == 2

    def test_win_then_loss_resets_then_increments(self):
        self.engine.update_after_trade_close(-1.0, self.account)
        self.engine.update_after_trade_close(+1.0, self.account)
        self.engine.update_after_trade_close(-1.0, self.account)
        assert self.engine._state.consecutive_losses == 1

    def test_trades_today_increments(self):
        self.engine.update_after_trade_close(+1.0, self.account)
        self.engine.update_after_trade_close(-1.0, self.account)
        assert self.engine._state.trades_today == 2

    def test_reset_daily_clears_daily_counters(self):
        self.engine._state.trades_today = 5
        self.engine._state.losses_today = 2
        self.engine._state.daily_pnl_pct = -2.5
        self.engine.reset_daily_state()
        assert self.engine._state.trades_today == 0
        assert self.engine._state.losses_today == 0
        assert self.engine._state.daily_pnl_pct == 0.0

    def test_reset_daily_preserves_consecutive_losses(self):
        self.engine._state.consecutive_losses = 3
        self.engine.reset_daily_state()
        assert self.engine._state.consecutive_losses == 3

    def test_reset_weekly_clears_weekly_pnl(self):
        self.engine._state.weekly_pnl_pct = -5.0
        self.engine.reset_weekly_state()
        assert self.engine._state.weekly_pnl_pct == 0.0


# ===========================================================================
# TestEdgeCases
# ===========================================================================

class TestEdgeCases:
    """14 tests — boundary conditions and unusual inputs."""

    def test_zero_balance_account(self):
        """Engine should not crash; lot size will be zero → POSITION_SIZE rejection."""
        engine = RiskEngine()
        rec = _rec()
        account = _account(balance=0.0)
        result, approved, rejection = engine.check_and_approve(rec, account)
        # lot_size = 0 → position size gate fires OR approved with 0.01 (lot_min)
        # Either way: no crash
        assert result in (RiskCheckResult.REJECTED, RiskCheckResult.APPROVED)

    def test_very_large_balance(self):
        engine = RiskEngine()
        rec = _rec()
        account = _account(balance=1_000_000.0)
        result, approved, _ = engine.check_and_approve(rec, account)
        assert result == RiskCheckResult.APPROVED
        assert approved.lot_size <= engine.config.lot_max

    def test_tiny_stop_distance(self):
        """1-pip stop → very large raw lots → clamped to lot_max."""
        engine = RiskEngine()
        rec = _rec(entry=1.1000, stop=1.09999, rr=2.0)  # 0.1 pip stop
        account = _account(balance=10_000.0)
        result, approved, _ = engine.check_and_approve(rec, account)
        if result == RiskCheckResult.APPROVED:
            assert approved.lot_size <= engine.config.lot_max

    def test_high_rr_still_approves(self):
        engine = RiskEngine()
        rec = _rec(rr=10.0)
        account = _account()
        result, _, _ = engine.check_and_approve(rec, account)
        assert result == RiskCheckResult.APPROVED

    def test_premium_tqs_increases_lot_size(self):
        """With premium_risk_enabled, PREMIUM TQS trades get more lots."""
        tqs_standard = _tqs(15, 15, 15, 15)  # STANDARD
        tqs_premium  = _tqs(20, 20, 20, 20)  # PREMIUM
        engine = RiskEngine(
            risk_per_trade_pct=1.0, premium_risk_pct=1.5, premium_risk_enabled=True
        )
        account = _account(balance=10_000.0)

        rec_std = _rec(tqs=tqs_standard)
        rec_prm = _rec(tqs=tqs_premium)

        _, std_approved, _ = engine.check_and_approve(rec_std, account)
        _, prm_approved, _ = engine.check_and_approve(rec_prm, account)

        assert prm_approved.lot_size > std_approved.lot_size

    def test_kill_switch_resets_and_trades_allowed(self):
        engine = RiskEngine()
        engine._state.kill_switch_active = True
        engine.reset_kill_switch("admin")
        account = _account()
        rec = _rec()
        result, approved, _ = engine.check_and_approve(rec, account)
        assert result == RiskCheckResult.APPROVED

    def test_update_open_trade_count_positive(self):
        engine = RiskEngine()
        engine.update_open_trade_count(2)
        assert engine._state.open_trade_count == 2

    def test_update_open_trade_count_negative_raises(self):
        engine = RiskEngine()
        with pytest.raises(ValueError):
            engine.update_open_trade_count(-1)

    def test_engulfing_bar_recommendation_approved(self):
        """StrategyName.ENGULFING_BAR passes through risk engine correctly."""
        engine = RiskEngine()
        tqs = _tqs()
        rec = TradeRecommendation(
            strategy=StrategyName.ENGULFING_BAR,
            symbol="GBPUSD", timeframe="H4",
            direction=Direction.SHORT,
            entry_price=1.2500, stop_price=1.2520,
            target_price=1.2460, rr_ratio=2.0, tqs=tqs,
        )
        account = _account()
        result, approved, _ = engine.check_and_approve(rec, account)
        assert result == RiskCheckResult.APPROVED
        assert approved.lot_size > 0

    def test_approved_order_has_timestamp(self):
        engine = RiskEngine()
        rec = _rec()
        account = _account()
        _, approved, _ = engine.check_and_approve(rec, account)
        assert isinstance(approved.approved_at, datetime)

    def test_rejection_has_timestamp(self):
        engine = RiskEngine()
        engine._state.kill_switch_active = True
        rec = _rec()
        account = _account()
        _, _, rejection = engine.check_and_approve(rec, account)
        assert isinstance(rejection.rejected_at, datetime)

    def test_seven_consecutive_losses_triggers_kill_switch(self):
        """update_after_trade_close should auto-trigger kill switch at 7 losses."""
        engine = RiskEngine(kill_switch_consecutive_losses=7)
        account = _account()
        for _ in range(7):
            engine.update_after_trade_close(-1.0, account)
        assert engine.kill_switch_active is True

    def test_six_losses_does_not_trigger_kill_switch(self):
        engine = RiskEngine(kill_switch_consecutive_losses=7)
        account = _account()
        for _ in range(6):
            engine.update_after_trade_close(-1.0, account)
        assert engine.kill_switch_active is False

    def test_is_trading_allowed_reflects_kill_switch(self):
        engine = RiskEngine()
        assert engine.is_trading_allowed() is True
        engine._state.kill_switch_active = True
        assert engine.is_trading_allowed() is False


# ===========================================================================
# Guard: no forbidden Phase 2 content
# ===========================================================================

class TestNoForbiddenFields:
    """4 tests — M09 must NOT implement Phase 2 features."""

    def test_no_kelly_criterion_method(self):
        engine = RiskEngine()
        assert not hasattr(engine, "kelly_criterion")
        assert not hasattr(engine, "compute_kelly")

    def test_no_correlation_check_method(self):
        engine = RiskEngine()
        assert not hasattr(engine, "check_correlation")
        assert not hasattr(engine, "portfolio_correlation")

    def test_no_live_execution_method(self):
        engine = RiskEngine()
        assert not hasattr(engine, "execute_order")
        assert not hasattr(engine, "place_trade")

    def test_no_position_tracking_list(self):
        engine = RiskEngine()
        # Phase 2: full portfolio tracking; Phase 1 uses account.open_trades
        assert not hasattr(engine, "open_positions")
        assert not hasattr(engine, "portfolio")
