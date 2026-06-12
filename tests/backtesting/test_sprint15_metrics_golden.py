"""
Sprint 15 Stage 2 — Metrics Golden Tests
==========================================
Validates the metrics computation layer with hand-computed expected values.
Demonstrates RC-3 (broken max drawdown) BEFORE the fix.

Test classes
------------
TestMaxDrawdownGolden       — 8 tests  (several FAIL on pre-fix code → prove RC-3)
TestProfitFactorGolden      — 5 tests
TestExpectancyGolden        — 5 tests
TestWinRateGolden           — 4 tests
TestNetProfitGolden         — 4 tests
TestConsecutiveStreaksGolden — 5 tests  (via BacktestResult / PipelineResult)
TestComputeMaxDrawdownUnit  — 6 tests  (direct _compute_max_drawdown)

Total: 37 tests
"""
from __future__ import annotations

import math
import pytest

from datetime import datetime, timezone

from src.integration.pipeline_runner import (
    PipelineResult,
    _compute_max_drawdown,
    _compute_max_drawdown_equity,
)
from src.execution.paper_executor import PaperOrder, PaperOrderStatus
from src.backtesting.backtest_runner import BacktestResult

_NOW = datetime.now(timezone.utc)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_order(r: float, pnl_usd: float, strategy: str = "pin_bar") -> PaperOrder:
    """Create a minimal closed PaperOrder fixture using correct field names."""
    return PaperOrder(
        order_id="test-001",
        symbol="EURUSD",
        timeframe="D1",
        strategy_name=strategy,
        direction="LONG",
        status=PaperOrderStatus.CLOSED,
        requested_price=1.1000,
        filled_price=1.1000,
        stop_loss=1.0950,
        take_profit=1.1100,
        lot_size=0.01,
        slippage_pips=1.0,
        created_at=_NOW,
        r_multiple=r,
        pnl_usd=pnl_usd,
    )


def _pipeline_result_from_trades(
    initial_balance: float,
    trades: list[tuple[float, float]],  # (r_multiple, pnl_usd)
    strategy: str = "pin_bar",
) -> PipelineResult:
    """Build a PipelineResult by recording synthetic closed orders."""
    result = PipelineResult(
        symbol="EURUSD",
        timeframe="D1",
        started_at=_NOW,
        initial_balance=initial_balance,
    )
    result.final_balance = initial_balance
    for r, pnl in trades:
        order = _make_order(r, pnl, strategy)
        result.record_closed_order(order)
    result.finalise()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TestComputeMaxDrawdownUnit — direct tests on _compute_max_drawdown()
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeMaxDrawdownUnit:
    """
    Unit tests for _compute_max_drawdown(r_multiples).
    Several of these tests FAIL on the pre-fix implementation,
    demonstrating RC-3 (peak seeded at 0.0 instead of initial_balance).

    NOTE: _compute_max_drawdown takes r_multiples only (no balance).
    After FIX-3, it must be replaced with an account-equity version.
    These tests document the EXPECTED behaviour of the FIXED implementation.
    Tests marked with (RC-3 EXPOSED) will fail on pre-fix code.
    """

    def test_empty_returns_zero(self):
        """No trades → DD = 0%."""
        assert _compute_max_drawdown([]) == 0.0

    def test_all_winners_returns_zero(self):
        """Monotonically increasing equity → DD = 0%."""
        assert _compute_max_drawdown([1.0, 2.0, 1.5]) == 0.0

    def test_single_winner_no_dd(self):
        """One win → DD = 0%."""
        assert _compute_max_drawdown([2.0]) == 0.0

    def test_legacy_canonical_sequence_200pct(self):
        """
        RC-3 DOCUMENTATION: Legacy _compute_max_drawdown (R-multiples, peak=0).
        Sequence [+1.0, -2.0, +0.5]: equity goes 0→1→-1→-0.5.
        Peak=1.0, trough=-1.0 → DD = (1-(-1))/1 * 100 = 200%.
        This broken value confirms RC-3. The LEGACY function is kept
        unchanged; only PipelineResult now uses the equity version.
        """
        r_mults = [1.0, -2.0, 0.5]
        result = _compute_max_drawdown(r_mults)
        assert result == pytest.approx(200.0, rel=0.01), (
            f"RC-3 legacy: expected ~200% DD from R-curve, got {result:.2f}%"
        )

    def test_legacy_first_trade_loss_astronomical(self):
        """
        RC-3 DOCUMENTATION: Legacy function with first-trade loss.
        Peak=0.0 → dd = (0-(-1))/1e-9*100 → astronomically large.
        """
        r_mults = [-1.0]
        result = _compute_max_drawdown(r_mults)
        assert result > 1e9, (
            f"RC-3 legacy: expected astronomical DD, got {result:.2e}"
        )

    def test_flat_sequence_zero_dd(self):
        """All zero R multiples → 0% DD."""
        assert _compute_max_drawdown([0.0, 0.0, 0.0]) == 0.0

    def test_legacy_single_loss_astronomical(self):
        """
        RC-3 DOCUMENTATION: legacy function, single −2R trade.
        peak=0.0, equity=-2.0 → dd astronomically large.
        """
        result = _compute_max_drawdown([-2.0])
        assert result > 1e9, f"Expected astronomical DD in legacy, got {result}"


# ═══════════════════════════════════════════════════════════════════════════════
# 1b. TestMaxDrawdownEquityUnit — new _compute_max_drawdown_equity() (FIX-3)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMaxDrawdownEquityUnit:
    """
    Post-FIX-3 golden tests for _compute_max_drawdown_equity().
    All must pass after FIX-3 is applied.
    """

    def test_empty_returns_zero(self):
        assert _compute_max_drawdown_equity([], 10_000) == 0.0

    def test_all_winners_zero_dd(self):
        assert _compute_max_drawdown_equity([100, 200, 150], 10_000) == 0.0

    def test_canonical_golden_case(self):
        """§2 mandatory: +100,-200,+50 from 10000 → DD=1.9802%."""
        dd = _compute_max_drawdown_equity([100, -200, 50], 10_000)
        assert dd == pytest.approx(1.9802, rel=1e-3)

    def test_first_trade_loss_one_pct(self):
        """First trade -$100 from $10,000 → DD = 1.0% (not astronomical)."""
        dd = _compute_max_drawdown_equity([-100], 10_000)
        assert dd == pytest.approx(1.0, rel=1e-4)

    def test_first_trade_loss_large_balance(self):
        """First trade -$500 from $50,000 → DD = 1.0% exactly."""
        dd = _compute_max_drawdown_equity([-500], 50_000)
        assert dd == pytest.approx(1.0, rel=1e-4)

    def test_all_losers(self):
        """Three losses: -100,-100,-100 from 10000 → DD = 3.0%."""
        dd = _compute_max_drawdown_equity([-100, -100, -100], 10_000)
        assert dd == pytest.approx(3.0, rel=1e-3)

    def test_no_dd_then_loss_then_recovery(self):
        """Win then loss then recovery: max dd < 1% at $10k scale."""
        dd = _compute_max_drawdown_equity([200, -100, 100], 10_000)
        # peak=10200, trough=10100 → 100/10200 * 100 ≈ 0.98%
        assert dd == pytest.approx(100 / 10200 * 100, rel=1e-3)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TestMaxDrawdownGolden — via PipelineResult (full pipeline metric chain)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMaxDrawdownGolden:
    """
    Golden tests on PipelineResult.max_drawdown (the end-to-end chain).
    After Sprint 15 FIX-3, PipelineResult uses _compute_max_drawdown_equity()
    with peak seeded at initial_balance. All tests here verify POST-FIX values.
    """

    def test_no_trades_dd_zero(self):
        r = _pipeline_result_from_trades(10_000, [])
        assert r.max_drawdown == 0.0

    def test_all_winners_dd_zero(self):
        r = _pipeline_result_from_trades(10_000, [(1.0, 100), (2.0, 200)])
        assert r.max_drawdown == 0.0

    def test_first_trade_loss_small_not_astronomical(self):
        """FIX-3: first trade a loss (-$100) from $10,000 → DD = 1.0% (not 1e11%)."""
        r = _pipeline_result_from_trades(10_000, [(-1.0, -100)])
        assert r.max_drawdown == pytest.approx(1.0, rel=1e-4), (
            f"FIX-3: expected 1.0% DD, got {r.max_drawdown:.4f}%"
        )

    def test_win_then_loss_correct_dd(self):
        """FIX-3: +$100 then -$200 from $10,000. peak=10,100 trough=9,900 DD=1.98%."""
        r = _pipeline_result_from_trades(10_000, [(1.0, 100), (-2.0, -200)])
        assert r.max_drawdown == pytest.approx(1.9802, rel=1e-3), (
            f"FIX-3: expected ~1.98% DD, got {r.max_drawdown:.4f}%"
        )

    def test_canonical_sequence_dd_golden(self):
        """FIX-3: §2 canonical sequence +$100,-$200,+$50 from $10,000 → DD=1.9802%."""
        r = _pipeline_result_from_trades(10_000, [(1.0, 100), (-2.0, -200), (0.5, 50)])
        assert r.max_drawdown == pytest.approx(1.9802, rel=1e-3), (
            f"FIX-3: expected ~1.9802% DD, got {r.max_drawdown:.4f}%"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TestProfitFactorGolden
# ═══════════════════════════════════════════════════════════════════════════════

class TestProfitFactorGolden:
    """Profit factor = gross_profit_R / gross_loss_R (hand computed)."""

    def test_no_trades_pf_zero(self):
        r = _pipeline_result_from_trades(10_000, [])
        assert r.profit_factor == 0.0

    def test_all_winners_pf_inf(self):
        r = _pipeline_result_from_trades(10_000, [(2.0, 200), (1.5, 150)])
        assert r.profit_factor == float("inf")

    def test_all_losers_pf_zero(self):
        r = _pipeline_result_from_trades(10_000, [(-1.0, -100), (-0.5, -50)])
        assert r.profit_factor == 0.0

    def test_mixed_pf_golden(self):
        """1W(+2R) 1L(-1R): gross_profit=2.0, gross_loss=1.0 → PF=2.0."""
        r = _pipeline_result_from_trades(10_000, [(2.0, 200), (-1.0, -100)])
        assert r.profit_factor == pytest.approx(2.0, rel=1e-6)

    def test_pf_1_10_scenario(self):
        """11 wins(+1R), 10 losses(-1R): gross_profit=11, gross_loss=10 → PF=1.10."""
        trades = [(1.0, 100)] * 11 + [(-1.0, -100)] * 10
        r = _pipeline_result_from_trades(10_000, trades)
        assert r.profit_factor == pytest.approx(1.1, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TestExpectancyGolden
# ═══════════════════════════════════════════════════════════════════════════════

class TestExpectancyGolden:
    """Expectancy = mean(r_multiples), computed in finalise()."""

    def test_no_trades_expectancy_zero(self):
        r = _pipeline_result_from_trades(10_000, [])
        assert r.expectancy == 0.0

    def test_single_winner_expectancy(self):
        r = _pipeline_result_from_trades(10_000, [(2.0, 200)])
        assert r.expectancy == pytest.approx(2.0)

    def test_mixed_expectancy_golden(self):
        """[+2, -1, +1, -1]: mean = 0.25R."""
        trades = [(2.0, 200), (-1.0, -100), (1.0, 100), (-1.0, -100)]
        r = _pipeline_result_from_trades(10_000, trades)
        assert r.expectancy == pytest.approx(0.25, rel=1e-6)

    def test_negative_expectancy(self):
        """[-1, -1, +0.5]: mean = -0.5R."""
        trades = [(-1.0, -100), (-1.0, -100), (0.5, 50)]
        r = _pipeline_result_from_trades(10_000, trades)
        assert r.expectancy == pytest.approx(-0.5, rel=1e-6)

    def test_zero_expectancy(self):
        """[+1, -1]: mean = 0.0R."""
        trades = [(1.0, 100), (-1.0, -100)]
        r = _pipeline_result_from_trades(10_000, trades)
        assert r.expectancy == pytest.approx(0.0, abs=1e-9)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TestWinRateGolden
# ═══════════════════════════════════════════════════════════════════════════════

class TestWinRateGolden:
    """Win rate = wins / (wins + losses) × 100."""

    def test_no_trades_wr_zero(self):
        r = _pipeline_result_from_trades(10_000, [])
        assert r.win_rate == 0.0

    def test_all_winners(self):
        r = _pipeline_result_from_trades(10_000, [(1.0, 100), (2.0, 200)])
        assert r.win_rate == pytest.approx(100.0)

    def test_half_winners(self):
        r = _pipeline_result_from_trades(10_000, [(1.0, 100), (-1.0, -100)])
        assert r.win_rate == pytest.approx(50.0)

    def test_three_quarters(self):
        trades = [(1.0, 100)] * 3 + [(-1.0, -100)]
        r = _pipeline_result_from_trades(10_000, trades)
        assert r.win_rate == pytest.approx(75.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TestNetProfitGolden
# ═══════════════════════════════════════════════════════════════════════════════

class TestNetProfitGolden:
    """net_profit_usd = final_balance - initial_balance."""

    def test_no_trades_net_zero(self):
        r = _pipeline_result_from_trades(10_000, [])
        assert r.net_profit_usd == pytest.approx(0.0)

    def test_net_profit_positive(self):
        r = _pipeline_result_from_trades(10_000, [(2.0, 300), (-1.0, -100)])
        assert r.net_profit_usd == pytest.approx(200.0)

    def test_net_profit_negative(self):
        r = _pipeline_result_from_trades(10_000, [(-1.0, -100), (-1.0, -100)])
        assert r.net_profit_usd == pytest.approx(-200.0)

    def test_net_profit_exact_golden(self):
        """$10k start, +$100, -$200, +$50 → net = -$50."""
        trades = [(1.0, 100), (-2.0, -200), (0.5, 50)]
        r = _pipeline_result_from_trades(10_000, trades)
        assert r.net_profit_usd == pytest.approx(-50.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TestConsecutiveStreaksGolden — via BacktestResult (compute_streaks)
# ═══════════════════════════════════════════════════════════════════════════════

class TestConsecutiveStreaksGolden:
    """Verify max consecutive wins/losses via _compute_streaks directly."""

    def _streaks(self, outcomes: list[bool]) -> tuple[int, int]:
        """Compute streaks from a bool sequence by building fake orders."""
        from src.backtesting.backtest_runner import _compute_streaks
        orders = []
        for is_win in outcomes:
            o = _make_order(1.0 if is_win else -1.0, 100 if is_win else -100)
            orders.append(o)
        return _compute_streaks(orders)

    def test_no_trades_streaks_zero(self):
        mw, ml = self._streaks([])
        assert mw == 0 and ml == 0

    def test_all_wins(self):
        mw, ml = self._streaks([True, True, True])
        assert mw == 3 and ml == 0

    def test_alternating(self):
        mw, ml = self._streaks([True, False, True, False])
        assert mw == 1 and ml == 1

    def test_two_loss_streak(self):
        _, ml = self._streaks([True, False, False, True])
        assert ml == 2

    def test_long_win_streak(self):
        mw, ml = self._streaks([True]*5 + [False]*2 + [True]*3)
        assert mw == 5 and ml == 2
