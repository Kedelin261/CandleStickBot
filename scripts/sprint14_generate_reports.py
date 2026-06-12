"""
Sprint 14 — Report Generation Script
=====================================
Runs all 3 backtest modes on real EURUSD D1 data and generates:
  - reports/pin_bar_scorecard.txt
  - reports/engulfing_scorecard.txt
  - reports/combined_scorecard.txt
  - reports/validation_lab_report.txt

Phase 1 system is FROZEN — no strategy modifications.
This script only orchestrates report generation.
"""
import sys
import logging
from pathlib import Path

# ── ensure project root is on PYTHONPATH ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sprint14")

from src.backtesting.backtest_runner import (
    BacktestConfig,
    BacktestRunner,
    StrategyValidationLab,
)
from src.backtesting.reports import (
    generate_scorecard,
    generate_validation_report,
)

DATA_FILE   = PROJECT_ROOT / "data" / "EURUSD_D1_2014_2026.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not DATA_FILE.exists():
        log.error("Data file not found: %s", DATA_FILE)
        sys.exit(1)

    log.info("Data file : %s", DATA_FILE)
    log.info("Reports   : %s", REPORTS_DIR)

    cfg = BacktestConfig(
        symbol="EURUSD",
        timeframe="D1",
        initial_balance=10_000.0,
        slippage_pips=1.0,
        lookback_window=50,
    )

    runner = BacktestRunner(cfg)

    # ── Mode 1 : Pin Bar Only ────────────────────────────────────────────────
    log.info("Running pin_bar_only …")
    pb_result = runner.run_pin_bar_only(str(DATA_FILE))
    log.info(
        "pin_bar_only  → trades_generated=%d  trades_executed=%d",
        pb_result.trades_generated,
        pb_result.trades_executed,
    )

    pb_scorecard = generate_scorecard(pb_result)
    sc_pb = REPORTS_DIR / "pin_bar_scorecard.txt"
    sc_pb.write_text(pb_scorecard, encoding="utf-8")
    log.info("Saved: %s", sc_pb)

    # ── Mode 2 : Engulfing Only ──────────────────────────────────────────────
    log.info("Running engulfing_only …")
    eng_result = runner.run_engulfing_only(str(DATA_FILE))
    log.info(
        "engulfing_only → trades_generated=%d  trades_executed=%d",
        eng_result.trades_generated,
        eng_result.trades_executed,
    )

    eng_scorecard = generate_scorecard(eng_result)
    sc_eng = REPORTS_DIR / "engulfing_scorecard.txt"
    sc_eng.write_text(eng_scorecard, encoding="utf-8")
    log.info("Saved: %s", sc_eng)

    # ── Mode 3 : Combined ───────────────────────────────────────────────────
    log.info("Running combined …")
    com_result = runner.run_combined(str(DATA_FILE))
    log.info(
        "combined      → trades_generated=%d  trades_executed=%d",
        com_result.trades_generated,
        com_result.trades_executed,
    )

    com_scorecard = generate_scorecard(com_result)
    sc_com = REPORTS_DIR / "combined_scorecard.txt"
    sc_com.write_text(com_scorecard, encoding="utf-8")
    log.info("Saved: %s", sc_com)

    # ── Validation Lab ───────────────────────────────────────────────────────
    log.info("Running StrategyValidationLab …")
    from src.backtesting.backtest_runner import ValidationReport
    lab_report = ValidationReport(
        pin_bar_result=pb_result,
        engulfing_result=eng_result,
        combined_result=com_result,
    )
    lab_report.rank()

    val_text = generate_validation_report(lab_report)
    vl_path = REPORTS_DIR / "validation_lab_report.txt"
    vl_path.write_text(val_text, encoding="utf-8")
    log.info("Saved: %s", vl_path)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SPRINT 14 — REPORT GENERATION COMPLETE")
    print("=" * 70)
    print(f"\n  Data file   : {DATA_FILE.name}")
    print(f"  Candles     : {pb_result.candles_processed}")
    print(f"  Date range  : {pb_result.date_range[0].date() if pb_result.date_range[0] else 'N/A'}"
          f" → {pb_result.date_range[1].date() if pb_result.date_range[1] else 'N/A'}")
    print()
    print(f"  {'Mode':<20} {'Generated':>10} {'Executed':>10} {'Baseline':>10}")
    print("  " + "-" * 55)
    for label, res in [
        ("pin_bar_only",   pb_result),
        ("engulfing_only", eng_result),
        ("combined",       com_result),
    ]:
        b = "PASS" if res.passes_baseline else "FAIL"
        print(f"  {label:<20} {res.trades_generated:>10} {res.trades_executed:>10} {b:>10}")

    print()
    print(f"  Rankings (best → worst): {' > '.join(lab_report.strategy_rankings)}")
    print()
    print("  Files written:")
    for p in [sc_pb, sc_eng, sc_com, vl_path]:
        print(f"    ✅  {p.relative_to(PROJECT_ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
