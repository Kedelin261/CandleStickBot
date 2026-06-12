"""
Sprint 14 — Official Backtest Run + Report Generation
======================================================
Runs all 3 modes using the unmodified BacktestRunner + PipelineRunner.
Generates all required Sprint 14 report files.
"""
import sys, logging
sys.path.insert(0, '.')
logging.disable(logging.CRITICAL)

from pathlib import Path
from src.backtesting.backtest_runner import (
    BacktestRunner, BacktestConfig, StrategyValidationLab
)
from src.backtesting.reports import (
    generate_scorecard, generate_comparison_report, generate_validation_report
)
from src.backtesting.audit import audit_csv, save_audit_report

DATA_PATH    = 'data/EURUSD_D1_2014_2026.csv'
REPORTS_DIR  = Path('reports')
REPORTS_DIR.mkdir(exist_ok=True)

cfg = BacktestConfig(
    symbol='EURUSD',
    timeframe='D1',
    initial_balance=10_000.0,
    slippage_pips=1.0,
    lookback_window=200,    # larger window → more swing points from M03
    minimum_tqs=0.0,
    minimum_rr=2.0,
)

runner = BacktestRunner(config=cfg)

print("Running Pin Bar only backtest...")
pin_result = runner.run_pin_bar_only(DATA_PATH)
print(f"  trades_generated={pin_result.trades_generated}  trades_executed={pin_result.trades_executed}")

print("Running Engulfing only backtest...")
eng_result = runner.run_engulfing_only(DATA_PATH)
print(f"  trades_generated={eng_result.trades_generated}  trades_executed={eng_result.trades_executed}")

print("Running Combined backtest...")
com_result = runner.run_combined(DATA_PATH)
print(f"  trades_generated={com_result.trades_generated}  trades_executed={com_result.trades_executed}")

# ── Generate scorecards ──────────────────────────────────────────────────────
print("\nGenerating scorecards...")

pin_card = generate_scorecard(pin_result)
eng_card = generate_scorecard(eng_result)
com_card = generate_scorecard(com_result)

(REPORTS_DIR / 'pin_bar_scorecard.txt').write_text(pin_card)
(REPORTS_DIR / 'engulfing_scorecard.txt').write_text(eng_card)
(REPORTS_DIR / 'combined_scorecard.txt').write_text(com_card)
print("  reports/pin_bar_scorecard.txt        ✓")
print("  reports/engulfing_scorecard.txt      ✓")
print("  reports/combined_scorecard.txt       ✓")

# ── Validation lab report ────────────────────────────────────────────────────
print("\nRunning StrategyValidationLab...")
lab = StrategyValidationLab(cfg)
lab_results = lab.run(DATA_PATH)

val_report = generate_validation_report(lab_results)
(REPORTS_DIR / 'validation_lab_report.txt').write_text(val_report)
print("  reports/validation_lab_report.txt    ✓")

# ── Comparison report ────────────────────────────────────────────────────────
comp_report = generate_comparison_report([pin_result, eng_result, com_result])
(REPORTS_DIR / 'comparison_report.txt').write_text(comp_report)
print("  reports/comparison_report.txt        ✓")

print("\nAll Sprint 14 report files generated.")
print(f"\nSummary:")
print(f"  Pin Bar  : trades={pin_result.trades_executed}, PF={pin_result.profit_factor:.3f}, ExpR={pin_result.expectancy_r:.4f}, DD={pin_result.max_drawdown_pct:.1f}%")
print(f"  Engulfing: trades={eng_result.trades_executed}, PF={eng_result.profit_factor:.3f}, ExpR={eng_result.expectancy_r:.4f}, DD={eng_result.max_drawdown_pct:.1f}%")
print(f"  Combined : trades={com_result.trades_executed}, PF={com_result.profit_factor:.3f}, ExpR={com_result.expectancy_r:.4f}, DD={com_result.max_drawdown_pct:.1f}%")
print(f"\nBaseline pass criteria: PF>=1.10, WR>=0.40, MaxDD<=20%, Trades>=10")
print(f"  Pin Bar   passes_baseline: {pin_result.passes_baseline}")
print(f"  Engulfing passes_baseline: {eng_result.passes_baseline}")
print(f"  Combined  passes_baseline: {com_result.passes_baseline}")
