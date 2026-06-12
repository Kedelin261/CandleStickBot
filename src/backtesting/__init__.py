"""
Sprint 13 — Backtesting Framework & Strategy Validation Lab
============================================================
Phase 1 backtesting components.  All strategy, risk, execution, and
analytics logic lives in the existing Phase 1 modules — this package
only adds CSV loading, orchestration, and report generation on top of
the existing PipelineRunner.

Public API
----------
from src.backtesting.data_loader    import load_candles_from_csv, DataQualityReport
from src.backtesting.backtest_runner import (
    BacktestConfig, BacktestResult, BacktestRunner, StrategyValidationLab
)
from src.backtesting.reports        import generate_scorecard, generate_comparison_report

No Phase 2 features. No optimization. No live execution.
"""
