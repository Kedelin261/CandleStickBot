# CandleStickBot

**Automated Forex Trading System — v3.1 (Phase 0)**

> _"The market is not random. It is a structured game of probability. Master the structure, master the outcome."_
> — The Candlestick Trading Bible

---

## Overview

CandleStickBot is a Python-based automated forex trading system built on the principles of **The Candlestick Trading Bible**. It combines candlestick pattern recognition, market structure analysis, and a rigorous trade quality scoring system (TQS) to trade EURUSD on the daily timeframe.

### Core Philosophy

1. **Structure First** — Never trade against the trend. Higher Highs + Higher Lows = Long only.
2. **Level Confluence** — Patterns must form at key S/R levels or the 21 SMA.
3. **Quality Gating** — Every potential trade receives a 0–100 Trade Quality Score. Only 60+ trades.
4. **Systematic Risk** — 1% risk per trade, 2% hard cap, 10% drawdown kill switch. No exceptions.

---

## Architecture

The system is structured as **19 modules across 4 layers**:

```
Layer 1 — Data Infrastructure
  M01  DataIngestionEngine     — MT5 OHLCV data fetch, backfill, CSV load
  M02  CandleStore             — SQLite/PostgreSQL persistent storage
  M13  AuditLogger             — Structured JSON decision audit trail
  M15  ConfigSystem            — Pydantic v2 YAML config with validation

Layer 2 — Analysis Engine
  M03  MarketStructureAnalyzer — Swing H/L detection, HH/HL/LH/LL, BOS
  M04  TrendDetectionEngine    — 21 SMA + swing structure trend classification
  M05  SRLevelEngine           — Swing S/R level detection, clustering
  M06  FibonacciEngine         — Fib retracement levels (Phase 2+)
  M16  RegimeClassifier        — ATR/ADX/BB-width market regime (TRENDING/RANGING/VOLATILE/QUIET)

Layer 3 — Strategy & Risk
  M07  PatternDetector         — Pin Bar, Engulfing Bar (Phase 1); Inside Bar, False Breakout (Phase 2+)
  M08  StrategyEngine          — TQS computation, signal generation, trade recommendation
  M09  RiskEngine              — Position sizing, kill switch, drawdown/loss limits
  M10  TradeExecutor           — Order management, MT5 EA bridge
  M11  BacktestEngine          — Historical simulation, walk-forward, Monte Carlo
  M12  OptimizationEngine      — Parameter optimization with governance policy (Phase 2+)
  M17  PortfolioEngine         — Multi-pair heat/correlation management (Phase 2+)

Layer 4 — Reporting & Governance
  M14  Dashboard               — CLI/web status monitor
  M18  PerformanceAnalytics    — Strategy scorecard, degradation alerts
  M19  TradeReviewClassifier   — Loss classification, systematic error detection
```

---

## Trade Quality Score (TQS)

Every potential trade is scored 0–100 before entry:

| Component | Weight | Criteria |
|-----------|--------|----------|
| **Trend** | 25 pts | Direction, strength, SMA position |
| **Level** | 25 pts | S/R confluence, zone proximity |
| **Pattern** | 25 pts | Pattern quality, wick ratios, body size |
| **Regime** | 25 pts | ATR/ADX/Choppiness market state |

**Tiers:**
- `REJECT` — Score < 60 → No trade
- `STANDARD` — Score 60–79 → Trade at 1.0% risk
- `PREMIUM` — Score ≥ 80 → Trade at 1.0% risk (1.5% opt-in disabled by default)

---

## Phase 1 MVP Scope

| Feature | Phase 1 | Phase 2+ |
|---------|---------|---------|
| Symbol | EURUSD only | GBPUSD, USDJPY, AUDUSD |
| Timeframe | D1 only | H4 added |
| Patterns | Pin Bar + Engulfing Bar | Inside Bar, False Breakout |
| Levels | Swing S/R + 21 SMA | + Fibonacci retracement |
| Execution | Backtest → Paper | → Demo → Live |
| Portfolio | Single pair | Multi-pair with heat/correlation |
| Optimization | Disabled | Walk-forward + Monte Carlo |

**Promotion Criteria (Paper → Live): 50 completed trades AND 3 calendar months** (both required)

---

## Risk Management

```yaml
risk_per_trade_pct: 1.0       # Default: 1% per trade
max_risk_per_trade_pct: 2.0   # Hard cap: cannot be exceeded
min_rr_ratio: 2.0             # Minimum 1:2 reward-to-risk
daily_loss_limit_pct: 3.0     # Max daily drawdown
weekly_loss_limit_pct: 6.0    # Max weekly drawdown
kill_switch_drawdown_pct: 10.0 # Emergency stop: 10% from peak
kill_switch_consecutive_losses: 7  # 7 losses in a row → halt
```

**Kill Switch Triggers** (any one):
1. 10% drawdown from equity peak
2. 7 consecutive losses
3. Daily loss limit AND weekly loss limit both hit

---

## Project Structure

```
CandleStickBot/
├── config/
│   └── default_config.yaml    # Master configuration (all parameters)
├── docs/
│   ├── PHASE0_BLUEPRINT.md    # Implementation plan and milestones
│   └── spec_v3.1.md           # Original specification document
├── migrations/
│   └── alembic.ini            # Database migration config (Alembic)
├── reports/                   # Backtest and performance reports
├── scripts/                   # Utility scripts (seed data, migrations)
├── src/
│   ├── __init__.py
│   ├── types.py               # Shared DTOs (CandleData, TQSComponents, etc.)
│   ├── analysis/              # M03, M04, M05, M16
│   ├── analytics/             # M18 performance analytics
│   ├── backtesting/           # M11 backtest engine
│   ├── config/                # M15 config system
│   ├── dashboard/             # M14 monitoring dashboard
│   ├── data/                  # M01 data ingestion
│   ├── db/                    # M02 ORM models, session, CandleStore
│   ├── execution/             # M10 trade executor
│   ├── logging/               # M13 audit logger
│   ├── optimization/          # M12 (Phase 2+)
│   ├── patterns/              # M07 pattern detectors
│   ├── risk/                  # M09 risk engine
│   ├── strategy/              # M08 strategy engine + TQS
│   └── trade_review/          # M19 loss classifier
└── tests/
    ├── conftest.py             # Shared fixtures
    └── unit/                  # Unit tests by module
        ├── config/             # 49 tests (config loader + validation)
        ├── db/                 # 53 tests (ORM + CandleStore + types)
        └── logging/            # 27 tests (audit logger)
```

---

## Getting Started

### Prerequisites

- Python 3.13+
- pip or pipx

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/CandleStickBot.git
cd CandleStickBot

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e ".[dev]"
```

### Running Tests

```bash
# All tests
pytest tests/unit/ -v

# Specific module
pytest tests/unit/config/ -v
pytest tests/unit/db/ -v
pytest tests/unit/logging/ -v
```

### Configuration

Copy the default config and customize:

```bash
cp config/default_config.yaml config/local_config.yaml
# Edit local_config.yaml with your settings
```

Or use environment variables (override any setting):

```bash
export CSBOT__EXECUTION__MODE=paper
export CSBOT__RISK__RISK_PER_TRADE_PCT=1.5
```

### MT5 Configuration

```yaml
# In local_config.yaml (never commit passwords)
execution:
  broker: mt5
  mt5:
    login: 107695703
    password: "!5UvKcSl"
    server: "YourBroker-Server"
```

---

## Database

The system uses **SQLite** for development and **PostgreSQL** for production.

```bash
# Initialize database (auto-created on first run)
# Tables: candles, swing_points, sr_levels, signals, trades,
#         trade_reviews, strategy_performance, strategy_summaries,
#         audit_logs, account_snapshots, backtest_results
```

Database migrations are managed with **Alembic** (prepared, not yet applied).

---

## Development Status

### Phase 0 — Foundation (Current)
- [x] Project structure and 19-module scaffold
- [x] Pydantic v2 configuration system with full validation
- [x] SQLAlchemy 2.0 ORM (11 tables)
- [x] CandleStore CRUD with gap detection
- [x] Structured audit logging (M13)
- [x] Shared type DTOs
- [x] Module stubs for all 19 modules
- [x] **141/141 unit tests passing**

### Phase 1 — Core Engine (Next)
- [ ] MT5 data ingestion (M01)
- [ ] Market structure analysis (M03)
- [ ] Trend detection (M04)
- [ ] S/R level engine (M05)
- [ ] Regime classifier (M16)
- [ ] Pin Bar + Engulfing pattern detectors (M07)
- [ ] Strategy engine with TQS computation (M08)
- [ ] Risk engine with kill switch (M09)
- [ ] Backtest engine (M11)

### Phase 2+ — Full System
- [ ] Additional pairs and H4 timeframe
- [ ] Inside Bar and False Breakout patterns
- [ ] Fibonacci retracement levels
- [ ] Portfolio management with heat/correlation
- [ ] Walk-forward optimization
- [ ] Paper trading execution bridge
- [ ] Live trading (after 50 trades + 3 months on paper)

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.13 |
| Config | Pydantic v2 + YAML |
| Database | SQLAlchemy 2.0 + SQLite/PostgreSQL |
| Migrations | Alembic |
| Logging | structlog (JSON audit trail) |
| Testing | pytest + pytest-mock + hypothesis |
| Broker API | MetaTrader5 (Python package) |
| Execution | MT5 Expert Advisor (EA) bridge |

---

## License

Private — All rights reserved.

---

## Disclaimer

This software is for educational and research purposes. Automated trading involves substantial risk of loss. Past performance does not guarantee future results. Always test thoroughly before risking real capital.
