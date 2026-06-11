# CandleStickBot 🕯️

**Automated Forex Trading Bot — Candlestick Bible Methodology**  
*Inspired by The Candlestick Trading Bible (Munehisa Homma / Steve Nison)*

[![Phase](https://img.shields.io/badge/Phase-0%20Foundation-blue)](docs/PHASE0_BLUEPRINT.md)
[![Tests](https://img.shields.io/badge/Tests-141%20passing-brightgreen)](#testing)
[![Spec](https://img.shields.io/badge/Spec-v3.1%20Final-green)](#)
[![Mode](https://img.shields.io/badge/Mode-Backtest%20Only-orange)](#execution-modes)

---

## Project Overview

CandleStickBot converts the discretionary price action methodology from *The Candlestick Trading Bible* into **mathematically objective, backtestable, and programmable trading rules**. The system implements the Trend + Level + Signal framework — every trade must simultaneously satisfy all three gates.

### Core Philosophy

```
EDGE = Market Regime + Location + Trend + Volatility + Risk Management + Execution Quality
Candlestick patterns are only trade triggers, not edge themselves.
```

### Three-Gate Decision Framework

```
GATE 1 — TREND:   Is the market in a defined, tradeable direction?
GATE 2 — LEVEL:   Is price at or near a significant key level?
GATE 3 — SIGNAL:  Has a valid candlestick pattern formed at that level?

ALL THREE must be true simultaneously → Trade considered
ANY ONE fails → No trade
```

---

## Current Status: Phase 0 — Foundation ✅

| Component | Status | Description |
|-----------|--------|-------------|
| M15 Config System | ✅ Complete | YAML + Pydantic; all parameters validated |
| M13 Logging/Audit | ✅ Complete | Structured logging; 44 event types; full audit trail |
| M02 Candle Storage | ✅ Complete | SQLAlchemy; 10-table schema; WAL SQLite |
| Domain Types | ✅ Complete | All inter-module data contracts defined |
| Test Suite | ✅ **141 passing** | Config, DB, logging, data type tests |

---

## Architecture — 19 Modules

```
DATA INFRASTRUCTURE:  M01 Ingestion · M02 Storage · M10 Execution · M15 Config
ANALYSIS LAYER:       M03 Structure · M04 Trend · M05 S/R · M06 Fibonacci* · M07 Patterns
                      M16 Market Regime · M19 Trade Review
STRATEGY LAYER:       M08 Strategy Engine · M09 Risk Engine · M11 Backtesting
                      M12 Optimization* · M17 Portfolio* · M18 Analytics
PRESENTATION:         M13 Logging · M14 Dashboard

* = Disabled in Phase 1 (Phase 2+ activation)
```

---

## Phase 1 MVP Scope (Active)

| | Detail |
|--|--------|
| **Pair** | EURUSD only |
| **Timeframe** | Daily (D1) primary; Weekly (W1) context |
| **Strategies** | Pin Bar + Engulfing Bar ONLY |
| **Levels** | Swing S/R + 21 SMA ONLY |
| **Mode** | Backtest → Paper (NO live trading) |
| **Risk** | 1% default; 2% hard cap; 3% daily limit; 6% weekly limit; 10% kill switch |

---

## Four Strategies (Spec v3.1)

| # | Strategy | Market Context | Phase |
|---|----------|----------------|-------|
| 1 | **Pin Bar** | Tail rejection at key level | Phase 1 ✅ |
| 2 | **Engulfing Bar** | Momentum shift at key level | Phase 1 ✅ |
| 3 | Inside Bar Breakout | Continuation in strong trend | Phase 2+ |
| 4 | Inside Bar False Breakout | Stop-hunt / institutional trap | Phase 2+ |

---

## Trade Quality Score (TQS) — 0 to 100

```
┌────────────────────────┬───────────┐
│  Trend Strength        │  0-25 pts │
│  Level Strength        │  0-25 pts │
│  Pattern Quality       │  0-25 pts │
│  Market Regime         │  0-25 pts │
├────────────────────────┼───────────┤
│  0-59  → REJECT        │  No trade │
│  60-79 → STANDARD      │  1% risk  │
│  80-100→ PREMIUM       │  1% risk* │
└────────────────────────┴───────────┘
* Premium risk increase to 1.5% requires explicit config opt-in
  Hard maximum: 2.0% — cannot be exceeded by any means
```

---

## Market Regime Engine (M16)

| Regime | Condition | Allowed Strategies | Risk |
|--------|-----------|-------------------|------|
| TRENDING | ADX≥25, ATR expanding, bands widening | Pin Bar, Engulfing, Inside Bar Breakout | 1.0× |
| RANGING | ADX<20, bands contracting | Pin Bar at extremes, Engulfing at extremes | 0.75× |
| VOLATILE | ATR>1.5×MA, ADX<25 | **NONE** | 0× |
| QUIET | ATR<0.6×MA, bands very narrow | Inside Bar only | 0.5× |
| CHOPPY | Choppiness Index≥61.8 | **NONE** | 0× |

---

## Risk Management Rules

| Parameter | Value | Notes |
|-----------|-------|-------|
| Default risk per trade | **1.0%** | Non-negotiable default |
| Premium trade risk | 1.5% (opt-in) | All 4 TQS components must be ≥15 pts |
| **Hard cap** | **2.0%** | Cannot be exceeded by ANY means |
| Min R:R ratio | 2.0:1 | Cannot be set below 2.0 |
| Daily loss limit | 3.0% | Blocks new entries for rest of day |
| Weekly loss limit | 6.0% | Blocks new entries until next week |
| Kill switch | 10.0% drawdown | Halts ALL trading; requires manual restart |
| Max open trades | 3 | Global maximum |

---

## Mode Promotion Policy

| Stage | Min Trades | Min Time | Key Criteria |
|-------|-----------|----------|-------------|
| Paper → Demo | **50 AND** | **3 months** | PF ≥ 1.3, DD ≤ 20% |
| Demo → Live | **50 AND** | **3 months** | PF > 1.3, DD ≤ 10% |

> **BOTH** trade count AND calendar time are required. OR logic is not permitted.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run tests (141 should pass)
python -m pytest tests/ -v

# 3. Verify config system
python -c "
from src.config import load_config
c = load_config()
print(f'Phase {c.system.phase} | Mode: {c.execution.mode.value} | Symbol: {c.symbols}')
"

# 4. Initialize database
python -c "
from src.db import get_database
db = get_database('sqlite:///data/candlestickbot.db')
print('Tables:', db.get_table_stats())
"
```

---

## Project Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **0** | Foundation — Config, Logging, DB Schema | ✅ Complete |
| **1** | Data Layer — M01 Ingestion + M02 CandleStore | 🔜 Next |
| **2** | Analysis Engine — M03 Structure + M04 Trend | |
| **3** | Level Detection — M05 S/R Engine | |
| **4** | Pattern Detection — M07 All 7 patterns | |
| **4.5** | Strategy Validation Lab — Independent backtests | |
| **5** | Strategy + Risk Engines — M08/M09/M16 | |
| **6** | Backtesting Engine — M11 + Full metrics | |
| **7** | Optimization + Walk-Forward — M12 | |
| **8** | Paper Trading — 50 trades + 3 months | |
| **9** | Dashboard — M14 FastAPI | |
| **10** | Demo + Live Preparation | |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Config | PyYAML + Pydantic v2 |
| Database (dev) | SQLite 3 + SQLAlchemy 2.0 |
| Database (prod) | PostgreSQL 14+ |
| Logging | structlog |
| Broker API | MetaTrader 5 Python library |
| MT5 Execution | MQL5 Expert Advisor (thin order router) |
| Backtesting | vectorbt (Phase 6+) |
| Optimization | optuna / Bayesian (Phase 7+) |
| Dashboard | FastAPI + uvicorn (Phase 9+) |
| Testing | pytest + hypothesis |

---

## MT5 Configuration

MT5 credentials are stored in `config/default_config.yaml` under `execution.mt5`. For security, override with environment variables in production:

```bash
export CSBOT__EXECUTION__MT5__LOGIN=107695703
export CSBOT__EXECUTION__MT5__PASSWORD=<your_password>
export CSBOT__EXECUTION__MT5__SERVER=<broker_server>
```

---

## Safety First

> ⚠️ **Default mode is `backtest`**. The bot will NOT place real orders unless execution mode is explicitly changed AND all promotion criteria are satisfied.
>
> Live trading requires: Phase 0→1→2→...→Phase 10 progression, 50 paper trades, 3 months, PF ≥ 1.3.

---

## Documentation

- [`docs/PHASE0_BLUEPRINT.md`](docs/PHASE0_BLUEPRINT.md) — Phase 0 implementation details
- [`config/default_config.yaml`](config/default_config.yaml) — All configurable parameters
- Planning Document v3.1 — The authoritative specification (awaiting final approval)

---

*Specification: Automated Forex Trading Bot Planning Document v3.1 (Build-Ready, Score: 96/100)*
