# CandleStickBot v3.1 — Automated Forex Trading Bot

> **Based on: The Candlestick Trading Bible**  
> Phase 0 Implementation — Repository Bootstrap & Architecture Foundation

---

## Overview

CandleStickBot is a professional automated forex trading system implementing the strategies and principles from *The Candlestick Trading Bible*. Built in Python 3.13 with a strict 19-module, 4-layer architecture enforcing disciplined risk management and evidence-based pattern trading.

**Phase 1 MVP Scope:** EURUSD D1 — Pin Bar + Engulfing Bar strategies — Backtest → Paper mode only.

---

## Architecture

### 4-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 4: PRESENTATION (M13, M14, M15, M19)                    │
│  Logging · Dashboard · Config · Trade Review                    │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 3: STRATEGY & EXECUTION (M08, M09, M10, M11, M12, M17) │
│  Strategy Engine · Risk Management · Execution · Backtesting   │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 2: ANALYSIS (M03, M04, M05, M06, M07, M16, M18)        │
│  Market Structure · Trend · S/R · Patterns · Regime · Analytics│
├─────────────────────────────────────────────────────────────────┤
│  LAYER 1: DATA INFRASTRUCTURE (M01, M02, M15)                  │
│  MT5 Ingestion · Candle Storage · Database                     │
└─────────────────────────────────────────────────────────────────┘
```

### 19-Module Map

| Module | Name | Layer | Phase 1 |
|--------|------|-------|---------|
| M01 | Data Ingestion (MT5) | Data | ✅ Stub |
| M02 | Candle Storage (CandleStore) | Data | ✅ Complete |
| M03 | Market Structure Analyzer | Analysis | ✅ Stub |
| M04 | Trend Detection (21 SMA + ADX) | Analysis | ✅ Stub |
| M05 | S/R Engine (Swing + SMA) | Analysis | ✅ Stub |
| M06 | Fibonacci Engine | Analysis | ❌ Phase 2 |
| M07 | Pattern Detection (Pin Bar + Engulfing) | Analysis | ✅ Stub |
| M08 | Strategy Engine (TQS Gate) | Strategy | ✅ Stub |
| M09 | Risk Management Engine | Strategy | ✅ Stub |
| M10 | Trade Executor | Strategy | ✅ Stub |
| M11 | Backtesting Engine | Strategy | ✅ Stub |
| M12 | Optimization Engine | Strategy | ❌ Phase 2 |
| M13 | Audit Logger (structlog) | Presentation | ✅ Complete |
| M14 | Dashboard Monitor | Presentation | ✅ Stub |
| M15 | Config System (Pydantic v2) | Presentation | ✅ Complete |
| M16 | Market Regime Engine | Analysis | ✅ Stub |
| M17 | Portfolio Engine | Strategy | ❌ Phase 2 |
| M18 | Performance Analytics | Analysis | ✅ Stub |
| M19 | Trade Review / Loss Classifier | Presentation | ✅ Stub |

---

## Trade Quality Score (TQS)

Every potential trade is scored 0-100 across 4 components:

| Component | Module | Max Points | Description |
|-----------|--------|-----------|-------------|
| Trend | M04 | 25 | SMA position + ADX strength |
| Level | M05 | 25 | S/R level quality and proximity |
| Pattern | M07 | 25 | Candlestick pattern quality |
| Regime | M16 | 25 | Market regime suitability |

**TQS Tiers:**
- 🔴 **REJECT** (< 60): No trade
- 🟡 **STANDARD** (60-79): Trade at 1.0% risk
- 🟢 **PREMIUM** (≥ 80): Eligible for 1.5% risk *(disabled by default)*

---

## Risk Management

| Parameter | Value | Notes |
|-----------|-------|-------|
| Default risk/trade | 1.0% | Fixed fractional |
| Premium risk (opt-in) | 1.5% | TQS ≥ 80, disabled by default |
| Hard cap | 2.0% | Cannot be overridden — enforced in Pydantic |
| Min R:R ratio | 2.0:1 | Cannot be set below 2.0 |
| Daily loss limit | 3.0% | Blocks new trades |
| Weekly loss limit | 6.0% | Blocks new trades |
| Kill switch — drawdown | 10.0% | Halts all trading |
| Kill switch — losses | 7 consecutive | Halts all trading |
| Max open trades | 3 | Phase 1 |

**Kill Switch:** Activates on ANY: 10% drawdown OR 7 consecutive losses OR both daily+weekly limits simultaneously. Manual reset required.

---

## Phase Scope

### Phase 1 (Current) — MVP
- ✅ EURUSD only (enforced by Pydantic validator)
- ✅ D1 timeframe only
- ✅ Pin Bar + Engulfing Bar strategies
- ✅ Swing S/R + 21 SMA levels
- ✅ Backtest → Paper modes
- ❌ Fibonacci (disabled)
- ❌ Inside Bar / False Breakout (disabled)
- ❌ Portfolio management (disabled)
- ❌ Optimization engine (disabled)
- ❌ Live trading (disabled)

### Phase 2 (Promotion criteria: 50 trades AND 3 months — both required)
- Additional pairs (GBPUSD, USDJPY, AUDUSD, USDCAD)
- H4 timeframe
- Fibonacci retracements (M06)
- Inside Bar + False Breakout strategies
- Portfolio engine with correlation management (M17)
- Optimization engine with baseline gate (M12)
- Live trading via MT5 EA

---

## MT5 Hybrid Architecture

```
Python (CandleStickBot)          MT5 Platform
┌─────────────────────┐         ┌──────────────────┐
│  M01 Data Fetch ────┼─────────┼→ Market Data      │
│  M03-M07 Analysis   │         │                  │
│  M08 Strategy       │         │                  │
│  M09 Risk Check     │         │                  │
│  M10 Order Params ──┼─────────┼→ Expert Advisor   │
│  M11 Backtest       │  IPC    │  (order exec)     │
│  M13 Audit Log      │         │                  │
└─────────────────────┘         └──────────────────┘
```

Python handles: All analysis, risk checks, position sizing, audit logging  
MT5 EA handles: Order placement, SL/TP management (Phase 2: live only)

---

## Project Structure

```
CandleStickBot/
├── config/
│   ├── default_config.yaml     # Master config (all parameters)
│   └── local_config.yaml       # Local overrides (gitignored)
├── docs/
│   ├── PHASE0_BLUEPRINT.md     # Implementation blueprint
│   └── spec_v3.1.md            # Original specification
├── migrations/                 # Alembic DB migrations
├── reports/                    # Generated backtest reports
├── scripts/                    # Utility scripts
├── src/
│   ├── analysis/               # M03, M04, M05, M16, M18
│   ├── analytics/              # M18 performance
│   ├── backtesting/            # M11
│   ├── config/                 # M15 (Pydantic models + loader)
│   ├── dashboard/              # M14
│   ├── data/                   # M01 ingestion
│   ├── db/                     # M02 (ORM models, CandleStore, session)
│   ├── execution/              # M10
│   ├── logging/                # M13 (AuditLogger)
│   ├── optimization/           # M12 (Phase 2)
│   ├── patterns/               # M07 (pin bar, engulfing, inside bar, false breakout)
│   ├── risk/                   # M09
│   ├── strategy/               # M08
│   ├── trade_review/           # M19
│   └── types.py                # Shared DTOs (CandleData, TQSComponents, etc.)
├── tests/
│   ├── conftest.py             # Shared fixtures
│   └── unit/
│       ├── config/             # Config loader + validation tests
│       ├── db/                 # CandleStore + ORM tests
│       └── logging/            # AuditLogger tests
├── requirements.txt
├── setup.py
├── pytest.ini
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.13+
- MetaTrader 5 terminal (for live/paper data — optional for backtesting with CSV)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/CandleStickBot.git
cd CandleStickBot
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

### Run Tests

```bash
python3 -m pytest tests/ -v
# Expected: 141 passed
```

### Configuration

Copy and customize:
```bash
cp config/default_config.yaml config/local_config.yaml
# Edit local_config.yaml — this file is gitignored
```

Environment variable overrides:
```bash
export CSBOT__EXECUTION__MODE=paper
export CSBOT__RISK__RISK_PER_TRADE_PCT=1.5
```

---

## Configuration Reference

Key parameters in `config/default_config.yaml`:

```yaml
system:
  phase: 1                    # Current phase (1 or 2)
  log_level: INFO

execution:
  mode: backtest              # backtest | paper | live

symbols:
  - EURUSD                    # Phase 1: only EURUSD allowed

risk:
  risk_per_trade_pct: 1.0     # Default risk per trade
  max_risk_per_trade_pct: 2.0 # Hard cap (cannot exceed)
  min_rr_ratio: 2.0           # Minimum R:R ratio
  daily_loss_limit_pct: 3.0
  weekly_loss_limit_pct: 6.0
  kill_switch_drawdown_pct: 10.0

tqs:
  min_score_to_trade: 60      # Minimum TQS to take a trade
  premium_threshold: 80       # Premium tier threshold
```

---

## Test Coverage

| Module | Tests | Status |
|--------|-------|--------|
| Config Loader (M15) | 20 | ✅ All passing |
| Config Validation (M15) | 19 | ✅ All passing |
| CandleStore (M02) | 27 | ✅ All passing |
| Database ORM | 14 | ✅ All passing |
| CandleData Types | 12 | ✅ All passing |
| AuditLogger (M13) | 27 | ✅ All passing |
| **Total** | **141** | **✅ 141/141** |

---

## Development Status

**Phase 0 (Complete):** Repository bootstrap, architecture, all module scaffolds, 141 tests passing.

**Phase 1 Sprint 1 (Next):** M01 data ingestion, MT5 connection, candle backfill.  
**Phase 1 Sprint 2:** M03/M04/M05/M16 analysis engines — full implementation.  
**Phase 1 Sprint 3:** M07/M08 pattern detection + strategy engine.  
**Phase 1 Sprint 4:** M09/M10/M11 risk + execution + backtesting.  
**Phase 1 Sprint 5:** M18/M19 analytics + first full backtest run.

See `docs/PHASE0_BLUEPRINT.md` for complete implementation plan.

---

## MT5 Credentials

Stored in `config/default_config.yaml` under `execution.mt5`:
- Login: `107695703`
- Server: configured in local_config.yaml (gitignored)

⚠️ **Never commit credentials to git.** Use `local_config.yaml` (gitignored) for sensitive overrides.

---

## License

Internal project. All rights reserved.
