# CandleStickBot — Phase 0 Implementation Blueprint
## Foundation Layer Complete

**Document Version**: 1.0  
**Spec Version**: 3.1 (Final)  
**Phase**: 0 — Foundation  
**Status**: ✅ COMPLETE — All 141 tests passing  
**Date**: 2026-06-11

---

## 1. What Was Built in Phase 0

Phase 0 implements the Foundation Layer of CandleStickBot as specified in the Technical Planning Document v3.1. Every deliverable maps directly to a module specification.

### Module Status

| Module | Name | Phase 0 Status | Tests |
|--------|------|----------------|-------|
| M15 | Config System | ✅ COMPLETE | 60+ tests |
| M13 | Logging / Audit | ✅ COMPLETE | 25+ tests |
| M02 | Candle Storage | ✅ COMPLETE (schema) | 35+ tests |
| Data Types | Domain Objects | ✅ COMPLETE | 21+ tests |

---

## 2. Repository Structure

```
CandleStickBot/
├── config/
│   └── default_config.yaml      # M15 — All 11+ strategy/risk parameters
├── src/
│   ├── config/
│   │   ├── models.py            # M15 — Pydantic validation models
│   │   └── loader.py            # M15 — YAML loading + env overrides
│   ├── db/
│   │   ├── models.py            # M02 — SQLAlchemy ORM (10 tables)
│   │   └── database.py          # M02 — DatabaseManager
│   ├── logging/
│   │   └── audit_logger.py      # M13 — AuditLogger + EventType enum
│   └── data/
│       └── types.py             # Shared domain objects (all modules)
├── tests/
│   ├── unit/
│   │   ├── config/              # 60+ config validation tests
│   │   ├── db/                  # 35+ database tests  
│   │   └── logging/             # 25+ logging tests
│   └── integration/
└── docs/
    └── PHASE0_BLUEPRINT.md      # This file
```

---

## 3. M15 — Config System

### What's Implemented

**File**: `src/config/models.py`  
**File**: `src/config/loader.py`  
**File**: `config/default_config.yaml`

#### Key Design Decisions

1. **Pydantic v2 validation** — Every parameter has type constraints, range checks, and cross-field validators
2. **Phase 1 scope enforcement** — `BotConfig.phase1_scope_enforcement()` prevents activating Phase 2+ features in Phase 1
3. **Hard caps enforced in code** — `RiskConfig.hard_cap_enforced()` prevents max_risk > 2.0% regardless of config
4. **Three-layer priority** — Environment vars > local_config.yaml > default_config.yaml
5. **Zero hardcoded values** — Every threshold in `default_config.yaml` maps to spec Section 14 Appendix

#### Parameters Covered

| Section | Parameters |
|---------|-----------|
| Risk | risk_per_trade_pct, max_risk_per_trade_pct (hard cap 2%), min_rr_ratio (floor 2.0), daily_loss_limit, weekly_loss_limit, kill_switch |
| TQS | min_score_to_trade (60), premium_threshold (80), all 4 component maxes |
| Promotion | paper→demo: 50 trades AND 3 months; demo→live: same + tighter criteria |
| Regime | ADX, ATR, BB Width, Choppiness Index thresholds for all 5 regimes |
| Strategy | All 4 strategies with quality thresholds; Phase 1/2 gating |
| Filters | Spread, session, news, correlation filters |
| Backtesting | In-sample/OOS split, walk-forward, Monte Carlo parameters |

#### Phase 1 Scope Guard (anti-scope-creep)

The following raise `ValidationError` in Phase 1:
- `symbols` contains anything other than `EURUSD`
- `strategies.inside_bar.enabled = true`
- `strategies.inside_bar_false_breakout.enabled = true`
- `levels.fibonacci.enabled = true`
- `levels.supply_demand_zones.enabled = true`
- `portfolio.enabled = true`
- `optimization.enabled = true`
- Primary timeframe other than D1

---

## 4. M13 — Logging / Audit Module

### What's Implemented

**File**: `src/logging/audit_logger.py`

#### EventType Enum — 44 Event Types

All decision types are covered:
- Data events: `DATA_FETCH`, `DATA_GAP_DETECTED`
- Analysis: `TREND_CLASSIFIED`, `REGIME_CLASSIFIED`, `PATTERN_DETECTED`
- Strategy: `TQS_COMPUTED`, `TRADE_RECOMMENDED`, `TRADE_REJECTED`
- Risk: `RISK_CHECK`, `KILL_SWITCH_TRIGGERED`, `DAILY_LIMIT_REACHED`
- Execution: `ORDER_PLACED`, `POSITION_CLOSED`, `SIGNAL_QUEUED`
- Analytics: `STRATEGY_SCORECARD_UPDATED`, `STRATEGY_DEGRADATION_ALERT`
- Review: `LOSS_CLASSIFIED`, `SYSTEMATIC_ERROR_ALERT`
- System: `BOT_STARTED`, `CONFIG_LOADED`, `MODE_CHANGED`

#### Schema Compliance

Every log entry follows:
```json
{
  "timestamp": "2026-06-11T00:00:00Z",
  "module": "M08",
  "event_type": "TRADE_RECOMMENDED",
  "symbol": "EURUSD",
  "trade_id": "uuid-here",
  ... full context payload
}
```

#### Output Channels

1. **Console** — Human-readable with structlog ConsoleRenderer
2. **Log file** — `logs/candlestickbot_YYYYMMDD.log` — structured text
3. **Audit JSONL** — `logs/audit_YYYYMMDD.jsonl` — machine-parseable, one entry per line

---

## 5. M02 — Candle Storage

### What's Implemented

**File**: `src/db/models.py`  
**File**: `src/db/database.py`

#### Database Schema (10 Tables)

| Table | Purpose | Module |
|-------|---------|--------|
| `candles` | OHLCV storage with composite unique index | M02 |
| `swing_points` | Detected swing highs/lows | M03 |
| `sr_levels` | Support/Resistance levels with strength scoring | M05 |
| `pattern_detections` | Every pattern evaluation (detected + rejected) | M07 |
| `trade_signals` | Full TQS scores, entry/stop/target, tier | M08 |
| `trades` | Complete trade lifecycle from entry to close | M10 |
| `strategy_performance` | Per-strategy scorecard (M18) | M18 |
| `monthly_failure_reports` | M19 loss taxonomy monthly aggregates | M19 |
| `audit_events` | DB-persisted audit trail | M13 |
| `bot_state` | Kill switch state, drawdown tracking | M09 |

#### Key Design: INDEX ON (symbol, timeframe, timestamp)

```python
__table_args__ = (
    UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_candle"),
    Index("idx_candle_lookup", "symbol", "timeframe", "timestamp"),
)
```

#### DatabaseManager Features

- **Transaction safety** — Context manager with auto-commit/rollback
- **WAL mode** — SQLite WAL journal for better concurrency
- **Foreign key enforcement** — `PRAGMA foreign_keys=ON`
- **Health check** — `db.health_check()` verifies connection
- **Table stats** — `db.get_table_stats()` for monitoring

---

## 6. Shared Domain Types

**File**: `src/data/types.py`

All modules communicate via these pure Python dataclasses:

```
CandleData         → Data contract for all modules (OHLCV + derived properties)
TrendSignal        → M04 output: direction, strength, tradeable
RegimeSignal       → M16 output: regime, allowed strategies, risk multiplier
SRLevelData        → M05 output: price, type, strength_score, zone bounds
PatternSignal      → M07 output: pattern_type, quality_score, entry/stop
TQSResult          → M08: 4-component score, tier, is_tradeable
TradeRecommendation → M08 output: complete trade parameters
RiskApprovedOrder  → M09 output: approved lot size and risk amount
```

**TQSResult auto-rejects when regime_score = 0** (VOLATILE/CHOPPY):
```python
@property
def is_tradeable(self) -> bool:
    return self.total >= 60 and self.regime_score > 0
```

---

## 7. Test Suite Summary

**Total: 141 tests, 100% passing**

| Test Module | Count | Pass Criteria (from spec) |
|------------|-------|--------------------------|
| Config validation | 60+ | Config loads with defaults; validation rejects invalid values; All 11+ parameters validated |
| Database / ORM | 35+ | Round-trip lossless; duplicate handling; 10K candle batch |
| Logging | 25+ | Logger outputs to file; all event types log without error |
| Data types | 21+ | Candle math correct; TQS scoring; boundary conditions |

**Phase 0 Pass/Fail from spec Section 11**:

✅ Config loads with defaults  
✅ Validation rejects invalid values  
✅ All 11+ strategy/risk parameters covered  
✅ pydantic validation bypass blocked  
✅ Database models create/migrate without errors  
✅ Logger outputs valid structured output  

---

## 8. Phase 1 Readiness

Phase 0 provides the foundation for Phase 1 (Data Layer):

### What Phase 1 Needs (already built)
- ✅ `Candle` ORM model with full schema
- ✅ `DatabaseManager` with session management
- ✅ `CandleData` domain type with all derived properties
- ✅ Config system with M01 parameters (broker, symbols, timeframes)
- ✅ Audit logging for M01 data events

### Phase 1 Remaining Work
- `src/data/ingestion.py` — M01 fetch_live_candles, fetch_historical_candles
- `src/db/candle_store.py` — M02 CandleStore CRUD operations
- MT5 Python API integration (when broker selected — Open Question #5)
- CSV import for historical data
- Gap detection implementation

---

## 9. Open Issues (Non-blocking for Phase 1)

From spec Section 12, none block Phase 1:

| # | Issue | Impact | Status |
|---|-------|--------|--------|
| 1 | Broker not selected | MT5 server name only | Non-blocking |
| 2 | News calendar API | Phase 2+ only | Non-blocking |
| 3 | D1 vs H4 primary TF | MVP is D1 only | Non-blocking |
| 4 | Correlation threshold | M17 deferred | Non-blocking |

---

## 10. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
python -m pytest tests/ -v

# Verify config loads
python -c "from src.config import load_config; c = load_config(); print(f'Phase {c.system.phase} | Mode: {c.execution.mode.value}')"

# Initialize database
python -c "from src.db import get_database; db = get_database(); print('Tables:', db.get_table_stats())"
```
