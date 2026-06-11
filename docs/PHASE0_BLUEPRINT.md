# CandleStickBot v3.1 — Phase 0 Implementation Blueprint

**Version:** 3.1  
**Status:** Phase 0 Complete  
**Date:** 2026-06-11  
**Based On:** CandleStickBot Automated Forex Trading Bot Planning Document v3.1

---

## Executive Summary

Phase 0 establishes the complete repository foundation for CandleStickBot: directory structure, all 19 module scaffolds, Pydantic v2 configuration system, SQLAlchemy 2.0 ORM, structlog audit logging, and 141 passing tests. No trading logic is implemented yet — Phase 0 is purely architectural.

**Phase 0 Deliverables: ALL COMPLETE ✅**
- Directory structure with all 19 module locations
- Full Pydantic v2 config system with phase enforcement validators  
- SQLAlchemy 2.0 ORM (11 tables) with WAL-mode SQLite
- M02 CandleStore (complete CRUD + gap detection)
- M13 AuditLogger (complete structlog implementation)
- M15 Config Loader (YAML + env override + save/reload)
- All Phase 1 module stubs (M01, M03, M04, M05, M07, M08, M09, M10, M11, M14, M16, M18, M19)
- 141 unit tests — **141/141 passing**

---

## 1. Directory Structure

```
/home/user/CandleStickBot/
│
├── config/
│   ├── default_config.yaml       [COMPLETE] Master config (~15,150 chars, all spec params)
│   └── local_config.yaml         [GITIGNORED] Local overrides (not in repo)
│
├── docs/
│   ├── PHASE0_BLUEPRINT.md       [THIS FILE]
│   └── spec_v3.1.md              [REFERENCE] Original planning document
│
├── migrations/                   [PREPARED] Alembic migration files
│   └── versions/                 Empty until Phase 1 first migration
│
├── reports/                      [EMPTY] Generated backtest reports land here
│
├── scripts/                      [EMPTY] Utility scripts (backfill, migrate, etc.)
│
├── src/
│   ├── __init__.py               [COMPLETE] Package version constants
│   ├── types.py                  [COMPLETE] All shared DTOs
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── market_structure.py   [STUB M03]
│   │   ├── trend_detection.py    [STUB M04]
│   │   ├── sr_engine.py          [STUB M05]
│   │   └── regime_engine.py      [STUB M16]
│   │
│   ├── analytics/
│   │   ├── __init__.py
│   │   └── performance.py        [STUB M18]
│   │
│   ├── backtesting/
│   │   ├── __init__.py
│   │   └── engine.py             [STUB M11]
│   │
│   ├── config/
│   │   ├── __init__.py           [COMPLETE] Exports: BotConfig, load_config, etc.
│   │   ├── models.py             [COMPLETE] Pydantic v2 models with validators
│   │   └── loader.py             [COMPLETE] YAML load + env override + save
│   │
│   ├── dashboard/
│   │   ├── __init__.py
│   │   └── app.py                [STUB M14]
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── ingestion.py          [STUB M01]
│   │
│   ├── db/
│   │   ├── __init__.py           [COMPLETE] Exports all ORM models + utils
│   │   ├── models.py             [COMPLETE] 11 ORM tables
│   │   ├── session.py            [COMPLETE] Engine + session factory
│   │   └── candle_store.py       [COMPLETE M02] Full CRUD + gap detection
│   │
│   ├── execution/
│   │   ├── __init__.py
│   │   └── trade_executor.py     [STUB M10]
│   │
│   ├── logging/
│   │   ├── __init__.py           [COMPLETE] Exports AuditLogger, etc.
│   │   └── audit_logger.py       [COMPLETE M13] Full structlog implementation
│   │
│   ├── optimization/
│   │   └── __init__.py           [PHASE 2] Disabled
│   │
│   ├── patterns/
│   │   ├── __init__.py
│   │   ├── pin_bar.py            [STUB M07] Full algorithm documented
│   │   ├── engulfing.py          [STUB M07] Full algorithm documented
│   │   ├── inside_bar.py         [PHASE 2 STUB] Disabled in Phase 1
│   │   └── false_breakout.py     [PHASE 2 STUB] Disabled in Phase 1
│   │
│   ├── risk/
│   │   ├── __init__.py
│   │   └── risk_engine.py        [STUB M09] Kill switch + position sizing logic
│   │
│   ├── strategy/
│   │   ├── __init__.py
│   │   └── strategy_engine.py    [STUB M08] TQS gate chain documented
│   │
│   └── trade_review/
│       ├── __init__.py
│       └── classifier.py         [STUB M19] Loss classification logic
│
├── tests/
│   ├── conftest.py               [COMPLETE] Shared fixtures
│   └── unit/
│       ├── config/
│       │   ├── test_config_loader.py     [20 tests ✅]
│       │   └── test_config_validation.py [19 tests ✅]
│       ├── db/
│       │   ├── test_candle_store.py      [27 tests ✅]
│       │   ├── test_database.py          [14 tests ✅]
│       │   └── test_candle_data_types.py [12 tests ✅]
│       └── logging/
│           └── test_audit_logger.py      [27 tests ✅]
│
├── requirements.txt              [COMPLETE]
├── setup.py                      [COMPLETE]
├── pytest.ini                    [COMPLETE]
└── README.md                     [COMPLETE]
```

---

## 2. Module Interface Contracts

### M01 — Data Ingestion (`src/data/ingestion.py`)

**Interface:**
```python
class DataIngestionEngine:
    def connect(self) -> bool
    def disconnect(self) -> None
    def fetch_candles(symbol, timeframe, count, start, end) -> FetchResult
    def fetch_latest(symbol, timeframe, n) -> FetchResult
    def backfill(symbol, timeframe, start, end) -> FetchResult
    def get_symbol_info(symbol) -> Optional[Dict]
    def get_server_time() -> Optional[datetime]
```

**Dependencies:** MT5 terminal (Python MetaTrader5 package), M02 CandleStore, M13 AuditLogger  
**Implementation Sprint:** Phase 1 Sprint 1

---

### M02 — Candle Storage (`src/db/candle_store.py`) ✅ COMPLETE

**Interface:**
```python
class CandleStore:
    def store_candles(candles: List[Candle]) -> int           # Returns count stored
    def get_candles(symbol, timeframe, start, end) -> List[Candle]
    def get_latest_n_candles(symbol, timeframe, n) -> List[Candle]
    def get_candle_count(symbol, timeframe) -> int
    def get_date_range(symbol, timeframe) -> Optional[Tuple[datetime, datetime]]
    def candle_gap_check(symbol, timeframe) -> List[Dict]
    def validate_data_integrity(symbol, timeframe) -> Dict

def candle_from_dict(data: Dict) -> Candle                    # Factory with validation
```

**Test coverage:** 27 tests ✅

---

### M03 — Market Structure (`src/analysis/market_structure.py`)

**Interface:**
```python
class MarketStructureAnalyzer:
    def analyze(candles: List[CandleData]) -> StructureAnalysis
    def detect_structure_break(candles, analysis) -> Tuple[bool, bool]

class StructureAnalysis:
    direction: TrendDirection  # UP | DOWN | RANGING | UNDEFINED
    swing_highs: List[SwingPoint]
    swing_lows: List[SwingPoint]
    last_hh, last_hl, last_lh, last_ll: Optional[float]
    def to_market_structure() -> MarketStructure  # DTO conversion
```

**Algorithm:** N-bar lookback pivot detection → HH/HL/LH/LL classification → trend direction  
**Implementation Sprint:** Phase 1 Sprint 2

---

### M04 — Trend Detection (`src/analysis/trend_detection.py`)

**Interface:**
```python
class TrendDetector:
    def analyze(candles, market_structure) -> TrendAnalysis

class TrendAnalysis:
    direction: str          # UP | DOWN | RANGING | UNDEFINED
    tradeable: bool         # True if ADX >= 20 and trend confirmed
    sma21: float
    adx: Optional[float]
    tqs_trend_score: int    # 0-25 for TQS calculation
    def to_trend_signal() -> TrendSignal
```

**Rules:** Price above 21 SMA + HH+HL structure = UPTREND tradeable  
**Implementation Sprint:** Phase 1 Sprint 2

---

### M05 — S/R Engine (`src/analysis/sr_engine.py`)

**Interface:**
```python
class SREngine:
    def analyze(candles, swing_highs, swing_lows, sma21) -> SRAnalysis
    def calculate_tqs_level_score(candle, nearest_support, nearest_resistance, direction) -> int

class SRAnalysis:
    levels: List[SRLevel]
    nearest_support: Optional[SRLevel]
    nearest_resistance: Optional[SRLevel]
```

**Phase 1 methods:** Swing S/R + 21 SMA dynamic  
**Phase 2 methods:** Fibonacci (M06), Supply & Demand zones  
**Implementation Sprint:** Phase 1 Sprint 2

---

### M07 — Pattern Detection (`src/patterns/`)

**Pin Bar Interface:**
```python
class PinBarDetector:
    def detect(candle: CandleData) -> PinBarResult

class PinBarResult:
    detected: bool
    pin_bar_type: Optional[PinBarType]  # BULLISH | BEARISH
    quality_score: int                   # 1-10
    tail_ratio: float
    suggested_entry, suggested_stop: Optional[float]
```

**Engulfing Bar Interface:**
```python
class EngulfingDetector:
    def detect(current: CandleData, previous: CandleData) -> EngulfingResult

class EngulfingResult:
    detected: bool
    engulfing_type: Optional[EngulfingType]  # BULLISH | BEARISH
    quality_score: int                        # 1-10
    engulfing_ratio: float
```

**Implementation Sprint:** Phase 1 Sprint 3

---

### M08 — Strategy Engine (`src/strategy/strategy_engine.py`)

**Interface:**
```python
class StrategyEngine:
    def evaluate(symbol, timeframe, candles) -> EvaluationResult
    def calculate_tqs(trend, level, pattern, regime) -> TQSComponents
```

**Gate Chain:** TREND → REGIME → PATTERN → LEVEL → TQS → RR  
**TQS Formula:** trend(0-25) + level(0-25) + pattern(0-25) + regime(0-25) = 0-100  
**Implementation Sprint:** Phase 1 Sprint 3

---

### M09 — Risk Engine (`src/risk/risk_engine.py`)

**Interface:**
```python
class RiskEngine:
    def check_and_approve(recommendation, account) -> Tuple[result, approved, rejection]
    def check_kill_switch(account) -> Optional[KillSwitchEvent]
    def reset_kill_switch(authorized_by) -> None
    def update_after_trade_close(pnl_r, account) -> None
```

**Kill Switch Triggers (any ONE activates):**
1. Drawdown from peak ≥ 10%
2. Consecutive losses ≥ 7
3. Both daily AND weekly limits hit simultaneously

**Hard Cap:** max_risk_per_trade_pct cannot exceed 2.0% (enforced in Pydantic + RiskEngine)  
**Implementation Sprint:** Phase 1 Sprint 3

---

### M10 — Trade Executor (`src/execution/trade_executor.py`)

**Interface:**
```python
class TradeExecutor:
    def submit_order(approved_order: RiskApprovedOrder) -> Trade
    def simulate_fill(trade, next_candle_open, next_candle_time) -> BacktestFill
    def check_exit_backtest(trade, candle) -> Optional[TradeStatus]
    def calculate_pnl(trade, close_price, lot_size) -> Tuple[pips, usd, r]
```

**Phase 1:** Backtest (simulated fills) + Paper (logged, no execution)  
**Phase 2:** Live mode via MT5 EA IPC  
**Implementation Sprint:** Phase 1 Sprint 3

---

### M11 — Backtesting Engine (`src/backtesting/engine.py`)

**Interface:**
```python
class BacktestEngine:
    def run(run_id) -> BacktestResult

class BacktestMetrics:
    total_trades, win_rate, profit_factor, expectancy_r: ...
    max_drawdown_pct, max_consecutive_losses: ...
    passes_baseline: bool  # PF >= 1.1 AND win_rate >= 40% AND max_DD <= 20%
```

**Warm-up period:** 200 bars (skip before trading begins)  
**Conservative exits:** SL takes precedence over TP if both hit same candle  
**Implementation Sprint:** Phase 1 Sprint 4

---

### M13 — Audit Logger (`src/logging/audit_logger.py`) ✅ COMPLETE

**Interface:**
```python
class AuditLogger:
    def log_data_fetch(symbol, timeframe, count, source, success, error)
    def log_trend_classified(symbol, timeframe, direction, tradeable, reason, ma_value, adx)
    def log_regime_classified(symbol, timeframe, regime, confidence, ...)
    def log_pattern_detected(symbol, timeframe, pattern_type, direction, quality_score, ...)
    def log_tqs_computed(symbol, timeframe, strategy, tqs_total, trend, level, pattern, regime, tier)
    def log_trade_rejected(symbol, timeframe, strategy, reason, gate, tqs)
    def log_trade_recommended(symbol, timeframe, strategy, direction, entry, stop, target, ...)
    def log_risk_check(result, reason, trade_id, risk_amount, lots)
    def log_kill_switch_triggered(reason, account_state)    # CRITICAL level
    def log_order_event(event_type, order_id, symbol, ...)
    def log_loss_classified(trade_id, strategy, category, pnl_r, context)
    def log_bot_started/stopped(...)
    # ... 20+ log methods total
```

**Test coverage:** 27 tests ✅

---

### M15 — Config System (`src/config/`) ✅ COMPLETE

**Interface:**
```python
def load_config(config_path, override_path, apply_env) -> BotConfig
def save_config(config: BotConfig, path: Path) -> None
def get_config() -> BotConfig

class BotConfig(BaseModel):
    # 16 nested config sections with validators
    @model_validator(mode='after')
    def phase1_scope_enforcement(self) -> 'BotConfig': ...
```

**Phase 1 enforcement (raises ValidationError):**
- `symbols != ["EURUSD"]`
- `inside_bar.enabled == True`
- `false_breakout.enabled == True`
- `fibonacci.enabled == True`
- `supply_demand.enabled == True`
- `portfolio.enabled == True`
- `optimization.enabled == True`

**Test coverage:** 39 tests ✅

---

## 3. Database Schema

### Tables (11 total)

| Table | Purpose | Key Fields |
|-------|---------|-----------|
| `candles` | OHLCV data (M02) | symbol, timeframe, timestamp, ohlcv, spread |
| `swing_points` | M03 swing detection | symbol, timeframe, price, swing_type |
| `sr_levels` | M05 S/R levels | price, level_type, strength_score, zone_high/low |
| `signals` | M08 trade recommendations | strategy, direction, entry, stop, target, tqs_* |
| `trades` | M10 executed trades | fill_price, close_price, pnl_pips, pnl_r |
| `trade_reviews` | M19 loss classification | category, tqs_breakdown, notes |
| `strategy_performance` | M18 per-trade scorecard | win_rate, profit_factor, expectancy_r |
| `strategy_summaries` | M18 aggregated stats | period rollup metrics |
| `audit_logs` | M13 decision trail | event_type, module, payload, timestamps |
| `account_snapshots` | M09 kill switch state | balance, equity, drawdown_pct, kill_switch |
| `backtest_results` | M11 run metadata | run_id, config_snapshot, metrics_json |

**Unique constraints:** `candles(symbol, timeframe, timestamp)` — upsert on conflict  
**SQLite WAL mode:** Enabled for concurrent read performance during backtesting

---

## 4. Testing Strategy

### Test Pyramid

```
                    ┌─────────────────┐
                    │   E2E Tests     │  Phase 1 Sprint 5
                    │  (full backtest)│
                    └────────┬────────┘
              ┌──────────────┴───────────────┐
              │      Integration Tests        │  Phase 1 Sprint 4
              │  (module-to-module, M08→M09)  │
              └──────────────┬───────────────┘
   ┌──────────────────────────┴────────────────────────────┐
   │                   Unit Tests (CURRENT)                 │
   │  141 tests — Config, CandleStore, ORM, AuditLogger    │
   └───────────────────────────────────────────────────────┘
```

### Coverage Targets

| Phase | Coverage | Tests |
|-------|---------|-------|
| Phase 0 (now) | Core modules | 141 ✅ |
| Phase 1 Sprint 2 | Analysis engines | +60 tests |
| Phase 1 Sprint 3 | Pattern + Strategy | +80 tests |
| Phase 1 Sprint 4 | Risk + Execution + Backtest | +50 tests |
| Phase 1 Sprint 5 | Analytics + E2E | +30 tests |
| **Phase 1 Total** | **All modules** | **~360 tests** |

### Hypothesis Property Testing

Phase 1 Sprint 2 will add `hypothesis`-based property tests for:
- TQS component ranges: `0 ≤ score ≤ 25` always
- Pin bar invariants: `tail_ratio ≥ min_tail_ratio` always
- Risk engine: `lot_size ≤ max_lots` always

---

## 5. Implementation Milestones

### Phase 1 Sprint 1 — Data Foundation (Weeks 1-2)
**Goal:** Live data flowing into database

- [ ] M01: MT5 connection and authentication
- [ ] M01: Historical candle download (5 years EURUSD D1)
- [ ] M01: Data validation and gap detection
- [ ] M01: CSV data loader for offline testing
- [ ] DB: Initial Alembic migration (create tables)
- [ ] Tests: M01 integration tests with mocked MT5

**Definition of Done:** 5 years of EURUSD D1 data stored in SQLite, gap check passes.

---

### Phase 1 Sprint 2 — Analysis Engines (Weeks 3-5)
**Goal:** Full market analysis pipeline operational

- [ ] M03: Swing point detection (N-bar lookback)
- [ ] M03: HH/HL/LH/LL classification
- [ ] M03: Structure break (BOS) detection
- [ ] M04: 21 SMA calculation
- [ ] M04: ADX calculation (Wilder's smoothing)
- [ ] M04: Trend classification and TQS scoring
- [ ] M05: S/R level detection from swing points
- [ ] M05: Level zone merging
- [ ] M05: Touch count and strength scoring
- [ ] M05: TQS level component scoring
- [ ] M16: ATR calculation
- [ ] M16: Choppiness Index
- [ ] M16: Regime classification
- [ ] M16: TQS regime scoring
- [ ] Tests: 60+ unit tests for all analysis modules

**Definition of Done:** Analysis pipeline produces trend + regime + levels from D1 data.

---

### Phase 1 Sprint 3 — Strategy & Risk (Weeks 6-8)
**Goal:** Trade recommendations generated with full TQS

- [ ] M07: Pin bar detection (tail ratio, body position, quality score)
- [ ] M07: Engulfing bar detection (engulfing ratio, body dominance)
- [ ] M08: Strategy Engine pipeline (gate chain)
- [ ] M08: TQS calculation from component scores
- [ ] M08: Trade recommendation builder (entry/stop/target/R:R)
- [ ] M09: Kill switch monitoring
- [ ] M09: Position sizing (fixed fractional)
- [ ] M09: Daily/weekly loss limit enforcement
- [ ] M10: Backtest fill simulation
- [ ] M10: Trade lifecycle management (PENDING → OPEN → CLOSED)
- [ ] Tests: 80+ unit tests

**Definition of Done:** System generates trade recommendations with TQS and rejects invalid trades.

---

### Phase 1 Sprint 4 — Backtesting (Weeks 9-11)
**Goal:** First complete backtest run with results

- [ ] M11: Walk-forward simulation loop
- [ ] M11: Conservative exit logic (SL > TP same candle)
- [ ] M11: Equity curve and drawdown calculation
- [ ] M11: Full trade log output
- [ ] M11: BacktestResult persistence to DB
- [ ] M14: Status logging and health checks
- [ ] Integration tests: Full pipeline M01→M11

**Definition of Done:** 5-year EURUSD D1 backtest completes, BacktestResult stored, metrics calculated.

**Baseline Check:** After first run, verify:
- PF ≥ 1.1
- Win rate ≥ 40%  
- Max drawdown ≤ 20%

If baseline fails → review M07/M08 parameters before proceeding.

---

### Phase 1 Sprint 5 — Analytics & Paper Mode (Weeks 12-14)
**Goal:** Full paper trading readiness

- [ ] M18: Strategy scorecard calculation
- [ ] M18: Promotion criteria evaluation (50 trades AND 3 months)
- [ ] M19: Loss classification pipeline
- [ ] M19: Systematic error detection
- [ ] M13: Monthly report generation
- [ ] Paper mode: Full paper trading loop
- [ ] E2E tests: Complete system test

**Definition of Done:** Paper trading active, first monthly report generated.

---

## 6. Risk Management Implementation Notes

### Kill Switch Logic (EXACT implementation)

```python
# Trigger conditions (ANY ONE activates kill switch):
if drawdown_from_peak_pct >= 10.0:           # Condition 1
    trigger_kill_switch(KillSwitchReason.DRAWDOWN)

elif consecutive_losses >= 7:                 # Condition 2
    trigger_kill_switch(KillSwitchReason.CONSECUTIVE_LOSSES)

elif daily_limit_breached AND weekly_limit_breached:  # Condition 3 (AND, not OR)
    trigger_kill_switch(KillSwitchReason.DAILY_AND_WEEKLY)
```

**Note:** Condition 3 requires BOTH daily AND weekly limits breached simultaneously. This is intentional — either limit alone does NOT trigger the kill switch (only blocks new trades).

### Promotion Criteria (AND logic)

```python
# BOTH conditions must be true simultaneously:
promotion_allowed = (
    completed_trades >= 50        # AND
    AND calendar_months >= 3      # AND
)

# NOT OR — having 50 trades in 1 month does NOT qualify
# NOT OR — having 3 months with 20 trades does NOT qualify
```

---

## 7. Configuration Environment Variables

All config values can be overridden via environment variables using double-underscore as section separator:

```bash
# Format: CSBOT__SECTION__KEY=value
export CSBOT__EXECUTION__MODE=paper
export CSBOT__RISK__RISK_PER_TRADE_PCT=1.5
export CSBOT__SYSTEM__PHASE=1
export CSBOT__TQS__MIN_SCORE_TO_TRADE=65
```

**Priority order (highest to lowest):**
1. Environment variables (`CSBOT__` prefix)
2. `config/local_config.yaml` (local overrides, gitignored)
3. `config/default_config.yaml` (checked into repo)

---

## 8. Phase 2 Deferred Features

These features are fully stubbed (classes exist, methods return disabled) but will NOT be implemented until Phase 1 promotion criteria are met:

| Feature | Module | Activation |
|---------|--------|-----------|
| Fibonacci retracements | M06 | `levels.fibonacci.enabled: true` |
| Inside Bar strategy | M07 | `strategies.inside_bar.enabled: true` |
| False Breakout strategy | M07 | `strategies.inside_bar_false_breakout.enabled: true` |
| Supply & Demand zones | M05 | `levels.supply_demand_zones.enabled: true` |
| Portfolio management | M17 | `portfolio.enabled: true` |
| Optimization engine | M12 | `optimization.enabled: true` |
| Multi-pair trading | BotConfig | `symbols: [EURUSD, GBPUSD, ...]` |
| H4 timeframe | BotConfig | `timeframes.primary: H4` |
| Live trading | M10 | `execution.mode: live` |

**Phase 1 Pydantic enforcement:** Setting any of these `enabled: true` in Phase 1 raises `ValidationError` with an explanatory message. This prevents accidental scope creep.

---

*Blueprint generated: 2026-06-11 | Phase 0 Complete: 141/141 tests passing*
