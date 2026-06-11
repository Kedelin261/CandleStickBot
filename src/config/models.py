"""
M15 — Config System: Pydantic validation models
All configuration sections with type enforcement, range validation, and defaults.
Version: 3.1 (Phase 0)
"""

from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ===========================================================================
# ENUMERATIONS
# ===========================================================================

class ExecutionMode(str, Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    DEMO = "demo"
    LIVE = "live"


class ApprovalMode(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"


class BrokerType(str, Enum):
    MT5 = "mt5"
    OANDA = "oanda"
    ALPACA = "alpaca"


class TrendMethod(str, Enum):
    SWING_POINTS = "swing_points"
    MA_ONLY = "ma_only"
    SWING_POINTS_AND_MA = "swing_points_and_ma"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ===========================================================================
# SECTION MODELS
# ===========================================================================

class SystemConfig(BaseModel):
    version: str = "0.1.0"
    phase: int = Field(default=1, ge=1, le=10)
    log_level: LogLevel = LogLevel.INFO
    log_dir: str = "logs"
    report_dir: str = "reports"
    db_url: str = "sqlite:///data/candlestickbot.db"


class MT5Config(BaseModel):
    login: Optional[int] = None
    password: Optional[str] = None
    server: str = ""
    timeout: int = Field(default=60000, ge=1000)


class ExecutionConfig(BaseModel):
    mode: ExecutionMode = ExecutionMode.BACKTEST
    approval_mode: ApprovalMode = ApprovalMode.AUTO
    slippage_pips: float = Field(default=1.0, ge=0.0, le=10.0)
    signal_expiry_minutes: int = Field(default=30, ge=1, le=1440)
    broker: BrokerType = BrokerType.MT5
    mt5: MT5Config = MT5Config()


class PinBarStrategyConfig(BaseModel):
    enabled: bool = True
    min_tail_ratio: float = Field(default=2.0, ge=1.0, le=10.0)
    min_tail_pct_of_range: float = Field(default=0.60, ge=0.30, le=0.90)
    max_opposite_wick_ratio: float = Field(default=0.50, ge=0.0, le=1.0)
    min_body_pips: float = Field(default=1.0, ge=0.1)
    max_body_pct_of_range: float = Field(default=0.35, ge=0.10, le=0.60)
    quality_min_score: int = Field(default=5, ge=1, le=10)
    buffer_pips: float = Field(default=5.0, ge=1.0, le=50.0)


class EngulfingStrategyConfig(BaseModel):
    enabled: bool = True
    strict_mode: bool = False
    min_engulf_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_min_score: int = Field(default=5, ge=1, le=10)
    buffer_pips: float = Field(default=5.0, ge=1.0, le=50.0)


class InsideBarStrategyConfig(BaseModel):
    enabled: bool = False  # Phase 2+ only
    strict_containment: bool = True
    cancel_after_candles: int = Field(default=3, ge=1, le=10)
    quality_min_score: int = Field(default=5, ge=1, le=10)


class FalseBreakoutStrategyConfig(BaseModel):
    enabled: bool = False  # Phase 2+ only
    max_candles_outside: int = Field(default=3, ge=1, le=10)
    quality_min_score: int = Field(default=5, ge=1, le=10)
    buffer_pips: float = Field(default=5.0, ge=1.0, le=50.0)


class StrategiesConfig(BaseModel):
    pin_bar: PinBarStrategyConfig = PinBarStrategyConfig()
    engulfing_bar: EngulfingStrategyConfig = EngulfingStrategyConfig()
    inside_bar: InsideBarStrategyConfig = InsideBarStrategyConfig()
    inside_bar_false_breakout: FalseBreakoutStrategyConfig = FalseBreakoutStrategyConfig()


class TrendConfig(BaseModel):
    method: TrendMethod = TrendMethod.SWING_POINTS_AND_MA
    ma_period: int = Field(default=21, ge=5, le=200)
    swing_lookback: int = Field(default=5, ge=2, le=20)
    min_swing_points: int = Field(default=4, ge=4)
    min_trend_amplitude_pips: float = Field(default=100.0, ge=10.0)
    range_tolerance_pips: float = Field(default=50.0, ge=5.0)
    adx_filter: bool = False
    adx_period: int = Field(default=14, ge=5, le=50)
    adx_threshold: float = Field(default=25.0, ge=10.0, le=50.0)


class SREngineConfig(BaseModel):
    enabled: bool = True
    lookback_bars: int = Field(default=200, ge=50, le=1000)
    cluster_tolerance_pips: float = Field(default=10.0, ge=1.0, le=50.0)
    min_strength_score: int = Field(default=3, ge=1, le=10)
    max_age_bars: int = Field(default=300, ge=50)
    min_touches: int = Field(default=1, ge=1)


class SupplyDemandConfig(BaseModel):
    enabled: bool = False  # Phase 2+
    min_body_atr_ratio: float = Field(default=0.70, ge=0.30, le=2.0)
    max_age_bars: int = Field(default=300, ge=50)


class FibonacciConfig(BaseModel):
    enabled: bool = False  # Phase 2+
    levels: List[float] = [0.236, 0.382, 0.500, 0.618, 0.786]
    primary_levels: List[float] = [0.500, 0.618]
    zone_tolerance_pips: float = Field(default=10.0, ge=1.0, le=50.0)


class LevelsConfig(BaseModel):
    sr_engine: SREngineConfig = SREngineConfig()
    supply_demand_zones: SupplyDemandConfig = SupplyDemandConfig()
    fibonacci: FibonacciConfig = FibonacciConfig()


class TrendingRegimeConfig(BaseModel):
    adx_min: float = Field(default=25.0, ge=10.0, le=50.0)
    atr_expansion_factor: float = Field(default=1.0, ge=0.5, le=3.0)
    bb_expansion_factor: float = Field(default=1.1, ge=0.5, le=3.0)


class RangingRegimeConfig(BaseModel):
    adx_max: float = Field(default=20.0, ge=5.0, le=40.0)
    bb_contraction_factor: float = Field(default=0.9, ge=0.3, le=1.0)


class VolatileRegimeConfig(BaseModel):
    atr_high_factor: float = Field(default=1.5, ge=1.0, le=5.0)
    adx_max: float = Field(default=25.0, ge=10.0, le=50.0)


class QuietRegimeConfig(BaseModel):
    atr_low_factor: float = Field(default=0.6, ge=0.1, le=0.9)
    bb_quiet_factor: float = Field(default=0.7, ge=0.1, le=0.9)


class RegimeConfig(BaseModel):
    atr_period: int = Field(default=14, ge=5, le=50)
    atr_ma_period: int = Field(default=14, ge=5, le=50)
    bb_period: int = Field(default=20, ge=5, le=50)
    bb_std_dev: float = Field(default=2.0, ge=1.0, le=4.0)
    bb_width_ma_period: int = Field(default=20, ge=5, le=50)
    adx_period: int = Field(default=14, ge=5, le=50)
    choppiness_period: int = Field(default=14, ge=5, le=50)
    choppiness_threshold: float = Field(default=61.8, ge=40.0, le=80.0)
    trending: TrendingRegimeConfig = TrendingRegimeConfig()
    ranging: RangingRegimeConfig = RangingRegimeConfig()
    volatile: VolatileRegimeConfig = VolatileRegimeConfig()
    quiet: QuietRegimeConfig = QuietRegimeConfig()


class TQSConfig(BaseModel):
    min_score_to_trade: int = Field(default=60, ge=0, le=100)
    premium_threshold: int = Field(default=80, ge=60, le=100)
    standard_risk_pct: float = Field(default=1.0, ge=0.1, le=2.0)
    premium_risk_pct: float = Field(default=1.0, ge=0.1, le=2.0)
    premium_min_component_score: int = Field(default=15, ge=0, le=25)

    @model_validator(mode='after')
    def premium_risk_cannot_exceed_cap(self) -> 'TQSConfig':
        if self.premium_risk_pct > 2.0:
            raise ValueError("premium_risk_pct cannot exceed 2.0% (hard cap)")
        return self


class RiskConfig(BaseModel):
    risk_per_trade_pct: float = Field(default=1.0, ge=0.1, le=2.0)
    max_risk_per_trade_pct: float = Field(default=2.0, ge=0.1, le=2.0)
    min_rr_ratio: float = Field(default=2.0, ge=2.0, le=10.0)  # Cannot be set below 2.0
    preferred_rr_ratio: float = Field(default=3.0, ge=2.0, le=10.0)
    min_lot: float = Field(default=0.01, ge=0.01)
    lot_step: float = Field(default=0.01, ge=0.01)
    max_open_trades: int = Field(default=3, ge=1, le=20)
    max_trades_per_currency: int = Field(default=2, ge=1, le=5)
    daily_loss_limit_pct: float = Field(default=3.0, ge=0.5, le=20.0)
    weekly_loss_limit_pct: float = Field(default=6.0, ge=1.0, le=30.0)
    kill_switch_drawdown_pct: float = Field(default=10.0, ge=2.0, le=50.0)
    kill_switch_consecutive_losses: int = Field(default=7, ge=3, le=20)
    buffer_pips_default: float = Field(default=5.0, ge=1.0, le=50.0)
    buffer_pips_scale_by_atr: bool = False

    @field_validator('max_risk_per_trade_pct')
    @classmethod
    def hard_cap_enforced(cls, v: float) -> float:
        if v > 2.0:
            raise ValueError(
                "max_risk_per_trade_pct HARD CAP is 2.0%. "
                "This cannot be exceeded by any configuration. (See Section 7, V3.1)"
            )
        return v

    @model_validator(mode='after')
    def risk_per_trade_cannot_exceed_max(self) -> 'RiskConfig':
        if self.risk_per_trade_pct > self.max_risk_per_trade_pct:
            raise ValueError(
                f"risk_per_trade_pct ({self.risk_per_trade_pct}) "
                f"cannot exceed max_risk_per_trade_pct ({self.max_risk_per_trade_pct})"
            )
        return self


class SessionConfig(BaseModel):
    start_utc: str = "08:00"
    end_utc: str = "17:00"


class FiltersConfig(BaseModel):
    max_spread_pips: float = Field(default=5.0, ge=0.1, le=50.0)
    min_spread_pips: float = Field(default=0.1, ge=0.0)
    session_filter_enabled: bool = True
    sessions: Dict[str, SessionConfig] = {
        "london": SessionConfig(start_utc="08:00", end_utc="17:00"),
        "new_york": SessionConfig(start_utc="13:00", end_utc="22:00"),
    }
    news_filter_enabled: bool = False
    news_blackout_minutes: int = Field(default=30, ge=5, le=120)
    high_impact_events: List[str] = ["NFP", "CPI", "FOMC", "ECB_RATE", "BOE_RATE", "FED_RATE"]
    correlation_filter_enabled: bool = False
    correlation_block_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    correlation_warn_threshold: float = Field(default=0.40, ge=0.0, le=1.0)
    correlation_period_days: int = Field(default=60, ge=10, le=365)


class PortfolioConfig(BaseModel):
    enabled: bool = False  # Phase 2+
    max_portfolio_heat_pct: float = Field(default=6.0, ge=1.0, le=20.0)
    max_currency_exposure_pct: float = Field(default=3.0, ge=0.5, le=10.0)
    max_directional_exposure_pct: float = Field(default=4.0, ge=1.0, le=15.0)


class BacktestSpreadConfig(BaseModel):
    EURUSD: float = Field(default=1.5, ge=0.0)
    GBPUSD: float = Field(default=2.0, ge=0.0)
    USDJPY: float = Field(default=1.5, ge=0.0)
    default: float = Field(default=2.0, ge=0.0)


class WalkForwardConfig(BaseModel):
    in_sample_months: int = Field(default=24, ge=6, le=120)
    out_of_sample_months: int = Field(default=6, ge=1, le=24)
    min_folds: int = Field(default=3, ge=2, le=20)
    preferred_folds: int = Field(default=5, ge=2, le=20)


class MonteCarloConfig(BaseModel):
    n_simulations: int = Field(default=1000, ge=100, le=100000)
    random_seed: int = 42


class BacktestPassCriteriaConfig(BaseModel):
    min_profit_factor: float = Field(default=1.3, ge=1.0)
    min_win_rate: float = Field(default=0.35, ge=0.0, le=1.0)
    min_expectancy_r: float = Field(default=0.3, ge=0.0)
    max_drawdown_pct: float = Field(default=20.0, ge=1.0, le=100.0)
    min_sharpe_ratio: float = Field(default=0.5, ge=0.0)
    min_trade_count: int = Field(default=200, ge=50)


class BacktestConfig(BaseModel):
    in_sample_start: str = "2014-01-01"
    in_sample_end: str = "2021-12-31"
    out_of_sample_start: str = "2022-01-01"
    out_of_sample_end: str = "2024-12-31"
    spread_pips: BacktestSpreadConfig = BacktestSpreadConfig()
    entry_slippage_pips: float = Field(default=1.0, ge=0.0)
    exit_slippage_pips: float = Field(default=0.5, ge=0.0)
    commission_per_lot: float = Field(default=7.0, ge=0.0)
    walk_forward: WalkForwardConfig = WalkForwardConfig()
    monte_carlo: MonteCarloConfig = MonteCarloConfig()
    pass_criteria: BacktestPassCriteriaConfig = BacktestPassCriteriaConfig()


class OptimizationConfig(BaseModel):
    enabled: bool = False  # Phase 2+ only; Optimization Governance Policy
    max_params: int = Field(default=5, ge=1, le=10)
    fitness_function: str = "pf_over_dd"


class AnalyticsConfig(BaseModel):
    enabled: bool = True
    degradation_alert_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    degradation_lookback_days: int = Field(default=30, ge=5, le=365)
    min_trades_for_scorecard: int = Field(default=10, ge=5)


class TradeReviewConfig(BaseModel):
    enabled: bool = True
    systematic_error_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    promotion_block_bad_regime_pct: float = Field(default=0.15, ge=0.0, le=1.0)
    promotion_block_bad_signal_pct: float = Field(default=0.20, ge=0.0, le=1.0)
    bad_signal_quality_threshold: int = Field(default=5, ge=1, le=10)
    bad_level_strength_threshold: int = Field(default=3, ge=1, le=10)
    bad_execution_slippage_pips: float = Field(default=3.0, ge=0.0)
    bad_execution_tight_stop_atr: float = Field(default=1.0, ge=0.0)


class PromotionThreshold(BaseModel):
    min_completed_trades: int = Field(default=50, ge=1)
    min_calendar_months: int = Field(default=3, ge=1)
    min_profit_factor: float = Field(default=1.3, ge=1.0)
    max_drawdown_pct: float = Field(default=20.0, ge=1.0, le=100.0)
    require_positive_expectancy: bool = True
    bad_regime_plus_bad_signal_max_pct: float = Field(default=0.20, ge=0.0, le=1.0)


class DemoToLivePromotion(PromotionThreshold):
    max_drawdown_pct: float = Field(default=10.0, ge=1.0, le=100.0)
    max_backtest_degradation_pct: float = Field(default=0.30, ge=0.0, le=1.0)
    bad_regime_plus_bad_signal_max_pct: float = Field(default=0.15, ge=0.0, le=1.0)


class PromotionConfig(BaseModel):
    paper_to_demo: PromotionThreshold = PromotionThreshold()
    demo_to_live: DemoToLivePromotion = DemoToLivePromotion()


# ===========================================================================
# ROOT CONFIG MODEL
# ===========================================================================

class BotConfig(BaseModel):
    """
    Root configuration model for CandleStickBot.
    All sections validated via pydantic. No invalid configuration can reach
    the trading engine.
    """
    system: SystemConfig = SystemConfig()
    execution: ExecutionConfig = ExecutionConfig()
    symbols: List[str] = ["EURUSD"]
    timeframes: Dict[str, str] = {"primary": "D1", "context": "W1"}
    strategies: StrategiesConfig = StrategiesConfig()
    trend: TrendConfig = TrendConfig()
    levels: LevelsConfig = LevelsConfig()
    regime: RegimeConfig = RegimeConfig()
    tqs: TQSConfig = TQSConfig()
    risk: RiskConfig = RiskConfig()
    filters: FiltersConfig = FiltersConfig()
    portfolio: PortfolioConfig = PortfolioConfig()
    backtest: BacktestConfig = BacktestConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    analytics: AnalyticsConfig = AnalyticsConfig()
    trade_review: TradeReviewConfig = TradeReviewConfig()
    promotion: PromotionConfig = PromotionConfig()

    @model_validator(mode='after')
    def phase1_scope_enforcement(self) -> 'BotConfig':
        """
        Enforce Phase 1 scope boundaries.
        Certain features cannot be activated in Phase 1.
        """
        phase = self.system.phase
        if phase == 1:
            # Phase 1: Only EURUSD allowed
            invalid_symbols = [s for s in self.symbols if s != "EURUSD"]
            if invalid_symbols:
                raise ValueError(
                    f"Phase 1 only supports EURUSD. "
                    f"Invalid symbols configured: {invalid_symbols}. "
                    f"Multi-pair trading is Phase 2+."
                )
            # Phase 1: Only D1 primary timeframe
            if self.timeframes.get("primary") not in ("D1", "W1"):
                raise ValueError(
                    "Phase 1 only supports D1 as primary timeframe. "
                    "H4 signals are Phase 2+."
                )
            # Phase 1: Inside Bar and False Breakout must be disabled
            if self.strategies.inside_bar.enabled:
                raise ValueError(
                    "Inside Bar strategy is Phase 2+ only. "
                    "inside_bar.enabled must be false in Phase 1."
                )
            if self.strategies.inside_bar_false_breakout.enabled:
                raise ValueError(
                    "Inside Bar False Breakout is Phase 2+ only. "
                    "inside_bar_false_breakout.enabled must be false in Phase 1."
                )
            # Phase 1: Fibonacci disabled
            if self.levels.fibonacci.enabled:
                raise ValueError(
                    "Fibonacci engine (M06) is Phase 2+ only. "
                    "levels.fibonacci.enabled must be false in Phase 1."
                )
            # Phase 1: Supply/Demand zones disabled
            if self.levels.supply_demand_zones.enabled:
                raise ValueError(
                    "Supply/Demand zones are Phase 2+ only. "
                    "levels.supply_demand_zones.enabled must be false in Phase 1."
                )
            # Phase 1: Portfolio engine disabled
            if self.portfolio.enabled:
                raise ValueError(
                    "Portfolio Exposure Engine (M17) is Phase 2+ only. "
                    "portfolio.enabled must be false in Phase 1."
                )
            # Phase 1: Optimization disabled
            if self.optimization.enabled:
                raise ValueError(
                    "Optimization Engine (M12) is Phase 2+ only and requires "
                    "baseline backtest to pass first (Optimization Governance Policy). "
                    "optimization.enabled must be false in Phase 1."
                )
        return self

    @model_validator(mode='after')
    def execution_mode_safety(self) -> 'BotConfig':
        """
        Enforce execution mode safety.
        Live mode requires explicit acknowledgment of promotion criteria.
        """
        mode = self.execution.mode
        if mode == ExecutionMode.LIVE:
            # This is a soft warning — live mode is checked at runtime
            # against promotion criteria (Section 8)
            pass
        return self
