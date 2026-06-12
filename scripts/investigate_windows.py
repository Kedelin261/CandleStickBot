"""
Sprint 14 — SR Engine window investigation.
Tests progressively larger lookback windows to find minimum needed for
M05 to produce S/R levels AND for trades to be generated.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import csv
from src.data.types import CandleData
from src.analysis.sr_engine import SREngine
from src.analysis.trend_detection import TrendDetector
from src.analysis.market_regime import MarketRegimeEngine

# ── Load CSV ─────────────────────────────────────────────────────────────────
candles = []
with open('data/EURUSD_D1_2014_2026.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        candles.append(CandleData(
            timestamp=row['date'],
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=float(row['volume']),
            symbol='EURUSD',
            timeframe='D1',
        ))

print(f"Loaded {len(candles)} candles\n")

sr_engine  = SREngine()
td_engine  = TrendDetector()
mr_engine  = MarketRegimeEngine()

print(f"{'Window':>8} | {'Positions':>9} | {'SR>0':>14} | {'Tradeable':>14} | {'Trending':>14} | {'All-3-gates':>14}")
print("-" * 90)

SAMPLE_STEP = 5   # every 5th candle — good balance of speed vs coverage

for window in [50, 100, 150, 200, 250, 300, 400, 500]:
    sr_ok = 0
    trend_ok = 0
    regime_ok = 0
    all3 = 0
    total = 0

    for i in range(window, len(candles), SAMPLE_STEP):
        ctx = candles[i - window: i]

        sr_result  = sr_engine.find_levels(ctx)
        td_result  = td_engine.analyze(ctx)
        mr_result  = mr_engine.analyze(ctx)

        has_sr    = (len(sr_result.support_levels) + len(sr_result.resistance_levels)) > 0
        tradeable = bool(getattr(td_result, 'tradeable', False))

        from src.analysis.market_regime import RegimeType
        regime_val = getattr(mr_result, 'regime', None)
        is_trend  = (regime_val == RegimeType.TRENDING)

        total += 1
        if has_sr:    sr_ok    += 1
        if tradeable: trend_ok += 1
        if is_trend:  regime_ok+= 1
        if has_sr and tradeable and is_trend:
            all3 += 1

    def pct(n, t):
        return f"{n}/{t} ({100*n/t:.1f}%)" if t else "0/0 (0%)"

    print(f"{window:>8} | {total:>9} | {pct(sr_ok,total):>20} | {pct(trend_ok,total):>20} | {pct(regime_ok,total):>20} | {pct(all3,total):>20}")

print()
print("Done — use the 'All-3-gates' column to pick the best window for BacktestRunner.")
