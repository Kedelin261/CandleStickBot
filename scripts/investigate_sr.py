"""
Sprint 14 SR Engine investigation — test progressively larger lookback windows.
Matches the EXACT pipeline call: sr.analyze(context_candles) with no swing args.
Goal: find minimum window where M05 begins producing S/R levels.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import csv
from src.data.types import CandleData
from src.analysis.sr_engine import SREngine
from src.analysis.trend_detection import TrendDetector
from src.analysis.market_regime import MarketRegimeEngine, RegimeType
from src.analysis.market_structure import MarketStructureAnalyzer

# Load CSV
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

print(f"Loaded {len(candles)} candles")
print()

sr_engine   = SREngine(pip_size=0.0001)
td_engine   = TrendDetector()
mr_engine   = MarketRegimeEngine()
ms_analyzer = MarketStructureAnalyzer()

print(f"{'Window':>8} | {'SR>0':>22} | {'Tradeable':>20} | {'TRENDING':>20} | {'All-3-gates':>20}")
print("-" * 100)

sample_step = 5  # sample every 5th candle across full dataset

for window in [50, 100, 150, 200, 300, 500]:
    sr_ok = 0; trend_ok = 0; regime_ok = 0; all3 = 0; total = 0
    sr_total_levels = 0

    for i in range(window, len(candles), sample_step):
        ctx = candles[i - window: i]

        try:
            structure_a = ms_analyzer.analyze(ctx)
            td_result   = td_engine.analyze(ctx, market_structure=structure_a.to_market_structure())
            sr_result   = sr_engine.analyze(ctx)
            mr_result   = mr_engine.analyze(ctx, adx=td_result.adx)
        except Exception as e:
            continue

        n_levels  = len(sr_result.support_levels) + len(sr_result.resistance_levels)
        has_sr    = n_levels > 0
        tradeable = getattr(td_result, 'tradeable', False)
        is_trend  = getattr(mr_result, 'regime', None) == RegimeType.TRENDING

        total      += 1
        sr_total_levels += n_levels
        if has_sr:                               sr_ok     += 1
        if tradeable:                            trend_ok  += 1
        if is_trend:                             regime_ok += 1
        if has_sr and tradeable and is_trend:    all3      += 1

    def pct(n, t=total):
        return f"{n}/{t} ({100*n/max(t,1):.1f}%)"

    avg_levels = sr_total_levels / max(total, 1)
    print(f"{window:>8} | {pct(sr_ok):>28} | {pct(trend_ok):>26} | {pct(regime_ok):>26} | {pct(all3):>24}  avg_sr_levels={avg_levels:.2f}")

print()
print("NOTE: All-3-gates = SR levels exist AND trend tradeable AND regime TRENDING")
print("      This is necessary but not sufficient for a trade — also needs pattern + level proximity.")
