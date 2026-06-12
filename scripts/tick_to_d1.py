"""
Stream a FX-Data year-branch tarball (Dukascopy hourly tick CSVs) and
aggregate to D1 bars using the NY-17:00-close convention.

Tick rows: 'YYYY.MM.DD HH:MM:SS.mmm,bid,ask,bidVol,askVol' (GMT).
Bar construction (declared convention, applied uniformly):
  - bucket date = calendar date of (tick GMT time -> America/New_York + 7h)
    => each candle runs 17:00 NY -> 17:00 NY; Sunday-evening reopen folds
       into Monday; no Sunday bars.
  - prices = BID (MT4/MT5 charting convention)
  - open  = first tick of bucket, close = last tick, high/low = extremes
  - volume = tick count
  - buckets falling on Saturday, Dec 25, or Jan 1 are non-trading days and
    are not emitted (standard platform behaviour)
Output: one CSV row per bar to stdout-named file, plus a stats line.
"""
import sys, io, tarfile, json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

GMT = ZoneInfo("UTC")
NY  = ZoneInfo("America/New_York")

def bucket_for(gmt_date_str: str, hour: int):
    """Map a GMT (date, hour) to its NY-close trading-day bucket."""
    y, m, d = int(gmt_date_str[0:4]), int(gmt_date_str[5:7]), int(gmt_date_str[8:10])
    dt = datetime(y, m, d, hour, 30, tzinfo=GMT)  # mid-hour avoids edge ambiguity
    ny = dt.astimezone(NY)
    b  = (ny + timedelta(hours=7)).date()
    return b

def aggregate_year(stream, days: dict):
    """Consume a tar.gz byte stream; update {bucket_date: state}."""
    tf = tarfile.open(fileobj=stream, mode="r|gz")
    n_files = 0
    for member in tf:
        name = member.name
        if not name.endswith("_ticks.csv"):
            continue
        base = name.rsplit("/", 1)[-1]            # 2015-01-02--12h_ticks.csv
        date_part, hour_part = base.split("--")
        hour = int(hour_part[0:2])
        f = tf.extractfile(member)
        if f is None:
            continue
        try:
            df = pd.read_csv(io.BytesIO(f.read()), header=None,
                             usecols=[1], engine="c")
        except Exception:
            continue
        if df.empty:
            continue
        bids = df[1].to_numpy()
        b = bucket_for(date_part, hour)
        key = (date_part, hour)                    # global GMT ordering key
        st = days.get(b)
        if st is None:
            days[b] = {"first_key": key, "open": float(bids[0]),
                       "last_key": key,  "close": float(bids[-1]),
                       "high": float(bids.max()), "low": float(bids.min()),
                       "vol": int(len(bids))}
        else:
            if key < st["first_key"]:
                st["first_key"], st["open"] = key, float(bids[0])
            if key > st["last_key"]:
                st["last_key"], st["close"] = key, float(bids[-1])
            hi, lo = float(bids.max()), float(bids.min())
            if hi > st["high"]: st["high"] = hi
            if lo < st["low"]:  st["low"]  = lo
            st["vol"] += int(len(bids))
        n_files += 1
    return n_files

def emit(days: dict, out_path: str):
    skipped = []
    rows = 0
    with open(out_path, "w") as out:
        out.write("date,open,high,low,close,volume\n")
        for b in sorted(days):
            if b.weekday() == 5 or (b.month, b.day) in ((12, 25), (1, 1)):
                skipped.append((str(b), days[b]["vol"]))
                continue
            st = days[b]
            out.write(f"{b.isoformat()},{st['open']:.5f},{st['high']:.5f},"
                      f"{st['low']:.5f},{st['close']:.5f},{st['vol']}\n")
            rows += 1
    return rows, skipped

if __name__ == "__main__":
    out_path = sys.argv[1]
    days: dict = {}
    n = aggregate_year(sys.stdin.buffer, days)
    rows, skipped = emit(days, out_path)
    print(json.dumps({"files": n, "bars": rows,
                      "skipped_nontrading": skipped[:6],
                      "n_skipped": len(skipped)}))
