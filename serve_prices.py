"""
serve_prices.py — Local price server for backtest_dashboard.html.

The dashboard fetches http://localhost:8765/price/{TICKER} and expects a JSON
array of {date, open, high, low, close, volume}, with date as "YYYY-MM-DD".

Two things this handles that a naive json.dumps does NOT:

  1. NaN. Python's json module emits bare `NaN`, which is NOT valid JSON, and
     the browser dies with:
         Unexpected token 'N', ..."open": NaN...
     Several parquets genuinely carry NaN open values (ASG has 55). Rows whose
     OHLC cannot be repaired are dropped; a missing `open` alone is backfilled
     from the previous close, which is what a chart wants anyway.

  2. CORS. The dashboard is opened from file:// or another origin, so the
     browser blocks the response without Access-Control-Allow-Origin.

Also drops close<=0 rows — the zero-price glitch that corrupted several
backtests in this project (it makes candles collapse to the axis).

Usage:
    python serve_prices.py            # port 8765
    python serve_prices.py --port N
Then open backtest_dashboard.html.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PRICE_DIR = os.path.join(BASE, "data", "price")
PORT = 8765

_cache: dict[str, list] = {}


def load_ticker(ticker: str):
    """Return a JSON-safe list of OHLCV dicts, or None if unavailable."""
    ticker = ticker.upper().strip()
    if ticker in _cache:
        return _cache[ticker]

    path = os.path.join(PRICE_DIR, f"{ticker}.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        print(f"  [error] {ticker}: {exc}")
        return None

    df.columns = [c.strip().lower() for c in df.columns]
    date_col = "time" if "time" in df.columns else "date"
    if date_col not in df.columns or "close" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    # zero/negative closes are bad data, not real bars
    df = df[df["close"] > 0]
    if df.empty:
        return None

    for col in ("open", "high", "low"):
        if col not in df.columns:
            df[col] = np.nan
        # Non-positive OHLC is bad data, not a real price. Treat it as MISSING
        # so the repair below fills it, rather than letting a 0.0 survive:
        # min(low, open, close) would otherwise happily keep a zero low and
        # draw a candle stretching to the axis (SSI's first bar did exactly
        # this: low=0.0 against close=6.66).
        df[col] = df[col].where(df[col] > 0, np.nan)

    # A missing open is recoverable: use the prior close, else this bar's close.
    df["open"] = df["open"].fillna(df["close"].shift(1)).fillna(df["close"])
    # high/low must bracket the bar; repair rather than drop where possible.
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    if "volume" not in df.columns:
        df["volume"] = 0
    df["volume"] = df["volume"].fillna(0)

    # anything still non-finite cannot be charted
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["open", "high", "low", "close"])
    if df.empty:
        return None

    out = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "open": round(float(o), 3),
            "high": round(float(h), 3),
            "low": round(float(lo), 3),
            "close": round(float(c), 3),
            "volume": int(v) if np.isfinite(v) else 0,
        }
        for d, o, h, lo, c, v in zip(
            df["date"], df["open"], df["high"], df["low"], df["close"], df["volume"]
        )
    ]
    _cache[ticker] = out
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload, content_type="application/json"):
        # allow_nan=False turns any residual NaN into a loud error here rather
        # than silently emitting invalid JSON to the browser
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        path = unquote(urlparse(self.path).path)

        if path in ("/", "/health"):
            n = len([f for f in os.listdir(PRICE_DIR) if f.endswith(".parquet")]) \
                if os.path.isdir(PRICE_DIR) else 0
            return self._send(200, {"status": "ok", "tickers_available": n,
                                    "cached": len(_cache)})

        if path.startswith("/price/"):
            ticker = path[len("/price/"):]
            try:
                data = load_ticker(ticker)
            except Exception as exc:
                print(f"  [error] {ticker}: {exc}")
                return self._send(500, {"error": str(exc)})
            if data is None:
                return self._send(404, {"error": f"no price data for {ticker}"})
            return self._send(200, data)

        self._send(404, {"error": "unknown path", "path": path})

    def log_message(self, fmt, *args):
        sys.stdout.write("  " + fmt % args + "\n")


def main():
    port = PORT
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    if not os.path.isdir(PRICE_DIR):
        print(f"price directory not found: {PRICE_DIR}")
        sys.exit(1)
    n = len([f for f in os.listdir(PRICE_DIR) if f.endswith(".parquet")])
    print(f"serving {n} tickers from {PRICE_DIR}")
    print(f"listening on http://localhost:{port}   (Ctrl+C to stop)")
    print(f"  health: http://localhost:{port}/health")
    print(f"  sample: http://localhost:{port}/price/SSI")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
