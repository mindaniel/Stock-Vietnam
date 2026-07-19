"""
build_dashboard.py — Regenerate backtest_dashboard.html from any trades CSV.

The dashboard embeds its trades directly in the HTML (const ALL_TRADES = [...]),
so pointing the price server at it is not enough — the markers stay whatever
was baked in. This rewrites the six places that reference strategy data:

    1. const ALL_TRADES        the trade objects themselves
    2. const TICKER_SUMMARY    drives the ticker dropdown
    3. let activeStrategies    which strategies start toggled on
    4. const STRAT_NAMES       display labels
    5. const STRAT_COLORS      marker colours
    6. the three toggle buttons in the HTML body

Writes to a NEW file by default so the original is never destroyed.

Required CSV columns:
    ticker, strategy, entry_date, entry_price, exit_date, exit_price,
    return_pct, days_held, exit_reason, entry_volume, trend_at_entry,
    volatility_at_entry

Usage:
    python build_dashboard.py
    python build_dashboard.py --csv fi_strategy_trades.csv --out my_dash.html
    python build_dashboard.py --template backtest_dashboard.html
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

PALETTE = [
    {"buy": "#4ecca3", "sell": "#2e8b6e", "bg": "rgba(78,204,163,0.15)"},
    {"buy": "#e9c46a", "sell": "#b8962a", "bg": "rgba(233,196,106,0.15)"},
    {"buy": "#e76f51", "sell": "#b8452f", "bg": "rgba(231,111,81,0.15)"},
    {"buy": "#8ab6f9", "sell": "#4a7fd4", "bg": "rgba(138,182,249,0.15)"},
    {"buy": "#c792ea", "sell": "#9a5fc4", "bg": "rgba(199,146,234,0.15)"},
]

REQUIRED = ["ticker", "strategy", "entry_date", "entry_price", "exit_date",
            "exit_price", "return_pct", "days_held", "exit_reason",
            "entry_volume", "trend_at_entry", "volatility_at_entry"]

PRETTY = {
    "FI_FlowExit": "FI + Flow Exit",
    "FI_PriceExit": "FI + Price Exit",
    "PriceOnly": "Price Only",
}


def replace_const(html: str, decl: str, new_value: str, terminator: str = ";") -> str:
    """Replace `const NAME = <anything>;` with a new value, matching the FIRST
    top-level terminator so a 5.8MB embedded array is handled without regex
    backtracking."""
    start = html.find(decl)
    if start == -1:
        raise SystemExit(f"could not find `{decl}` in template")
    eq = html.index("=", start)
    depth, i, n = 0, eq + 1, len(html)
    while i < n:
        ch = html[i]
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == terminator and depth == 0:
            break
        i += 1
    return html[:eq + 1] + " " + new_value + html[i:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(BASE, "fi_strategy_trades.csv"))
    ap.add_argument("--template", default=os.path.join(BASE, "backtest_dashboard.html"))
    ap.add_argument("--out", default=os.path.join(BASE, "fi_dashboard.html"))
    a = ap.parse_args()

    if not os.path.exists(a.template):
        raise SystemExit(f"template not found: {a.template}")
    df = pd.read_csv(a.csv)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}")

    # NaN is not valid JSON and will break the browser parse with
    # "Unexpected token 'N'". Coerce before serialising.
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in ["entry_price", "exit_price", "return_pct", "volatility_at_entry"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).round(3)
    for c in ["days_held", "entry_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in ["ticker", "strategy", "entry_date", "exit_date", "exit_reason",
              "trend_at_entry"]:
        df[c] = df[c].astype(str)

    strategies = sorted(df["strategy"].unique())
    print(f"{len(df):,} trades | {df['ticker'].nunique()} tickers | strategies: {strategies}")

    trades = df[REQUIRED].to_dict(orient="records")

    summary = []
    for t, g in df.groupby("ticker"):
        summary.append({
            "ticker": t,
            "total_trades": int(len(g)),
            "win_rate": round(float((g["return_pct"] > 0).mean() * 100), 1),
            "avg_return": round(float(g["return_pct"].mean()), 1),
            "best_trade": round(float(g["return_pct"].max()), 1),
            "strategies": sorted(g["strategy"].unique().tolist()),
        })
    summary.sort(key=lambda s: s["ticker"])

    names = {s: PRETTY.get(s, s.replace("_", " ")) for s in strategies}
    colors = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(strategies)}
    active = {s: True for s in strategies}

    html = open(a.template, encoding="utf-8", errors="ignore").read()
    html = replace_const(html, "const ALL_TRADES",
                         json.dumps(trades, allow_nan=False))
    html = replace_const(html, "const TICKER_SUMMARY",
                         json.dumps(summary, allow_nan=False))
    html = replace_const(html, "let activeStrategies",
                         json.dumps(active, allow_nan=False))
    html = replace_const(html, "const STRAT_NAMES",
                         json.dumps(names, allow_nan=False))
    html = replace_const(html, "const STRAT_COLORS",
                         json.dumps(colors, allow_nan=False))

    # Rebuild the toggle buttons. The template hardcodes exactly three
    # (btnA/btnB/btnC); emit one per strategy instead.
    buttons = "\n        ".join(
        f'<button class="btn-strat active" id="btn{i}" '
        f"onclick=\"toggleStrategy('{s}')\">{names[s]}</button>"
        for i, s in enumerate(strategies))
    html = re.sub(
        r'<button class="btn-strat active" id="btn[ABC]"[^>]*>.*?</button>\s*',
        "", html, flags=re.S)
    html = html.replace("egies:</span>", "egies:</span>\n        " + buttons, 1)

    # toggleStrategy flips a button's `active` class by id; ids changed, so
    # make the lookup name-based to stay in sync.
    html = re.sub(r"document\.getElementById\('btn'\s*\+\s*[^)]+\)",
                  "document.querySelector(`[onclick=\"toggleStrategy('${s}')\"]`)",
                  html)

    # The sidebar's "By Strategy" panel iterates a HARDCODED strategy array.
    # Left alone it renders one empty row per old strategy name.
    strat_list = ", ".join(f"'{s}'" for s in strategies)
    html, n_sub = re.subn(r"\['A_TrendFollow',\s*'B_Contrarian',\s*'C_Combined'\]",
                          f"[{strat_list}]", html)
    if n_sub:
        print(f"  patched {n_sub} hardcoded strategy list(s) in the sidebar")

    leftover = re.findall(r"A_TrendFollow|B_Contrarian|C_Combined", html)
    if leftover:
        print(f"  WARNING: {len(leftover)} old strategy reference(s) still present")

    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)

    size = os.path.getsize(a.out) / 1e6
    print(f"wrote {a.out}  ({size:.1f} MB)")
    print("\nstrategy breakdown:")
    for s in strategies:
        g = df[df["strategy"] == s]
        print(f"  {names[s]:<18} {len(g):>6,} trades  "
              f"win {100*(g['return_pct']>0).mean():>5.1f}%  "
              f"avg {g['return_pct'].mean():>+6.2f}%")
    print("\nstart the price server, then open the file:")
    print("  python serve_prices.py")


if __name__ == "__main__":
    main()
