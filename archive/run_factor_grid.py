"""
// run_factor_grid.py
// Purpose:
// 1) Run all ON/OFF combinations for factor_long_only_backtest.py
// 2) Collect metrics (Months, CAGR, Vol, Sharpe, MaxDD)
// 3) Save one comparison table to results/factor_variations_summary.csv
// 4) Print top configurations ranked by lower drawdown / higher Sharpe
"""

import itertools
import math
import os
import re
import subprocess

import pandas as pd


def parse_pct(label: str, text: str) -> float:
    mt = re.search(label + r"\s*([-+]?[0-9]*\.?[0-9]+)%", text)
    return float(mt.group(1)) / 100 if mt else math.nan


def parse_num(label: str, text: str) -> float:
    mt = re.search(label + r"\s*([-+]?[0-9]*\.?[0-9]+)", text)
    return float(mt.group(1)) if mt else math.nan


def main():
    base = [
        "python",
        "factor_long_only_backtest.py",
        "--start", "2018-01-01",
        "--min-liq", "1000000",
        "--top", "0.25",
        "--cost-bps", "15",
    ]

    rows = []
    combos = []
    for reg in [0, 1]:
        for v, q, i, m in itertools.product([0, 1], [0, 1], [0, 1], [0, 1]):
            if (v, q, i, m) == (0, 0, 0, 0):
                continue
            combos.append((reg, v, q, i, m))

    print(f"Running {len(combos)} variations...")

    for idx, (reg, v, q, i, m) in enumerate(combos, 1):
        cmd = base.copy()
        if reg:
            cmd.append("--use-regime")
        if v:
            cmd.append("--use-value")
        if q:
            cmd.append("--use-quality")
        if i:
            cmd.append("--use-investment")
        if m:
            cmd.append("--use-momentum")

        p = subprocess.run(cmd, capture_output=True, text=True)
        out = (p.stdout or "") + "\n" + (p.stderr or "")

        rows.append(
            {
                "regime": reg,
                "value": v,
                "quality": q,
                "investment": i,
                "momentum": m,
                "months": parse_num("Months:", out),
                "cagr": parse_pct("Net CAGR:", out),
                "vol": parse_pct("Net Vol:", out),
                "sharpe": parse_num("Net Sharpe:", out),
                "mdd": parse_pct("Net MaxDD:", out),
                "ok": "Saved returns:" in out,
            }
        )
        print(f"[{idx:02d}/{len(combos)}] reg={reg} v={v} q={q} i={i} m={m} done")

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out_file = "results/factor_variations_summary.csv"
    df.to_csv(out_file, index=False)

    print("\nSaved:", out_file)
    print("\nTop 15 by lower drawdown / higher Sharpe:")
    print(df.sort_values(["mdd", "sharpe"], ascending=[False, False]).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
