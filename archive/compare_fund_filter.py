"""
compare_fund_filter.py
----------------------
Compare 4sectors.py backtest results WITH vs WITHOUT the fundamental filter.

Usage:
  1. Run 4sectors.py with FUNDAMENTAL_FILTER_ENABLED = True  → backtest_results.csv
     cp backtest_results.csv backtest_results_fund_on.csv
  2. Run 4sectors.py with FUNDAMENTAL_FILTER_ENABLED = False → backtest_results.csv
  3. python compare_fund_filter.py

Or pass custom paths:
  python compare_fund_filter.py --on backtest_results_fund_on.csv --off backtest_results.csv
"""

import argparse
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def summary_row(df: pd.DataFrame) -> pd.Series:
    return df[df["type"] == "summary"].iloc[0]


def annual_df(df: pd.DataFrame) -> pd.DataFrame:
    a = df[df["type"] == "annual"][["label", "total_return_pct"]].copy()
    a["label"] = a["label"].astype(int)
    return a.set_index("label")


def print_table(on: pd.DataFrame, off: pd.DataFrame):
    s_on  = summary_row(on)
    s_off = summary_row(off)

    print("\n" + "=" * 65)
    print(f"{'Metric':<28} {'Fund ON':>10} {'Fund OFF':>10} {'Delta':>10}")
    print("=" * 65)

    metrics = [
        ("Total return (%)",    "total_return_pct"),
        ("Ann. return (%)",     "ann_return_pct"),
        ("Sharpe",              "sharpe"),
        ("Max drawdown (%)",    "max_dd_pct"),
        ("Time invested (%)",   "time_invested_pct"),
        ("N trades",            "n_trades"),
        ("Win rate (%)",        "win_rate_pct"),
        ("Avg winner (%)",      "avg_winner_pct"),
        ("Avg loser (%)",       "avg_loser_pct"),
    ]
    for label, col in metrics:
        try:
            v_on  = float(s_on[col])
            v_off = float(s_off[col])
            delta = v_on - v_off
            sign  = "+" if delta >= 0 else ""
            print(f"  {label:<26} {v_on:>10.2f} {v_off:>10.2f} {sign}{delta:>9.2f}")
        except Exception:
            pass

    print("=" * 65)

    # Annual breakdown
    a_on  = annual_df(on).rename(columns={"total_return_pct": "on"})
    a_off = annual_df(off).rename(columns={"total_return_pct": "off"})
    ann   = a_on.join(a_off, how="outer").sort_index()
    ann["delta"] = ann["on"] - ann["off"]

    print(f"\n{'Year':<6} {'Fund ON':>9} {'Fund OFF':>9} {'Delta':>9}  {'Winner'}")
    print("-" * 46)
    for year, row in ann.iterrows():
        on_v  = row.get("on",    float("nan"))
        off_v = row.get("off",   float("nan"))
        delta = row.get("delta", float("nan"))
        winner = ""
        if not pd.isna(delta):
            winner = "ON  +" if delta > 0 else ("OFF +" if delta < 0 else "TIE")
        sign = "+" if delta > 0 else ""
        print(f"  {year:<4} {on_v:>9.1f} {off_v:>9.1f} {sign}{delta:>8.1f}  {winner}")

    # Count wins
    on_wins  = (ann["delta"] > 0).sum()
    off_wins = (ann["delta"] < 0).sum()
    print("-" * 46)
    print(f"  Fund ON wins: {on_wins} years  |  Fund OFF wins: {off_wins} years")

    # Sector breakdown
    print("\n── Sector breakdown ──────────────────────────────────────────")
    sectors_on  = on[on["type"]  == "sector"].set_index("label")
    sectors_off = off[off["type"] == "sector"].set_index("label")

    sec_cols = ["end_capital_M", "n_trades", "win_rate_pct"]
    for sec in sectors_on.index.union(sectors_off.index):
        print(f"\n  {sec}")
        for col in sec_cols:
            try:
                v_on  = float(sectors_on.loc[sec, col])
                v_off = float(sectors_off.loc[sec, col])
                delta = v_on - v_off
                sign  = "+" if delta >= 0 else ""
                print(f"    {col:<20} ON={v_on:>10.2f}  OFF={v_off:>10.2f}  Δ={sign}{delta:.2f}")
            except Exception:
                pass

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--on",  default=os.path.join(BASE, "backtest_results_fund_on.csv"))
    parser.add_argument("--off", default=os.path.join(BASE, "backtest_results.csv"))
    args = parser.parse_args()

    for p, label in [(args.on, "--on"), (args.off, "--off")]:
        if not os.path.exists(p):
            print(f"ERROR: {label} file not found: {p}")
            print("  Run 4sectors.py with the appropriate FUNDAMENTAL_FILTER_ENABLED setting first.")
            return

    on  = load(args.on)
    off = load(args.off)
    print(f"  ON  file : {args.on}")
    print(f"  OFF file : {args.off}")
    print_table(on, off)


if __name__ == "__main__":
    main()
