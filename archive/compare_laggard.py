"""
compare_laggard.py
==================
Tests the fund-style laggard exit rule:
  "If a stock isn't running after 2 weeks, sell it"

Compares 4 variants:
  OFF       — no laggard (baseline)
  FLAT_ONLY — flat laggard only  (20d < +3%)
  LOSS_ONLY — loss laggard only  (10d < -10%)
  BOTH      — both rules active (default config)

Prints year-by-year and sector breakdown to see if laggard exits
improve 2022 / reduce drawdown without hurting 2020-2021.
"""
import sys, os, importlib, importlib.util
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.abspath(__file__))

print("Loading 4sectors module...")
spec = importlib.util.spec_from_file_location("4sectors", os.path.join(BASE, "4sectors.py"))
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.modules["4sectors"] = mod
print(f"  Data loaded. Running 4 variants...\n")

VARIANTS = {
    "OFF":       dict(flat=None,  loss=None),
    "FLAT_ONLY": dict(flat=0.03,  loss=None),
    "LOSS_ONLY": dict(flat=None,  loss=0.10),
    "BOTH":      dict(flat=0.03,  loss=0.10),
}

all_results = {}

for label, cfg in VARIANTS.items():
    mod.LAGGARD_FLAT_THRESH = cfg["flat"]
    mod.LAGGARD_LOSS_THRESH = cfg["loss"]
    mod.VNINDEX_GATE        = "OFF"
    mod.VNINDEX_EXIT_ENABLED = False

    print(f"{'─'*60}")
    flat_str = f"+{cfg['flat']*100:.0f}%" if cfg["flat"] else "off"
    loss_str = f"-{cfg['loss']*100:.0f}%" if cfg["loss"] else "off"
    print(f"  LAGGARD: {label}  (flat={flat_str} after {mod.LAGGARD_FLAT_DAYS}d, "
          f"loss={loss_str} after {mod.LAGGARD_LOSS_DAYS}d)")
    print(f"{'─'*60}")
    mod.main()

    df = pd.read_csv(os.path.join(BASE, "backtest_results.csv"))
    summary = df[df["type"] == "summary"].iloc[0]
    annual  = df[df["type"] == "annual"].copy()
    sectors = df[df["type"] == "sector"].copy()

    all_results[label] = {
        "cagr":      summary["ann_return_pct"],
        "sharpe":    summary["sharpe"],
        "maxdd":     summary["max_dd_pct"],
        "total_ret": summary["total_return_pct"],
        "n_trades":  summary["n_trades"],
        "win_rate":  summary["win_rate_pct"],
        "annual":    annual[["label","ann_return_pct","max_dd_pct"]].set_index("label"),
        "sectors":   sectors[["label","end_capital_M","n_trades","win_rate_pct"]].set_index("label"),
    }

# ── Comparison table ─────────────────────────────────────────────────────────
COLS = list(VARIANTS.keys())
off  = all_results["OFF"]

print()
print("=" * 78)
print("  LAGGARD EXIT COMPARISON")
print(f"  Flat: +3% after 20d   Loss: -10% after 10d")
print("=" * 78)
print(f"  {'Metric':<22} {'OFF':>12} {'FLAT_ONLY':>12} {'LOSS_ONLY':>12} {'BOTH':>12}")
print(f"  {'-'*70}")
for key, label in [
    ("cagr",     "CAGR %"),
    ("sharpe",   "Sharpe"),
    ("maxdd",    "Max DD %"),
    ("total_ret","Total Return %"),
    ("n_trades", "# Trades"),
    ("win_rate", "Win Rate %"),
]:
    row = f"  {label:<22}"
    for c in COLS:
        v = all_results[c][key]
        row += f" {v:>12.2f}"
    print(row)

print()
print("  ANNUAL RETURNS (%)")
print(f"  {'Year':<6} {'OFF':>8} {'FLAT':>8} {'LOSS':>8} {'BOTH':>8}  "
      f"{'FL-OFF':>8} {'LO-OFF':>8} {'BO-OFF':>8}")
print(f"  {'-'*66}")
years = sorted(off["annual"].index)
for yr in years:
    try:
        vals = {c: float(all_results[c]["annual"].loc[yr,"ann_return_pct"]) for c in COLS}
    except Exception:
        continue
    if all(v == 0 for v in vals.values()):
        continue
    deltas = {c: vals[c] - vals["OFF"] for c in COLS if c != "OFF"}
    flags  = {c: "▲" if deltas[c]>1 else ("▼" if deltas[c]<-1 else " ") for c in deltas}
    print(f"  {yr:<6} {vals['OFF']:>8.1f} {vals['FLAT_ONLY']:>8.1f} "
          f"{vals['LOSS_ONLY']:>8.1f} {vals['BOTH']:>8.1f}  "
          f"{deltas['FLAT_ONLY']:>+7.1f}{flags['FLAT_ONLY']} "
          f"{deltas['LOSS_ONLY']:>+7.1f}{flags['LOSS_ONLY']} "
          f"{deltas['BOTH']:>+7.1f}{flags['BOTH']}")

print()
print("  SECTOR FINAL CAPITAL (M VND, starting 100M each)")
print(f"  {'Sector':<22} {'OFF':>12} {'FLAT_ONLY':>12} {'LOSS_ONLY':>12} {'BOTH':>12}")
print(f"  {'-'*62}")
for sec in ["Banks", "Basic Resources", "Food & Beverage", "Real Estate"]:
    row = f"  {sec:<22}"
    for c in COLS:
        try:
            v = float(all_results[c]["sectors"].loc[sec,"end_capital_M"])
        except Exception:
            v = 0.0
        row += f" {v:>12,.0f}"
    print(row)

print()
print("  ▲ = laggard exit improved  ▼ = reduced (>1pp)")
print("  FLAT: exit stocks still <+3% after 20 days held")
print("  LOSS: exit stocks still <-10% after 10 days held")
print("=" * 78)
