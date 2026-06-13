"""
compare_vn_gate.py
==================
Runs 4sectors backtest 3 times with different VN-Index gate settings:
  OFF  — no gate (baseline)
  SOFT — half size when VN-Index 63d return < -5%, skip if < -10%
  HARD — skip entry when VN-Index 63d return < -10%

Prints a side-by-side comparison of annual returns and sector PnL.
"""
import sys, os, io, importlib, importlib.util, contextlib
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.abspath(__file__))
GATES = ["OFF", "SOFT", "HARD"]

# ── Load module once, patch gate each run ────────────────────────────────────
print("Loading 4sectors module...")
spec = importlib.util.spec_from_file_location("4sectors", os.path.join(BASE, "4sectors.py"))
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.modules["4sectors"] = mod

print(f"  Data loaded. Running {len(GATES)} backtests...\n")

all_results = {}

for gate in GATES:
    mod.VNINDEX_GATE = gate
    print(f"{'─'*60}")
    print(f"  VN-GATE: {gate}")
    print(f"{'─'*60}")
    mod.main()

    # Read results CSV saved by main()
    csv_path = os.path.join(BASE, "backtest_results.csv")
    df = pd.read_csv(csv_path)

    summary = df[df["type"] == "summary"].iloc[0]
    annual  = df[df["type"] == "annual"].copy()
    sectors = df[df["type"] == "sector"].copy()

    all_results[gate] = {
        "cagr":     summary["ann_return_pct"],
        "sharpe":   summary["sharpe"],
        "maxdd":    summary["max_dd_pct"],
        "total_ret": summary["total_return_pct"],
        "n_trades": summary["n_trades"],
        "win_rate": summary["win_rate_pct"],
        "annual":   annual[["label", "ann_return_pct", "max_dd_pct"]].set_index("label"),
        "sectors":  sectors[["label", "end_capital_M", "n_trades", "win_rate_pct"]].set_index("label"),
    }

# ── Print comparison table ────────────────────────────────────────────────────
print()
print("=" * 70)
print("  VN-INDEX GATE COMPARISON")
print("=" * 70)
print(f"  {'Metric':<22} {'OFF':>12} {'SOFT':>12} {'HARD':>12}")
print(f"  {'-'*58}")
for key, label in [
    ("cagr",     "CAGR %"),
    ("sharpe",   "Sharpe"),
    ("maxdd",    "Max DD %"),
    ("total_ret","Total Return %"),
    ("n_trades", "# Trades"),
    ("win_rate", "Win Rate %"),
]:
    vals = [all_results[g][key] for g in GATES]
    print(f"  {label:<22} {vals[0]:>12.2f} {vals[1]:>12.2f} {vals[2]:>12.2f}")

print()
print("  ANNUAL RETURNS (%)")
print(f"  {'Year':<8} {'OFF':>8} {'SOFT':>8} {'HARD':>8}  {'SOFT-OFF':>10} {'HARD-OFF':>10}")
print(f"  {'-'*56}")
years = sorted(all_results["OFF"]["annual"].index)
for yr in years:
    try:
        off_r  = float(all_results["OFF"]["annual"].loc[yr,  "ann_return_pct"])
        soft_r = float(all_results["SOFT"]["annual"].loc[yr, "ann_return_pct"])
        hard_r = float(all_results["HARD"]["annual"].loc[yr, "ann_return_pct"])
    except Exception:
        continue
    if off_r == 0 and soft_r == 0 and hard_r == 0:
        continue
    delta_soft = soft_r - off_r
    delta_hard = hard_r - off_r
    flag_soft = "▲" if delta_soft > 0 else ("▼" if delta_soft < -1 else " ")
    flag_hard = "▲" if delta_hard > 0 else ("▼" if delta_hard < -1 else " ")
    print(f"  {yr:<8} {off_r:>8.1f} {soft_r:>8.1f} {hard_r:>8.1f}  "
          f"{delta_soft:>+9.1f}{flag_soft} {delta_hard:>+9.1f}{flag_hard}")

print()
print("  SECTOR FINAL CAPITAL (M VND, starting 100M)")
print(f"  {'Sector':<24} {'OFF':>12} {'SOFT':>12} {'HARD':>12}")
print(f"  {'-'*62}")
for gate in GATES:
    pass  # header only
for sec in ["Banks", "Basic Resources", "Food & Beverage", "Real Estate"]:
    vals = []
    for gate in GATES:
        try:
            vals.append(float(all_results[gate]["sectors"].loc[sec, "end_capital_M"]))
        except Exception:
            vals.append(0.0)
    print(f"  {sec:<24} {vals[0]:>12,.0f} {vals[1]:>12,.0f} {vals[2]:>12,.0f}")

print()
print("  ▲ = gate improved return  ▼ = gate reduced return (>1pp)")
print("  SOFT: half size when VN 63d < -5%,  skip when < -10%")
print("  HARD: skip entry when VN 63d < -10%")
print("=" * 70)
