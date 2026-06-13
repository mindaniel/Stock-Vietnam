"""
compare_vn_exit.py
==================
Runs 4sectors backtest twice:
  OFF — no VN-Index exit accelerator (VNINDEX_EXIT_ENABLED = False)
  ON  — VN-Index exit accelerator active (5d flash -7%, 20d sustained -12%)

Prints side-by-side comparison of annual returns, Sharpe, MaxDD, and sector PnL
so we can see whether the accelerator helps 2022 without hurting 2020/2021/2023.
"""
import sys, os, importlib, importlib.util
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Load module once, patch exit toggle for each run ─────────────────────────
print("Loading 4sectors module...")
spec = importlib.util.spec_from_file_location("4sectors", os.path.join(BASE, "4sectors.py"))
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.modules["4sectors"] = mod
print(f"  Data loaded. Running 2 backtests (OFF → ON)...\n")

VARIANTS = [
    ("OFF",  False),
    ("ON",   True),
]

all_results = {}

for label, enabled in VARIANTS:
    mod.VNINDEX_EXIT_ENABLED = enabled
    # Also make sure the gate stays OFF so it doesn't interfere
    mod.VNINDEX_GATE = "OFF"

    print(f"{'─'*60}")
    print(f"  VN-EXIT: {label}  (thresholds: {mod.VNINDEX_EXIT_FLASH_DAYS}d<{mod.VNINDEX_EXIT_FLASH_THRESH:.0%}, "
          f"{mod.VNINDEX_EXIT_SUSTAINED_DAYS}d<{mod.VNINDEX_EXIT_SUSTAINED_THRESH:.0%})")
    print(f"{'─'*60}")
    mod.main()

    csv_path = os.path.join(BASE, "backtest_results.csv")
    df = pd.read_csv(csv_path)

    summary = df[df["type"] == "summary"].iloc[0]
    annual  = df[df["type"] == "annual"].copy()
    sectors = df[df["type"] == "sector"].copy()

    all_results[label] = {
        "cagr":       summary["ann_return_pct"],
        "sharpe":     summary["sharpe"],
        "maxdd":      summary["max_dd_pct"],
        "total_ret":  summary["total_return_pct"],
        "n_trades":   summary["n_trades"],
        "win_rate":   summary["win_rate_pct"],
        "annual":     annual[["label", "ann_return_pct", "max_dd_pct"]].set_index("label"),
        "sectors":    sectors[["label", "end_capital_M", "n_trades", "win_rate_pct"]].set_index("label"),
    }

# ── Print comparison table ────────────────────────────────────────────────────
print()
print("=" * 70)
print("  VN-INDEX EXIT ACCELERATOR COMPARISON")
print("=" * 70)

off = all_results["OFF"]
on  = all_results["ON"]

print(f"  {'Metric':<26} {'OFF (baseline)':>16} {'ON (accelerator)':>16}  {'Delta':>8}")
print(f"  {'-'*68}")
for key, label in [
    ("cagr",     "CAGR %"),
    ("sharpe",   "Sharpe"),
    ("maxdd",    "Max DD %"),
    ("total_ret","Total Return %"),
    ("n_trades", "# Trades"),
    ("win_rate", "Win Rate %"),
]:
    v_off = off[key]
    v_on  = on[key]
    delta = v_on - v_off
    flag  = "▲" if delta > 0 else ("▼" if delta < 0 else " ")
    print(f"  {label:<26} {v_off:>16.2f} {v_on:>16.2f}  {delta:>+7.2f}{flag}")

print()
print("  ANNUAL RETURNS (%)")
print(f"  {'Year':<8} {'OFF':>8} {'ON':>8}  {'Delta':>8}  {'MaxDD-OFF':>10} {'MaxDD-ON':>10}")
print(f"  {'-'*58}")
years = sorted(off["annual"].index)
for yr in years:
    try:
        r_off  = float(off["annual"].loc[yr, "ann_return_pct"])
        r_on   = float(on["annual"].loc[yr,  "ann_return_pct"])
        dd_off = float(off["annual"].loc[yr, "max_dd_pct"])
        dd_on  = float(on["annual"].loc[yr,  "max_dd_pct"])
    except Exception:
        continue
    if r_off == 0 and r_on == 0:
        continue
    delta = r_on - r_off
    flag  = "▲" if delta > 1 else ("▼" if delta < -1 else " ")
    print(f"  {yr:<8} {r_off:>8.1f} {r_on:>8.1f}  {delta:>+7.1f}{flag}  {dd_off:>10.1f} {dd_on:>10.1f}")

print()
print("  SECTOR FINAL CAPITAL (M VND, starting 100M each)")
print(f"  {'Sector':<24} {'OFF':>14} {'ON':>14}  {'Delta':>8}")
print(f"  {'-'*64}")
for sec in ["Banks", "Basic Resources", "Food & Beverage", "Real Estate"]:
    try:
        v_off = float(off["sectors"].loc[sec, "end_capital_M"])
        v_on  = float(on["sectors"].loc[sec,  "end_capital_M"])
    except Exception:
        v_off, v_on = 0.0, 0.0
    delta = v_on - v_off
    flag  = "▲" if delta > 0 else ("▼" if delta < 0 else " ")
    print(f"  {sec:<24} {v_off:>14,.0f} {v_on:>14,.0f}  {delta:>+7,.0f}{flag}")

print()
print("  Thresholds: flash = VN-Index 5d < -7%,  sustained = VN-Index 20d < -12%")
print("  Both OFF and ON use VNINDEX_GATE=OFF (entry gate disabled)")
print("  ▲ = accelerator improved metric   ▼ = accelerator reduced metric")
print("=" * 70)
