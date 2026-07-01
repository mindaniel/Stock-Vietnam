"""
Compare 4 exit-rule variants — backtest window: Sep 2024 → present.

Start date pinned to flow data availability (Sep 2024) so every variant
runs on exactly the same trades and the comparison is apples-to-apples.

Variants:
  A  Baseline  — no individual stock exits beyond 25% TP
  B  SWING_LL  — exit on confirmed LL (price only)
  C  FLOW_DIST — exit on FlowSignalEngine distribution alert (flow only)
  D  SWING+FLOW— exit only when BOTH LL confirmed AND flow distributing

Usage:  python run_flow_compare.py
"""

import importlib.util, sys, io, os
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE  = Path(__file__).parent
SPEC  = importlib.util.spec_from_file_location("_4s", BASE / "archive" / "4sectors.py")

SUB_START = "2024-09-16"   # flow data start — compare sub-window separately


def run_variant(name, patches: dict):
    """Load 4sectors module fresh, apply patches, run main(), capture stdout."""
    mod = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(mod)
    for attr, val in patches.items():
        setattr(mod, attr, val)

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        mod.main()
    finally:
        sys.stdout = old_stdout

    output = buf.getvalue()
    return output


def extract_metrics(output):
    """Pull key numbers out of the printed backtest output."""
    metrics = {}
    for line in output.splitlines():
        line = line.strip()
        if "Total return" in line:
            try: metrics["total_ret"] = float(line.split(":")[1].replace("%","").replace("M","").strip().split()[0])
            except: pass
        if "Annualised" in line:
            try: metrics["ann_ret"] = float(line.split(":")[1].replace("%","").strip().split()[0])
            except: pass
        if "Sharpe" in line and ":" in line:
            try: metrics["sharpe"] = float(line.split(":")[1].strip().split()[0])
            except: pass
        if "Max drawdown" in line:
            try: metrics["mdd"] = float(line.split(":")[1].replace("%","").strip().split()[0])
            except: pass
        if "Win rate" in line and "%" in line:
            try: metrics["win_rate"] = float(line.split(":")[1].replace("%","").strip().split()[0].replace("(",""))
            except: pass
        if "Avg winner" in line:
            try: metrics["avg_win"] = float(line.split(":")[1].replace("%","").replace("+","").strip().split()[0])
            except: pass
        if "Avg loser" in line:
            try: metrics["avg_loss"] = float(line.split(":")[1].replace("%","").strip().split()[0])
            except: pass
        if "stock-trades)" in line:
            try: metrics["n_trades"] = int(line.split("(")[1].split()[0])
            except: pass
    return metrics


FLOW_START = "2024-09-16"   # pin all variants to flow data start

VARIANTS = [
    ("A  Baseline",   {"BACKTEST_START": FLOW_START,
                       "SWING_LL_EXIT": False, "SWING_FLOW_EXIT": False,
                       "FLOW_DIST_EXIT": False, "FLOW_SIGNAL_ENABLED": False}),
    ("B  SWING_LL",   {"BACKTEST_START": FLOW_START,
                       "SWING_LL_EXIT": True,  "SWING_FLOW_EXIT": False,
                       "FLOW_DIST_EXIT": False, "FLOW_SIGNAL_ENABLED": False}),
    ("C  FLOW_DIST",  {"BACKTEST_START": FLOW_START,
                       "SWING_LL_EXIT": False, "SWING_FLOW_EXIT": False,
                       "FLOW_DIST_EXIT": True,  "FLOW_SIGNAL_ENABLED": True}),
    ("D  SWING+FLOW", {"BACKTEST_START": FLOW_START,
                       "SWING_LL_EXIT": False, "SWING_FLOW_EXIT": True,
                       "FLOW_DIST_EXIT": False, "FLOW_SIGNAL_ENABLED": True}),
]


def main():
    results = []
    for name, patches in VARIANTS:
        print(f"\n  Running variant {name}...", flush=True)
        output = run_variant(name, patches)
        m = extract_metrics(output)
        results.append((name, m, output))
        print(f"  Done: ann={m.get('ann_ret','?')}%  sharpe={m.get('sharpe','?')}", flush=True)

    # Print comparison table
    print(f"\n{'═'*80}")
    print(f"  VARIANT COMPARISON")
    print(f"{'─'*80}")
    print(f"  {'Variant':<18} {'Ann%':>6} {'Sharpe':>7} {'MDD%':>7} "
          f"{'WinR%':>6} {'AvgW%':>6} {'AvgL%':>6} {'#Trades':>8}")
    print(f"  {'-'*78}")
    for name, m, _ in results:
        print(f"  {name:<18} "
              f"{m.get('ann_ret', 0):>+6.2f}  "
              f"{m.get('sharpe', 0):>7.3f}  "
              f"{m.get('mdd', 0):>7.1f}%  "
              f"{m.get('win_rate', 0):>5.1f}%  "
              f"{m.get('avg_win', 0):>5.1f}%  "
              f"{m.get('avg_loss', 0):>6.2f}%  "
              f"{m.get('n_trades', 0):>7}")

    print(f"{'─'*80}")
    base_ann = results[0][1].get("ann_ret", 0)
    for name, m, _ in results[1:]:
        diff = m.get("ann_ret", 0) - base_ann
        sym  = "▲" if diff > 0 else "▼"
        print(f"  {name:<18} vs baseline: {sym} {diff:>+.2f}pp ann")

    print(f"\n  Note: all variants start {FLOW_START} — same trades, fair comparison.")
    print(f"        Tick data (3 months) too short for backtest — not included.")
    print(f"{'═'*80}")


if __name__ == "__main__":
    main()
