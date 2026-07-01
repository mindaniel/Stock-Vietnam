"""
Run trailing-stop comparison: 3 configs, each saved to results/
Configs tested (all with WONHAM_EXIT=True, STOCK_TP_PCT=None):
  A) trail=0.10, activate=0.15
  B) trail=0.15, activate=0.15
  C) trail=0.20, activate=0.20
"""
import re, shutil, subprocess, sys, os, time

BASE    = os.path.dirname(os.path.abspath(__file__))
SRC     = os.path.join(BASE, "archive", "4sectors.py")
RESULTS = os.path.join(BASE, "results")

CONFIGS = [
    # trail10 already done — uncomment to re-run
    # dict(label="trail10", trail=0.10, activate=0.15, tp="None",  wonham=True),
    dict(label="trail15", trail=0.15, activate=0.15, tp="None",  wonham=True),
    dict(label="trail20", trail=0.20, activate=0.20, tp="None",  wonham=True),
]

def patch_and_run(cfg):
    src = open(SRC, encoding="utf-8").read()

    trail_val  = str(cfg["trail"]) if cfg["trail"] is not None else "None"
    tp_val     = str(cfg["tp"])
    wonham_val = "True" if cfg["wonham"] else "False"
    act_val    = str(cfg["activate"])

    src = re.sub(r'(STOCK_TRAILING_STOP_PCT\s*=\s*).*',
                 rf'\g<1>{trail_val}   # patched by run_trail_compare', src)
    src = re.sub(r'(TRAIL_ACTIVATE_PCT\s*=\s*).*',
                 rf'\g<1>{act_val}   # patched', src)
    src = re.sub(r'(STOCK_TP_PCT\s*=\s*).*',
                 rf'\g<1>{tp_val}   # patched', src)
    src = re.sub(r'(WONHAM_EXIT\s*=\s*).*',
                 rf'\g<1>{wonham_val}   # patched', src)

    # Must live in archive/ — the script uses __file__ to resolve data paths
    tmp = os.path.join(BASE, "archive", f"_tmp_{cfg['label']}.py")
    open(tmp, "w", encoding="utf-8").write(src)

    print(f"\n{'='*60}")
    print(f"  Running: {cfg['label']}  trail={trail_val}  activate={act_val}  TP={tp_val}  wonham={wonham_val}")
    print(f"{'='*60}")
    t0 = time.time()
    ret = subprocess.run([sys.executable, "-u", tmp], cwd=BASE, text=True)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f}min  (exit code {ret.returncode})")

    for fname in ["backtest_results.csv", "walkforward_results.csv",
                  "trades_option_c.csv", "option_c_momentum_rank.png"]:
        src_file = os.path.join(BASE, fname)
        if os.path.exists(src_file):
            dest = os.path.join(RESULTS, f"{cfg['label']}_{fname}")
            shutil.copy(src_file, dest)
            print(f"  Saved -> {dest}")

    os.remove(tmp)

if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    for cfg in CONFIGS:
        patch_and_run(cfg)
    print("\nAll configs done.")
