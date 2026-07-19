"""
test_foreign_inst_threshold_signal.py — Replicate the DeepSeek dashboard's
exact signal definition (foreign institutional net flow > 90th-percentile
threshold -> buy, < -90th percentile -> sell, forward 10-day return), but
fix its three methodological problems and see if the effect survives:

  1. LOOK-AHEAD: DeepSeek's threshold = full-sample quantile of abs_flow,
     computed ONCE using the whole history including future dates. Here the
     threshold is a TRAILING 252-day rolling quantile, shifted by 1 day —
     genuinely known at the time, no look-ahead.
  2. SMALL/CHERRY-PICKED SAMPLE: DeepSeek tested 20 stocks, highlighting
     whichever had the best in-sample Sharpe per ticker (multiple-comparisons
     trap: 20 stocks x 5 players = 100 draws, some will look great by chance).
     Here: the FULL liquid+flow universe (~600 tickers), ONE fixed signal
     definition applied uniformly, no cherry-picking.
  3. NO TIME CONTROL: DeepSeek's backtest has no period fixed effect, so a
     "buy when foreign institutions buy" signal is contaminated by foreign
     institutions' well-documented procyclical/momentum-following behavior —
     it may just be detecting "the market was rallying that month." Here:
     run the SAME regression twice, once naive (no controls, matching
     DeepSeek) and once with sector + period fixed effects + clustered SEs,
     to directly show how much of the naive effect is a market-timing
     artifact vs a genuine ticker-level flow signal.

Usage:  python archive/test_foreign_inst_threshold_signal.py
"""

import glob, os, sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOW_DIR = os.path.join(BASE, "data", "investor_flow")
PRICE_DIR = os.path.join(BASE, "data", "price")

FORWARD_DAYS = 10          # matches DeepSeek's 10-day forward window
THRESHOLD_PCT = 0.90       # matches DeepSeek's 90th-percentile cutoff
ROLLING_WINDOW = 252       # trailing ~1yr for the point-in-time quantile
MIN_LIQUIDITY_VND = 500_000_000  # loose filter, just excludes dead/halted names


def liquid_flow_universe():
    keep = set()
    for fpath in glob.glob(os.path.join(PRICE_DIR, "*.parquet")):
        ticker = os.path.splitext(os.path.basename(fpath))[0].upper()
        try:
            df = pd.read_parquet(fpath)
            df.columns = [c.strip().lower() for c in df.columns]
            if "close" not in df.columns or "volume" not in df.columns:
                continue
            med_to = (df["close"] * df["volume"] * 1000).median()
            if med_to >= MIN_LIQUIDITY_VND:
                keep.add(ticker)
        except Exception:
            pass
    return keep


def build_signal_frame(ticker: str, sector: str) -> pd.DataFrame:
    fpath = os.path.join(FLOW_DIR, f"{ticker}.parquet")
    if not os.path.exists(fpath):
        return pd.DataFrame()
    df = pd.read_parquet(fpath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "close" not in df.columns or "to_chuc_nuocngoai_net" not in df.columns:
        return pd.DataFrame()
    if len(df) < ROLLING_WINDOW + FORWARD_DAYS + 10:
        return pd.DataFrame()

    flow = df["to_chuc_nuocngoai_net"].fillna(0)
    abs_flow = flow.abs()

    # Point-in-time threshold: trailing rolling quantile, SHIFTED by 1 day so
    # today's threshold only ever uses data strictly before today.
    threshold = abs_flow.rolling(ROLLING_WINDOW, min_periods=60).quantile(THRESHOLD_PCT).shift(1)

    signal = pd.Series(0, index=df.index)
    signal[(flow > 0) & (abs_flow > threshold)] = 1
    signal[(flow < 0) & (abs_flow > threshold)] = -1

    fwd_ret = df["close"].shift(-FORWARD_DAYS) / df["close"] - 1

    out = pd.DataFrame({
        "ticker": ticker, "sector": sector, "date": df["date"],
        "signal": signal, "fwd_ret": fwd_ret,
    })
    out["period"] = out["date"].dt.strftime("%Y-%m")
    return out[out["signal"] != 0].dropna(subset=["fwd_ret"])


def main():
    print("Building liquid+flow universe...")
    liquid = liquid_flow_universe()
    flow_tickers = {os.path.splitext(os.path.basename(f))[0].upper()
                    for f in glob.glob(os.path.join(FLOW_DIR, "*.parquet"))}
    mapping = pd.read_csv(os.path.join(BASE, "ticker_sectors.csv"))
    mapping.columns = [c.strip().lower() for c in mapping.columns]
    mapping = mapping[mapping["exchange"].isin(["HOSE", "HNX"])]
    ticker_to_sector = dict(zip(mapping["ticker"].str.upper(), mapping["industry"]))

    universe = sorted(liquid & flow_tickers & set(ticker_to_sector))
    print(f"  {len(universe)} tickers")

    frames = []
    for i, t in enumerate(universe):
        sec = ticker_to_sector.get(t)
        if sec is None or sec == "Unknown":
            continue
        f = build_signal_frame(t, sec)
        if not f.empty:
            frames.append(f)
        if (i + 1) % 300 == 0:
            print(f"  ...{i + 1}/{len(universe)} processed")

    panel = pd.concat(frames, ignore_index=True)
    print(f"\n{panel['ticker'].nunique()} tickers with signal days, "
          f"{len(panel):,} total signal-days (BUY={ (panel['signal']==1).sum() }, "
          f"SELL={ (panel['signal']==-1).sum() })")
    print(f"Date range: {panel['date'].min().date()} to {panel['date'].max().date()}")

    # Match DeepSeek's "strategy_return" convention: positive when the
    # signal correctly predicted direction (buy signal + positive return,
    # or sell signal + negative return both count as a "win").
    panel["strategy_return"] = panel["signal"] * panel["fwd_ret"]

    print(f"\n{'='*90}")
    print("  NAIVE REPLICATION (DeepSeek-style: no period/sector control, full universe)")
    print(f"{'-'*90}")
    buy = panel[panel["signal"] == 1]
    sell = panel[panel["signal"] == -1]
    print(f"  BUY  signals: {len(buy):,}  avg fwd_ret={buy['fwd_ret'].mean()*100:+.2f}%  "
          f"win_rate={ (buy['fwd_ret']>0).mean()*100:.1f}%")
    print(f"  SELL signals: {len(sell):,}  avg fwd_ret={sell['fwd_ret'].mean()*100:+.2f}%  "
          f"(win if negative) win_rate={ (sell['fwd_ret']<0).mean()*100:.1f}%")
    overall_sr = panel["strategy_return"].mean() * 100
    overall_win = (panel["strategy_return"] > 0).mean() * 100
    t_naive = panel["strategy_return"].mean() / (panel["strategy_return"].std() / np.sqrt(len(panel)))
    print(f"\n  Pooled strategy_return: {overall_sr:+.2f}%  win_rate={overall_win:.1f}%  "
          f"naive t-stat (no clustering) = {t_naive:.2f}")

    print(f"\n{'='*90}")
    print("  CONTROLLED VERSION (sector + period fixed effects, clustered SE by ticker)")
    print(f"{'-'*90}")
    model_a = smf.ols("fwd_ret ~ signal", data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel["ticker"]})
    print("\n  Model A: fwd_ret ~ signal  (no FE, still clustered by ticker)")
    print(f"    signal coef = {model_a.params['signal']:+.5f}  "
          f"t = {model_a.tvalues['signal']:.2f}  p = {model_a.pvalues['signal']:.4f}")

    sec_counts = panel["sector"].value_counts()
    ok_sectors = sec_counts[sec_counts >= 15].index
    panel_fe = panel[panel["sector"].isin(ok_sectors)]
    model_b = smf.ols("fwd_ret ~ signal + C(sector) + C(period)", data=panel_fe).fit(
        cov_type="cluster", cov_kwds={"groups": panel_fe["ticker"]})
    print("\n  Model B: fwd_ret ~ signal + C(sector) + C(period)  (clustered by ticker)")
    print(f"    signal coef = {model_b.params['signal']:+.5f}  "
          f"t = {model_b.tvalues['signal']:.2f}  p = {model_b.pvalues['signal']:.4f}  "
          f"N = {int(model_b.nobs):,}")

    print(f"\n{'='*90}")
    print("  INTERPRETATION")
    print(f"{'-'*90}")
    print("  Model A (no time control) is the closest fair replication of DeepSeek's claim,")
    print("  applied to the full universe with a point-in-time (no look-ahead) threshold.")
    print("  Model B adds sector + period fixed effects: if the signal coefficient shrinks")
    print("  toward zero / loses significance from A to B, the naive effect was largely")
    print("  foreign institutions' flow correlating with market-wide/sector-wide moves that")
    print("  period, not a genuine ticker-specific predictive signal.")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
