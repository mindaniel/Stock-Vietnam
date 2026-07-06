import glob
import os
import sys
import numpy as np
import pandas as pd

# Fix console encoding for Windows environments if needed
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Setup paths to match your directory structure
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "archive" in os.getcwd() else os.getcwd()
sys.path.insert(0, os.path.join(BASE, "lib"))

# Try to import fundamental builder if available, otherwise we will use a fallback/mock for testing
try:
    from factor_stock_ranker import build_factor_features
    HAS_FUNDAMENTALS = True
except ImportError:
    HAS_FUNDAMENTALS = False

FLOW_DIR = os.path.join(BASE, "data", "investor_flow")
PRICE_DIR = os.path.join(BASE, "data", "price")

# Strategy Parameters
MIN_LIQUIDITY_VND = 1_000_000_000
ACCUM_WINDOW = 60
FWD_HORIZON = 120
SAMPLE_STRIDE = 20
MIN_STOCKS_PER_SECTOR_PERIOD = 3
TOP_QUINTILE = 0.8  # Buy top 20% highest retail accumulation
MIN_Q0_NP_YOY = 1.0  # Fundamental filter: Q0 Net Profit YoY must be > 100% (+1.0)


def get_liquid_universe():
    """Filters tickers meeting minimum liquidity thresholds over the last 60 days."""
    liquid = set()
    for fpath in glob.glob(os.path.join(PRICE_DIR, "*.parquet")):
        ticker = os.path.splitext(os.path.basename(fpath))[0].upper()
        try:
            df = pd.read_parquet(fpath)
            df.columns = [c.strip().lower() for c in df.columns]
            if "close" not in df.columns or "volume" not in df.columns:
                continue
            # Approximate turnover in VND (assuming price is scaled by 1,000)
            med_to = (df["close"] * df["volume"] * 1000).tail(60).median()
            if med_to >= MIN_LIQUIDITY_VND:
                liquid.add(ticker)
        except Exception:
            pass
    return liquid


def load_and_sync_signals(ticker: str, sector: str) -> pd.DataFrame:
    """Loads flow and price data, computes accumulation signals and forward returns."""
    fpath = os.path.join(FLOW_DIR, f"{ticker}.parquet")
    if not os.path.exists(fpath):
        return pd.DataFrame()

    df = pd.read_parquet(fpath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "close" not in df.columns or len(df) < ACCUM_WINDOW + FWD_HORIZON + 5:
        return pd.DataFrame()

    # Build the normalized foreign retail accumulation factor
    retail = df.get("ca_nhan_nuocngoai_net", pd.Series(0, index=df.index)).fillna(0)
    cum = retail.rolling(ACCUM_WINDOW).sum()
    scale = retail.abs().rolling(ACCUM_WINDOW).mean() * ACCUM_WINDOW
    
    out = pd.DataFrame({"date": df["date"], "ticker": ticker, "sector": sector})
    out["accum_retail"] = cum / scale.replace(0, np.nan)
    out["fwd_ret"] = df["close"].shift(-FWD_HORIZON) / df["close"] - 1
    
    return out


def main():
    print("=" * 90)
    print(" GIAI DOAN 1: KHOI TAO VU TRU DU LIEU & TIN HIEU")
    print("-" * 90)

    liquid_symbols = get_liquid_universe()
    flow_symbols = {os.path.splitext(os.path.basename(f))[0].upper() for f in glob.glob(os.path.join(FLOW_DIR, "*.parquet"))}
    universe = sorted(liquid_symbols & flow_symbols)

    # Load sector mappings
    map_path = os.path.join(BASE, "ticker_sectors.csv")
    if not os.path.exists(map_path):
        print(f"❌ Khong tim thay file mapping nganh tai: {map_path}")
        return

    mapping = pd.read_csv(map_path)
    mapping.columns = [c.strip().lower() for c in mapping.columns]
    mapping = mapping[mapping["exchange"].isin(["HOSE", "HNX"])]
    ticker_to_sector = dict(zip(mapping["ticker"].str.upper(), mapping["industry"]))

    frames = []
    for t in universe:
        sec = ticker_to_sector.get(t)
        if not sec or sec == "Unknown":
            continue
        f = load_and_sync_signals(t, sec)
        if not f.empty:
            frames.append(f)

    if not frames:
        print("❌ Khong co du lieu tin hieu hop le de xu ly.")
        return

    all_df = pd.concat(frames, ignore_index=True)
    print(f"  Da tai: {all_df['ticker'].nunique()} tickers thuoc {all_df['sector'].nunique()} nganh.")

    # Load and map fundamentals (Q0)
    print("\n" + "=" * 90)
    print(" GIAI DOAN 2: TICH HOP DU LIEU CO BAN QUY (Q0 FUNDAMENTALS)")
    print("-" * 90)
    
    if HAS_FUNDAMENTALS:
        print("  Dang chay build_factor_features de trich xuat np_yoy...")
        qfeat = build_factor_features(symbols=universe)
        qfeat = qfeat.sort_values("avail_date")
    else:
        print("  ⚠️ Khong thay factor_stock_ranker. Tu dong tao ngau nhien np_yoy de test code...")
        # Fallback dummy frame if factor engine isn't in path
        qfeat = pd.DataFrame({
            "symbol": np.random.choice(universe, len(universe) * 10),
            "avail_date": pd.to_datetime(np.random.choice(all_df["date"].unique(), len(universe) * 10)),
            "np_yoy": np.random.uniform(-0.5, 3.0, len(universe) * 10)
        }).sort_values("avail_date")

    print("\n" + "=" * 90)
    print(" GIAI DOAN 3: CHAY BACKTEST THEO TUNG CHU KY (CROSS-SECTIONAL RUN)")
    print("-" * 90)

    sample_dates = sorted(all_df["date"].unique())[::SAMPLE_STRIDE]
    portfolio_history = []

    for d in sample_dates:
        # 1. Get cross section of signals
        cross = all_df[all_df["date"] == d].dropna(subset=["accum_retail", "fwd_ret"]).copy()
        if len(cross) < 15:
            continue

        # 2. Minimum ticker per sector filter
        sec_counts = cross.groupby("sector")["ticker"].transform("count")
        cross = cross[sec_counts >= MIN_STOCKS_PER_SECTOR_PERIOD]
        if cross.empty:
            continue

        # 3. Calculate sector-neutral returns (Benchmark)
        cross["sector_avg_ret"] = cross.groupby("sector")["fwd_ret"].transform("mean")
        cross["sector_neutral_ret"] = cross["fwd_ret"] - cross["sector_avg_ret"]

        # 4. Point-In-Time merge for Q0 Fundamentals
        q0_list = []
        for _, row in cross.iterrows():
            sym = row["ticker"]
            sub_q = qfeat[(qfeat["symbol"] == sym) & (qfeat["avail_date"] <= d)]
            np_yoy = sub_q["np_yoy"].iloc[-1] if not sub_q.empty else np.nan
            q0_list.append(np_yoy)
        cross["q0_np_yoy"] = q0_list

        # 5. Apply Strategy Rules: Heavy Retail Accumulation AND High Growth Narrative
        thresh = cross["accum_retail"].quantile(TOP_QUINTILE)
        
        # Screen triggers
        strategy_portfolio = cross[
            (cross["accum_retail"] >= thresh) & 
            (cross["q0_np_yoy"] >= MIN_Q0_NP_YOY)
        ]

        if not strategy_portfolio.empty:
            for _, row in strategy_portfolio.iterrows():
                portfolio_history.append({
                    "date": d,
                    "ticker": row["ticker"],
                    "sector": row["sector"],
                    "fwd_ret": row["fwd_ret"],
                    "sector_neutral_ret": row["sector_neutral_ret"],
                    "q0_np_yoy": row["q0_np_yoy"]
                })

    if not portfolio_history:
        print("❌ Chien thuat khong tim thay ma nao thoa man tieu chi trong suot lich su.")
        return

    perf_df = pd.DataFrame(portfolio_history)

    # ====================================================================
    # GIAI DOAN 4: TONG HOP & HIEN THI KET QUA
    # ====================================================================
    print("\n" + "=" * 90)
    print(" KET QUA HIEN THI TONG HOP CHIEN THUAT (STRATEGY PERFORMANCE METRICS)")
    print("-" * 90)

    total_trades = len(perf_df)
    win_rate = (perf_df["sector_neutral_ret"] > 0).sum() / total_trades
    avg_raw_ret = perf_df["fwd_ret"].mean()
    avg_neutral_ret = perf_df["sector_neutral_ret"].mean()

    print(f"  Tong so tin hieu kich hoat (Trades): {total_trades}")
    print(f"  Ti le Thang so voi Nganh (Win Rate vs Sector): {win_rate * 100:.2f}%")
    print(f"  Loi nhuan Ky vong Tho (Avg Raw Forward Return): {avg_raw_ret * 100:.2f}%")
    print(f"  Loi nhuan Thuan Alpha (Avg Sector-Neutral Return): {avg_neutral_ret * 100:.2f}%")

    print("\n" + "-" * 90)
    print(" TOP 10 GIAO DICH THANG LON NHAT SO VOI NGANH (TOP ALPHA GENERATORS)")
    print("-" * 90)
    top_10 = perf_df.sort_values(by="sector_neutral_ret", ascending=False).head(10)
    print(top_10[["date", "ticker", "sector", "q0_np_yoy", "fwd_ret", "sector_neutral_ret"]].to_string(index=False))

    # Save to disk for deep-dive auditing
    out_path = os.path.join(BASE, "strategy_backtest_results.csv")
    perf_df.to_csv(out_path, index=False)
    print(f"\n✅ Da xuat chi tiet ket qua backtest ra file: {out_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()