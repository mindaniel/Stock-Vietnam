"""
Fama-French style factor test for Vietnam equities (local data).

Builds monthly factors from local price + fundamentals:
  - MKT_RF (market minus risk-free)
  - SMB    (small minus big)
  - HML    (high B/M minus low B/M)
  - RMW    (robust profitability minus weak)
  - CMA    (conservative investment minus aggressive)
  - MOM    (12-1 momentum spread)

Then runs OLS regressions for a target portfolio excess return:
  - FF3: MKT_RF + SMB + HML
  - FF5: FF3 + RMW + CMA
  - FF6: FF5 + MOM

Usage examples:
  python ff_vietnam.py --portfolio equal --start 2016-01-01
  python ff_vietnam.py --portfolio sector --sector "Banks" --start 2018-01-01
  python ff_vietnam.py --portfolio ticker --ticker FPT
"""

import argparse
import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "price")
FA_DIR = os.path.join(DATA_DIR, "financials_fa")
TICKER_SECTOR_CSV = os.path.join(BASE_DIR, "ticker_sectors.csv")
VNINDEX_CSV = os.path.join(BASE_DIR, "VNINDEX.csv")
OUT_DIR = os.path.join(BASE_DIR, "results")


def load_sector_map() -> pd.DataFrame:
    df = pd.read_csv(TICKER_SECTOR_CSV)
    # normalize to: symbol, sector
    cols = {c.lower(): c for c in df.columns}
    sym_col = cols.get("ticker") or cols.get("symbol")
    sec_col = cols.get("industry") or cols.get("sector")
    out = df[[sym_col, sec_col]].copy()
    out.columns = ["symbol", "sector"]
    out["symbol"] = out["symbol"].astype(str).str.upper()
    return out.dropna(subset=["symbol"]).drop_duplicates("symbol")


def load_prices() -> pd.DataFrame:
    rows = []
    for f in glob.glob(os.path.join(DATA_DIR, "*.parquet")):
        sym = os.path.basename(f).replace(".parquet", "").upper()
        try:
            df = pd.read_parquet(f, columns=["time", "close", "value"])
            df = df.rename(columns={"time": "date"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["date", "close"])
            df = df[df["close"] > 0]
            df["symbol"] = sym
            rows.append(df[["date", "symbol", "close", "value"]])
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "close", "value"])
    out = pd.concat(rows, ignore_index=True).sort_values(["symbol", "date"])
    return out


def load_vnindex_monthly() -> pd.Series:
    vn = pd.read_csv(VNINDEX_CSV)
    vn["date"] = pd.to_datetime(vn["date"], errors="coerce")
    vn["close"] = pd.to_numeric(vn["close"], errors="coerce")
    vn = vn.dropna(subset=["date", "close"]).sort_values("date")
    m = vn.set_index("date")["close"].resample("ME").last().pct_change(fill_method=None)
    return m.rename("mkt").dropna()


def load_fundamentals_latest_by_year() -> pd.DataFrame:
    rows = []
    for f in glob.glob(os.path.join(FA_DIR, "*.parquet")):
        if "indicators_snapshot" in f:
            continue
        try:
            df = pd.read_parquet(f)
            if "quarter" in df.columns:
                df = df[df["quarter"] == 0].copy()
            if df.empty:
                continue
            keep = [
                "symbol", "year", "equity", "net_profit", "total_assets",
                "revenue", "sector", "pb", "pe"
            ]
            for c in keep:
                if c not in df.columns:
                    df[c] = np.nan
            df = df[keep].copy()
            df["symbol"] = df["symbol"].astype(str).str.upper()
            rows.append(df)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["symbol", "year", "equity", "net_profit", "total_assets", "revenue", "sector", "pb", "pe"])
    rows = [r for r in rows if not r.empty]
    fin = pd.concat(rows, ignore_index=True)
    fin = fin.dropna(subset=["symbol", "year"]).copy()
    fin["year"] = fin["year"].astype(int)
    for c in ["equity", "net_profit", "total_assets", "revenue", "pb", "pe"]:
        fin[c] = pd.to_numeric(fin[c], errors="coerce")
    return fin


def monthly_stock_panel(prices: pd.DataFrame) -> pd.DataFrame:
    px = prices.copy()
    monthly = (
        px.groupby(["symbol", pd.Grouper(key="date", freq="ME")])
        .agg(close=("close", "last"), value=("value", "mean"))
        .reset_index()
        .sort_values(["symbol", "date"])
    )
    monthly["ret"] = monthly.groupby("symbol")["close"].pct_change(fill_method=None)
    monthly["mom_12_1"] = (
        monthly.groupby("symbol")["close"].pct_change(11, fill_method=None).shift(1)
    )
    # proxy size by average traded value in month
    monthly["size_proxy"] = monthly["value"]
    return monthly.dropna(subset=["ret"])


def attach_accounting(monthly: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:
    df = monthly.copy()
    df["fyear"] = df["date"].dt.year - 1  # use prior-year report
    # drop sector from fin to avoid collision with sector already on df from sec_map merge
    fin_cols = [c for c in fin.columns if c != "sector"]
    x = fin[fin_cols].copy().rename(columns={"year": "fyear"})
    df = df.merge(x, on=["symbol", "fyear"], how="left")

    # Use accounting + valuation fields to infer market cap and B/M:
    # market_cap ≈ equity * PB   (PB from fundamentals snapshot)
    # book-to-market = equity / market_cap = 1 / PB
    if "pb" not in x.columns:
        x["pb"] = np.nan
    if "pe" not in x.columns:
        x["pe"] = np.nan

    df["pb"] = pd.to_numeric(df.get("pb"), errors="coerce")
    df["pe"] = pd.to_numeric(df.get("pe"), errors="coerce")
    df["mkt_cap_proxy"] = df["equity"] * df["pb"]
    df["bm"] = 1.0 / df["pb"].replace(0, np.nan)
    df["profitability"] = df["net_profit"] / df["equity"].replace(0, np.nan)

    # investment proxy: asset growth from y-2 to y-1
    assets = fin[["symbol", "year", "total_assets"]].copy()
    assets = assets.sort_values(["symbol", "year"])
    assets["asset_growth"] = assets.groupby("symbol")["total_assets"].pct_change(fill_method=None)
    assets = assets.rename(columns={"year": "fyear"})
    df = df.merge(assets[["symbol", "fyear", "asset_growth"]], on=["symbol", "fyear"], how="left")

    # Earnings yield (inverse PE) from financial + price-linked valuation
    df["earnings_yield"] = 1.0 / df["pe"].replace(0, np.nan)

    return df


def _spread(top: pd.Series, bot: pd.Series) -> float:
    if len(top) == 0 or len(bot) == 0:
        return np.nan
    return top.mean() - bot.mean()


def build_factors(panel: pd.DataFrame, rf_annual: float = 0.03, min_stocks: int = 10) -> pd.DataFrame:
    rf_m = (1 + rf_annual) ** (1 / 12) - 1
    out = []
    for dt, g in panel.groupby("date"):
        g = g.dropna(subset=["ret"])
        if len(g) < min_stocks:
            continue

        # SMB: prefer mkt_cap_proxy (equity×PB); fall back to size_proxy (traded value)
        size_col = "mkt_cap_proxy" if g["mkt_cap_proxy"].notna().sum() >= min_stocks else "size_proxy"
        g_size = g.dropna(subset=[size_col])
        smb = np.nan
        if len(g_size) >= min_stocks:
            size_med = g_size[size_col].median()
            small = g_size[g_size[size_col] <= size_med]
            big   = g_size[g_size[size_col] >  size_med]
            smb   = _spread(small["ret"], big["ret"])

        g_bm = g.dropna(subset=["bm"])
        hml = np.nan
        if len(g_bm) >= min_stocks:
            q30, q70 = g_bm["bm"].quantile([0.3, 0.7])
            hml = _spread(g_bm[g_bm["bm"] >= q70]["ret"], g_bm[g_bm["bm"] <= q30]["ret"])

        g_prof = g.dropna(subset=["profitability"])
        rmw = np.nan
        if len(g_prof) >= min_stocks:
            q30, q70 = g_prof["profitability"].quantile([0.3, 0.7])
            rmw = _spread(g_prof[g_prof["profitability"] >= q70]["ret"],
                          g_prof[g_prof["profitability"] <= q30]["ret"])

        g_inv = g.dropna(subset=["asset_growth"])
        cma = np.nan
        if len(g_inv) >= min_stocks:
            q30, q70 = g_inv["asset_growth"].quantile([0.3, 0.7])
            cma = _spread(g_inv[g_inv["asset_growth"] <= q30]["ret"],
                          g_inv[g_inv["asset_growth"] >= q70]["ret"])

        g_mom = g.dropna(subset=["mom_12_1"])
        mom = np.nan
        if len(g_mom) >= min_stocks:
            q30, q70 = g_mom["mom_12_1"].quantile([0.3, 0.7])
            mom = _spread(g_mom[g_mom["mom_12_1"] >= q70]["ret"],
                          g_mom[g_mom["mom_12_1"] <= q30]["ret"])

        mkt = g["ret"].mean() - rf_m
        out.append({"date": dt, "RF": rf_m, "MKT_RF": mkt,
                    "SMB": smb, "HML": hml, "RMW": rmw, "CMA": cma, "MOM": mom})

    if not out:
        return pd.DataFrame(columns=["date", "RF", "MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"])
    ff = pd.DataFrame(out).sort_values("date").dropna(subset=["MKT_RF"])
    return ff


def portfolio_return(panel: pd.DataFrame, mode: str, ticker: str | None, sector: str | None) -> pd.Series:
    g = panel.copy()
    if mode == "ticker":
        if not ticker:
            raise ValueError("--ticker required for --portfolio ticker")
        g = g[g["symbol"] == ticker.upper()]
    elif mode == "sector":
        if not sector:
            raise ValueError("--sector required for --portfolio sector")
        g = g[g["sector"].fillna("") == sector]
    # equal weighted by month
    p = g.groupby("date")["ret"].mean().rename("port_ret")
    return p.dropna()


@dataclass
class OLSResult:
    model: str
    n: int
    alpha: float
    r2: float
    betas: dict


def run_ols(y: pd.Series, X: pd.DataFrame, model_name: str) -> OLSResult:
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(df) < 24:
        return OLSResult(model_name, len(df), np.nan, np.nan, {})

    yv = df["y"].values
    xv = df[X.columns].values
    x = np.column_stack([np.ones(len(xv)), xv])
    beta = np.linalg.lstsq(x, yv, rcond=None)[0]
    yhat = x @ beta
    ss_res = np.sum((yv - yhat) ** 2)
    ss_tot = np.sum((yv - yv.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return OLSResult(
        model=model_name,
        n=len(df),
        alpha=float(beta[0]),
        r2=float(r2),
        betas={c: float(beta[i + 1]) for i, c in enumerate(X.columns)},
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--portfolio", choices=["equal", "sector", "ticker"], default="equal")
    p.add_argument("--ticker", default=None)
    p.add_argument("--sector", default=None)
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--rf-annual", type=float, default=0.03, help="Annual risk-free rate proxy, decimal")
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    prices = load_prices()
    if prices.empty:
        print("No price data found in data/*.parquet")
        return

    sec_map = load_sector_map()
    monthly = monthly_stock_panel(prices).merge(sec_map, on="symbol", how="left")

    fin = load_fundamentals_latest_by_year()
    panel = attach_accounting(monthly, fin)
    panel = panel[panel["date"] >= pd.Timestamp(args.start)].copy()

    ff = build_factors(panel, rf_annual=args.rf_annual)
    if ff.empty:
        print("No factor observations could be built from current financial + price coverage in this date range.")
        return
    port = portfolio_return(panel, mode=args.portfolio, ticker=args.ticker, sector=args.sector)

    ds = pd.concat([port, ff.set_index("date")], axis=1).dropna(subset=["port_ret", "RF", "MKT_RF"])
    ds["port_excess"] = ds["port_ret"] - ds["RF"]

    r3 = run_ols(ds["port_excess"], ds[["MKT_RF", "SMB", "HML"]], "FF3")
    r5 = run_ols(ds["port_excess"], ds[["MKT_RF", "SMB", "HML", "RMW", "CMA"]], "FF5")
    r6 = run_ols(ds["port_excess"], ds[["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]], "FF6")

    res_rows = []
    for r in [r3, r5, r6]:
        row = {"model": r.model, "n": r.n, "alpha_monthly": r.alpha, "alpha_annualized": (1 + r.alpha) ** 12 - 1 if pd.notna(r.alpha) else np.nan, "r2": r.r2}
        row.update(r.betas)
        res_rows.append(row)
    res = pd.DataFrame(res_rows)

    prefix = f"ff_{args.portfolio}"
    if args.ticker:
        prefix += f"_{args.ticker.upper()}"
    if args.sector:
        prefix += "_" + args.sector.lower().replace(" ", "_")

    factors_path = os.path.join(OUT_DIR, f"{prefix}_factors.csv")
    regress_path = os.path.join(OUT_DIR, f"{prefix}_regression.csv")
    ff.to_csv(factors_path, index=False)
    res.to_csv(regress_path, index=False)

    print("=" * 80)
    print("FAMA-FRENCH TEST (VIETNAM, MONTHLY)")
    print("=" * 80)
    print(f"Portfolio mode: {args.portfolio} | start: {args.start}")
    if args.ticker:
        print(f"Ticker: {args.ticker.upper()}")
    if args.sector:
        print(f"Sector: {args.sector}")
    print(f"Observations used: {len(ds)} months")
    print("\nRegression summary:")
    print(res.to_string(index=False))
    print("\nSaved:")
    print(" -", factors_path)
    print(" -", regress_path)


if __name__ == "__main__":
    main()
