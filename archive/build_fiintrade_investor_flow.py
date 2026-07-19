"""
build_fiintrade_investor_flow.py — Parse the raw fiintrade_*.json /
fiintrade_gapfill_*.json capture files (from archive/fiintrade_harvest.js)
into per-ticker parquet files matching data/investor_flow's schema, so they
can extend/replace the nguoiquansat data (Sep 2024+) with much longer
history (validated back to 2016 for some tickers — see the ACB cross-check
done earlier in this session: FiinTrade's net flow values matched
nguoiquansat's to 2 decimal places for overlapping dates).

Each raw JSON file is a list of {url, capturedAt, data} capture records from
the harvest script's network patch. Only GetPriceData responses carry the
investor-classification fields; other captured endpoints (GetLatestPrice,
etc.) are ignored. Both normal (fiintrade_{TICKER}_*.json) and gap-fill
(fiintrade_gapfill_{TICKER}_*.json) files are unioned per ticker before
saving — same underlying data, just fetched by different runs.

Output schema matches data/investor_flow/{TICKER}.parquet:
  date, ticker, close, tu_doanh_net, tu_doanh_net_total,
  ca_nhan_trongnuoc_net, ca_nhan_trongnuoc_total,
  to_chuc_trongnuoc_net, to_chuc_trongnuoc_total,
  ca_nhan_nuocngoai_net, ca_nhan_nuocngoai_total,
  to_chuc_nuocngoai_net, to_chuc_nuocngoai_total
(net = buy - sell, total = buy + sell, both in tỷ VND — closePrice etc are
raw VND in the source, /1e9 converts value fields to tỷ, close itself is
/1000 to match the "thousand VND" convention data/price already uses.)

Usage:
  python archive/build_fiintrade_investor_flow.py --src "<folder with json files>" --out data/investor_flow_fiintrade
"""

import argparse
import glob
import json
import os
import re
import sys

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(BASE, "data", "investor_flow_fiintrade")

FILENAME_RE = re.compile(r"^fiintrade_(?:gapfill_)?([A-Za-z0-9]+)_\d+\.json$")


def ticker_from_filename(filepath: str) -> str:
    """The harvest script names each download after the ticker it switched
    to (switchTicker(code) in fiintrade_harvest.js), e.g.
    fiintrade_ABB_169....json or fiintrade_gapfill_ABB_169....json.

    This is the ground truth for ticker identity — NOT item['code'] from the
    JSON body. FiinTrade's own 'code' field sometimes returns a different
    internal identifier for the same requested ticker (long org codes,
    company short-names, or even an unrelated-looking ticker), confirmed via
    ticker_mapping.csv (filename_ticker vs json_ticker often differ, e.g.
    ABB -> AAV, AAM -> a bare numeric code). Trusting the filename avoids
    silently mis-attributing rows to whatever code FiinTrade happened to
    echo back.
    """
    m = FILENAME_RE.match(os.path.basename(filepath))
    if not m:
        raise ValueError(f"Unexpected filename format: {filepath}")
    return m.group(1).upper()


def extract_rows(filepath: str) -> list:
    """Return list of dicts, one per (ticker, tradingDate) item found in
    this capture file's GetPriceData responses. Ticker is taken from the
    FILENAME (see ticker_from_filename), not from item['code']."""
    ticker = ticker_from_filename(filepath)
    try:
        with open(filepath, encoding="utf-8") as f:
            captures = json.load(f)
    except Exception as e:
        print(f"  [WARN] failed to load {filepath}: {e}")
        return []

    rows = []
    for cap in captures:
        url = cap.get("url", "")
        if "GetPriceData" not in url:
            continue
        data = cap.get("data")
        if not isinstance(data, dict):
            continue
        items = data.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict) or "tradingDate" not in item:
                continue
            item = dict(item)
            item["ticker"] = ticker  # overrides/ignores item.get('code')
            rows.append(item)
    return rows


def build_dataframe(items: list) -> pd.DataFrame:
    df = pd.DataFrame(items)
    if df.empty:
        return df

    # 'ticker' was already set from the filename in extract_rows (not from
    # the JSON's own 'code' field — see ticker_from_filename's docstring).
    df["date"] = pd.to_datetime(df["tradingDate"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "ticker"])

    # IMPORTANT: use the *Match-suffixed fields, not the plain ones.
    # FiinTrade's plain local{Individual,Institutional}{Buy,Sell}Value INCLUDE
    # put-through/negotiated deals (totalDealValue) — nguoiquansat's domestic
    # breakdown tracks MATCHED (khớp lệnh) trades only. Cross-checked against
    # CTG: plain fields gave a median 8.0 tỷ / max 149 tỷ discrepancy vs
    # nguoiquansat; the Match-suffixed fields gave median 0.003 tỷ (rounding
    # noise), matching the already-good foreign columns' quality (those were
    # correct from the start because their field names — e.g.
    # foreignIndividualBuyTradingMatchValue — are Match variants already).
    #
    # Also: FiinTrade's "institutional" bucket INCLUDES proprietary (tự doanh)
    # desk trading as a subset, not a separate category (confirmed: on days
    # with no put-through, localInstitutionalSellMatchValue exactly equalled
    # proprietaryTotalMatchSellTradeValue). Subtract proprietary-Match from
    # institutional-Match to isolate true institutional flow, matching
    # nguoiquansat's 5 non-overlapping categories.
    num_cols = [
        "closePrice",
        "localIndividualBuyMatchValue", "localIndividualSellMatchValue",
        "localInstitutionalBuyMatchValue", "localInstitutionalSellMatchValue",
        "foreignIndividualBuyTradingMatchValue", "foreignIndividualSellTradingMatchValue",
        "foreignInstitutionalBuyTradingMatchValue", "foreignInstitutionalSellTradingMatchValue",
        "proprietaryTotalMatchBuyTradeValue", "proprietaryTotalMatchSellTradeValue",
    ]
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    inst_buy = df["localInstitutionalBuyMatchValue"] - df["proprietaryTotalMatchBuyTradeValue"]
    inst_sell = df["localInstitutionalSellMatchValue"] - df["proprietaryTotalMatchSellTradeValue"]

    out = pd.DataFrame({
        "date": df["date"],
        "ticker": df["ticker"].str.upper(),
        "close": df["closePrice"] / 1000.0,
        "ca_nhan_trongnuoc_net": (df["localIndividualBuyMatchValue"] - df["localIndividualSellMatchValue"]) / 1e9,
        "ca_nhan_trongnuoc_total": (df["localIndividualBuyMatchValue"] + df["localIndividualSellMatchValue"]) / 1e9,
        "to_chuc_trongnuoc_net": (inst_buy - inst_sell) / 1e9,
        "to_chuc_trongnuoc_total": (inst_buy + inst_sell) / 1e9,
        "ca_nhan_nuocngoai_net": (df["foreignIndividualBuyTradingMatchValue"] - df["foreignIndividualSellTradingMatchValue"]) / 1e9,
        "ca_nhan_nuocngoai_total": (df["foreignIndividualBuyTradingMatchValue"] + df["foreignIndividualSellTradingMatchValue"]) / 1e9,
        "to_chuc_nuocngoai_net": (df["foreignInstitutionalBuyTradingMatchValue"] - df["foreignInstitutionalSellTradingMatchValue"]) / 1e9,
        "to_chuc_nuocngoai_total": (df["foreignInstitutionalBuyTradingMatchValue"] + df["foreignInstitutionalSellTradingMatchValue"]) / 1e9,
        "tu_doanh_net": (df["proprietaryTotalMatchBuyTradeValue"] - df["proprietaryTotalMatchSellTradeValue"]) / 1e9,
        "tu_doanh_net_total": (df["proprietaryTotalMatchBuyTradeValue"] + df["proprietaryTotalMatchSellTradeValue"]) / 1e9,
    })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Folder containing fiintrade_*.json capture files")
    p.add_argument("--out", default=DEFAULT_OUT, help="Output directory for per-ticker parquet files")
    args = p.parse_args()

    files = glob.glob(os.path.join(args.src, "fiintrade_*.json"))
    print(f"Found {len(files)} capture files in {args.src}")
    if not files:
        sys.exit("No files found.")

    os.makedirs(args.out, exist_ok=True)

    all_rows = []
    skipped_files = []
    for i, fp in enumerate(files):
        try:
            rows = extract_rows(fp)
        except ValueError as e:
            skipped_files.append(str(e))
            continue
        all_rows.extend(rows)
        if (i + 1) % 200 == 0:
            print(f"  ...processed {i + 1}/{len(files)} files, {len(all_rows):,} rows so far")

    if skipped_files:
        print(f"  Skipped {len(skipped_files)} file(s) with unparseable names: {skipped_files}")

    print(f"Total raw item rows extracted: {len(all_rows):,}")
    df = build_dataframe(all_rows)
    if df.empty:
        sys.exit("No valid GetPriceData rows found across all files.")

    df = df.sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker", "date"], keep="last")
    print(f"After dedup: {len(df):,} (ticker, date) rows, {df['ticker'].nunique()} tickers")

    # Safety net (ticker identity now comes from the filename, not
    # item['code'] — see ticker_from_filename): drop anything that still
    # isn't in the real market universe, in case a harvest run ever switched
    # to a bad/non-ticker string.
    sectors_path = os.path.join(BASE, "ticker_sectors.csv")
    valid_tickers = set(pd.read_csv(sectors_path)["ticker"].str.upper())
    before = df["ticker"].nunique()
    df = df[df["ticker"].isin(valid_tickers)]
    after = df["ticker"].nunique()
    print(f"Filtered to real tickers (ticker_sectors.csv): {before} -> {after} tickers "
          f"({before - after} non-ticker codes dropped)")

    n_saved = 0
    for ticker, g in df.groupby("ticker"):
        g = g.drop(columns=["ticker"]).sort_values("date").reset_index(drop=True)
        g.to_parquet(os.path.join(args.out, f"{ticker}.parquet"), index=False)
        n_saved += 1

    print(f"Saved {n_saved} per-ticker parquet files to {args.out}")

    # Quick coverage summary
    span = df.groupby("ticker")["date"].agg(["min", "max", "count"])
    span["span_days"] = (span["max"] - span["min"]).dt.days
    print("\nCoverage summary:")
    print(span["span_days"].describe())
    print(f"\nTickers with >1000-day span: {(span['span_days'] > 1000).sum()}")


if __name__ == "__main__":
    main()
