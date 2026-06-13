"""
fetch_financials.py  —  Financial data fetcher (FireAnt)
=========================================================
Two jobs, one source, one output folder.

  Job B  FireAnt HTML    →  data/financials_fa/{SYM}.parquet
           Rich ratios: margins, ROCE/ROIC, growth metrics, sector benchmarks

  Job C  FireAnt REST API →  data/financials_fa/{SYM}.parquet
                           +  data/financials_fa/indicators_snapshot.parquet
           Raw BS + IS + CF statements, derived ratios, daily snapshot

Usage:
  python fetch_financials.py                   # run both jobs B + C
  python fetch_financials.py --jobs B          # only FireAnt HTML
  python fetch_financials.py --jobs C          # only FireAnt API
  python fetch_financials.py --indicators-only # Job C snapshot only (fast)
  python fetch_financials.py --all             # all tickers, not just strategy sectors
  python fetch_financials.py --force           # re-fetch even if fresh
  python fetch_financials.py --days 14         # freshness threshold (default 30)
  python fetch_financials.py --workers 4       # parallel threads (default 4)
"""

import argparse
import datetime as dt
import json
import os
import re
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

warnings.filterwarnings("ignore", category=FutureWarning)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TICKER_CSV  = os.path.join(BASE_DIR, "ticker_sectors.csv")
VNDIRECT_XLSX = os.path.join(BASE_DIR, "vndirect_listing.xlsx")
DIR_FA      = os.path.join(BASE_DIR, "data", "financials_fa")  # Job B + C output
SNAPSHOT_PATH = os.path.join(DIR_FA, "indicators_snapshot.parquet")

STRATEGY_SECTORS = {"Banks", "Food & Beverage", "Basic Resources", "Real Estate"}
TODAY = dt.date.today()


def load_tickers(use_vndirect: bool, include_all: bool) -> pd.DataFrame:
    """
    Load ticker universe from either:
      - ticker_sectors.csv (default)
      - vndirect_listing.xlsx (when --from-vndirect is set)
    Returns standardized columns: ticker, industry, exchange.
    """
    if use_vndirect:
        if not os.path.exists(VNDIRECT_XLSX):
            raise FileNotFoundError(f"Missing file: {VNDIRECT_XLSX}")
        df = pd.read_excel(VNDIRECT_XLSX)
        required = {"code", "IndustryEN", "floor"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"vndirect_listing.xlsx missing required columns: {sorted(missing)}")
        tickers = df.rename(columns={"code": "ticker", "IndustryEN": "industry", "floor": "exchange"})
    else:
        if not os.path.exists(TICKER_CSV):
            raise FileNotFoundError(f"Missing file: {TICKER_CSV}")
        tickers = pd.read_csv(TICKER_CSV)

    if "ticker" not in tickers.columns:
        raise ValueError("Ticker source must contain a 'ticker' column")
    if "industry" not in tickers.columns:
        tickers["industry"] = "Unknown"
    if "exchange" not in tickers.columns:
        tickers["exchange"] = ""

    tickers["ticker"] = tickers["ticker"].astype(str).str.strip().str.upper()
    tickers["industry"] = tickers["industry"].fillna("Unknown").astype(str).str.strip()
    tickers["exchange"] = tickers["exchange"].fillna("").astype(str).str.strip().str.upper()
    tickers = tickers[tickers["ticker"].str.match(r"^[A-Z0-9]{1,10}$", na=False)].copy()

    if not include_all:
        tickers = tickers[tickers["industry"].isin(STRATEGY_SECTORS)].copy()

    tickers = tickers.drop_duplicates("ticker").reset_index(drop=True)
    return tickers[["ticker", "industry", "exchange"]]


def filter_exchange(tickers: pd.DataFrame, no_upcom: bool) -> pd.DataFrame:
    """Drop UPCOM tickers when --no-upcom is set."""
    if not no_upcom:
        return tickers
    before = len(tickers)
    tickers = tickers[tickers["exchange"].str.upper() != "UPCOM"].copy()
    print(f"  [exchange filter] Dropped {before - len(tickers)} UPCOM tickers → {len(tickers)} remain (HOSE + HNX)")
    return tickers

# ── Shared: browser profiles ───────────────────────────────────────────────────
BROWSER_PROFILES = [
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
     "sec-ch-ua": '"Google Chrome";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
     "sec-ch-ua-platform": '"Windows"', "Accept-Language": "en-US,en;q=0.9"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
     "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
     "sec-ch-ua-platform": '"macOS"', "Accept-Language": "en-GB,en;q=0.9"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
     "sec-ch-ua": '"Microsoft Edge";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
     "sec-ch-ua-platform": '"Windows"', "Accept-Language": "en-US,en;q=0.8"},
    {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
     "sec-ch-ua": '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
     "sec-ch-ua-platform": '"Linux"', "Accept-Language": "en-US,en;q=0.7"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
     "Accept-Language": "en-US,en;q=0.5"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
     "sec-ch-ua": '"Chromium";v="146", "Not-A.Brand";v="24", "Google Chrome";v="146"',
     "sec-ch-ua-platform": '"Windows"', "Accept-Language": "en-US,en;q=0.9"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
     "Accept-Language": "en-US,en;q=0.9"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
     "sec-ch-ua": '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
     "sec-ch-ua-platform": '"Windows"', "Accept-Language": "en-AU,en;q=0.9"},
]

# FireAnt Bearer token (long-lived public token)
FA_TOKEN = (
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6IkdYdExONzViZlZQakdvNERWdjV4"
    "QkRITHpnSSIsImtpZCI6IkdYdExONzViZlZQakdvNERWdjV4QkRITHpnSSJ9.eyJpc3MiOiJo"
    "dHRwczovL2FjY291bnRzLmZpcmVhbnQudm4iLCJhdWQiOiJodHRwczovL2FjY291bnRzLmZp"
    "cmVhbnQudm4vcmVzb3VyY2VzIiwiZXhwIjoxODg5NjIyNTMwLCJuYmYiOjE1ODk2MjI1MzAs"
    "ImNsaWVudF9pZCI6ImZpcmVhbnQudHJhZGVzdGF0aW9uIiwic2NvcGUiOlsiYWNhZGVteS1y"
    "ZWFkIiwiYWNhZGVteS13cml0ZSIsImFjY291bnRzLXJlYWQiLCJhY2NvdW50cy13cml0ZSIs"
    "ImJsb2ctcmVhZCIsImNvbXBhbmllcy1yZWFkIiwiZmluYW5jZS1yZWFkIiwiaW5kaXZpZHVh"
    "bHMtcmVhZCIsImludmVzdG9wZWRpYS1yZWFkIiwib3JkZXJzLXJlYWQiLCJvcmRlcnMtd3Jp"
    "dGUiLCJwb3N0cy1yZWFkIiwicG9zdHMtd3JpdGUiLCJzZWFyY2giLCJzeW1ib2xzLXJlYWQi"
    "LCJ1c2VyLWRhdGEtcmVhZCIsInVzZXItZGF0YS13cml0ZSIsInVzZXJzLXJlYWQiXSwianRp"
    "IjoiMjYxYTZhYWQ2MTQ5Njk1ZmJiYzcwODM5MjM0Njc1NWQifQ.dA5-HVzWv-BRfEiAd24u"
    "NBiBxASO-PAyWeWESovZm_hj4aXMAZA1-bWNZeXt88dqogo18AwpDQ-h6gefLPdZSFrG5umC1"
    "dVWaeYvUnGm62g4XS29fj6p01dhKNNqrsu5KrhnhdnKYVv9VdmbmqDfWR8wDgglk5cJFqalzq"
    "6dJWJInFQEPmUs9BW_Zs8tQDn-i5r4tYq2U8vCdqptXoM7YgPllXaPVDeccC9QNu2Xlp9WUv"
    "oROzoQXg25lFub1IYkTrM66gJ6t9fJRZToewCt495WNEOQFa_rwLCZ1QwzvL0iYkONHS_jZ0B"
    "OhBCdW9dWSawD6iF1SIQaFROvMDH1rg"
)

# ── Shared: threading helpers ──────────────────────────────────────────────────
_thread_local = threading.local()
_print_lock   = threading.Lock()


def log(msg: str):
    with _print_lock:
        print(msg, flush=True)


def is_fresh(path: str, max_age_days: int) -> bool:
    if not os.path.exists(path):
        return False
    return (time.time() - os.path.getmtime(path)) / 86_400 < max_age_days


def merge_and_save(path: str, new_df: pd.DataFrame, dedup_cols: list) -> int:
    if os.path.exists(path):
        try:
            old = pd.read_parquet(path)
            combined = (
                pd.concat([new_df, old], ignore_index=True)
                  .drop_duplicates(subset=dedup_cols, keep="first")
                  .reset_index(drop=True)
            )
        except Exception:
            combined = new_df
    else:
        combined = new_df
    combined.to_parquet(path, index=False, engine="pyarrow")
    return len(combined)


def run_with_retry(tasks, worker_fn, workers, tickers_df=None):
    """Run tasks in parallel, retry failures up to 3×."""
    ok, skipped, failed = 0, 0, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker_fn, t): t[0] for t in tasks}
        for future in as_completed(futures):
            sym, status = future.result()
            if   status == "ok":      ok      += 1
            elif status == "skipped": skipped += 1
            else:                     failed.append(sym)

    wait = 15
    for attempt in range(1, 4):
        if not failed or tickers_df is None:
            break
        print(f"\n  Retry {attempt}/3 — {len(failed)} symbols (waiting {wait}s...)")
        time.sleep(wait)
        wait *= 2
        still_failed = []
        for i, sym in enumerate(failed, 1):
            row = tickers_df[tickers_df["ticker"] == sym].iloc[0]
            # rebuild task with same shape as original but force=True at position 6
            task = list(tasks[0])  # get shape reference
            # caller must handle retry via a rebuild — we just re-call sequentially
            _, st = worker_fn(_rebuild_task(sym, row, i, len(failed), task))
            if st == "ok": ok += 1
            else:          still_failed.append(sym)
            time.sleep(2)
        failed = still_failed

    return ok, skipped, failed


# ══════════════════════════════════════════════════════════════════════════════
#  JOB B — FireAnt HTML: rich ratios + growth metrics + sector benchmarks
#  Output: data/financials_fa/{SYM}.parquet
# ══════════════════════════════════════════════════════════════════════════════

_FA_PAGE_URL  = "https://fireant.vn/ma-chung-khoan/{symbol}"
_NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S)


def _jobB_session(profile: dict) -> requests.Session:
    if not hasattr(_thread_local, "sessionB"):
        s = requests.Session()
        s.headers.update({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "sec-ch-ua-mobile": "?0",
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "none",
            "upgrade-insecure-requests": "1",
            **profile,
        })
        _thread_local.sessionB = s
    return _thread_local.sessionB


def _jobB_fetch_next_data(symbol: str, profile: dict) -> dict | None:
    session = _jobB_session(profile)
    url = _FA_PAGE_URL.format(symbol=symbol)
    for attempt in range(3):
        try:
            r = session.get(url, timeout=25)
            if r.status_code == 429:
                time.sleep(20 * (attempt + 1))
                continue
            if r.status_code != 200:
                return None
            m = _NEXT_DATA_RE.search(r.text)
            return json.loads(m.group(1)) if m else None
        except Exception as e:
            if attempt == 2:
                log(f"  [ERR] {symbol} fireant-html: {e}")
            time.sleep(5)
    return None


def _pct_fa(fv, key):
    v = fv.get(key)
    return round(v * 100, 4) if v is not None else None


def _jobB_parse(symbol, industry, exchange, next_data) -> pd.DataFrame | None:
    try:
        fd = (next_data.get("props", {})
                       .get("initialState", {})
                       .get("symbols", {})
                       .get("financialData", {}))
    except Exception:
        return None
    if not fd:
        return None

    rows = []
    for period_key in ("Y", "Q"):
        for rec in fd.get(period_key, {}).get(symbol, []):
            fv      = rec.get("financialValues", {})
            year    = rec.get("year")
            quarter = rec.get("quarter", 0)
            if year is None:
                continue
            np_ = fv.get("ParentCompanyShareholderProfitAfterTax") or fv.get("ProfitAfterTax")
            ocf = fv.get("CashflowFromOperatingActivity")
            rows.append({
                "symbol": symbol, "sector": industry, "platform": exchange,
                "year": year, "quarter": quarter,
                "roe": _pct_fa(fv, "ROE"), "roa": _pct_fa(fv, "ROA"),
                "roce": _pct_fa(fv, "ROCE"), "roic": _pct_fa(fv, "ROIC"),
                "gross_margin": _pct_fa(fv, "GrossMargin"),
                "ebit_margin":  _pct_fa(fv, "EBITMargin"),
                "op_margin":    _pct_fa(fv, "OperatingMargin"),
                "net_margin":   _pct_fa(fv, "ROS"),
                "pe": fv.get("PE"), "pb": fv.get("PB"), "ps": fv.get("PS"),
                "eps_basic": fv.get("BasicEPS"), "eps_diluted": fv.get("DilutedEPS"),
                "debt_equity":       fv.get("TotalDebtOverEquity"),
                "lt_debt_equity":    fv.get("LongtermDebtOverEquity"),
                "current_ratio":     fv.get("CurrentRatio"),
                "quick_ratio":       fv.get("QuickRatio"),
                "interest_coverage": fv.get("InterestCoverageRatio"),
                "sector_roe":  _pct_fa(fv, "SectorROE"),
                "sector_roa":  _pct_fa(fv, "SectorROA"),
                "sector_roce": _pct_fa(fv, "SectorROCE"),
                "sector_pe":   fv.get("SectorPE"),
                "sector_pb":   fv.get("SectorPB"),
                "revenue_growth_lfy": fv.get("SaleGrowth_LFY"),
                "revenue_growth_3yr": fv.get("SaleGrowth_03Yr"),
                "profit_growth_lfy":  fv.get("ProfitGrowth_LFY"),
                "profit_growth_3yr":  fv.get("ProfitGrowth_03Yr"),
                "eps_growth_lfy":     fv.get("DilutedEPSGrowth_LFY"),
                "eps_growth_3yr":     fv.get("DilutedEPSGrowth_03Yr"),
                "ocf": ocf, "capex": fv.get("CAPEX"), "fcf": fv.get("FCF"),
                "ocf_to_netprofit": round(ocf / np_, 4) if ocf and np_ else None,
                "revenue":      fv.get("NetSale"),
                "net_profit":   np_,
                "ebitda":       fv.get("EBITDA"),
                "total_assets": fv.get("TotalAsset"),
                "equity":       fv.get("StockHolderEquity"),
                "total_debt":   fv.get("TotalDebt"),
            })
    return pd.DataFrame(rows) if rows else None


def _jobB_worker(args):
    symbol, industry, exchange, profile_idx, total, position, force, max_age = args
    profile  = BROWSER_PROFILES[profile_idx % len(BROWSER_PROFILES)]
    out_path = os.path.join(DIR_FA, f"{symbol}.parquet")

    if not force and is_fresh(out_path, max_age):
        log(f"[{position:>4}/{total}] {symbol:>6}  SKIP (B)")
        return symbol, "skipped"

    nd = _jobB_fetch_next_data(symbol, profile)
    if nd is None:
        log(f"[{position:>4}/{total}] {symbol:>6}  NO PAGE (B)")
        return symbol, "failed"

    df = _jobB_parse(symbol, industry, exchange, nd)
    if df is None or df.empty:
        log(f"[{position:>4}/{total}] {symbol:>6}  EMPTY (B)")
        return symbol, "failed"

    n = merge_and_save(out_path, df, ["symbol", "year", "quarter"])
    log(f"[{position:>4}/{total}] {symbol:>6}  OK {n:>3}r (B)")
    time.sleep(0.3)
    return symbol, "ok"


# ══════════════════════════════════════════════════════════════════════════════
#  JOB C — FireAnt REST API: raw BS + IS + CF statements + daily snapshot
#  Output: data/financials_fa/{SYM}.parquet  +  indicators_snapshot.parquet
# ══════════════════════════════════════════════════════════════════════════════

_FA_API_URL = "https://restv2.fireant.vn/symbols/{symbol}/full-financial-reports"

_INDICATOR_MAP = {
    "P/E": "pe", "P/S": "ps", "P/B": "pb", "EPS": "eps",
    "%Lãi ròng": "net_margin", "%Lãi gộp": "gross_margin",
    "%EBIT": "ebit_margin", "%Lãi HĐKD": "op_margin",
    "TT Hiện hành": "current_ratio", "TT Nhanh": "quick_ratio",
    "TT Lãi vay": "interest_coverage", "Nợ/VCSH": "debt_equity",
    "ROA": "roa", "ROE": "roe", "ROIC": "roic", "ROCE": "roce",
    "VQ Tổng TS": "asset_turnover", "VQ HTK": "inventory_turnover",
    "VQ KPT": "receivables_turnover", "VQ TSNH": "current_asset_turnover",
}


def _jobC_session() -> requests.Session:
    if not hasattr(_thread_local, "sessionC"):
        s = requests.Session()
        s.headers.update({
            "accept":         "application/json, text/plain, */*",
            "authorization":  f"Bearer {FA_TOKEN}",
            "origin":         "https://fireant.vn",
            "referer":        "https://fireant.vn/",
            "user-agent":     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/147.0.0.0 Safari/537.36",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
        })
        _thread_local.sessionC = s
    return _thread_local.sessionC


def _jobC_fetch_stmt(symbol: str, stmt_type: int, quarter_mode: int, limit: int) -> list | None:
    session = _jobC_session()
    url = _FA_API_URL.format(symbol=symbol)
    for attempt in range(3):
        try:
            r = session.get(url, params={"type": stmt_type, "year": TODAY.year,
                                          "quarter": quarter_mode, "limit": limit}, timeout=20)
            if r.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            if r.status_code != 200:
                return None
            data = r.json()
            return data if isinstance(data, list) else None
        except Exception as e:
            log(f"  [ERR] {symbol} type={stmt_type}: {e}")
            time.sleep(2)
    return None


def _find_row(rows, *keywords):
    kw = [k.lower() for k in keywords]
    for row in rows:
        if isinstance(row, dict) and all(k in (row.get("name") or "").lower() for k in kw):
            return row
    return None


def _extract_values(row):
    if row is None:
        return {}
    return {(int(v["year"]), int(v["quarter"])): v.get("value")
            for v in (row.get("values") or []) if isinstance(v, dict) and v.get("year") is not None}


def _parse_bs(rows):
    if not rows:
        return {}
    ta  = _extract_values(_find_row(rows, "tổng cộng tài sản"))
    eq  = _extract_values(_find_row(rows, "i. vốn chủ sở hữu") or _find_row(rows, "vốn chủ sở hữu"))
    tl  = _extract_values(_find_row(rows, "a. nợ phải trả"))
    std = _extract_values(_find_row(rows, "vay và nợ", "ngắn hạn"))
    ltd = _extract_values(_find_row(rows, "vay và nợ", "dài hạn"))
    out = {}
    for p in set(ta) | set(eq) | set(tl):
        out[p] = {"total_assets": ta.get(p), "equity": eq.get(p), "total_liab": tl.get(p),
                  "short_term_debt": std.get(p), "long_term_debt": ltd.get(p)}
    return out


def _parse_is(rows):
    if not rows:
        return {}
    rv  = _extract_values(_find_row(rows, "doanh thu thuần"))
    gp  = _extract_values(_find_row(rows, "lợi nhuận gộp"))
    op  = _extract_values(_find_row(rows, "lợi nhuận thuần", "kinh doanh"))
    np_ = _extract_values(_find_row(rows, "lợi nhuận sau thuế", "công ty mẹ") or
                          _find_row(rows, "lợi nhuận sau thuế thu nhập"))
    out = {}
    for p in set(rv) | set(np_):
        out[p] = {"revenue": rv.get(p), "gross_profit": gp.get(p),
                  "operating_profit": op.get(p), "net_profit": np_.get(p)}
    return out


def _parse_cf(rows):
    if not rows:
        return {}
    ocf  = _extract_values(_find_row(rows, "lưu chuyển tiền thuần", "kinh doanh"))
    capx = _extract_values(_find_row(rows, "tiền chi", "mua sắm", "tscđ"))
    icf  = _extract_values(_find_row(rows, "lưu chuyển tiền thuần", "đầu tư"))
    fcf  = _extract_values(_find_row(rows, "lưu chuyển tiền thuần", "tài chính"))
    out  = {}
    for p in set(ocf) | set(capx):
        ov, cv = ocf.get(p), capx.get(p)
        out[p] = {"ocf": ov, "capex": abs(cv) if cv is not None else None,
                  "free_cf": (ov or 0) + (cv or 0) if (ov is not None or cv is not None) else None,
                  "investing_cf": icf.get(p), "financing_cf": fcf.get(p)}
    return out


def _safe_pct(a, b):
    try:
        return round(a / b * 100, 4) if a is not None and b and b != 0 else None
    except Exception:
        return None


def _safe_ratio(a, b):
    try:
        return round(a / b, 4) if a is not None and b and b != 0 else None
    except Exception:
        return None


def _build_rows(symbol, industry, bs, is_, cf):
    rows = []
    for (year, quarter) in sorted(set(bs) | set(is_) | set(cf)):
        b = bs.get((year, quarter), {}); i = is_.get((year, quarter), {}); c = cf.get((year, quarter), {})
        ta = b.get("total_assets"); eq = b.get("equity"); tl = b.get("total_liab")
        np_ = i.get("net_profit"); rv = i.get("revenue"); gp = i.get("gross_profit")
        op = i.get("operating_profit"); ocf = c.get("ocf")
        rows.append({
            "symbol": symbol, "industry": industry, "year": year, "quarter": quarter,
            "total_assets": ta, "equity": eq, "total_liab": tl,
            "short_term_debt": b.get("short_term_debt"), "long_term_debt": b.get("long_term_debt"),
            "revenue": rv, "gross_profit": gp, "operating_profit": op, "net_profit": np_,
            "ocf": ocf, "capex": c.get("capex"), "free_cf": c.get("free_cf"),
            "investing_cf": c.get("investing_cf"), "financing_cf": c.get("financing_cf"),
            "roe": _safe_pct(np_, eq), "roa": _safe_pct(np_, ta),
            "net_margin": _safe_pct(np_, rv), "gross_margin": _safe_pct(gp, rv),
            "op_margin": _safe_pct(op, rv), "debt_equity": _safe_ratio(tl, eq),
            "ocf_to_netprofit": _safe_ratio(ocf, np_),
        })
    return rows


def _jobC_worker(args):
    symbol, industry, position, total, force, max_age, q_limit = args
    out_path = os.path.join(DIR_FA, f"{symbol}.parquet")

    if not force and is_fresh(out_path, max_age):
        log(f"[{position:>4}/{total}] {symbol:>6}  SKIP (C)")
        return symbol, "skipped"

    log(f"[{position:>4}/{total}] {symbol:>6}  fetching (C)")
    all_rows = []
    # q_mode=0 → annual, q_mode=4 → quarterly
    # limit controls how many periods are returned
    # annual limit = q_limit // 4, minimum 20 (covers same history as quarterly)
    a_limit = max(20, q_limit // 4)
    for q_mode, limit in [(0, a_limit), (4, q_limit)]:
        bs  = _parse_bs(_jobC_fetch_stmt(symbol, 1, q_mode, limit) or [])
        is_ = _parse_is(_jobC_fetch_stmt(symbol, 2, q_mode, limit) or [])
        cf  = _parse_cf(_jobC_fetch_stmt(symbol, 4, q_mode, limit) or [])
        all_rows.extend(_build_rows(symbol, industry, bs, is_, cf))
        time.sleep(0.3)

    if not all_rows:
        log(f"[{position:>4}/{total}] {symbol:>6}  NO DATA (C)")
        return symbol, "failed"

    n = merge_and_save(out_path, pd.DataFrame(all_rows), ["symbol", "year", "quarter"])
    log(f"[{position:>4}/{total}] {symbol:>6}  OK {n:>3}r (C)")
    return symbol, "ok"


def _jobC_indicators(tickers: pd.DataFrame, workers: int):
    """Fetch daily financial-indicators snapshot for all tickers."""
    print(f"\n  Fetching indicator snapshot ({len(tickers)} tickers)...")

    def _fetch(args):
        sym, ind = args
        session = _jobC_session()
        url = f"https://restv2.fireant.vn/symbols/{sym}/financial-indicators"
        for attempt in range(3):
            try:
                r = session.get(url, timeout=15)
                if r.status_code == 429:
                    time.sleep(15 * (attempt + 1))
                    continue
                if r.status_code != 200:
                    return None
                data = r.json()
                if not isinstance(data, list):
                    return None
                row = {"symbol": sym, "industry": ind, "date": str(TODAY)}
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    col = _INDICATOR_MAP.get((item.get("shortName") or "").strip())
                    if col is None:
                        continue
                    row[col] = item.get("value")
                    row[f"{col}_vs_industry"] = (
                        (item.get("value") or 0) - (item.get("industryValue") or 0)
                        if item.get("value") is not None and item.get("industryValue") is not None
                        else None
                    )
                return row
            except Exception as e:
                log(f"  [ERR] {sym} indicators: {e}")
                time.sleep(2)
        return None

    tasks  = [(row["ticker"], row["industry"]) for _, row in tickers.iterrows()]
    rows   = [r for r in (ThreadPoolExecutor(max_workers=workers)
                          .map(_fetch, tasks)) if r]

    if not rows:
        print("  No indicator data returned.")
        return

    new_df = pd.DataFrame(rows)
    if os.path.exists(SNAPSHOT_PATH):
        try:
            old  = pd.read_parquet(SNAPSHOT_PATH)
            mask = ~((old["date"] == str(TODAY)) & (old["symbol"].isin(new_df["symbol"])))
            combined = pd.concat([old[mask], new_df], ignore_index=True)
        except Exception:
            combined = new_df
    else:
        combined = new_df

    combined.to_parquet(SNAPSHOT_PATH, index=False, engine="pyarrow")
    print(f"  Snapshot saved: {len(rows)} stocks, {len(combined)} total rows → {SNAPSHOT_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Financial data fetcher — FireAnt Jobs B + C.")
    parser.add_argument("--jobs",            default="BC",   help="Which jobs to run: B, C, or BC (default: BC)")
    parser.add_argument("--all",             action="store_true", help="All tickers, not just strategy sectors")
    parser.add_argument("--from-vndirect",   action="store_true", help="Load ticker universe from vndirect_listing.xlsx")
    parser.add_argument("--force",           action="store_true", help="Re-fetch even if parquet is fresh")
    parser.add_argument("--days",            type=int, default=30, help="Freshness threshold in days (default 30)")
    parser.add_argument("--workers",         type=int, default=4,  help="Parallel threads (default 4)")
    parser.add_argument("--quarters",        type=int, default=80, help="Number of quarterly periods to fetch in Job C (default 80 = full API history ~19 years)")
    parser.add_argument("--indicators-only", action="store_true",  help="Job C snapshot only, skip full statements")
    parser.add_argument("--no-upcom",        action="store_true",  help="Exclude UPCOM-listed tickers (keep HOSE + HNX only)")
    args = parser.parse_args()

    jobs = args.jobs.upper()
    os.makedirs(DIR_FA, exist_ok=True)

    tickers = load_tickers(use_vndirect=args.from_vndirect, include_all=args.all)
    tickers = filter_exchange(tickers, no_upcom=args.no_upcom)

    source = "vndirect_listing.xlsx" if args.from_vndirect else "ticker_sectors.csv"
    scope  = "ALL tickers" if args.all else "strategy sectors"
    exch   = " (HOSE+HNX only)" if args.no_upcom else ""
    print(f"[{TODAY}]  fetch_financials.py  jobs={jobs}  scope={scope}{exch}  source={source}")
    print(f"  Tickers={len(tickers)}  days={args.days}  force={args.force}  workers={args.workers}\n")

    # ── Job B ──────────────────────────────────────────────────────────────────
    if "B" in jobs:
        print("── Job B: FireAnt HTML fundamentals ─────────────────────────")
        tasks = [
            (row["ticker"], row["industry"], row.get("exchange", ""),
             i % args.workers, len(tickers), i + 1, args.force, args.days)
            for i, row in tickers.iterrows()
        ]
        ok = skipped = 0; failed = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_jobB_worker, t): t[0] for t in tasks}
            for f in as_completed(futures):
                sym, st = f.result()
                if st == "ok": ok += 1
                elif st == "skipped": skipped += 1
                else: failed.append(sym)
        wait = 20
        for attempt in range(1, 4):
            if not failed: break
            print(f"  Retry {attempt}/3 — {len(failed)} failed (waiting {wait}s...)")
            time.sleep(wait); wait *= 2
            still = []
            for i, sym in enumerate(failed, 1):
                row = tickers[tickers["ticker"] == sym].iloc[0]
                _, st = _jobB_worker((sym, row["industry"], row.get("exchange", ""), 0,
                                      len(failed), i, True, args.days))
                if st == "ok": ok += 1
                else: still.append(sym)
                time.sleep(2)
            failed = still
        print(f"  B done: ok={ok}  skipped={skipped}  failed={len(failed)}\n")

    # ── Job C ──────────────────────────────────────────────────────────────────
    if "C" in jobs:
        print("── Job C: FireAnt API statements + daily snapshot ───────────")
        _jobC_indicators(tickers, args.workers)

        if not args.indicators_only:
            tasks = [
                (row["ticker"], row["industry"], i + 1, len(tickers), args.force, args.days, args.quarters)
                for i, row in tickers.iterrows()
            ]
            ok = skipped = 0; failed = []
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_jobC_worker, t): t[0] for t in tasks}
                for f in as_completed(futures):
                    sym, st = f.result()
                    if st == "ok": ok += 1
                    elif st == "skipped": skipped += 1
                    else: failed.append(sym)
            wait = 15
            for attempt in range(1, 4):
                if not failed: break
                print(f"  Retry {attempt}/3 — {len(failed)} failed (waiting {wait}s...)")
                time.sleep(wait); wait *= 2
                still = []
                for i, sym in enumerate(failed, 1):
                    row = tickers[tickers["ticker"] == sym].iloc[0]
                    _, st = _jobC_worker((sym, row["industry"], i, len(failed), True, args.days, args.quarters))
                    if st == "ok": ok += 1
                    else: still.append(sym)
                    time.sleep(2)
                failed = still
            print(f"  C done: ok={ok}  skipped={skipped}  failed={len(failed)}\n")

    print("── All done ──────────────────────────────────────────────────")
    print(f"  data/financials_fa/  (Job B+C fundamentals + snapshot)")


if __name__ == "__main__":
    main()
