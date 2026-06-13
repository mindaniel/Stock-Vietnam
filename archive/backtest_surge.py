#!/usr/bin/env python3
"""
backtest_surge.py
Backtests the surge_detector signals over the available tick history.
Computes win rates by score bucket.
"""
import os, sys, pandas as pd, numpy as np, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

TICK_DIR = os.path.join(BASE_DIR, "data", "tick_data")
DATA_DIR = os.path.join(BASE_DIR, "data", "price")
TD_FILE  = os.path.join(BASE_DIR, "tudoanh", "tudoanh_all.csv")

# ── Collect all trading dates ──────────────────────────────────────────────────
print("Scanning tick dates...")
all_dates = set()
for f in os.listdir(TICK_DIR):
    if not f.endswith(".parquet"):
        continue
    df = pd.read_parquet(os.path.join(TICK_DIR, f), columns=["td"])
    all_dates.update(df["td"].dropna().unique())

dates_vn  = sorted(all_dates, key=lambda d: dt.datetime.strptime(d, "%d/%m/%Y"))
dates_iso = [dt.datetime.strptime(d, "%d/%m/%Y").strftime("%Y-%m-%d") for d in dates_vn]
symbols   = [f.replace(".parquet", "") for f in os.listdir(TICK_DIR) if f.endswith(".parquet")]

print(f"  Trading days : {len(dates_vn)}  ({dates_vn[0]} to {dates_vn[-1]})")
print(f"  Symbols      : {len(symbols)}")

# ── Load tu doanh ──────────────────────────────────────────────────────────────
td_raw = pd.DataFrame()
if os.path.exists(TD_FILE):
    td_raw = pd.read_csv(TD_FILE, encoding="utf-8-sig")
    parsed = pd.to_datetime(td_raw["date"], format="%d/%m/%Y", errors="coerce")
    mask   = parsed.isna()
    if mask.any():
        parsed[mask] = pd.to_datetime(td_raw["date"][mask], format="%Y-%m-%d", errors="coerce")
    td_raw["date"] = parsed.dt.strftime("%Y-%m-%d")

# ── Cache price CSVs ───────────────────────────────────────────────────────────
print("Loading price CSVs...")
price_cache = {}
for sym in symbols:
    path = os.path.join(DATA_DIR, f"{sym}.parquet")
    if not os.path.exists(path):
        continue
    df = pd.read_parquet(path)
    if "time" not in df.columns or df.empty:
        continue
    df = df.sort_values("time").reset_index(drop=True)
    price_cache[sym] = df

# ── Cache tick data ────────────────────────────────────────────────────────────
print("Loading tick parquets...")
tick_cache = {}
for sym in symbols:
    path = os.path.join(TICK_DIR, f"{sym}.parquet")
    df = pd.read_parquet(path)
    df["t"] = df["t"].astype(str)
    df["v"] = pd.to_numeric(df["v"], errors="coerce").fillna(0)
    df["s"] = df["s"].astype(str).str.strip().str.lower()
    tick_cache[sym] = df

print("Data loaded. Running backtest...\n")

# ── Scoring function (mirrors surge_detector logic) ───────────────────────────
def score_one(sym, date_iso, date_vn, td_today):
    score = 0.0

    # Price signals
    pdf = price_cache.get(sym)
    if pdf is not None:
        idx = pdf.index[pdf["time"] == date_iso]
        if len(idx) > 0:
            i    = idx[-1]
            row  = pdf.iloc[i]
            past = pdf.iloc[max(0, i - 20):i]
            if len(past) >= 5:
                close   = float(row.get("close",  0) or 0)
                high    = float(row.get("high",   0) or 0)
                low     = float(row.get("low",    0) or 0)
                vol     = float(row.get("volume", 0) or 0)
                avg_vol = past["volume"].mean()

                if close > 0 and vol > 0 and avg_vol > 0:
                    vol_x = vol / avg_vol
                    score += (20 if vol_x >= 3 else 14 if vol_x >= 2
                              else 8 if vol_x >= 1.5 else 4 if vol_x >= 1.2 else 0)

                    if "high" in past.columns:
                        h20 = past["high"].max()
                        h10 = past.tail(10)["high"].max()
                        h5  = past.tail(5)["high"].max()
                        score += (15 if close > h20 else 10 if close > h10
                                  else 5 if close > h5 else 0)

                        if "low" in past.columns:
                            rng5  = ((past.tail(5)["high"] - past.tail(5)["low"])
                                     / past.tail(5)["close"]).mean()
                            today_rng = (high - low) / close
                            if rng5 > 0:
                                score += min(10, max(0, (today_rng - rng5) / rng5 * 10))

                    fb    = float(row.get("foreign_buy_vol",  0) or 0)
                    fs    = float(row.get("foreign_sell_vol", 0) or 0)
                    f_net = fb - fs
                    streak = 0
                    for _, r in past.iloc[::-1].iterrows():
                        if (float(r.get("foreign_buy_vol",  0) or 0)
                                - float(r.get("foreign_sell_vol", 0) or 0)) > 0:
                            streak += 1
                        else:
                            break
                    score += min(6, max(0, f_net / avg_vol) * 6) + min(2, streak * 0.5)

    # Tick signals
    tdf = tick_cache.get(sym)
    if tdf is not None:
        t = tdf[tdf["td"] == date_vn]
        if len(t) >= 10:
            day_vol = t["v"].sum()
            if day_vol > 0:
                late  = t[t["t"] >= "14:00:00"]
                lb    = late[late["s"] == "buy"]["v"].sum()
                ls    = late[late["s"] == "sell"]["v"].sum()
                lr    = lb / (lb + ls) if (lb + ls) > 0 else 0.5
                score += max(0, (lr - 0.5) / 0.5) * 20

                first_hr = t[(t["t"] >= "09:00:00") & (t["t"] < "10:00:00")]["v"].sum()
                last_hr  = t[t["t"] >= "13:45:00"]["v"].sum()
                if first_hr > 0:
                    score += min(10, max(0, (last_hr / first_hr - 1) / 3) * 10)

                avg_v   = t["v"].mean()
                thresh  = max(avg_v * 5, 100_000)
                big2h   = t[(t["t"] >= "13:00:00") & (t["s"] == "buy")
                             & (t["v"] >= thresh)]["v"].sum()
                score  += min(10, (big2h / day_vol) * 30)

    # Tu doanh
    if not td_today.empty and sym in td_today.index:
        row_td = td_today.loc[[sym]].iloc[0]
        net_val = float(row_td.get("net_value", 0) or 0)
        if net_val > 0:
            score += min(7, (net_val / 1e9) * 1.5)

    return round(score, 1)


def next_day_return(sym, idx_today):
    pdf = price_cache.get(sym)
    if pdf is None:
        return None
    # Find next row with valid (non-zero) close
    for j in range(idx_today + 1, min(idx_today + 3, len(pdf))):
        c0 = float(pdf.iloc[idx_today].get("close", 0) or 0)
        c1 = float(pdf.iloc[j].get("close", 0) or 0)
        if c0 > 0 and c1 > 0:
            return (c1 - c0) / c0 * 100
    return None


def next_day_open_return(sym, idx_today):
    pdf = price_cache.get(sym)
    if pdf is None:
        return None
    for j in range(idx_today + 1, min(idx_today + 3, len(pdf))):
        c0 = float(pdf.iloc[idx_today].get("close", 0) or 0)
        o1 = float(pdf.iloc[j].get("open",  0) or 0)
        if c0 > 0 and o1 > 0:
            return (o1 - c0) / c0 * 100
    return None


# ── Backtest loop ─────────────────────────────────────────────────────────────
records = []

# Exclude the last date (no next day to evaluate)
for date_vn, date_iso in zip(dates_vn[:-1], dates_iso[:-1]):
    td_sub   = td_raw[td_raw["date"] == date_iso] if not td_raw.empty else pd.DataFrame()
    td_today = td_sub.set_index("symbol") if not td_sub.empty else pd.DataFrame()

    for sym in symbols:
        sc = score_one(sym, date_iso, date_vn, td_today)
        if sc <= 0:
            continue

        pdf = price_cache.get(sym)
        if pdf is None:
            continue
        idx = pdf.index[pdf["time"] == date_iso]
        if len(idx) == 0:
            continue
        i = idx[-1]

        ret_close = next_day_return(sym, i)
        ret_open  = next_day_open_return(sym, i)
        if ret_close is None:
            continue

        records.append({
            "sym":       sym,
            "date":      date_iso,
            "score":     sc,
            "ret_close": round(ret_close, 3),
            "ret_open":  round(ret_open, 3) if ret_open is not None else None,
        })

df = pd.DataFrame(records)
print(f"Observations (score > 0): {len(df):,}")
print(f"Dates tested            : {len(dates_vn) - 1}")
print(f"Symbols tested          : {len(symbols)}")
print()

# ── Win rate table ────────────────────────────────────────────────────────────
buckets = [(0, 10, "0–10"), (10, 20, "10–20"), (20, 30, "20–30"),
           (30, 45, "30–45"), (45, 200, "45+")]

print(f"{'Score bucket':<14}  {'N':>5}  {'WR>0%':>7}  {'WR>1%':>7}  "
      f"{'WR>2%':>7}  {'WR>3%':>7}  {'Avg ret':>8}  {'Med ret':>8}")
print("-" * 75)
for lo, hi, label in buckets:
    sub = df[(df["score"] >= lo) & (df["score"] < hi)]
    if sub.empty:
        continue
    r = sub["ret_close"]
    print(f"{label:<14}  {len(sub):>5}  "
          f"{(r > 0).mean()*100:>6.1f}%  {(r > 1).mean()*100:>6.1f}%  "
          f"{(r > 2).mean()*100:>6.1f}%  {(r > 3).mean()*100:>6.1f}%  "
          f"{r.mean():>7.2f}%  {r.median():>7.2f}%")

print()
# Compare: random baseline (all stocks all days)
all_rets = []
for sym in symbols:
    pdf = price_cache.get(sym)
    if pdf is None:
        continue
    pdf2 = pdf[pdf["time"].isin(dates_iso[:-1])].copy()
    for i in pdf2.index:
        if i + 1 < len(pdf):
            c0 = float(pdf.iloc[i].get("close", 0) or 0)
            c1 = float(pdf.iloc[i+1].get("close", 0) or 0)
            if c0 > 0:
                all_rets.append((c1-c0)/c0*100)

ar = pd.Series(all_rets)
print(f"BASELINE (all stocks, all test days, no filter):")
print(f"  N={len(ar):,}  WR>0%={(ar>0).mean()*100:.1f}%  WR>1%={(ar>1).mean()*100:.1f}%  "
      f"Avg={ar.mean():.2f}%  Med={ar.median():.2f}%")

print()
print("── TOP INSTANCES (score >= 30) ─────────────────────────────────────────")
top = df[df["score"] >= 30].sort_values("score", ascending=False).head(40)
if not top.empty:
    print(top[["date", "sym", "score", "ret_close", "ret_open"]].to_string(index=False))

print()
print("── SCORE >= 30: RETURN DISTRIBUTION ────────────────────────────────────")
top30 = df[df["score"] >= 30]["ret_close"]
if not top30.empty:
    for pct in [10, 25, 50, 75, 90]:
        print(f"  P{pct}: {top30.quantile(pct/100):.2f}%")
    print(f"  Min : {top30.min():.2f}%")
    print(f"  Max : {top30.max():.2f}%")
