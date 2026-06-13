"""
commitment_levels.py  — Volume Profile + Commitment Level Analysis
===================================================================
Shows three things:
  1. TARGET LEVELS  — where to consider selling (HVN resistance above price)
  2. COMMITMENT ZONES — where most people bought/sold (High/Low Volume Nodes)
  3. BIG MONEY SIGNAL — is institutional money buying or distributing?

Big money sources used:
  • Foreign net flow (foreign_buy_vol - foreign_sell_vol) — foreigners in VN
    are 90%+ institutional funds. Best proxy for "smart money".
  • Block trade ratio (tick data) — unusually large single ticks vs average
    = institutional block orders, not retail small lots.
  • Tick absorption (tick data) — tvb/tvs shows the buy/sell queue. When
    sell queue (tvs) is shrinking faster than buy queue → buyers absorbing
    supply = accumulation. Reverse = distribution.
  • Order book ratio (order_history) — ob_ratio = buy_vol/sell_vol per day.
    Trending down while price rises = hidden distribution.

Usage:
  python commitment_levels.py ACB
  python commitment_levels.py VCB --entry 85.0
  python commitment_levels.py HPG --lookback 120
"""
import sys, os, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.stdout.reconfigure(encoding="utf-8")

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE, "data")
PRICE_DIR = os.path.join(DATA_DIR, "price")

# ── Args ─────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("ticker")
ap.add_argument("--entry",    type=float, default=None, help="Entry price (k VND)")
ap.add_argument("--lookback", type=int,   default=180,  help="Days of history")
ap.add_argument("--bins",     type=int,   default=50,   help="Price bins")
ap.add_argument("--no-plot",  action="store_true")
args = ap.parse_args()

TICKER   = args.ticker.upper()
ENTRY    = args.entry
LOOKBACK = args.lookback
N_BINS   = args.bins
VA_PCT   = 0.70

# ── Load OHLCV ───────────────────────────────────────────────────────────────
raw = pd.read_parquet(os.path.join(PRICE_DIR, f"{TICKER}.parquet"))
raw = raw.rename(columns={"time": "date"})
raw["date"] = pd.to_datetime(raw["date"])
raw = raw.sort_values("date").reset_index(drop=True)

cutoff = raw["date"].max() - pd.Timedelta(days=LOOKBACK)
df     = raw[raw["date"] >= cutoff].copy().reset_index(drop=True)

current = float(df["close"].iloc[-1])
today   = df["date"].iloc[-1].date()

print(f"\n{'='*62}")
print(f"  {TICKER}  |  {today}  |  Current: {current:.2f}k VND", end="")
if ENTRY:
    gain = (current - ENTRY) / ENTRY
    print(f"  |  Entry: {ENTRY:.2f}k  ({gain:+.1%})", end="")
print(f"\n{'='*62}")

# ════════════════════════════════════════════════════════════════
# SECTION 1 — VOLUME PROFILE (Commitment Zones)
# ════════════════════════════════════════════════════════════════
price_lo  = df["low"].min()
price_hi  = df["high"].max()
bins      = np.linspace(price_lo * 0.998, price_hi * 1.002, N_BINS + 1)
bc        = (bins[:-1] + bins[1:]) / 2   # bin centres
vp        = np.zeros(N_BINS)             # volume profile

for _, r in df.iterrows():
    lo, hi, cl, vol = r["low"], r["high"], r["close"], r["volume"]
    if hi <= lo or vol == 0:
        continue
    for i in range(N_BINS):
        if bins[i+1] < lo or bins[i] > hi:
            continue
        dist   = abs(bc[i] - cl) / max(hi - lo, 0.01)
        weight = max(1.0 - dist, 0.1)
        vp[i] += vol * weight

# Blend with recent tick data if available
tick_path = os.path.join(DATA_DIR, "tick_data", f"{TICKER}.parquet")
tick_vp   = np.zeros(N_BINS)
ticks_df  = None

if os.path.exists(tick_path):
    try:
        ticks_df = pd.read_parquet(tick_path)
        ticks_df["td"] = pd.to_datetime(ticks_df["td"], dayfirst=True)
        recent_ticks   = ticks_df[ticks_df["td"] >= ticks_df["td"].max() - pd.Timedelta(days=30)]
        price_k        = recent_ticks["p"] / 1000.0
        for i in range(N_BINS):
            mask         = (price_k >= bins[i]) & (price_k < bins[i+1])
            tick_vp[i]   = recent_ticks.loc[mask, "v"].sum()
    except Exception:
        ticks_df = None

if tick_vp.sum() > 0:
    vp_norm = vp / vp.sum()
    tv_norm = tick_vp / tick_vp.sum()
    combined = 0.55 * tv_norm + 0.45 * vp_norm   # weight recent ticks more
else:
    combined = vp / vp.sum()

# POC
poc_i   = int(np.argmax(combined))
poc     = bc[poc_i]

# Value Area (70%)
va_vol  = combined[poc_i]
lo_i    = poc_i
hi_i    = poc_i
rem     = combined.copy()
rem[poc_i] = 0

while va_vol < VA_PCT and (lo_i > 0 or hi_i < N_BINS - 1):
    nxt_lo = rem[lo_i - 1] if lo_i > 0       else -1
    nxt_hi = rem[hi_i + 1] if hi_i < N_BINS-1 else -1
    if nxt_hi >= nxt_lo and hi_i < N_BINS - 1:
        hi_i += 1; va_vol += rem[hi_i]; rem[hi_i] = 0
    elif lo_i > 0:
        lo_i -= 1; va_vol += rem[lo_i]; rem[lo_i] = 0
    else:
        break

VAH = bc[hi_i]
VAL = bc[lo_i]

# Top-5 HVN above and below current price
sorted_i   = np.argsort(combined)[::-1]
hvn_all    = [bc[i] for i in sorted_i if i != poc_i][:8]
hvn_above  = sorted([p for p in hvn_all if p > current])
hvn_below  = sorted([p for p in hvn_all if p < current], reverse=True)

# LVN above current (fast-move zones = price passes quickly)
lvn_thresh = np.percentile(combined[combined > 0], 25)
lvn_above  = sorted([bc[i] for i in range(N_BINS)
                     if bc[i] > current and combined[i] <= lvn_thresh])

# Anchored VWAP (from 60d low)
swing = df.tail(min(60, len(df)))
vwap_start = df[df.index >= swing["low"].idxmin()].copy()
if len(vwap_start) >= 3:
    tp    = (vwap_start["high"] + vwap_start["low"] + vwap_start["close"]) / 3
    vwap  = (tp * vwap_start["volume"]).sum() / vwap_start["volume"].sum()
else:
    vwap  = current

# ════════════════════════════════════════════════════════════════
# SECTION 2 — BIG MONEY ANALYSIS
# ════════════════════════════════════════════════════════════════

# ── A. Foreign net flow (best big-money proxy) ───────────────────
fdf = df[["date","close","foreign_buy_vol","foreign_sell_vol"]].dropna(
        subset=["foreign_buy_vol","foreign_sell_vol"]).copy()
fdf["net_f"]   = fdf["foreign_buy_vol"] - fdf["foreign_sell_vol"]
fdf["net_f_5"] = fdf["net_f"].rolling(5).mean()
fdf["net_f_20"]= fdf["net_f"].rolling(20).mean()

foreign_recent_5d   = float(fdf["net_f"].tail(5).sum())
foreign_recent_20d  = float(fdf["net_f"].tail(20).sum())
foreign_5d_avg      = float(fdf["net_f_5"].iloc[-1])   if not fdf["net_f_5"].isna().all()  else 0
foreign_20d_avg     = float(fdf["net_f_20"].iloc[-1])  if not fdf["net_f_20"].isna().all() else 0
foreign_trend       = "SELLING" if foreign_5d_avg < 0 else "BUYING"
foreign_accel       = "accelerating" if abs(foreign_5d_avg) > abs(foreign_20d_avg) else "decelerating"

# Cumulative foreign flow over lookback (did they accumulate or distribute?)
foreign_cumulative = float(fdf["net_f"].sum())
fdf["cum_net"] = fdf["net_f"].cumsum()
# Find where foreigners started net selling consistently
sell_streak = 0
for v in reversed(fdf["net_f"].values):
    if v < 0:
        sell_streak += 1
    else:
        break

# ── B. Block trade detection (tick data) ─────────────────────────
block_signal = None
block_detail = ""

if ticks_df is not None and len(ticks_df) > 0:
    try:
        recent_t   = ticks_df[ticks_df["td"] >= ticks_df["td"].max() - pd.Timedelta(days=10)]
        avg_tick   = recent_t["v"].mean()
        block_thresh = avg_tick * 10        # 10x average tick = block trade
        blocks     = recent_t[recent_t["v"] >= block_thresh]

        if len(blocks) > 0:
            # Classify block direction: if tvs decreasing rapidly = block sell
            # Approximate: use surrounding ticks
            block_buy_vol  = 0
            block_sell_vol = 0
            for _, row in blocks.iterrows():
                day_ticks = recent_t[recent_t["td"] == row["td"]]
                # tvs decreasing = sell queue being consumed = buyers dominant
                tvs_change = day_ticks["tvs"].diff().mean()
                tvb_change = day_ticks["tvb"].diff().mean()
                if tvs_change < tvb_change:        # sell queue shrinking faster
                    block_buy_vol  += row["v"]     # buyers absorbing = bullish
                else:
                    block_sell_vol += row["v"]

            total_block = block_buy_vol + block_sell_vol
            if total_block > 0:
                buy_ratio = block_buy_vol / total_block
                if buy_ratio > 0.60:
                    block_signal = "ACCUMULATION"
                    block_detail = f"{buy_ratio:.0%} of block trades buyer-initiated"
                elif buy_ratio < 0.40:
                    block_signal = "DISTRIBUTION"
                    block_detail = f"{1-buy_ratio:.0%} of block trades seller-initiated"
                else:
                    block_signal = "NEUTRAL"
                    block_detail = f"balanced block flow ({buy_ratio:.0%} buy)"

            block_count = len(blocks)
            block_pct   = block_count / len(recent_t) * 100
    except Exception:
        pass

# ── C. Tick absorption (today's queue) ───────────────────────────
absorption_signal = None
absorption_detail = ""

if ticks_df is not None:
    try:
        today_ticks = ticks_df[ticks_df["td"] == ticks_df["td"].max()]
        if len(today_ticks) > 10:
            # tvb/tvs at end of day vs start
            tvb_end   = float(today_ticks["tvb"].iloc[-1])
            tvs_end   = float(today_ticks["tvs"].iloc[-1])
            tvb_start = float(today_ticks["tvb"].iloc[0])
            tvs_start = float(today_ticks["tvs"].iloc[0])

            tvb_consumed = tvb_start - tvb_end    # buy orders consumed (sellers ate bids)
            tvs_consumed = tvs_start - tvs_end    # sell orders consumed (buyers ate asks)

            if tvs_consumed > 0 and tvb_consumed > 0:
                ratio = tvs_consumed / (tvs_consumed + tvb_consumed)
                if ratio > 0.60:
                    absorption_signal = "BUYERS ABSORBING"
                    absorption_detail = f"Today: {tvs_consumed/1e6:.1f}M sell orders absorbed vs {tvb_consumed/1e6:.1f}M buy orders"
                elif ratio < 0.40:
                    absorption_signal = "SELLERS ABSORBING"
                    absorption_detail = f"Today: {tvb_consumed/1e6:.1f}M buy orders absorbed vs {tvs_consumed/1e6:.1f}M sell orders"
                else:
                    absorption_signal = "BALANCED"
                    absorption_detail = f"Sell absorbed: {tvs_consumed/1e6:.1f}M  Buy absorbed: {tvb_consumed/1e6:.1f}M"
    except Exception:
        pass

# ── D. Order book ratio trend ─────────────────────────────────────
ob_signal = None
ob_detail = ""
ob_trend_dir = None

try:
    oh = pd.read_parquet(os.path.join(DATA_DIR, "order_history", f"{TICKER}.parquet"))
    oh["date"] = pd.to_datetime(oh["date"])
    oh = oh.sort_values("date")
    recent_oh = oh.dropna(subset=["ob_ratio"]).tail(20)
    if len(recent_oh) >= 5:
        ob_5d  = float(recent_oh["ob_ratio"].tail(5).mean())
        ob_20d = float(recent_oh["ob_ratio"].mean())
        ob_trend_dir = "declining" if ob_5d < ob_20d * 0.97 else ("rising" if ob_5d > ob_20d * 1.03 else "stable")

        # Price trending up but ob_ratio trending down = distribution
        price_trend = df["close"].tail(10).iloc[-1] > df["close"].tail(10).iloc[0]
        if price_trend and ob_5d < ob_20d * 0.95:
            ob_signal = "⚠️  DIVERGENCE: price up, buy/sell ratio down → possible distribution"
        elif ob_5d > 1.05:
            ob_signal = "✓  Buy-dominant order flow"
        elif ob_5d < 0.95:
            ob_signal = "⚠️  Sell-dominant order flow"
        else:
            ob_signal = "~  Balanced order flow"

        ob_detail = f"ob_ratio 5d avg: {ob_5d:.3f}  |  20d avg: {ob_20d:.3f}  |  trend: {ob_trend_dir}"
except Exception:
    pass

# ════════════════════════════════════════════════════════════════
# PRINT REPORT
# ════════════════════════════════════════════════════════════════

# ── 1. Commitment Zones ──────────────────────────────────────────
print(f"\n  ① COMMITMENT ZONES  (where people bought/sold most)")
print(f"  {'─'*55}")
print(f"  POC  (max volume, strongest level) : {poc:.2f}k VND")
print(f"  VAH  (value area top,  70% vol)    : {VAH:.2f}k VND")
print(f"  VAL  (value area base, 70% vol)    : {VAL:.2f}k VND")
print(f"  Anchored VWAP (from 60d low)       : {vwap:.2f}k VND")
print()

pos_tag = ("ABOVE Value Area → stretched / distribution zone" if current > VAH
           else "Inside Value Area → balanced zone" if current >= VAL
           else "BELOW Value Area → undervalued / accumulation zone")
print(f"  Current {current:.2f}k : {pos_tag}")

print()
print(f"  High Volume Nodes — RESISTANCE above {current:.2f}k:")
if hvn_above:
    for p in hvn_above[:3]:
        pct = (p - current) / current * 100
        print(f"    {p:.2f}k  (+{pct:.1f}%)  ← where sellers likely to appear")
else:
    print(f"    (none in range — price at all-time high territory)")

print()
print(f"  High Volume Nodes — SUPPORT below {current:.2f}k:")
for p in hvn_below[:3]:
    pct = (current - p) / current * 100
    print(f"    {p:.2f}k  (-{pct:.1f}%)  ← where buyers likely to step in")

if lvn_above:
    print()
    print(f"  Low Volume Nodes above (thin air, price moves fast through):")
    for p in lvn_above[:2]:
        pct = (p - current) / current * 100
        print(f"    {p:.2f}k  (+{pct:.1f}%)")

# ── 2. Target Levels ─────────────────────────────────────────────
print(f"\n  ② TARGET LEVELS  (where to consider selling)")
print(f"  {'─'*55}")

targets = []
# T1: first HVN resistance
if hvn_above:
    targets.append((hvn_above[0],  "T1 — first resistance HVN",     "take partial profit"))
# T2: second HVN or VAH
if len(hvn_above) >= 2:
    targets.append((hvn_above[1],  "T2 — second resistance HVN",    "take more profit"))
elif VAH > current:
    targets.append((VAH,           "T2 — Value Area High",          "top of value zone"))
# T3: extension (if price breaks above VAH)
if VAH > current:
    ext = VAH + (VAH - poc) * 0.5
    targets.append((ext,           "T3 — extension (VAH + 0.5×range)", "if strong breakout"))

# Support / stop
stop = hvn_below[0] if hvn_below else VAL

print(f"  Entry reference  : {ENTRY:.2f}k" if ENTRY else f"  Current price    : {current:.2f}k")
for price, label, note in targets:
    pct_from_cur = (price - current) / current * 100
    pct_from_ent = (price - ENTRY)   / ENTRY   * 100 if ENTRY else None
    entry_str    = f"  |  {pct_from_ent:+.1f}% from entry" if pct_from_ent else ""
    print(f"  {label:<34}: {price:.2f}k  ({pct_from_cur:+.1f}% from now{entry_str})")

print(f"  {'─'*55}")
print(f"  Stop / support   : {stop:.2f}k  ({(stop-current)/current*100:+.1f}% — key support if breaks)")

# ── 3. Big Money Signal ──────────────────────────────────────────
print(f"\n  ③ BIG MONEY SIGNAL")
print(f"  {'─'*55}")

# A. Foreign flow
flow_tag  = "🔴 NET SELLING" if foreign_5d_avg < 0 else "🟢 NET BUYING"
flow_5d   = f"{foreign_recent_5d/1e6:+.1f}M shares"
flow_20d  = f"{foreign_recent_20d/1e6:+.1f}M shares"
cum_tag   = f"{foreign_cumulative/1e6:+.1f}M shares ({LOOKBACK}d)"

print(f"  [FOREIGN FLOW — best big-money proxy in VN]")
print(f"  Why: 90%+ of foreign investors in VN are institutional funds.")
print(f"  Foreign = fund managers, ETFs, hedge funds. Retail almost never.")
print()
print(f"  Last 5 days  : {flow_5d}  →  {flow_tag}")
print(f"  Last 20 days : {flow_20d}")
print(f"  {LOOKBACK}d cumulative : {cum_tag}")

if sell_streak > 0:
    print(f"  ⚠️  Foreigners sold {sell_streak} consecutive days")
else:
    print(f"  ✓  No consecutive sell streak")

if foreign_5d_avg < 0 and foreign_20d_avg < 0:
    print(f"  → SIGNAL: Sustained foreign distribution. Big money exiting.")
elif foreign_5d_avg < 0 < foreign_20d_avg:
    print(f"  → SIGNAL: Recent foreign selling (reversal from buying). Watch carefully.")
elif foreign_5d_avg > 0 and foreign_20d_avg < 0:
    print(f"  → SIGNAL: Recent foreign buying after period of selling. Possible re-entry.")
else:
    print(f"  → SIGNAL: Foreigners net buying. Big money supportive.")

# B. Block trades
if block_signal:
    print()
    print(f"  [BLOCK TRADES — institutional-size orders from tick data]")
    print(f"  Why: single orders of 10x+ average trade size = fund/prop desk.")
    print()
    icon = "🟢" if block_signal == "ACCUMULATION" else ("🔴" if block_signal == "DISTRIBUTION" else "⚪")
    print(f"  {icon} {block_signal}  —  {block_detail}")

# C. Tick absorption
if absorption_signal:
    print()
    print(f"  [TICK ABSORPTION — who is consuming the order queue today]")
    print(f"  Why: when sellers' queue (tvs) shrinks fast = buyers eating supply.")
    print(f"       When buyers' queue (tvb) shrinks = sellers eating demand.")
    print()
    icon = "🟢" if "BUYERS" in absorption_signal else ("🔴" if "SELLERS" in absorption_signal else "⚪")
    print(f"  {icon} {absorption_signal}  —  {absorption_detail}")

# D. Order book
if ob_signal:
    print()
    print(f"  [ORDER BOOK RATIO — buy vs sell volume balance]")
    print(f"  Why: ob_ratio = buy_vol/sell_vol. >1.0 = buyers dominate that day.")
    print(f"       Rising price + falling ratio = stealth distribution.")
    print()
    print(f"  {ob_signal}")
    print(f"  {ob_detail}")

# ── Overall big money summary ─────────────────────────────────────
bm_signals  = []
bm_bearish  = 0
bm_bullish  = 0

if foreign_5d_avg < 0:   bm_bearish += 2; bm_signals.append("foreign selling")
else:                     bm_bullish += 2; bm_signals.append("foreign buying")

if block_signal == "DISTRIBUTION": bm_bearish += 1; bm_signals.append("block distribution")
if block_signal == "ACCUMULATION": bm_bullish += 1; bm_signals.append("block accumulation")

if absorption_signal and "SELLERS" in absorption_signal:
    bm_bearish += 1; bm_signals.append("sellers absorbing")
if absorption_signal and "BUYERS" in absorption_signal:
    bm_bullish += 1; bm_signals.append("buyers absorbing")

if ob_trend_dir == "declining" and df["close"].tail(5).iloc[-1] > df["close"].tail(5).iloc[0]:
    bm_bearish += 1; bm_signals.append("price↑/ratio↓ divergence")

print()
print(f"  {'─'*55}")
if bm_bearish > bm_bullish:
    print(f"  🔴 OVERALL BIG MONEY: DISTRIBUTING  ({', '.join(bm_signals)})")
    print(f"     → Institutional money is net exiting. Be cautious.")
elif bm_bullish > bm_bearish:
    print(f"  🟢 OVERALL BIG MONEY: ACCUMULATING  ({', '.join(bm_signals)})")
    print(f"     → Smart money buying. Dips likely supported.")
else:
    print(f"  ⚪ OVERALL BIG MONEY: MIXED / NEUTRAL")

# ════════════════════════════════════════════════════════════════
# CHART
# ════════════════════════════════════════════════════════════════
if not args.no_plot:
    fig = plt.figure(figsize=(16, 9), facecolor="#0d1117")
    gs  = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.05,
                           height_ratios=[3, 1, 1])

    ax_p  = fig.add_subplot(gs[0, :2])   # price + levels
    ax_vp = fig.add_subplot(gs[0, 2])    # volume profile
    ax_f  = fig.add_subplot(gs[1, :2])   # foreign flow
    ax_ob = fig.add_subplot(gs[2, :2])   # ob ratio

    for ax in fig.get_axes():
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333333")

    # ── Price chart ──────────────────────────────────────────────
    plot_df = df.tail(90).reset_index(drop=True)
    x       = np.arange(len(plot_df))

    ax_p.plot(x, plot_df["close"], color="#58a6ff", lw=1.5)
    ax_p.fill_between(x, plot_df["close"], plot_df["close"].min() * 0.99,
                      alpha=0.12, color="#58a6ff")

    # Value area band
    ax_p.axhspan(VAL, VAH, alpha=0.08, color="#ffd700", label="_nolegend_")

    lvl_styles = [
        (poc,  "#ffd700", "--",  1.8, f"POC {poc:.2f}k"),
        (VAH,  "#ff6b6b", "-.",  1.2, f"VAH {VAH:.2f}k"),
        (VAL,  "#51cf66", "-.",  1.2, f"VAL {VAL:.2f}k"),
        (vwap, "#cc77ff", ":",   1.2, f"VWAP {vwap:.2f}k"),
    ]
    for price, col, ls, lw, lbl in lvl_styles:
        ax_p.axhline(price, color=col, ls=ls, lw=lw, label=lbl)

    for p in hvn_above[:3]:
        ax_p.axhline(p, color="#ff9966", ls=":", lw=0.8, alpha=0.7)
        ax_p.text(x[-1]+0.5, p, f" R {p:.1f}", color="#ff9966",
                  fontsize=7, va="center")

    for p in hvn_below[:2]:
        ax_p.axhline(p, color="#66cc88", ls=":", lw=0.8, alpha=0.7)
        ax_p.text(x[-1]+0.5, p, f" S {p:.1f}", color="#66cc88",
                  fontsize=7, va="center")

    if ENTRY:
        ax_p.axhline(ENTRY, color="#ffffff", ls="--", lw=1, alpha=0.55,
                     label=f"Entry {ENTRY:.2f}k")

    ax_p.axhline(current, color="#ffffff", lw=2, label=f"Now {current:.2f}k")

    # Target annotations
    for t_price, t_label, _ in targets:
        if t_price <= plot_df["close"].max() * 1.15:
            ax_p.annotate(f"► {t_label.split('—')[0].strip()}",
                          xy=(x[-1], t_price),
                          xytext=(x[-1] - 12, t_price),
                          color="#ffcc44", fontsize=7,
                          arrowprops=dict(arrowstyle="->", color="#ffcc44", lw=0.8))

    step = max(1, len(plot_df)//8)
    ax_p.set_xticks(x[::step])
    ax_p.set_xticklabels([d.strftime("%d/%m") for d in plot_df["date"].iloc[::step]],
                          rotation=30, fontsize=7)
    ax_p.set_ylabel("k VND", color="#aaaaaa", fontsize=8)
    ax_p.set_title(f"{TICKER} — Price & Commitment Levels (last 90d)",
                   color="#ffffff", fontsize=11, pad=6)
    ax_p.legend(fontsize=7, loc="upper left",
                facecolor="#1a1a2e", labelcolor="#cccccc", framealpha=0.8)
    ax_p.set_xlim(-1, len(x) + 8)

    # Big money label
    bm_color = "#ff6b6b" if bm_bearish > bm_bullish else (
               "#51cf66" if bm_bullish > bm_bearish else "#aaaaaa")
    bm_text  = ("BIG MONEY: DISTRIBUTING 🔴" if bm_bearish > bm_bullish else
                "BIG MONEY: ACCUMULATING 🟢" if bm_bullish > bm_bearish else
                "BIG MONEY: NEUTRAL ⚪")
    ax_p.text(0.99, 0.97, bm_text, transform=ax_p.transAxes,
              ha="right", va="top", color=bm_color, fontsize=9, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#111122", alpha=0.85))

    # ── Volume profile ────────────────────────────────────────────
    colors_vp = []
    for i, b in enumerate(bc):
        if abs(b - poc) < (bins[1]-bins[0]):
            colors_vp.append("#ffd700")
        elif b > current:
            v = min(combined[i] / combined.max(), 1.0)
            colors_vp.append((*plt.cm.Reds(0.3 + 0.7*v)[:3], 0.85))
        else:
            v = min(combined[i] / combined.max(), 1.0)
            colors_vp.append((*plt.cm.Greens(0.3 + 0.7*v)[:3], 0.85))

    ax_vp.barh(bc, combined, height=(bins[1]-bins[0])*0.88, color=colors_vp)
    for price, col, ls, lw, _ in lvl_styles:
        ax_vp.axhline(price, color=col, ls=ls, lw=lw*0.8)
    ax_vp.axhline(current, color="#ffffff", lw=2)
    if ENTRY:
        ax_vp.axhline(ENTRY, color="#ffffff", lw=1, ls="--", alpha=0.5)
    ax_vp.set_ylim(ax_p.get_ylim())
    ax_vp.yaxis.set_visible(False)
    ax_vp.set_xlabel("Vol weight", color="#aaaaaa", fontsize=7)
    ax_vp.set_title("Volume\nProfile", color="#ffffff", fontsize=9)

    # ── Foreign flow chart ────────────────────────────────────────
    fdf_plot = fdf.tail(90).reset_index(drop=True)
    xf       = np.arange(len(fdf_plot))
    colors_f = ["#51cf66" if v >= 0 else "#ff6b6b" for v in fdf_plot["net_f"]]
    ax_f.bar(xf, fdf_plot["net_f"] / 1e6, color=colors_f, alpha=0.7, width=0.8)
    if "net_f_5" in fdf_plot.columns:
        ax_f.plot(xf, fdf_plot["net_f_5"] / 1e6, color="#ffcc44", lw=1.5, label="5d MA")
    ax_f.axhline(0, color="#555555", lw=0.8)
    ax_f.set_ylabel("M shares", color="#aaaaaa", fontsize=7)
    ax_f.set_title("Foreign Net Flow  (+ = buying,  − = selling)",
                   color="#ffffff", fontsize=9)
    ax_f.set_xticks([])
    ax_f.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="#cccccc", framealpha=0.7)

    # ── Order book ratio chart ────────────────────────────────────
    try:
        oh_plot = oh.dropna(subset=["ob_ratio"]).tail(90).reset_index(drop=True)
        xo      = np.arange(len(oh_plot))
        ob_cols = ["#51cf66" if v >= 1.0 else "#ff6b6b" for v in oh_plot["ob_ratio"]]
        ax_ob.bar(xo, oh_plot["ob_ratio"], color=ob_cols, alpha=0.7, width=0.8)
        ax_ob.axhline(1.0, color="#ffd700", lw=1.2, ls="--", label="balance (1.0)")
        ax_ob.set_ylabel("buy/sell vol", color="#aaaaaa", fontsize=7)
        ax_ob.set_title("Order Book Ratio  (>1 = buyers dominant,  <1 = sellers)",
                        color="#ffffff", fontsize=9)
        ax_ob.set_xticks([])
        ax_ob.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="#cccccc", framealpha=0.7)
    except Exception:
        ax_ob.text(0.5, 0.5, "order history unavailable",
                   transform=ax_ob.transAxes, ha="center", color="#555555")

    out = os.path.join(BASE, f"commitment_{TICKER}.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"\n  Chart → commitment_{TICKER}.png")

print(f"\n{'='*62}")
