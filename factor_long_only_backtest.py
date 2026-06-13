import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ff_vietnam import load_prices, load_sector_map, load_fundamentals_latest_by_year, monthly_stock_panel, attach_accounting


def load_vnindex_regime() -> pd.DataFrame:
    vn = pd.read_csv("VNINDEX.csv")
    vn["date"] = pd.to_datetime(vn["date"], errors="coerce")
    vn["close"] = pd.to_numeric(vn["close"], errors="coerce")
    vn = vn.dropna(subset=["date", "close"]).sort_values("date")
    m = vn.set_index("date")["close"].resample("ME").last().to_frame("vn_close")
    m["ma10"] = m["vn_close"].rolling(10, min_periods=10).mean()
    m["risk_on"] = (m["vn_close"] >= m["ma10"]).astype(int)
    return m.reset_index()[["date", "risk_on"]]


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    m, sd = s.mean(), s.std()
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - m) / sd


def perf_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return {"months": 0, "cagr": np.nan, "vol": np.nan, "sharpe": np.nan, "maxdd": np.nan}
    eq = (1 + r).cumprod()
    years = len(r) / 12
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * np.sqrt(12)
    sharpe = (r.mean() * 12) / vol if vol and vol > 0 else np.nan
    dd = eq / eq.cummax() - 1
    return {"months": len(r), "cagr": cagr, "vol": vol, "sharpe": sharpe, "maxdd": dd.min()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--top", type=float, default=0.25, help="Top quantile to hold")
    p.add_argument("--min-liq", type=float, default=2_000_000, help="Min monthly avg traded value")
    p.add_argument("--cost-bps", type=float, default=15.0, help="One-way transaction cost bps")
    p.add_argument("--use-regime", action="store_true", help="Use VNINDEX trend filter (10M MA)")
    p.add_argument("--use-momentum", action="store_true", help="Include momentum in score")
    p.add_argument("--use-value", action="store_true", help="Include value (BM) in score")
    p.add_argument("--use-quality", action="store_true", help="Include profitability in score")
    p.add_argument("--use-investment", action="store_true", help="Include conservative investment in score")
    p.add_argument("--save-trades", action="store_true", help="Save ticker-level buy/sell ledger and holdings snapshots")
    p.add_argument("--initial-capital", type=float, default=100_000_000, help="Initial capital (VND)")
    args = p.parse_args()

    prices = load_prices()
    sec = load_sector_map()
    fin = load_fundamentals_latest_by_year()
    panel = monthly_stock_panel(prices).merge(sec, on="symbol", how="left")
    panel = attach_accounting(panel, fin)
    panel = panel[panel["date"] >= pd.Timestamp(args.start)].copy()

    regime = load_vnindex_regime() if args.use_regime else None

    rets = []
    prev_weights = {}
    trades = []
    holdings_rows = []
    capital = float(args.initial_capital)
    pos_shares = {}
    pos_cost = {}

    for dt, g in panel.groupby("date"):
        if regime is not None:
            rr = regime.loc[regime["date"] == dt, "risk_on"]
            if len(rr) == 0 or int(rr.iloc[0]) == 0:
                for s, w0 in prev_weights.items():
                    if w0 > 0:
                        trades.append({"date": dt, "symbol": s, "action": "SELL", "w_prev": w0, "w_new": 0.0, "dw": -w0})
                prev_weights = {}
                rets.append({"date": dt, "gross_ret": 0.0, "net_ret": 0.0, "n_hold": 0, "turnover": 0.0})
                continue
        g = g.copy()
        g = g[(g["value"] >= args.min_liq)]
        g = g.dropna(subset=["ret"])
        if len(g) < 20:
            continue

        # Adaptive factor score: use available factors per stock
        comp = pd.DataFrame(index=g.index)
        comp["value"] = zscore(g["bm"])
        comp["quality"] = zscore(g["profitability"])
        comp["invest"] = zscore(-g["asset_growth"])
        comp["mom"] = zscore(g["mom_12_1"])

        use_value = args.use_value
        use_quality = args.use_quality
        use_investment = args.use_investment
        use_momentum = args.use_momentum
        # default: if none selected, use all
        if not any([use_value, use_quality, use_investment, use_momentum]):
            use_value = use_quality = use_investment = use_momentum = True

        weights = {}
        if use_value:
            weights["value"] = 0.35
        if use_quality:
            weights["quality"] = 0.30
        if use_investment:
            weights["invest"] = 0.15
        if use_momentum:
            weights["mom"] = 0.20
        wsum = pd.Series(0.0, index=g.index)
        num = pd.Series(0.0, index=g.index)
        n_avail = pd.Series(0, index=g.index)
        for k, w in weights.items():
            v = comp[k]
            ok = v.notna()
            num.loc[ok] += w * v.loc[ok]
            wsum.loc[ok] += w
            n_avail.loc[ok] += 1

        g["score"] = num / wsum.replace(0, np.nan)
        g["n_factors"] = n_avail
        g = g[g["n_factors"] >= 2].dropna(subset=["score"])
        if len(g) < 20:
            continue

        q = g["score"].quantile(1 - args.top)
        picks = g[g["score"] >= q][["symbol", "ret", "close", "value"]].copy()
        if picks.empty:
            continue

        w = {s: 1 / len(picks) for s in picks["symbol"]}
        gross = picks.set_index("symbol")["ret"].mean()

        # Trade ledger + holdings snapshot (capital-based)
        if args.save_trades:
            px_map = picks.set_index("symbol")["close"].to_dict()
            val_map = picks.set_index("symbol")["value"].to_dict()
            all_syms = set(prev_weights) | set(w)
            for s in sorted(all_syms):
                w0 = prev_weights.get(s, 0.0)
                w1 = w.get(s, 0.0)
                dw = w1 - w0
                if abs(dw) > 1e-12:
                    action = "BUY" if dw > 0 else "SELL"
                    price = float(px_map.get(s, np.nan))
                    liq = float(val_map.get(s, np.nan))
                    trade_value = abs(dw) * capital
                    shares = np.floor(trade_value / price) if pd.notna(price) and price > 0 else 0.0
                    cashflow = (-shares * price) if action == "BUY" else (shares * price)
                    realized = np.nan
                    if action == "SELL":
                        avg_cost = pos_cost.get(s, price)
                        realized = (price - avg_cost) * shares
                        pos_shares[s] = max(0.0, pos_shares.get(s, 0.0) - shares)
                    else:
                        old_sh = pos_shares.get(s, 0.0)
                        old_cost = pos_cost.get(s, price)
                        new_sh = old_sh + shares
                        if new_sh > 0:
                            pos_cost[s] = (old_sh * old_cost + shares * price) / new_sh
                        pos_shares[s] = new_sh
                    trades.append({"date": dt, "time": "month_end", "symbol": s, "action": action, "price": price, "liquidity_value": liq,
                                   "w_prev": w0, "w_new": w1, "dw": dw, "trade_value": trade_value, "shares": shares,
                                   "cashflow": cashflow, "realized_pnl": realized})
            for s, w1 in w.items():
                px = float(picks.loc[picks["symbol"] == s, "close"].iloc[0]) if (picks["symbol"] == s).any() else np.nan
                holdings_rows.append({"date": dt, "symbol": s, "weight": w1, "price": px,
                                      "shares": pos_shares.get(s, np.nan), "avg_cost": pos_cost.get(s, np.nan)})

        # turnover-based cost
        all_syms = set(prev_weights) | set(w)
        turnover = sum(abs(w.get(s, 0) - prev_weights.get(s, 0)) for s in all_syms)
        cost = turnover * (args.cost_bps / 10000.0)
        net = gross - cost
        prev_weights = w
        capital = capital * (1 + net)

        rets.append({"date": dt, "gross_ret": gross, "net_ret": net, "n_hold": len(picks), "turnover": turnover})

    if not rets:
        print("No backtest observations. Try lower min-liq or earlier start.")
        return
    out = pd.DataFrame(rets).sort_values("date")
    out["date"] = pd.to_datetime(out["date"])
    out["pnl_net"] = (1 + out["net_ret"]).cumprod()
    out["pnl_gross"] = (1 + out["gross_ret"]).cumprod()

    os.makedirs("results", exist_ok=True)
    config_tag = f"s{args.start}_liq{int(args.min_liq)}_top{args.top}_cost{args.cost_bps}_reg{int(args.use_regime)}_v{int(use_value)}q{int(use_quality)}i{int(use_investment)}m{int(use_momentum)}"
    out_path = f"results/factor_long_only_returns_{config_tag}.csv"
    out.to_csv(out_path, index=False)

    if args.save_trades:
        trades_df = pd.DataFrame(trades)
        holds_df = pd.DataFrame(holdings_rows)
        trades_path = f"results/factor_long_only_trades_{config_tag}.csv"
        holds_path = f"results/factor_long_only_holdings_{config_tag}.csv"
        trades_df.to_csv(trades_path, index=False)
        holds_df.to_csv(holds_path, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(out["date"], out["pnl_net"], label="Net PnL")
    plt.plot(out["date"], out["pnl_gross"], label="Gross PnL", alpha=0.6)
    plt.title("Factor Long-Only PnL Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig_path = f"results/factor_long_only_pnl_{config_tag}.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=140)
    plt.close()

    strat = perf_stats(out["net_ret"])
    ew = perf_stats(out["gross_ret"])

    print("=== FACTOR LONG-ONLY BACKTEST ===")
    print(f"Regime filter: {'ON' if args.use_regime else 'OFF'}")
    print(f"Months: {strat['months']}")
    print(f"Net CAGR:   {strat['cagr']:.2%}")
    print(f"Net Vol:    {strat['vol']:.2%}")
    print(f"Net Sharpe: {strat['sharpe']:.2f}")
    print(f"Net MaxDD:  {strat['maxdd']:.2%}")
    print(f"Avg holdings: {out['n_hold'].mean():.1f} | Avg turnover: {out['turnover'].mean():.2f}")
    print(f"Saved returns: {out_path}")
    print(f"Saved chart:   {fig_path}")
    if args.save_trades:
        print(f"Saved trades:  {trades_path}")
        print(f"Saved holdings:{holds_path}")


if __name__ == "__main__":
    main()
