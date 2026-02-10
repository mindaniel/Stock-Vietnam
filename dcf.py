import requests
import numpy as np
import pandas as pd
import os
import streamlit as st
from vnstock import Finance, Quote, Company

TAX_RATE = 0.20  # 20% corporate tax rate in Vietnam

def compute_cashflows(symbol="ACB"):
    """
    Unified function:
    - Fetches income, balance sheet, cash flow from vnstock (VCI → TCBS)
    - Computes FCFE, FCFF
    - Fetches current price & market cap
    - Returns clean DataFrame
    """
    def safe_get(df, keys):
        """Find first column containing any of the keywords."""
        for c in df.columns:
            if any(k.lower() in c.lower() for k in keys):
                return c
        return None

    # --- Try VCI first, fallback to TCBS ---
    finance = None
    for src in ["VCI", "TCBS"]:
        try:
            finance = Finance(symbol=symbol, source=src)
            inc = finance.income_statement(period="year", lang="en")
            bal = finance.balance_sheet(period="year", lang="en")
            cf = finance.cash_flow(period="year", lang="en")
            if not inc.empty and not bal.empty and not cf.empty:
                break
        except Exception:
            finance = None
            continue
    if finance is None:
        raise RuntimeError(f"❌ Failed to fetch data for {symbol} from vnstock")

    # --- Income ---
    net_col = (safe_get(inc, ["net", "profit"]) 
               or safe_get(inc, ["profit", "year"]) 
               or safe_get(inc, ["after", "tax"]))
    net_income = inc.set_index("yearReport")[net_col]

    interest_col = safe_get(inc, ["interest", "expense"])
    interest = inc.set_index("yearReport")[interest_col] if interest_col else pd.Series(0, index=inc["yearReport"])

    # --- Cash Flow ---
    op_cf_col = safe_get(cf, ["operating"])
    inv_cf_col = safe_get(cf, ["investing"])
    fin_cf_col = safe_get(cf, ["financial"])

    op_cf = cf.set_index("yearReport")[op_cf_col] if op_cf_col else pd.Series(0, index=cf["yearReport"])
    inv_cf = cf.set_index("yearReport")[inv_cf_col] if inv_cf_col else pd.Series(0, index=cf["yearReport"])
    fin_cf = cf.set_index("yearReport")[fin_cf_col] if fin_cf_col else pd.Series(0, index=cf["yearReport"])

    # --- Balance Sheet ---
    short_col = safe_get(bal, ["short", "borrow"])
    long_col = safe_get(bal, ["long", "borrow"])
    equity_col = safe_get(bal, ["owner", "equity"]) or safe_get(bal, ["shareholders", "equity"])

    short_debt = bal.set_index("yearReport")[short_col] if short_col else pd.Series(0, index=bal["yearReport"])
    long_debt = bal.set_index("yearReport")[long_col] if long_col else pd.Series(0, index=bal["yearReport"])
    equity = bal.set_index("yearReport")[equity_col] if equity_col else pd.Series(0, index=bal["yearReport"])

    # --- Approximate Depreciation & Amortisation (D&A) ---
    da_col = safe_get(bal, ["fixed", "asset"]) or safe_get(bal, ["tangible"])
    if da_col:
        try:
            fa = pd.to_numeric(bal.set_index("yearReport")[da_col], errors="coerce")
            df_da = (0.05 * fa).fillna(0)  # assume 5% of fixed assets
        except Exception:
            df_da = pd.Series(0, index=bal["yearReport"])
    else:
        df_da = pd.Series(0, index=bal["yearReport"])

    # --- Combine all ---
    df = pd.concat([net_income, op_cf, inv_cf, fin_cf,
                    short_debt, long_debt, equity, interest, df_da], axis=1)
    df.columns = [
        "Net Income", "Operating CF", "Investing CF", "Financing CF",
        "Short Debt", "Long Debt", "Equity", "Interest", "Approx D&A"
    ]
    df = df.reset_index().rename(columns={"yearReport": "Year"}).sort_values("Year")
    interest_adj = df["Interest"].fillna(0).abs()

    # --- Derived metrics ---
    df["Total Debt"] = df["Short Debt"].fillna(0) + df["Long Debt"].fillna(0)
    df["ΔDebt"] = df["Total Debt"].diff().fillna(0)
    df["FCFE"] = df["Operating CF"].fillna(0) + df["Investing CF"].fillna(0)
    df["FCFF"] = df["FCFE"] + interest_adj * (1 - TAX_RATE) + df["ΔDebt"]

    # --- Convert to billions (₫) ---
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 1e9

    # --- Fetch current price and market cap ---
    current_price, shares = np.nan, np.nan
    try:
        quote = Quote(symbol=symbol, source="VCI")
        price_df = quote.history()
        if not price_df.empty:
            current_price = float(price_df["close"].iloc[-1])
    except Exception:
        pass

    try:
        company = Company(symbol=symbol, source="TCBS")
        info = company.overview()
        if not info.empty:
            shares = float(info.get("outstanding_share", [np.nan])[0])
    except Exception:
        pass

    # ✅ Fallbacks if data missing
    if np.isnan(current_price) or current_price <= 0:
        current_price = 10_000.0  # assume 10k VND
    if np.isnan(shares) or shares <= 0:
        shares = 100_000_000  # assume 100 million shares

    market_cap = current_price * shares

    # --- Attach meta info ---
    df.attrs["price"] = current_price
    df.attrs["market_cap"] = market_cap
    df.attrs["shares"] = shares

    # --- Fix names for compatibility ---
    df.columns = df.columns.str.replace(" ", "_").str.replace("Δ", "Delta")

    return df.round(2)


# ========== CONFIG ==========
rf_rate = 0.03937
erp = 0.06  # keep constant for now

# ---------- Helper functions ----------
def estimate_growth(series):
    if len(series) < 3:
        return 0.04
    try:
        s = pd.Series(series).astype(float)
        if s.iloc[-3] <= 0:
            return 0.04
        g = (s.iloc[-1] / s.iloc[-3]) ** (1/2) - 1
        return float(np.clip(g, -0.05, 0.10))
    except Exception:
        return 0.04


def infer_growth_and_terminal(df, market_cap):
    g_explicit = estimate_growth(df["Net_Income"])
    if market_cap is None:
        market_cap = 0
    if market_cap > 100_000:
        g_terminal = 0.027
    elif market_cap > 10_000:
        g_terminal = 0.018
    else:
        g_terminal = 0.010
    return g_explicit, g_terminal


def estimate_wacc(debt, equity):
    try:
        de_ratio = np.mean(np.array(debt, dtype=float) / (np.array(equity, dtype=float) + 1e-9))
    except Exception:
        de_ratio = 0.5
    cost_debt = rf_rate + 0.02
    cost_equity = rf_rate + erp
    if de_ratio < 0.3: adj = 0.00
    elif de_ratio < 0.7: adj = 0.01
    else: adj = 0.02
    wacc = cost_equity*(1/(1+de_ratio)) + cost_debt*(1-TAX_RATE)*(de_ratio/(1+de_ratio))
    return float(wacc + adj)


def dcf_project(values, g_explicit, g_terminal, discount_rate, years=5):
    base = float(pd.Series(values).dropna().astype(float).iloc[-1])
    if base < 0:
        positives = [x for x in values if x > 0]
        base = float(np.mean(positives)) if positives else abs(base) * 0.3
        print("⚠️ Negative base FCF detected. Using normalized mean of positive years.")
    if not np.isfinite(base) or base == 0:
        base = float(np.nanmean(values))

    fcfs, pv = [], []
    for t in range(1, years+1):
        f = base * ((1 + g_explicit) ** t)
        fcfs.append(f)
        pv.append(f / ((1 + discount_rate) ** t))
    tv = fcfs[-1] * (1 + g_terminal) / max((discount_rate - g_terminal), 1e-6)
    pv_tv = tv / ((1 + discount_rate) ** years)
    total_value = sum(pv) + pv_tv
    return pd.DataFrame({"Year": range(1, years+1), "FCF": fcfs, "PV": pv}), float(total_value)


def run_auto_dcf(symbol, df, market_cap=None):
    if hasattr(df, "attrs"):
        if market_cap is None or market_cap == 0:
            market_cap = df.attrs.get("market_cap", 0)

    """Auto DCF — returns projections & summary table."""
    shares = 400.0
    meta_market_cap = market_cap
    try:
        company = Company(symbol=symbol, source="TCBS")
        meta = company.overview().iloc[0]
        shares = float(meta.get("outstanding_share", shares))
        # attempt to get market cap if not provided
        if market_cap is None:
            meta_market_cap = float(meta.get("market_cap", 0))
    except Exception as e:
        print(f"⚠️ Could not fetch TCBS meta: {e}")
        if market_cap is None:
            meta_market_cap = 0

    g_explicit, g_terminal = infer_growth_and_terminal(df, meta_market_cap)
    wacc = estimate_wacc(df["Total_Debt"], df["Equity"])

    proj_fcfe, val_fcfe = dcf_project(df["FCFE"], g_explicit, g_terminal, wacc)
    proj_fcff, val_fcff = dcf_project(df["FCFF"], g_explicit, g_terminal, wacc)

    # shares assumed to be in millions in some sources; keep as number here
    price_fcfe = val_fcfe / (shares if shares else 1.0)
    price_fcff = val_fcff / (shares if shares else 1.0)

    summary = pd.DataFrame({
        "Metric": ["symbol", "market_cap", "shares", "WACC",
                   "g_explicit", "g_terminal",
                   "Equity Value (FCFE)", "Equity Value (FCFF)",
                   "Price/Share (FCFE)", "Price/Share (FCFF)"],
        "Value": [symbol, meta_market_cap, shares, wacc,
                  g_explicit, g_terminal,
                  val_fcfe, val_fcff, price_fcfe, price_fcff]
    })
    return proj_fcfe, proj_fcff, summary


# ---------- Sensitivity Matrix ----------
def build_sensitivity(base_value, shares, g_terms, wacc_terms, last_fcf, years=5):
    """Generate sensitivity matrix for terminal growth × WACC (price per share)."""
    rows = []
    for g_t in g_terms:
        row = []
        for w in wacc_terms:
            # avoid invalid denominators
            if w <= g_t + 1e-6:
                row.append(np.nan)
                continue
            tv = last_fcf * (1 + g_t) / (w - g_t)
            pv_tv = tv / ((1 + w) ** years)
            equity_value = base_value + pv_tv
            price = equity_value / (shares if shares else 1.0)
            row.append(price)
        rows.append(row)
    df = pd.DataFrame(rows,
                      index=[f"{x:.1%}" for x in g_terms],
                      columns=[f"{x:.1%}" for x in wacc_terms])
    return df


# ---------- Scenario runner (with margin + matrices) ----------
def run_three_scenarios(symbol, df, market_cap, margin=0.10):
    """Run pessimistic / base / optimistic DCFs automatically with ±margin and matrices."""
    base_proj_fcfe, base_proj_fcff, base_summary = run_auto_dcf(symbol, df, market_cap)

    base_wacc = float(base_summary.loc[base_summary["Metric"] == "WACC", "Value"].values[0])
    base_g_explicit = float(base_summary.loc[base_summary["Metric"] == "g_explicit", "Value"].values[0])
    base_g_terminal = float(base_summary.loc[base_summary["Metric"] == "g_terminal", "Value"].values[0])
    shares = float(base_summary.loc[base_summary["Metric"] == "shares", "Value"].values[0])

    scenarios = {
        "⚠️ Pessimistic": {"wacc": base_wacc + 0.02, "g_exp": base_g_explicit / 2, "g_term": base_g_terminal / 2},
        "⚖️ Base":        {"wacc": base_wacc,       "g_exp": base_g_explicit,       "g_term": base_g_terminal},
        "🚀 Optimistic":  {"wacc": base_wacc - 0.02, "g_exp": base_g_explicit * 1.5, "g_term": base_g_terminal * 1.5}
    }

    results = []
    for name, p in scenarios.items():
        fcfe_proj, fcfe_val = dcf_project(df["FCFE"], p["g_exp"], p["g_term"], p["wacc"])
        fcff_proj, fcff_val = dcf_project(df["FCFF"], p["g_exp"], p["g_term"], p["wacc"])
        price_fcfe = fcfe_val / (shares if shares else 1.0)
        price_fcff = fcff_val / (shares if shares else 1.0)

        # ±margin range (on price)
        fcfe_low, fcfe_high = price_fcfe * (1 - margin), price_fcfe * (1 + margin)
        fcff_low, fcff_high = price_fcff * (1 - margin), price_fcff * (1 + margin)

        results.append([
            name,
            round(price_fcfe, 2),
            f"{fcfe_low:.2f} – {fcfe_high:.2f}",
            round(price_fcff, 2),
            f"{fcff_low:.2f} – {fcff_high:.2f}"
        ])

    scen_df = pd.DataFrame(results, columns=[
        "Scenario",
        "Price (FCFE)",
        "FCFE Range (±margin)",
        "Price (FCFF)",
        "FCFF Range (±margin)"
    ])

    # --- Sensitivity Matrices for base case ---
    wacc_range = np.arange(0.06, 0.141, 0.005)
    g_range = np.arange(0.000, 0.041, 0.005)
    last_fcfe = base_proj_fcfe["FCF"].iloc[-1]
    last_fcff = base_proj_fcff["FCF"].iloc[-1]

    base_pv_fcfe = base_proj_fcfe["PV"].sum()
    base_pv_fcff = base_proj_fcff["PV"].sum()

    fcfe_mat = build_sensitivity(base_pv_fcfe, shares, g_range, wacc_range, last_fcfe)
    fcff_mat = build_sensitivity(base_pv_fcff, shares, g_range, wacc_range, last_fcff)

    return scen_df, base_proj_fcfe, base_proj_fcff, base_summary, fcfe_mat, fcff_mat


