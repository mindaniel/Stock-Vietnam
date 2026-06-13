import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression


def perf_stats(r: pd.Series):
    r = r.dropna()
    eq = (1 + r).cumprod()
    years = len(r) / 12
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * np.sqrt(12)
    sharpe = (r.mean() * 12) / vol if vol > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    return cagr, vol, sharpe, mdd


def main():
    src = "results/factor_long_only_returns_s2010-01-01_liq1000000_top0.25_cost15.0_reg1_v0q1i0m1.csv"
    out = "results/forecast_eval_walkforward.csv"

    df = pd.read_csv(src, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    x = df.copy()
    # Add market context features (known at month-end t)
    vni = pd.read_csv("VNINDEX.csv")
    vni["date"] = pd.to_datetime(vni["date"], errors="coerce")
    vni["close"] = pd.to_numeric(vni["close"], errors="coerce")
    vni = vni.dropna(subset=["date", "close"]).sort_values("date")
    vm = vni.set_index("date")["close"].resample("ME").last().to_frame("vni_close")
    vm["vni_ret_1"] = vm["vni_close"].pct_change()
    vm["vni_ret_3m"] = vm["vni_close"].pct_change(3)
    vm["vni_vol_6m"] = vm["vni_ret_1"].rolling(6).std()
    vm["vni_ma10"] = vm["vni_close"].rolling(10).mean()
    vm["vni_regime"] = (vm["vni_close"] >= vm["vni_ma10"]).astype(int)
    vm = vm.reset_index()[["date", "vni_ret_1", "vni_ret_3m", "vni_vol_6m", "vni_regime"]]
    x = x.merge(vm, on="date", how="left")
    x["ret_1"] = x["net_ret"].shift(1)
    x["ret_3m"] = x["net_ret"].rolling(3).mean().shift(1)
    x["vol_6m"] = x["net_ret"].rolling(6).std().shift(1)
    x["turn_1"] = x["turnover"].shift(1)
    x["hold_1"] = x["n_hold"].shift(1)
    x["target"] = x["net_ret"]
    x = x.dropna().reset_index(drop=True)

    start_train = 60
    preds = []
    for i in range(start_train, len(x)):
        tr = x.iloc[:i]
        te = x.iloc[i:i + 1]
        feat_cols = ["ret_1", "ret_3m", "vol_6m", "turn_1", "hold_1", "vni_ret_1", "vni_ret_3m", "vni_vol_6m", "vni_regime"]
        Xtr = tr[feat_cols].values
        ytr = tr["target"].values
        Xte = te[feat_cols].values

        m = Ridge(alpha=1.0)
        m.fit(Xtr, ytr)
        p = float(m.predict(Xte)[0])
        # classification probability for positive month
        ybin = (ytr > 0).astype(int)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, ybin)
        p_up = float(clf.predict_proba(Xte)[0, 1])
        a = float(te["target"].iloc[0])
        preds.append((te["date"].iloc[0], p, p_up, a))

    res = pd.DataFrame(preds, columns=["date", "pred", "p_up", "actual"])
    # adaptive overlay: only invest when model confidence is strong
    res["overlay_ret"] = np.where((res["pred"] > 0) & (res["p_up"] >= 0.55), res["actual"], 0.0)

    os.makedirs("results", exist_ok=True)
    res.to_csv(out, index=False)

    hit = (np.sign(res["pred"]) == np.sign(res["actual"])).mean()
    corr = res[["pred", "actual"]].corr().iloc[0, 1]
    r2 = 1 - (((res.actual - res.pred) ** 2).sum() / ((res.actual - res.actual.mean()) ** 2).sum())

    bc, bv, bs, bm = perf_stats(res["actual"])
    oc, ov, o_sharpe, om = perf_stats(res["overlay_ret"])

    print("Saved:", os.path.abspath(out))
    print("Exists:", os.path.exists(out))
    print("Rows:", len(res))
    print("sign_hit:", round(hit, 4))
    print("corr:", round(float(corr), 4))
    print("R2_oos:", round(float(r2), 4))
    print("BASE cagr/vol/sharpe/mdd:", round(bc, 4), round(bv, 4), round(bs, 4), round(bm, 4))
    print("OVERLAY cagr/vol/sharpe/mdd:", round(oc, 4), round(ov, 4), round(o_sharpe, 4), round(om, 4))


if __name__ == "__main__":
    main()
