"""Section 7 analyses (report-only; nothing here feeds the paper until approved).

  7.1  Day-level dependence at the 15-min horizon:
       (a) day-block bootstrap (resample whole test days, 2000 resamples) of
           dMSE and dMAE for each learner and the 6-feature RF vs CLIPER, and
           RF vs LR;
       (b) DM tests with (i) the Andrews (1991) AR(1) plug-in automatic
           bandwidth for the Bartlett kernel and (ii) a Newey-West lag equal to
           the mean number of daylight intervals per test day; the existing Holm
           families (8 vs persistence, 8 vs CLIPER) are re-applied.
  7.2  Weather-only leaked comparator in the formulation of [6]-[8]: target P
       at tau from G, T^m, T^a read at tau (optionally + s, c at tau), no power
       lags; Random Forest tuned with the same protocol (last 25% of training).

Usage: PLANT=1|2 python analysis/day_dependence.py
The tuned causal settings are read from the existing run, not re-searched, so
results/ is never written; outputs go to analysis/out/plant<N>/.
"""
import ast, os, re, sys
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import revision_round3 as R  # noqa: E402  (reads PLANT from the environment)
from sklearn.ensemble import RandomForestRegressor  # noqa: E402

OUT = os.path.join(ROOT, "analysis", "out", "plant%d" % R.PLANT)
os.makedirs(OUT, exist_ok=True)
R.OUT = OUT  # any helper that writes goes here, never to results/

log = open(os.path.join(R.ROOT, "results", "run_log.txt" if R.PLANT == 1
                        else os.path.join("plant2", "run_log.txt"))).read()
R.RF_PARAMS = ast.literal_eval(re.search(r"RF params: (\{.*\})", log).group(1))
R.XGB_PARAMS = ast.literal_eval(re.search(r"XGB params: (\{.*\})", log).group(1))

raw, df = R.load_grid()
peak = float(raw["AC_POWER"][raw["N_INV"] == R.N_INVERTERS].max())
f = R.build_features(df)
causal = f.dropna(subset=R.FULL + ["target"])
main_set = causal[causal["elev_tau"] > R.ELEV_MIN]
n_tr = int(len(main_set) * 0.80)
cutoff = main_set.index[n_tr]
tr, te = main_set.iloc[:n_tr], main_set.iloc[n_tr:]
clim = tr.groupby("mod")["target"].mean()
preds, _ = R.fit_causal(tr, te, clim)
y = te["target"].values

# reproduction check against the published Table I csv
t1 = pd.read_csv(os.path.join(R.ROOT, "results", "table1_causal_daylight.csv") if R.PLANT == 1
                 else os.path.join(R.ROOT, "results", "plant2", "table1_causal_daylight.csv"), index_col=0)
repro = {k: (np.sqrt(np.mean((v - y) ** 2)), t1.loc[k, "RMSE"]) for k, v in preds.items()}
maxdiff = max(abs(a - b) for a, b in repro.values())
print("reproduction max |RMSE diff| vs results/: %.3g kW" % maxdiff)

LEARN = ["Linear Regression", "XGBoost", "Random Forest (full)", "Random Forest (reduced)"]
days = pd.DatetimeIndex(te.index).normalize()
uday = days.unique()
per_day = pd.Series(1, index=days).groupby(level=0).sum()
print("test days: %d, intervals/day: %s" % (len(uday), per_day.values.tolist()))


# ------------------------------------------------------------------ 7.1 (b)
def lrv_bartlett(d, bw):
    """Bartlett long-run variance with weights 1 - k/bw for k < bw."""
    dc = d - d.mean()
    n = len(d)
    v = np.sum(dc * dc) / n
    k = 1
    while k < bw and k < n:
        v += 2 * (1 - k / bw) * np.sum(dc[k:] * dc[:-k]) / n
        k += 1
    return v


def andrews_bw(d):
    """Andrews (1991) AR(1) plug-in bandwidth for the Bartlett kernel."""
    dc = d - d.mean()
    rho = np.sum(dc[1:] * dc[:-1]) / np.sum(dc[:-1] ** 2)
    a1 = 4 * rho ** 2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
    return 1.1447 * (a1 * len(d)) ** (1 / 3), rho


def dm(y, p1, p2, loss, bw, h=1):
    e1, e2 = y - p1, y - p2
    d = e1 ** 2 - e2 ** 2 if loss == "SE" else np.abs(e1) - np.abs(e2)
    n = len(d)
    st = d.mean() / np.sqrt(lrv_bartlett(d, bw) / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return st, 2 * stats.t.sf(abs(st), df=n - 1), d


L_DAY = int(round(per_day.mean()))
rows = []
pairs = [("Persistence", m) for m in LEARN] + [("CLIPER", m) for m in LEARN] + \
        [("Linear Regression", "Random Forest (full)")]
for a, b in pairs:
    for loss in ("SE", "AE"):
        e1, e2 = y - preds[a], y - preds[b]
        d = e1 ** 2 - e2 ** 2 if loss == "SE" else np.abs(e1) - np.abs(e2)
        bwA, rho = andrews_bw(d)
        stA, pA, _ = dm(y, preds[a], preds[b], loss, bwA)
        stD, pD, _ = dm(y, preds[a], preds[b], loss, L_DAY + 1)   # lags 1..L_DAY
        st5, p5, _ = dm(y, preds[a], preds[b], loss, 6)           # existing: lags 1..5
        rows.append(dict(ref=a, model=b, loss=loss, rho1=rho, andrews_bw=bwA,
                         p_NW5=p5, p_andrews=pA, p_NWday=pD, NWday_lag=L_DAY))
D = pd.DataFrame(rows)
for ref in ("Persistence", "CLIPER"):
    m = D["ref"].eq(ref)
    for c in ("p_NW5", "p_andrews", "p_NWday"):
        D.loc[m, c + "_Holm"] = R.holm(D.loc[m, c].values)

# ------------------------------------------------------------------ 7.1 (a)
rng = np.random.default_rng(R.SEED)
E = {k: y - v for k, v in preds.items()}
day_idx = [np.where(days == d_)[0] for d_ in uday]
bpairs = [("CLIPER", m) for m in LEARN] + [("Linear Regression", "Random Forest (full)")]
draws = {p: {"dMSE": [], "dMAE": []} for p in bpairs}
for _ in range(R.BOOT_N):
    pick = rng.integers(0, len(uday), size=len(uday))
    idx = np.concatenate([day_idx[j] for j in pick])
    for a, b in bpairs:
        ea, eb = E[a][idx], E[b][idx]
        draws[(a, b)]["dMSE"].append(np.mean(eb ** 2) - np.mean(ea ** 2))
        draws[(a, b)]["dMAE"].append(np.mean(np.abs(eb)) - np.mean(np.abs(ea)))
brow = []
for a, b in bpairs:
    pt = {"dMSE": np.mean(E[b] ** 2) - np.mean(E[a] ** 2),
          "dMAE": np.mean(np.abs(E[b])) - np.mean(np.abs(E[a]))}
    for q, v in draws[(a, b)].items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        brow.append(dict(ref=a, model=b, quantity=q, point=pt[q], ci_lo=lo, ci_hi=hi,
                         excludes_zero=not (lo <= 0 <= hi),
                         share_draws_below_0=float(np.mean(np.array(v) < 0))))
B = pd.DataFrame(brow)

# ------------------------------------------------------------------ 7.2
WX = ["IRRADIATION_tau", "AMBIENT_TEMPERATURE_tau", "MODULE_TEMPERATURE_tau"]
WXSC = WX + ["time_sin_tau", "time_cos_tau"]
wrows = []
for regime in ("all hours", "daylight"):
    base_c = causal if regime == "all hours" else causal[causal["elev_tau"] > R.ELEV_MIN]
    wall = f.dropna(subset=WXSC + ["target"])
    base_w = wall if regime == "all hours" else wall[wall["elev_tau"] > R.ELEV_MIN]
    common = base_c.index.intersection(base_w.index)
    c, w = base_c.loc[common], base_w.loc[common]
    trc, tec = c[c.index < cutoff], c[c.index >= cutoff]
    trw, tew = w[w.index < cutoff], w[w.index >= cutoff]
    yt = tec["target"].values
    pc = np.clip(R.rf_model().fit(trc[R.FULL], trc["target"]).predict(tec[R.FULL]), 0, None)
    out = {"Causal RF forecast (13 features, P_{t+1} from I_t)": pc}
    for name, cols in (("Weather-only leaked RF (G, Tm, Ta at target)", WX),
                       ("Weather-only leaked RF (G, Tm, Ta, s, c at target)", WXSC)):
        lp, _ = R.tune_leaked(trw, regime, cols)
        out[name] = np.clip(RandomForestRegressor(**lp, random_state=R.SEED, n_jobs=-1)
                            .fit(trw[cols], trw["target"]).predict(tew[cols]), 0, None)
        out[name + " params"] = lp
    for name, p in out.items():
        if name.endswith("params"):
            continue
        s = R.scores(yt, p, peak)
        wrows.append(dict(regime=regime, model=name, n_train=len(trc), N=s["N"], RMSE=s["RMSE"],
                          MAE=s["MAE"], R2=s["R2"],
                          params=repr(out.get(name + " params", R.RF_PARAMS))))
W = pd.DataFrame(wrows)
for regime in ("all hours", "daylight"):
    m = W["regime"] == regime
    cr = W[m & W["model"].str.startswith("Causal")].iloc[0]
    W.loc[m, "RMSE_ratio_causal_over_this"] = cr["RMSE"] / W.loc[m, "RMSE"]
    W.loc[m, "MAE_ratio_causal_over_this"] = cr["MAE"] / W.loc[m, "MAE"]

D.to_csv(os.path.join(OUT, "dm_bandwidth.csv"), index=False)
B.to_csv(os.path.join(OUT, "day_block_bootstrap.csv"), index=False)
W.to_csv(os.path.join(OUT, "weather_only_leak.csv"), index=False)
pd.set_option("display.width", 250)
print(D.round(4).to_string())
print(B.round(4).to_string())
print(W.drop(columns="params").round(4).to_string())
print(W[["regime", "model", "params"]].to_string())
