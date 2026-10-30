"""Expanding-window (rolling-origin) evaluation, 15-minute horizon. Report-only.

Folds: four test windows of whole calendar days, each preceded by an
expanding training block holding every earlier daylight row:
  F1 26 May-31 May, F2 1-6 Jun, F3 7-12 Jun, F4 13-17 Jun (5 days).
Per fold the forest is re-tuned on that fold's validation slice (last 25%
of its training block, same 36-point grid as the paper) and refitted on the
whole training block; CLIPER's climatology and alpha are refitted on it.
Models: persistence, CLIPER, Linear Regression, Random Forest (13 features).

Reported: per-fold metrics and DM tests (Newey-West 5 lags + HLN), and
pooled over all folds: DM (5 lags; one-day lag) and a day-block bootstrap
(2000 resamples of whole test days) for RF vs CLIPER and RF vs LR.
Usage: PLANT=1|2 python analysis/rolling_origin.py
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import revision_round3 as R  # noqa: E402

OUT = os.path.join(ROOT, "analysis", "out", "plant%d" % R.PLANT)
os.makedirs(OUT, exist_ok=True)
R.OUT = OUT

FOLDS = [("F1", "2020-05-26", "2020-06-01"), ("F2", "2020-06-01", "2020-06-07"),
         ("F3", "2020-06-07", "2020-06-13"), ("F4", "2020-06-13", "2020-06-18")]

raw, df = R.load_grid()
peak = float(raw["AC_POWER"][raw["N_INV"] == R.N_INVERTERS].max())
f = R.build_features(df)
c = f.dropna(subset=R.FULL + ["target"])
main = c[c["elev_tau"] > R.ELEV_MIN]


def tune_rf(tr):
    nv = int(len(tr) * 0.75)
    fit, val = tr.iloc[:nv], tr.iloc[nv:]
    best, best_r = None, np.inf
    for depth in (6, 10, 14, None):
        for leaf in (1, 5, 10):
            for mf in (1.0, 0.5, "sqrt"):
                p = dict(n_estimators=300, max_depth=depth, min_samples_leaf=leaf, max_features=mf)
                m = RandomForestRegressor(**p, random_state=R.SEED, n_jobs=-1).fit(fit[R.FULL], fit["target"])
                r = np.sqrt(np.mean((np.clip(m.predict(val[R.FULL]), 0, None) - val["target"]) ** 2))
                if r < best_r:
                    best, best_r = p, r
    return best


def dm_lag(y, p1, p2, loss, lags):
    return R.dm_test(y, p1, p2, loss, lags=lags)[1]


rows, pooled = [], {k: [] for k in ("y", "Persistence", "CLIPER", "Linear Regression", "Random Forest", "day")}
for name, a, b in FOLDS:
    tr = main[main.index < a]
    te = main[(main.index >= a) & (main.index < b)]
    y, y_tr = te["target"].values, tr["target"].values
    best = tune_rf(tr)
    clim = tr.groupby("mod")["target"].mean()
    c_tr, c_te = R.clim_lookup(clim, tr["mod"]), R.clim_lookup(clim, te["mod"])
    alpha = float(np.clip(np.sum((tr["ac_now"].values - c_tr) * (y_tr - c_tr))
                          / np.sum((tr["ac_now"].values - c_tr) ** 2), 0, 1))
    P = {"Persistence": te["ac_now"].values,
         "CLIPER": alpha * te["ac_now"].values + (1 - alpha) * c_te,
         "Linear Regression": np.clip(LinearRegression().fit(tr[R.FULL], y_tr).predict(te[R.FULL]), 0, None),
         "Random Forest": np.clip(RandomForestRegressor(**best, random_state=R.SEED, n_jobs=-1)
                                  .fit(tr[R.FULL], y_tr).predict(te[R.FULL]), 0, None)}
    r = {"fold": name, "test": f"{a}..{b}", "n_train": len(tr), "n_test": len(te),
         "n_days": pd.DatetimeIndex(te.index).normalize().nunique(), "alpha": alpha, "rf_params": repr(best)}
    for k, v in P.items():
        r["RMSE " + k] = float(np.sqrt(np.mean((v - y) ** 2)))
        r["MAE " + k] = float(np.mean(np.abs(v - y)))
    r["MAE skill RF vs CLIPER"] = 1 - r["MAE Random Forest"] / r["MAE CLIPER"]
    r["RMSE skill RF vs CLIPER"] = 1 - r["RMSE Random Forest"] / r["RMSE CLIPER"]
    for ref in ("CLIPER", "Linear Regression"):
        for loss in ("SE", "AE"):
            r[f"p {loss} RF vs {ref}"] = dm_lag(y, P[ref], P["Random Forest"], loss, R.NW_LAGS)
    rows.append(r)
    pooled["y"].append(y)
    pooled["day"].append(pd.DatetimeIndex(te.index).normalize())
    for k, v in P.items():
        pooled[k].append(v)

F = pd.DataFrame(rows)
y = np.concatenate(pooled["y"])
Pp = {k: np.concatenate(pooled[k]) for k in ("Persistence", "CLIPER", "Linear Regression", "Random Forest")}
days = pd.DatetimeIndex(np.concatenate([d.values for d in pooled["day"]]))
uday = days.unique()
L_DAY = int(round(len(y) / len(uday)))
prow = {"N": len(y), "days": len(uday), "one_day_lag": L_DAY}
for k, v in Pp.items():
    prow["RMSE " + k] = float(np.sqrt(np.mean((v - y) ** 2)))
    prow["MAE " + k] = float(np.mean(np.abs(v - y)))
prow["MAE skill RF vs CLIPER"] = 1 - prow["MAE Random Forest"] / prow["MAE CLIPER"]
prow["RMSE skill RF vs CLIPER"] = 1 - prow["RMSE Random Forest"] / prow["RMSE CLIPER"]
for ref in ("CLIPER", "Linear Regression", "Persistence"):
    for loss in ("SE", "AE"):
        prow[f"p5 {loss} RF vs {ref}"] = dm_lag(y, Pp[ref], Pp["Random Forest"], loss, R.NW_LAGS)
        prow[f"pDay {loss} RF vs {ref}"] = dm_lag(y, Pp[ref], Pp["Random Forest"], loss, L_DAY)
E = {k: y - v for k, v in Pp.items()}
idx_day = [np.where(days == d)[0] for d in uday]
rng = np.random.default_rng(R.SEED)
dr = {ref: {"dMSE": [], "dMAE": []} for ref in ("CLIPER", "Linear Regression")}
for _ in range(R.BOOT_N):
    ii = np.concatenate([idx_day[j] for j in rng.integers(0, len(uday), size=len(uday))])
    for ref in dr:
        ea, eb = E[ref][ii], E["Random Forest"][ii]
        dr[ref]["dMSE"].append(np.mean(eb ** 2) - np.mean(ea ** 2))
        dr[ref]["dMAE"].append(np.mean(np.abs(eb)) - np.mean(np.abs(ea)))
for ref in dr:
    for q in ("dMSE", "dMAE"):
        pt = (np.mean(E["Random Forest"] ** 2) - np.mean(E[ref] ** 2)) if q == "dMSE" else \
             (np.mean(np.abs(E["Random Forest"])) - np.mean(np.abs(E[ref])))
        lo, hi = np.percentile(dr[ref][q], [2.5, 97.5])
        prow[f"dayboot {q} RF-{ref}"] = f"{pt:.0f} [{lo:.0f}, {hi:.0f}]"

F.to_csv(os.path.join(OUT, "rolling_origin_folds.csv"), index=False)
pd.Series(prow).to_csv(os.path.join(OUT, "rolling_origin_pooled.csv"))
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print(F.drop(columns=["rf_params"]).round(4).T.to_string())
print(F[["fold", "rf_params"]].to_string())
print(pd.Series(prow).to_string())
