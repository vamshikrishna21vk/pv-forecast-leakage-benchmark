"""Rolling-origin (expanding-window) evaluation, 15-minute horizon, v2.

Adds XGBoost (re-tuned per fold on the paper's grid) to persistence, CLIPER,
Linear Regression and the 13-feature Random Forest, runs a per-fold leakage
check, and applies Holm correction to pooled DM tests: within each plant,
3 learners x 2 losses vs CLIPER (6 tests) and vs persistence (6 tests).
RF vs LR is reported unadjusted.

Folds (test windows of whole calendar days; training = all earlier daylight
rows; validation slice = last 25% of training, strictly before the test):
  F1 26-31 May, F2 1-6 Jun, F3 7-12 Jun, F4 13-17 Jun.

Outputs: results/rolling_origin/plant<N>/{folds.csv, pooled.csv,
pooled_dm_holm.csv, dayblock.csv}, results/rolling_origin/leakage_check.txt
(appended per plant). Usage: PLANT=1|2 python analysis/rolling_origin_v2.py
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import revision_round3 as R  # noqa: E402

FOLDS = [("F1", "2020-05-26", "2020-06-01"), ("F2", "2020-06-01", "2020-06-07"),
         ("F3", "2020-06-07", "2020-06-13"), ("F4", "2020-06-13", "2020-06-18")]
# RO_FOLDS=F1,F2 restricts the run to a subset (written to rolling_origin_F1F2/).
_sub = os.environ.get("RO_FOLDS")
if _sub:
    FOLDS = [f_ for f_ in FOLDS if f_[0] in _sub.split(",")]

BASE = os.path.join(ROOT, "results", "rolling_origin" + ("_" + _sub.replace(",", "") if _sub else ""))
OUT = os.path.join(BASE, "plant%d" % R.PLANT)
os.makedirs(OUT, exist_ok=True)
R.OUT = OUT
LEARN = ["Linear Regression", "XGBoost", "Random Forest"]

raw, df = R.load_grid()
f = R.build_features(df)
c = f.dropna(subset=R.FULL + ["target"])
main = c[c["elev_tau"] > R.ELEV_MIN]
P_grid = df["AC_POWER"]
G_grid = df["IRRADIATION"]


def tune(tr):
    nv = int(len(tr) * 0.75)
    fit, val = tr.iloc[:nv], tr.iloc[nv:]
    best_rf, br = None, np.inf
    for depth in (6, 10, 14, None):
        for leaf in (1, 5, 10):
            for mf in (1.0, 0.5, "sqrt"):
                p = dict(n_estimators=300, max_depth=depth, min_samples_leaf=leaf, max_features=mf)
                m = RandomForestRegressor(**p, random_state=R.SEED, n_jobs=-1).fit(fit[R.FULL], fit["target"])
                r = np.sqrt(np.mean((np.clip(m.predict(val[R.FULL]), 0, None) - val["target"]) ** 2))
                if r < br:
                    best_rf, br = p, r
    best_x, bx = None, np.inf
    for depth in (2, 3, 4, 6):
        for lr_ in (0.03, 0.05, 0.1):
            for n in (150, 300):
                p = dict(n_estimators=n, max_depth=depth, learning_rate=lr_, subsample=0.8, colsample_bytree=0.8)
                m = XGBRegressor(**p, random_state=R.SEED, n_jobs=-1).fit(fit[R.FULL], fit["target"])
                r = np.sqrt(np.mean((np.clip(m.predict(val[R.FULL]), 0, None) - val["target"]) ** 2))
                if r < bx:
                    best_x, bx = p, r
    return best_rf, best_x, fit.index, val.index


def leak_check(name, tr, te, fit_idx, val_idx):
    """Return a list of (check, passed, detail)."""
    out = []
    out.append(("train ends before test", tr.index.max() < te.index.min(),
                f"train max {tr.index.max()}, test min {te.index.min()}"))
    out.append(("validation slice is inside training and before test",
                val_idx.min() > fit_idx.max() and val_idx.max() < te.index.min()
                and val_idx.isin(tr.index).all(),
                f"fit {fit_idx.min()}..{fit_idx.max()}, val {val_idx.min()}..{val_idx.max()}"))
    out.append(("no test stamp in training", not te.index.isin(tr.index).any(), ""))
    # every causal feature of a test row equals the grid value at an earlier stamp
    dt = pd.Timedelta("15min")
    ok = True
    for col, lag in (("ac_now", 1), ("lag_15m", 2), ("lag_30m", 3), ("lag_1h", 5), ("lag_24h", 97)):
        ref = P_grid.reindex(te.index - lag * dt).values
        ok &= np.allclose(te[col].values, ref, equal_nan=False)
    ok &= np.allclose(te["IRRADIATION"].values, G_grid.reindex(te.index - dt).values)
    ok &= np.allclose(te["d_irr_15m"].values,
                      G_grid.reindex(te.index - dt).values - G_grid.reindex(te.index - 2 * dt).values)
    roll = np.mean([P_grid.reindex(te.index - k * dt).values for k in (1, 2, 3, 4)], axis=0)
    ok &= np.allclose(te["roll_1h"].values, roll)
    out.append(("test features read only stamps <= tau - 15 min (P lags, G, dG, roll_1h verified)", bool(ok), ""))
    out.append(("no feature column read at target time tau",
                not any(col.endswith("_tau") for col in R.FULL), f"FULL = {R.FULL}"))
    return out


fold_rows, checks = [], []
pool = {k: [] for k in ["y", "day", "Persistence", "CLIPER"] + LEARN}
for name, a, b in FOLDS:
    tr = main[main.index < a]
    te = main[(main.index >= a) & (main.index < b)]
    y, y_tr = te["target"].values, tr["target"].values
    rfp, xgp, fit_idx, val_idx = tune(tr)
    for chk in leak_check(name, tr, te, fit_idx, val_idx):
        checks.append((name,) + chk)
    clim = tr.groupby("mod")["target"].mean()
    c_tr, c_te = R.clim_lookup(clim, tr["mod"]), R.clim_lookup(clim, te["mod"])
    alpha = float(np.clip(np.sum((tr["ac_now"].values - c_tr) * (y_tr - c_tr))
                          / np.sum((tr["ac_now"].values - c_tr) ** 2), 0, 1))
    P = {"Persistence": te["ac_now"].values,
         "CLIPER": alpha * te["ac_now"].values + (1 - alpha) * c_te,
         "Linear Regression": np.clip(LinearRegression().fit(tr[R.FULL], y_tr).predict(te[R.FULL]), 0, None),
         "XGBoost": np.clip(XGBRegressor(**xgp, random_state=R.SEED, n_jobs=-1)
                            .fit(tr[R.FULL], y_tr).predict(te[R.FULL]), 0, None),
         "Random Forest": np.clip(RandomForestRegressor(**rfp, random_state=R.SEED, n_jobs=-1)
                                  .fit(tr[R.FULL], y_tr).predict(te[R.FULL]), 0, None)}
    r = {"fold": name, "test_from": a, "test_to_excl": b, "n_train": len(tr), "n_val": len(val_idx),
         "n_test": len(te), "n_days": pd.DatetimeIndex(te.index).normalize().nunique(),
         "cliper_alpha": alpha, "rf_params": repr(rfp), "xgb_params": repr(xgp)}
    for k, v in P.items():
        r["RMSE " + k] = float(np.sqrt(np.mean((v - y) ** 2)))
        r["MAE " + k] = float(np.mean(np.abs(v - y)))
    for m in LEARN:
        for ref in ("Persistence", "CLIPER"):
            r[f"RMSE skill {m} vs {ref}"] = 1 - r["RMSE " + m] / r["RMSE " + ref]
            r[f"MAE skill {m} vs {ref}"] = 1 - r["MAE " + m] / r["MAE " + ref]
    for ref in ("CLIPER", "Linear Regression"):
        for loss in ("SE", "AE"):
            r[f"p {loss} RF vs {ref}"] = R.dm_test(y, P[ref], P["Random Forest"], loss)[1]
    fold_rows.append(r)
    pool["y"].append(y)
    pool["day"].append(pd.DatetimeIndex(te.index).normalize())
    for k, v in P.items():
        pool[k].append(v)

F = pd.DataFrame(fold_rows)
y = np.concatenate(pool["y"])
Pp = {k: np.concatenate(pool[k]) for k in ["Persistence", "CLIPER"] + LEARN}
days = pd.DatetimeIndex(np.concatenate([d.values for d in pool["day"]]))
uday = days.unique()
L_DAY = int(round(len(y) / len(uday)))

pooled = {"N": len(y), "test_days": len(uday), "one_day_lag": L_DAY}
for k, v in Pp.items():
    pooled["RMSE " + k] = float(np.sqrt(np.mean((v - y) ** 2)))
    pooled["MAE " + k] = float(np.mean(np.abs(v - y)))
for m in LEARN:
    for ref in ("Persistence", "CLIPER"):
        pooled[f"RMSE skill {m} vs {ref}"] = 1 - pooled["RMSE " + m] / pooled["RMSE " + ref]
        pooled[f"MAE skill {m} vs {ref}"] = 1 - pooled["MAE " + m] / pooled["MAE " + ref]

dm_rows = []
for ref in ("Persistence", "CLIPER"):
    for m in LEARN:
        for loss in ("SE", "AE"):
            dm_rows.append(dict(ref=ref, model=m, loss=loss,
                                p_NW5=R.dm_test(y, Pp[ref], Pp[m], loss)[1],
                                p_NWday=R.dm_test(y, Pp[ref], Pp[m], loss, lags=L_DAY)[1]))
for loss in ("SE", "AE"):
    dm_rows.append(dict(ref="Linear Regression", model="Random Forest", loss=loss,
                        p_NW5=R.dm_test(y, Pp["Linear Regression"], Pp["Random Forest"], loss)[1],
                        p_NWday=R.dm_test(y, Pp["Linear Regression"], Pp["Random Forest"], loss, lags=L_DAY)[1]))
D = pd.DataFrame(dm_rows)
for ref in ("Persistence", "CLIPER"):
    msk = D["ref"].eq(ref)
    D.loc[msk, "p_NW5_Holm6"] = R.holm(D.loc[msk, "p_NW5"].values)
    D.loc[msk, "p_NWday_Holm6"] = R.holm(D.loc[msk, "p_NWday"].values)

E = {k: y - v for k, v in Pp.items()}
idx_day = [np.where(days == d_)[0] for d_ in uday]
rng = np.random.default_rng(R.SEED)
bpairs = [("CLIPER", "Random Forest"), ("Linear Regression", "Random Forest"),
          ("Persistence", "Linear Regression"), ("Persistence", "XGBoost"), ("Persistence", "Random Forest")]
dr = {p_: {"dMSE": [], "dMAE": []} for p_ in bpairs}
for _ in range(R.BOOT_N):
    ii = np.concatenate([idx_day[j] for j in rng.integers(0, len(uday), size=len(uday))])
    for a_, b_ in bpairs:
        ea, eb = E[a_][ii], E[b_][ii]
        dr[(a_, b_)]["dMSE"].append(np.mean(eb ** 2) - np.mean(ea ** 2))
        dr[(a_, b_)]["dMAE"].append(np.mean(np.abs(eb)) - np.mean(np.abs(ea)))
brows = []
for a_, b_ in bpairs:
    for q in ("dMSE", "dMAE"):
        pt = (np.mean(E[b_] ** 2) - np.mean(E[a_] ** 2)) if q == "dMSE" else \
             (np.mean(np.abs(E[b_])) - np.mean(np.abs(E[a_])))
        lo, hi = np.percentile(dr[(a_, b_)][q], [2.5, 97.5])
        brows.append(dict(ref=a_, model=b_, quantity=q, point=pt, ci_lo=lo, ci_hi=hi,
                          excludes_zero=not (lo <= 0 <= hi)))
B = pd.DataFrame(brows)

F.to_csv(os.path.join(OUT, "folds.csv"), index=False)
pd.Series(pooled).to_csv(os.path.join(OUT, "pooled.csv"), header=["value"])
D.to_csv(os.path.join(OUT, "pooled_dm_holm.csv"), index=False)
B.to_csv(os.path.join(OUT, "dayblock.csv"), index=False)
with open(os.path.join(BASE, "leakage_check.txt"), "a", encoding="utf-8") as fh:
    fh.write(f"=== Plant {R.PLANT} ===\n")
    for name, chk, ok, det in checks:
        fh.write(f"[{'PASS' if ok else 'FAIL'}] {name}: {chk} {det}\n")
    fh.write("Training-only fits: CLIPER climatology and alpha, LR, XGB, RF are fitted on the fold's "
             "training rows; tuning scores only the validation slice (last 25% of training).\n"
             "Daylight filter uses solar elevation at tau, a deterministic function of the clock.\n\n")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
cols = ["fold", "n_train", "n_val", "n_test", "n_days"] + [c_ for c_ in F.columns if "skill" in c_ or c_.startswith("p ")]
print(F[cols].round(4).T.to_string())
print(pd.Series(pooled).round(4).to_string())
print(D.round(5).to_string())
print(B.round(0).to_string())
print("leakage checks:", sum(ok for _, _, ok, _ in checks), "/", len(checks), "pass")
