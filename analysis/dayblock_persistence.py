"""Day-block bootstrap (whole test days, 2000 resamples) of dMSE and dMAE for
each learner and the 6-feature RF against naive persistence (15 min).
Report-only; tuned settings read from results/*/run_log.txt.
Usage: PLANT=1|2 python analysis/dayblock_persistence.py"""
import ast, os, re, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import revision_round3 as R  # noqa: E402

OUT = os.path.join(ROOT, "analysis", "out", "plant%d" % R.PLANT)
os.makedirs(OUT, exist_ok=True)
R.OUT = OUT
log = open(os.path.join(ROOT, "results", "run_log.txt" if R.PLANT == 1
                        else os.path.join("plant2", "run_log.txt"))).read()
R.RF_PARAMS = ast.literal_eval(re.search(r"RF params: (\{.*\})", log).group(1))
R.XGB_PARAMS = ast.literal_eval(re.search(r"XGB params: (\{.*\})", log).group(1))

raw, df = R.load_grid()
f = R.build_features(df)
c = f.dropna(subset=R.FULL + ["target"])
m = c[c["elev_tau"] > R.ELEV_MIN]
k = int(len(m) * 0.8)
tr, te = m.iloc[:k], m.iloc[k:]
preds, _ = R.fit_causal(tr, te, tr.groupby("mod")["target"].mean())
y = te["target"].values
E = {n: y - p for n, p in preds.items()}
days = pd.DatetimeIndex(te.index).normalize()
uday = days.unique()
day_idx = [np.where(days == d)[0] for d in uday]
models = ["Linear Regression", "XGBoost", "Random Forest (full)", "Random Forest (reduced)"]
rng = np.random.default_rng(R.SEED)
draws = {mm: {"dMSE": [], "dMAE": []} for mm in models}
for _ in range(R.BOOT_N):
    idx = np.concatenate([day_idx[j] for j in rng.integers(0, len(uday), size=len(uday))])
    for mm in models:
        ea, eb = E["Persistence"][idx], E[mm][idx]
        draws[mm]["dMSE"].append(np.mean(eb ** 2) - np.mean(ea ** 2))
        draws[mm]["dMAE"].append(np.mean(np.abs(eb)) - np.mean(np.abs(ea)))
rows = []
for mm in models:
    for q in ("dMSE", "dMAE"):
        lo, hi = np.percentile(draws[mm][q], [2.5, 97.5])
        pt = (np.mean(E[mm] ** 2) - np.mean(E["Persistence"] ** 2) if q == "dMSE"
              else np.mean(np.abs(E[mm])) - np.mean(np.abs(E["Persistence"])))
        rows.append(dict(model=mm, quantity=q, point=pt, ci_lo=lo, ci_hi=hi,
                         excludes_zero=not (lo <= 0 <= hi)))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, "day_block_bootstrap_vs_persistence.csv"), index=False)
print(out.round(1).to_string())
