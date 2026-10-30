"""Regenerate Fig. 2(a) only (Plant 1) from the tuned settings in results/run_log.txt,
without re-running the pipeline (results/ is not written)."""
import ast, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PLANT"] = "1"
import revision_round3 as R
log = open(os.path.join(ROOT, "results", "run_log.txt")).read()
R.RF_PARAMS = ast.literal_eval(re.search(r"RF params: (\{.*\})", log).group(1))
R.XGB_PARAMS = ast.literal_eval(re.search(r"XGB params: (\{.*\})", log).group(1))
raw, df = R.load_grid()
f = R.build_features(df)
c = f.dropna(subset=R.FULL + ["target"]); m = c[c["elev_tau"] > R.ELEV_MIN]
k = int(len(m) * 0.8); tr, te = m.iloc[:k], m.iloc[k:]
preds, _ = R.fit_causal(tr, te, tr.groupby("mod")["target"].mean())
y = te["target"].values
import numpy as np
for n in ("Persistence", "Linear Regression", "XGBoost", "Random Forest (full)"):
    print(n, "sigma %.0f" % np.std(y - preds[n], ddof=1))
R.plot_residuals(y, preds)
