"""
SANKET 2026 -- ROUND-2 REVISION EVALUATION

Adds to the leakage-corrected pipeline:
  1. Reduced Random Forest (drops the four negative/zero-importance features).
  2. MBE and nRMSE (normalized by 28.0 MW rated capacity) for every model.
  3. Diebold-Mariano tests (h=1, rectangular kernel) on squared and absolute
     loss differentials, pairwise across Persistence / LR / RF.
  4. Empirical 95th-percentile absolute residuals, and the operational
     BESS-throughput and reserve figures they imply against BOTH
     Persistence and Linear Regression.
  5. The leaked (nowcasting) model, recomputed so the paper's Table I can
     show it on the same footing as the causal models.

Writes results/revision_metrics_round2.txt.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = os.path.dirname(os.path.abspath(__file__))
DC_SCALE = 10.0
CAPACITY_KW = 28000.0      # 28.0 MW rated, for nRMSE
DT_H = 0.25                # dispatch interval in hours
SEED = 42

# ------------------------------------------------------------------ load
gen = pd.read_csv(os.path.join(ROOT, "data", "Plant_1_Generation_Data.csv"))
wx = pd.read_csv(os.path.join(ROOT, "data", "Plant_1_Weather_Sensor_Data.csv"))
gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], dayfirst=True)
wx["DATE_TIME"] = pd.to_datetime(wx["DATE_TIME"])

plant = gen.groupby("DATE_TIME").agg(
    AC_POWER=("AC_POWER", "sum"), DC_POWER=("DC_POWER", "sum")).reset_index()
plant["DC_POWER"] /= DC_SCALE
wxf = wx[["DATE_TIME", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
          "IRRADIATION"]].drop_duplicates("DATE_TIME")
df = pd.merge(plant, wxf, on="DATE_TIME", how="inner").sort_values(
    "DATE_TIME").reset_index(drop=True)

mins = df["DATE_TIME"].dt.hour * 60 + df["DATE_TIME"].dt.minute
df["time_sin"] = np.sin(2 * np.pi * mins / 1440.0)
df["time_cos"] = np.cos(2 * np.pi * mins / 1440.0)
df["ac_now"] = df["AC_POWER"]
df["lag_15m"] = df["AC_POWER"].shift(1)
df["lag_30m"] = df["AC_POWER"].shift(2)
df["lag_1h"] = df["AC_POWER"].shift(4)
df["lag_24h"] = df["AC_POWER"].shift(96)
df["roll_1h"] = df["AC_POWER"].rolling(4).mean()
df["d_ac_15m"] = df["ac_now"] - df["lag_15m"]
df["d_irr_15m"] = df["IRRADIATION"] - df["IRRADIATION"].shift(1)
df["target"] = df["AC_POWER"].shift(-1)

FULL = ["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
        "time_sin", "time_cos", "ac_now", "lag_15m", "lag_30m", "lag_1h",
        "lag_24h", "roll_1h", "d_ac_15m", "d_irr_15m"]
# Dropped by ablation: lag_30m, roll_1h, lag_24h, lag_1h (negative / ~zero PI)
REDUCED = ["ac_now", "IRRADIATION", "MODULE_TEMPERATURE",
           "time_sin", "time_cos", "lag_15m"]

d = df.dropna(subset=FULL + ["target"]).reset_index(drop=True)

# Daylight filter actually applied: positive target. Report how a
# G_t-based filter would compare, since the two are not identical.
mask_target = d["target"] > 0
mask_irr = d["IRRADIATION"] > 0
print("daylight filter comparison on %d rows: target>0 -> %d | G_t>0 -> %d | both -> %d"
      % (len(d), mask_target.sum(), mask_irr.sum(), (mask_target & mask_irr).sum()))

d = d[mask_target].reset_index(drop=True)
split = int(len(d) * 0.80)
tr, te = d.iloc[:split], d.iloc[split:]
y_tr, y_te = tr["target"].values, te["target"].values
print("records=%d train=%d test=%d" % (len(d), len(tr), len(te)))

# ------------------------------------------------------------ causal models
preds = {"Persistence": te["ac_now"].values}

preds["Linear Regression"] = np.clip(
    LinearRegression().fit(tr[FULL], y_tr).predict(te[FULL]), 0, None)

rf_full = RandomForestRegressor(n_estimators=150, max_depth=14,
                               min_samples_split=4, random_state=SEED,
                               n_jobs=-1).fit(tr[FULL], y_tr)
preds["Random Forest (full)"] = np.clip(rf_full.predict(te[FULL]), 0, None)

preds["XGBoost"] = np.clip(
    XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                 n_jobs=-1).fit(tr[FULL], y_tr).predict(te[FULL]), 0, None)

rf_red = RandomForestRegressor(n_estimators=150, max_depth=14,
                              min_samples_split=4, random_state=SEED,
                              n_jobs=-1).fit(tr[REDUCED], y_tr)
preds["Random Forest (reduced)"] = np.clip(rf_red.predict(te[REDUCED]), 0, None)

# ------------------------------------------------- leaked comparator (P_t)
# Original flawed setup: predict P_t from contemporaneous G_t, all hours.
lk = df.dropna(subset=["IRRADIATION", "AMBIENT_TEMPERATURE",
                       "MODULE_TEMPERATURE", "time_sin", "time_cos",
                       "lag_15m", "lag_30m", "lag_1h", "lag_24h",
                       "roll_1h"]).reset_index(drop=True)
LKF = ["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
       "time_sin", "time_cos", "lag_15m", "lag_30m", "lag_1h", "lag_24h",
       "roll_1h"]
lsplit = int(len(lk) * 0.80)
lk_tr, lk_te = lk.iloc[:lsplit], lk.iloc[lsplit:]
rf_leak = RandomForestRegressor(n_estimators=150, max_depth=14,
                               min_samples_split=4, random_state=SEED,
                               n_jobs=-1).fit(lk_tr[LKF], lk_tr["AC_POWER"])
leak_pred = np.clip(rf_leak.predict(lk_te[LKF]), 0, None)
leak_true = lk_te["AC_POWER"].values


def scores(yt, yp):
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    return {
        "RMSE": rmse,
        "nRMSE%": 100.0 * rmse / CAPACITY_KW,
        "MAE": float(mean_absolute_error(yt, yp)),
        "MBE": float(np.mean(yp - yt)),
        "R2": float(r2_score(yt, yp)),
        "P95": float(np.percentile(np.abs(yt - yp), 95)),
    }


rows = {k: scores(y_te, v) for k, v in preds.items()}
leak_row = scores(leak_true, leak_pred)


# ------------------------------------------------------ Diebold-Mariano
def dm_test(y, p1, p2, loss="SE", h=1):
    """Diebold-Mariano statistic with the Harvey-Leybourne-Newbold (1997)
    small-sample correction. h=1 -> rectangular kernel, gamma_0 only.

    Returns (DM, p_normal, DM_HLN, p_HLN). The HLN variant rescales by
    sqrt([N + 1 - 2h + h(h-1)/N] / N) and refers to t with N-1 df, which is
    the conservative choice at N=328. Negative statistic favours model 1."""
    e1, e2 = y - p1, y - p2
    dl = e1 ** 2 - e2 ** 2 if loss == "SE" else np.abs(e1) - np.abs(e2)
    n = len(dl)
    dbar = dl.mean()
    gamma = [np.sum((dl - dbar) * (dl - dbar)) / n]
    for k in range(1, h):
        gamma.append(np.sum((dl[k:] - dbar) * (dl[:-k] - dbar)) / n)
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return (float("nan"),) * 4
    stat = dbar / np.sqrt(var)
    adj = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat_hln = stat * adj
    p_norm = 2 * (1 - stats.norm.cdf(abs(stat)))
    p_hln = 2 * (1 - stats.t.cdf(abs(stat_hln), df=n - 1))
    return float(stat), float(p_norm), float(stat_hln), float(p_hln)


PAIRS = [("Persistence", "Linear Regression"),
         ("Persistence", "XGBoost"),
         ("Persistence", "Random Forest (full)"),
         ("Persistence", "Random Forest (reduced)"),
         ("Linear Regression", "Random Forest (full)"),
         ("Random Forest (full)", "Random Forest (reduced)")]

# ------------------------------------------------------------------ report
L = []
A = L.append
A("=" * 86)
A("  SANKET 2026 -- ROUND-2 REVISION METRICS (leak-free causal benchmark)")
A("=" * 86)
A("Target P_{t+15m}; features in I_t. Daylight (target>0). "
  "records=%d train=%d test=%d" % (len(d), len(tr), len(te)))
A("nRMSE normalized by rated capacity %.1f MW." % (CAPACITY_KW / 1000.0))
A("-" * 86)
A("%-26s %9s %8s %9s %9s %8s %9s"
  % ("Model", "RMSE", "nRMSE%", "MAE", "MBE", "R2", "P95|e|"))
A("-" * 86)
A("%-26s %9.1f %8.2f %9.1f %9.1f %8.4f %9.1f"
  % ("[LEAKED] RF nowcast", leak_row["RMSE"], leak_row["nRMSE%"],
     leak_row["MAE"], leak_row["MBE"], leak_row["R2"], leak_row["P95"]))
for k in ("Persistence", "Linear Regression", "XGBoost",
          "Random Forest (full)", "Random Forest (reduced)"):
    r = rows[k]
    A("%-26s %9.1f %8.2f %9.1f %9.1f %8.4f %9.1f"
      % (k, r["RMSE"], r["nRMSE%"], r["MAE"], r["MBE"], r["R2"], r["P95"]))
A("-" * 86)
A("Leakage delta: R2 %.4f -> %.4f ; RMSE x%.2f ; MAE x%.2f"
  % (leak_row["R2"], rows["Random Forest (full)"]["R2"],
     rows["Random Forest (full)"]["RMSE"] / leak_row["RMSE"],
     rows["Random Forest (full)"]["MAE"] / leak_row["MAE"]))
A("")
A("DIEBOLD-MARIANO (h=1, rectangular kernel) with Harvey-Leybourne-Newbold")
A("small-sample correction; p_HLN referred to t(N-1), N=%d. "
  "Negative stat favours model 1." % len(y_te))
A("%-44s %9s %9s %9s %9s"
  % ("Pair", "DM(SE)", "p_HLN(SE)", "DM(AE)", "p_HLN(AE)"))
A("-" * 86)
for a, b in PAIRS:
    s1, pn1, h1, ph1 = dm_test(y_te, preds[a], preds[b], "SE")
    s2, pn2, h2, ph2 = dm_test(y_te, preds[a], preds[b], "AE")
    A("%-44s %9.3f %9.4f %9.3f %9.4f"
      % ("%s vs %s" % (a, b), h1, ph1, h2, ph2))
A("  (uncorrected normal p-values for reference:)")
for a, b in PAIRS:
    _, pn1, _, _ = dm_test(y_te, preds[a], preds[b], "SE")
    _, pn2, _, _ = dm_test(y_te, preds[a], preds[b], "AE")
    A("  %-42s %9s %9.4f %9s %9.4f"
      % ("%s vs %s" % (a, b), "", pn1, "", pn2))
A("")
A("OPERATIONAL FIGURES (dispatch interval %.2f h)" % DT_H)
A("%-26s %14s %16s %16s"
  % ("Model", "kWh/interval", "vs Persistence", "vs LR"))
A("-" * 86)
base_p = rows["Persistence"]["MAE"] * DT_H
base_l = rows["Linear Regression"]["MAE"] * DT_H
for k in ("Persistence", "Linear Regression", "XGBoost",
          "Random Forest (full)", "Random Forest (reduced)"):
    e = rows[k]["MAE"] * DT_H
    A("%-26s %14.1f %16.1f %16.1f" % (k, e, base_p - e, base_l - e))
A("")
A("RESERVE from EMPIRICAL 95th-percentile |residual| (no Gaussian assumption)")
A("%-26s %14s %16s %16s"
  % ("Model", "R95 (kW)", "vs Persistence", "vs LR"))
A("-" * 86)
rp, rl = rows["Persistence"]["P95"], rows["Linear Regression"]["P95"]
for k in ("Persistence", "Linear Regression", "XGBoost",
          "Random Forest (full)", "Random Forest (reduced)"):
    v = rows[k]["P95"]
    A("%-26s %14.1f %16.1f %16.1f" % (k, v, rp - v, rl - v))
A("=" * 86)

report = "\n".join(L)
print("\n" + report)
with open(os.path.join(ROOT, "results", "revision_metrics_round2.txt"), "w") as f:
    f.write(report + "\n")
