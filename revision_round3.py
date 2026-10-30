"""
SANKET 2026 -- ROUND-3 REVISION ANALYSIS (REVIEW.md section 2)

Changes against eval_revision_round2.py:

  1. REGULAR TIME INDEX. Telemetry is reindexed onto a complete 15-minute
     grid before any lag is built, so P_{t-96} is exactly 24 h earlier and a
     missing stamp yields NaN rather than a silently shifted neighbour.
  2. DAYLIGHT FILTER ON SOLAR ELEVATION at the target time (> 5 deg), which is
     known at issue time. The dataset documents no coordinates: longitude is
     estimated from the irradiance onset/offset midpoint (78.5 E, timestamps
     taken as IST), latitude is assumed (20 N) and varied in a sensitivity run.
  3. 2x2 LEAKAGE TABLE {leaked nowcast, causal forecast} x {all hours,
     daylight} on a common chronological cutoff, indexed by target time.
  4. CLIPER and clear-sky smart persistence as additional references.
  5. Diebold-Mariano with Newey-West (Bartlett, 5 lags) long-run variance and
     the HLN correction; Ljung-Box on the loss differentials; Holm correction.
  6. Paired moving-block bootstrap CIs (block 6, 2000 resamples).
  7. MBE on the training block and on a time-ordered validation block.
  8. Permutation importance with 50 repeats plus grouped importance.
  9. nRMSE normalized by the observed peak aggregate AC output.

All outputs are written to results/ and the two diagnostic figures to
figures/.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

ROOT = os.path.dirname(os.path.abspath(__file__))
PLANT = int(os.environ.get("PLANT", "1"))
EXCL_OUTAGE = os.environ.get("EXCL_OUTAGE", "0") == "1"
OUT = os.path.join(ROOT, "results") if PLANT == 1 else os.path.join(ROOT, "results", "plant%d" % PLANT)
if EXCL_OUTAGE:
    OUT += "_no_outage" if PLANT != 1 else os.sep + "no_outage"
# Only Plant 1 writes the paper's figures; other plants keep theirs in OUT.
FIG_DIR = os.path.join(ROOT, "figures") if PLANT == 1 else OUT
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Plant 1 publishes DC at ten times the AC scale; Plant 2 does not (DC/AC ~1.02).
DC_SCALE = {1: 10.0, 2: 1.0}[PLANT]
DT_H = 0.25
SEED = 42
# Estimated from the irradiance onset/offset midpoint (lon_estimate.py).
LON_EST = {1: 78.5, 2: 78.3}[PLANT]
LAT_ASSUMED = 20.0         # not documented; varied in sensitivity run
TZ_OFFSET_H = 5.5          # timestamps taken as IST
ELEV_MIN = 5.0
NW_LAGS = 5
BOOT_N, BOOT_BLOCK = 2000, 6
PI_REPEATS = 50
N_INVERTERS = 22

FULL = ["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
        "time_sin", "time_cos", "ac_now", "lag_15m", "lag_30m", "lag_1h",
        "lag_24h", "roll_1h", "d_ac_15m", "d_irr_15m"]
REDUCED = ["ac_now", "IRRADIATION", "MODULE_TEMPERATURE",
           "time_sin", "time_cos", "lag_15m"]
# Leaked nowcast as the paper describes it: contemporaneous weather at the
# target time plus strictly lagged output (rolling mean excludes the target).
LEAK = ["IRRADIATION_tau", "AMBIENT_TEMPERATURE_tau", "MODULE_TEMPERATURE_tau",
        "time_sin_tau", "time_cos_tau", "lag_15m_tau", "lag_30m_tau",
        "lag_1h_tau", "lag_24h_tau", "roll_1h_excl_tau"]
# Round-2 leaked comparator: its rolling mean includes the target P_tau.
LEAK_R2 = LEAK[:-1] + ["roll_1h_incl_tau"]
# Matched leaked comparator: exactly the 13 causal features, except that the
# four weather-sensor features are read at the target time tau instead of at
# the issue time t = tau-1. Everything else (lags, encodings, tuning) is equal.
MATCHED = ["IRRADIATION_tau", "AMBIENT_TEMPERATURE_tau", "MODULE_TEMPERATURE_tau",
           "time_sin", "time_cos", "ac_now", "lag_15m", "lag_30m", "lag_1h",
           "lag_24h", "roll_1h", "d_ac_15m", "d_irr_tau"]
LAG_GROUP = ["lag_15m", "lag_30m", "lag_1h", "lag_24h", "roll_1h",
             "d_ac_15m"]


# ------------------------------------------------------------ solar geometry
def solar_elevation(ts, lat=LAT_ASSUMED, lon=LON_EST):
    """NOAA solar-position approximation (accuracy ~0.1 deg); ts local IST."""
    ts = pd.DatetimeIndex(ts)
    doy = ts.dayofyear.values
    hour = ts.hour.values + ts.minute.values / 60.0
    g = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
            - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hour * 60 + eot + 4 * lon - 60 * TZ_OFFSET_H
    ha = np.radians(tst / 4.0 - 180.0)
    phi = np.radians(lat)
    cosz = (np.sin(phi) * np.sin(decl)
            + np.cos(phi) * np.cos(decl) * np.cos(ha))
    return np.degrees(np.arcsin(np.clip(cosz, -1, 1)))


def clear_sky_ghi(elev_deg):
    """Haurwitz clear-sky GHI (W/m^2), a generic model needing no inputs."""
    cosz = np.sin(np.radians(np.asarray(elev_deg)))
    return np.where(cosz > 0, 1098.0 * cosz * np.exp(-0.057 / np.maximum(cosz, 1e-3)), 0.0)


# ------------------------------------------------------- load onto 15-min grid
def load_grid(drop_partial=False):
    gen = pd.read_csv(os.path.join(ROOT, "data", "Plant_%d_Generation_Data.csv" % PLANT))
    wx = pd.read_csv(os.path.join(ROOT, "data", "Plant_%d_Weather_Sensor_Data.csv" % PLANT))
    # Plant 1 generation stamps are dd-mm-yyyy; Plant 2 stamps are ISO.
    gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], dayfirst=(PLANT == 1))
    wx["DATE_TIME"] = pd.to_datetime(wx["DATE_TIME"])
    plant = gen.groupby("DATE_TIME").agg(
        AC_POWER=("AC_POWER", "sum"), DC_POWER=("DC_POWER", "sum"),
        N_INV=("SOURCE_KEY", "nunique"))
    plant["DC_POWER"] /= DC_SCALE
    wxf = wx.drop_duplicates("DATE_TIME").set_index("DATE_TIME")[
        ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]
    raw = plant.join(wxf, how="inner")
    # Outage stamps: any inverter reporting zero AC while plant irradiance
    # exceeds 0.3 (strong sun). With EXCL_OUTAGE=1 they are set missing.
    g_ = gen.join(wxf["IRRADIATION"], on="DATE_TIME")
    outage = pd.DatetimeIndex(g_.loc[(g_["AC_POWER"] <= 0) & (g_["IRRADIATION"] > 0.3),
                                     "DATE_TIME"].unique())
    raw["OUTAGE"] = raw.index.isin(outage)
    grid = pd.date_range(raw.index.min().floor("D"),
                         raw.index.max().floor("D") + pd.Timedelta("23h45min"),
                         freq="15min")
    df = raw.reindex(grid)
    df.index.name = "DATE_TIME"
    if drop_partial:
        df.loc[df["N_INV"] < N_INVERTERS, :] = np.nan
    if EXCL_OUTAGE:
        df.loc[df["OUTAGE"] == True, :] = np.nan  # noqa: E712 (NaN-safe)
    return raw, df


def build_features(df, lat=LAT_ASSUMED, h=1):
    """One row per TARGET time tau. Causal features are taken at t = tau-h
    (h = 1, 2, 4 for 15, 30, 60 min), leaked features at tau. shift() on the
    regular grid is time-exact."""
    P = df["AC_POWER"]
    mins = df.index.hour * 60 + df.index.minute
    f = pd.DataFrame(index=df.index)
    f["target"] = P
    f["elev_tau"] = solar_elevation(df.index, lat=lat)
    f["ghi_cs_tau"] = clear_sky_ghi(f["elev_tau"])
    sin_, cos_ = np.sin(2 * np.pi * mins / 1440.0), np.cos(2 * np.pi * mins / 1440.0)

    # causal: information set at t = tau - h
    Pt = P.shift(h)
    f["ac_now"] = Pt
    f["lag_15m"] = P.shift(h + 1)
    f["lag_30m"] = P.shift(h + 2)
    f["lag_1h"] = P.shift(h + 4)
    f["lag_24h"] = P.shift(h + 96)
    f["roll_1h"] = P.shift(h).rolling(4, min_periods=4).mean()
    f["d_ac_15m"] = Pt - P.shift(h + 1)
    f["IRRADIATION"] = df["IRRADIATION"].shift(h)
    f["AMBIENT_TEMPERATURE"] = df["AMBIENT_TEMPERATURE"].shift(h)
    f["MODULE_TEMPERATURE"] = df["MODULE_TEMPERATURE"].shift(h)
    f["d_irr_15m"] = df["IRRADIATION"].shift(h) - df["IRRADIATION"].shift(h + 1)
    f["time_sin"] = pd.Series(sin_, index=df.index).shift(h)
    f["time_cos"] = pd.Series(cos_, index=df.index).shift(h)
    f["elev_t"] = pd.Series(f["elev_tau"].values, index=df.index).shift(h)
    f["ghi_cs_t"] = clear_sky_ghi(f["elev_t"].fillna(-90))

    # leaked: contemporaneous with the target
    f["IRRADIATION_tau"] = df["IRRADIATION"]
    f["AMBIENT_TEMPERATURE_tau"] = df["AMBIENT_TEMPERATURE"]
    f["MODULE_TEMPERATURE_tau"] = df["MODULE_TEMPERATURE"]
    f["time_sin_tau"], f["time_cos_tau"] = sin_, cos_
    f["lag_15m_tau"] = P.shift(1)
    f["lag_30m_tau"] = P.shift(2)
    f["lag_1h_tau"] = P.shift(4)
    f["lag_24h_tau"] = P.shift(96)
    f["roll_1h_excl_tau"] = P.shift(1).rolling(4, min_periods=4).mean()
    f["roll_1h_incl_tau"] = P.rolling(4, min_periods=4).mean()
    f["d_irr_tau"] = df["IRRADIATION"] - df["IRRADIATION"].shift(1)
    f["mod"] = mins
    return f


# ---------------------------------------------------------------- scoring
def scores(y, p, peak):
    e = p - y
    rmse = float(np.sqrt(np.mean(e ** 2)))
    return {"N": len(y), "RMSE": rmse, "nRMSE%": 100 * rmse / peak,
            "MAE": float(np.mean(np.abs(e))), "MBE": float(np.mean(e)),
            "R2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)),
            "P95": float(np.percentile(np.abs(e), 95)),
            "sigma_ddof1": float(np.std(y - p, ddof=1))}


def dm_test(y, p1, p2, loss="SE", lags=NW_LAGS, h=1):
    """DM with Newey-West (Bartlett) long-run variance and the HLN
    small-sample factor for horizon h, referred to t(N-1). Negative stat
    favours model 1."""
    e1, e2 = y - p1, y - p2
    d = e1 ** 2 - e2 ** 2 if loss == "SE" else np.abs(e1) - np.abs(e2)
    n, dbar = len(d), d.mean()
    dc = d - dbar
    lrv = np.sum(dc * dc) / n
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * np.sum(dc[k:] * dc[:-k]) / n
    stat = dbar / np.sqrt(lrv / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def ljung_box(x, lags=10):
    x = x - x.mean()
    n = len(x)
    acf = np.array([np.sum(x[k:] * x[:-k]) / np.sum(x * x) for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(acf ** 2 / (n - np.arange(1, lags + 1)))
    return float(q), float(stats.chi2.sf(q, lags)), acf


def holm(pvals):
    order = np.argsort(pvals)
    m, adj, run = len(pvals), np.empty(len(pvals)), 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * pvals[j])
        adj[j] = min(run, 1.0)
    return adj


def block_bootstrap_idx(n, rng, block=BOOT_BLOCK):
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


# ------------------------------------------------------------------- models
# Round-2 settings. With TUNE=1 (default) they are replaced by a grid search
# on a time-ordered validation slice (last 25% of the training block); the
# test block is never touched during selection.
RF_PARAMS = dict(n_estimators=150, max_depth=14, min_samples_split=4)
XGB_PARAMS = dict(n_estimators=150, max_depth=6, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8)
TUNE = os.environ.get("TUNE", "1") == "1"
if not TUNE:
    OUT = os.path.join(OUT, "untuned")
    FIG_DIR = OUT  # keep the paper figures in figures/ untouched
    os.makedirs(OUT, exist_ok=True)


def rf_model():
    return RandomForestRegressor(**RF_PARAMS, random_state=SEED, n_jobs=-1)


def xgb_model():
    return XGBRegressor(**XGB_PARAMS, random_state=SEED, n_jobs=-1)


def tune(tr):
    """Grid search on the last 25% of the training block, by validation MSE."""
    nv = int(len(tr) * 0.75)
    fit, val = tr.iloc[:nv], tr.iloc[nv:]
    rows = []
    for depth in (6, 10, 14, None):
        for leaf in (1, 5, 10):
            for mf in (1.0, 0.5, "sqrt"):
                p = dict(n_estimators=300, max_depth=depth, min_samples_leaf=leaf,
                         max_features=mf)
                m = RandomForestRegressor(**p, random_state=SEED, n_jobs=-1).fit(
                    fit[FULL], fit["target"])
                mse = np.mean((np.clip(m.predict(val[FULL]), 0, None) - val["target"]) ** 2)
                rows.append({"model": "RF", "params": repr(p), "val_RMSE": np.sqrt(mse)})
    for depth in (2, 3, 4, 6):
        for lr_ in (0.03, 0.05, 0.1):
            for n in (150, 300):
                p = dict(n_estimators=n, max_depth=depth, learning_rate=lr_,
                         subsample=0.8, colsample_bytree=0.8)
                m = XGBRegressor(**p, random_state=SEED, n_jobs=-1).fit(
                    fit[FULL], fit["target"])
                mse = np.mean((np.clip(m.predict(val[FULL]), 0, None) - val["target"]) ** 2)
                rows.append({"model": "XGB", "params": repr(p), "val_RMSE": np.sqrt(mse)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "hyperparameter_search.csv"), index=False)
    best = {k: eval(g.sort_values("val_RMSE")["params"].iloc[0])
            for k, g in res.groupby("model")}
    return best["RF"], best["XGB"]


def tune_leaked(trl, regime, cols=None):
    """Same RF grid for a leaked comparator, on the last 25% of its own
    training block (features cols, default LEAK); the test block is never
    touched."""
    cols = LEAK if cols is None else cols
    nv = int(len(trl) * 0.75)
    fit, val = trl.iloc[:nv], trl.iloc[nv:]
    rows = []
    for depth in (6, 10, 14, None):
        for leaf in (1, 5, 10):
            for mf in (1.0, 0.5, "sqrt"):
                p = dict(n_estimators=300, max_depth=depth, min_samples_leaf=leaf,
                         max_features=mf)
                m = RandomForestRegressor(**p, random_state=SEED, n_jobs=-1).fit(
                    fit[cols], fit["target"])
                mse = np.mean((np.clip(m.predict(val[cols]), 0, None) - val["target"]) ** 2)
                rows.append({"regime": regime, "params": repr(p), "val_RMSE": np.sqrt(mse)})
    res = pd.DataFrame(rows).sort_values("val_RMSE")
    return eval(res["params"].iloc[0]), res


def clim_lookup(clim, mods):
    """Time-of-day climatology; a minute-of-day absent from the training
    block (e.g. an 18:30 stamp that only clears the elevation filter late in
    the record) takes the nearest training minute-of-day."""
    idx = clim.index.values
    m = np.asarray(mods)
    near = idx[np.abs(idx[None, :] - m[:, None]).argmin(axis=1)]
    return clim.loc[near].values


def fit_causal(tr, te, clim):
    """Return dict of test predictions, plus fitted models."""
    y_tr = tr["target"].values
    preds, models = {}, {}
    preds["Persistence"] = te["ac_now"].values

    # Smart persistence: carry the clear-sky index of P_t forward; fall back
    # to plain persistence when the sun is too low for a stable ratio.
    ok = te["elev_t"].values > ELEV_MIN
    ratio = np.where(ok, te["ghi_cs_tau"].values / np.maximum(te["ghi_cs_t"].values, 1e-6), 1.0)
    preds["Smart persistence"] = te["ac_now"].values * ratio

    # CLIPER: optimal convex combination of persistence and time-of-day
    # climatology, alpha fitted by least squares on the training block.
    c_tr, c_te = clim_lookup(clim, tr["mod"]), clim_lookup(clim, te["mod"])
    a_num = np.sum((tr["ac_now"].values - c_tr) * (y_tr - c_tr))
    a_den = np.sum((tr["ac_now"].values - c_tr) ** 2)
    alpha = float(np.clip(a_num / a_den, 0, 1))
    preds["CLIPER"] = alpha * te["ac_now"].values + (1 - alpha) * c_te
    models["cliper_alpha"] = alpha

    lr = LinearRegression().fit(tr[FULL], y_tr)
    preds["Linear Regression"] = np.clip(lr.predict(te[FULL]), 0, None)
    xgb = xgb_model().fit(tr[FULL], y_tr)
    preds["XGBoost"] = np.clip(xgb.predict(te[FULL]), 0, None)
    rf = rf_model().fit(tr[FULL], y_tr)
    preds["Random Forest (full)"] = np.clip(rf.predict(te[FULL]), 0, None)
    rfr = rf_model().fit(tr[REDUCED], y_tr)
    preds["Random Forest (reduced)"] = np.clip(rfr.predict(te[REDUCED]), 0, None)
    models.update(lr=lr, xgb=xgb, rf=rf, rfr=rfr)
    return preds, models


def main():
    log = []

    def say(s=""):
        print(s)
        log.append(s)

    raw, df = load_grid()
    peak = float(raw["AC_POWER"][raw["N_INV"] == N_INVERTERS].max())
    say("Synchronized stamps (inner join): %d" % len(raw))
    say("Regular 15-min grid: %d stamps, %d missing"
        % (len(df), df["AC_POWER"].isna().sum()))
    say("Stamps with fewer than %d inverters: %d"
        % (N_INVERTERS, int((raw["N_INV"] < N_INVERTERS).sum())))
    say("Observed peak aggregate AC (all inverters reporting): %.1f kW" % peak)
    say("Outage stamps (an inverter at AC=0 with G>0.3): %d; excluded: %s"
        % (int(raw["OUTAGE"].sum()), EXCL_OUTAGE))
    dl = df[(solar_elevation(df.index) > ELEV_MIN)].dropna(subset=["AC_POWER", "IRRADIATION"])
    say("Same-time corr(AC, G) on daylight stamps (elev > 5): %.4f (n = %d)"
        % (np.corrcoef(dl["AC_POWER"], dl["IRRADIATION"])[0, 1], len(dl)))

    f = build_features(df)
    causal = f.dropna(subset=FULL + ["target"])
    say("Constructible causal rows (all hours): %d" % len(causal))

    day_masks = {
        "elev_tau>5": causal["elev_tau"] > ELEV_MIN,
        "target>0": causal["target"] > 0,
        "G_t>0": causal["IRRADIATION"] > 0,
    }
    for k, m in day_masks.items():
        say("  daylight rows, %-11s: %d" % (k, int(m.sum())))

    main_set = causal[day_masks["elev_tau>5"]]
    n_tr = int(len(main_set) * 0.80)
    cutoff = main_set.index[n_tr]
    tr, te = main_set.iloc[:n_tr], main_set.iloc[n_tr:]
    say("Main daylight set: %d rows, train %d, test %d; test starts %s"
        % (len(main_set), len(tr), len(te), cutoff))
    say("Test rows with target == 0: %d" % int((te["target"] == 0).sum()))

    global RF_PARAMS, XGB_PARAMS
    if TUNE:
        RF_PARAMS, XGB_PARAMS = tune(tr)
    say("RF params: %r" % RF_PARAMS)
    say("XGB params: %r" % XGB_PARAMS)

    clim = tr.groupby("mod")["target"].mean()
    preds, models = fit_causal(tr, te, clim)
    y = te["target"].values
    say("CLIPER alpha (train): %.4f" % models["cliper_alpha"])

    # ---------------------------------------------------------- Table I
    table = pd.DataFrame({k: scores(y, v, peak) for k, v in preds.items()}).T
    p_rmse = table.loc["Persistence", "RMSE"]
    table["skill_vs_pers"] = 1 - table["RMSE"] / p_rmse
    table["skill_vs_cliper"] = 1 - table["RMSE"] / table.loc["CLIPER", "RMSE"]

    # ------------------------------------------------- DM / Ljung-Box / Holm
    pairs = [("Persistence", m) for m in
             ["Linear Regression", "XGBoost", "Random Forest (full)",
              "Random Forest (reduced)", "CLIPER", "Smart persistence"]]
    pairs += [("Linear Regression", "Random Forest (full)"),
              ("Random Forest (full)", "Random Forest (reduced)"),
              ("CLIPER", "Linear Regression"),
              ("CLIPER", "XGBoost"),
              ("CLIPER", "Random Forest (full)"),
              ("CLIPER", "Random Forest (reduced)")]
    dm_rows = []
    for a, b in pairs:
        for loss in ("SE", "AE"):
            st, p = dm_test(y, preds[a], preds[b], loss)
            e1, e2 = y - preds[a], y - preds[b]
            d = e1 ** 2 - e2 ** 2 if loss == "SE" else np.abs(e1) - np.abs(e2)
            q, pq, acf = ljung_box(d, 10)
            st0, p0 = dm_test(y, preds[a], preds[b], loss, lags=0)
            dm_rows.append({"model_1": a, "model_2": b, "loss": loss,
                            "DM_NW5_HLN": st, "p_NW5_HLN": p,
                            "DM_rect_HLN": st0, "p_rect_HLN": p0,
                            "LjungBox_Q10": q, "LjungBox_p": pq,
                            "acf1_d": acf[0]})
    dm = pd.DataFrame(dm_rows)
    fam = dm["model_1"].eq("Persistence") & dm["model_2"].isin(
        ["Linear Regression", "XGBoost", "Random Forest (full)",
         "Random Forest (reduced)"])
    dm.loc[fam, "p_Holm_family8"] = holm(dm.loc[fam, "p_NW5_HLN"].values)
    famc = dm["model_1"].eq("CLIPER")
    dm.loc[famc, "p_Holm_cliper_family8"] = holm(dm.loc[famc, "p_NW5_HLN"].values)
    dm.to_csv(os.path.join(OUT, "dm_tests.csv"), index=False)

    for m in table.index:
        for loss in ("SE", "AE"):
            r = dm[(dm.model_1 == "Persistence") & (dm.model_2 == m) & (dm.loss == loss)]
            table.loc[m, "DM_p_%s_vs_pers" % loss] = r["p_NW5_HLN"].iloc[0] if len(r) else np.nan
    table.to_csv(os.path.join(OUT, "table1_causal_daylight.csv"))

    # ------------------------------------------------- block bootstrap CIs
    rng = np.random.default_rng(SEED)
    bpairs = [("Persistence", "Linear Regression"),
              ("Persistence", "Random Forest (full)"),
              ("Persistence", "XGBoost"),
              ("Linear Regression", "Random Forest (full)"),
              ("Random Forest (full)", "Random Forest (reduced)"),
              ("CLIPER", "Linear Regression")]
    E = {k: y - v for k, v in preds.items()}
    boot = {f"{a} -> {b}": {"dMSE": [], "dMAE": [], "dP95": [], "dThroughput_kWh": []}
            for a, b in bpairs}
    for _ in range(BOOT_N):
        idx = block_bootstrap_idx(len(y), rng)
        for a, b in bpairs:
            ea, eb = E[a][idx], E[b][idx]
            r = boot[f"{a} -> {b}"]
            r["dMSE"].append(np.mean(eb ** 2) - np.mean(ea ** 2))
            r["dMAE"].append(np.mean(np.abs(eb)) - np.mean(np.abs(ea)))
            r["dP95"].append(np.percentile(np.abs(eb), 95) - np.percentile(np.abs(ea), 95))
            r["dThroughput_kWh"].append(DT_H * (np.mean(np.abs(eb)) - np.mean(np.abs(ea))))
    brows = []
    for a, b in bpairs:
        ea, eb = E[a], E[b]
        point = {"dMSE": np.mean(eb ** 2) - np.mean(ea ** 2),
                 "dMAE": np.mean(np.abs(eb)) - np.mean(np.abs(ea)),
                 "dP95": np.percentile(np.abs(eb), 95) - np.percentile(np.abs(ea), 95),
                 "dThroughput_kWh": DT_H * (np.mean(np.abs(eb)) - np.mean(np.abs(ea)))}
        for k, v in boot[f"{a} -> {b}"].items():
            lo, hi = np.percentile(v, [2.5, 97.5])
            brows.append({"from": a, "to": b, "quantity": k, "point (to - from)": point[k],
                          "ci_lo": lo, "ci_hi": hi, "spans_zero": lo <= 0 <= hi})
    pd.DataFrame(brows).to_csv(os.path.join(OUT, "bootstrap_ci.csv"), index=False)

    # --------------------------------------------------- operational table
    ops = pd.DataFrame({
        "throughput_kWh_per_interval": table["MAE"] * DT_H,
        "R95_kW": table["P95"]})
    ops["thr_saving_vs_pers"] = ops.loc["Persistence", "throughput_kWh_per_interval"] - ops["throughput_kWh_per_interval"]
    ops["thr_saving_vs_LR"] = ops.loc["Linear Regression", "throughput_kWh_per_interval"] - ops["throughput_kWh_per_interval"]
    ops["R95_saving_vs_pers"] = ops.loc["Persistence", "R95_kW"] - ops["R95_kW"]
    ops["R95_saving_vs_LR"] = ops.loc["Linear Regression", "R95_kW"] - ops["R95_kW"]
    ops.to_csv(os.path.join(OUT, "operational.csv"))

    # -------------------------------------------------------- 2x2 leakage
    leak_all = f.dropna(subset=LEAK + LEAK_R2 + MATCHED + ["target"])
    rows2 = []
    sets = {}
    leak_search, leak_params = [], {}
    for regime, base_c, base_l in (
            ("all hours", causal, leak_all),
            ("daylight", causal[causal["elev_tau"] > ELEV_MIN],
             leak_all[leak_all["elev_tau"] > ELEV_MIN])):
        common = base_c.index.intersection(base_l.index)
        c, l_ = base_c.loc[common], base_l.loc[common]
        trc, tec = c[c.index < cutoff], c[c.index >= cutoff]
        trl, tel = l_[l_.index < cutoff], l_[l_.index >= cutoff]
        yt = tec["target"].values
        pc = np.clip(rf_model().fit(trc[FULL], trc["target"]).predict(tec[FULL]), 0, None)
        pl_shared = np.clip(rf_model().fit(trl[LEAK], trl["target"]).predict(tel[LEAK]), 0, None)
        if TUNE:
            lp, lres = tune_leaked(trl, regime)
            leak_search.append(lres)
        else:
            lp = RF_PARAMS
        leak_params[regime] = lp
        pl = np.clip(RandomForestRegressor(**lp, random_state=SEED, n_jobs=-1).fit(
            trl[LEAK], trl["target"]).predict(tel[LEAK]), 0, None)
        pl2 = np.clip(rf_model().fit(trl[LEAK_R2], trl["target"]).predict(tel[LEAK_R2]), 0, None)
        if TUNE:
            mp, mres = tune_leaked(trl, regime + " (matched)", MATCHED)
            leak_search.append(mres)
        else:
            mp = RF_PARAMS
        leak_params[regime + " (matched)"] = mp
        pm = np.clip(RandomForestRegressor(**mp, random_state=SEED, n_jobs=-1).fit(
            trl[MATCHED], trl["target"]).predict(tel[MATCHED]), 0, None)
        pp = tec["ac_now"].values
        for name, p in (("Matched leaked RF (target-time sensors)", pm),
                        ("Leaked RF nowcast (as described)", pl),
                        ("Leaked RF nowcast (forecaster settings)", pl_shared),
                        ("Leaked RF nowcast (round-2, rolling mean incl. target)", pl2),
                        ("Causal RF forecast", pc),
                        ("Causal persistence", pp)):
            s = scores(yt, p, peak)
            s.update(model=name, regime=regime, n_train=len(trc))
            rows2.append(s)
        sets[regime] = (yt, pl, pc)
    t2 = pd.DataFrame(rows2)[["regime", "model", "n_train", "N", "RMSE", "MAE", "R2", "MBE", "P95"]]
    t2.to_csv(os.path.join(OUT, "leakage_2x2.csv"), index=False)
    if leak_search:
        pd.concat(leak_search).to_csv(os.path.join(OUT, "hyperparameter_search_leaked.csv"), index=False)
    say("Leaked RF params: %r" % leak_params)

    def cell(regime, model, col):
        return float(t2[(t2.regime == regime) & (t2.model == model)][col].iloc[0])
    LK, CR = "Leaked RF nowcast (as described)", "Causal RF forecast"
    A, B = cell("all hours", LK, "R2"), cell("daylight", LK, "R2")
    C, D = cell("all hours", CR, "R2"), cell("daylight", CR, "R2")
    leak_share = ((B - D) + (A - C)) / 2
    zero_share = ((A - B) + (C - D)) / 2
    MK = "Matched leaked RF (target-time sensors)"
    Am, Bm = cell("all hours", MK, "R2"), cell("daylight", MK, "R2")
    decomp_matched = [
        {"quantity": "MATCHED R2 leaked all-hours (A)", "value": Am},
        {"quantity": "MATCHED R2 leaked daylight (B)", "value": Bm},
        {"quantity": "MATCHED R2 gap A-D", "value": Am - D},
        {"quantity": "MATCHED leakage component (Shapley avg of B-D, A-C)", "value": ((Bm - D) + (Am - C)) / 2},
        {"quantity": "MATCHED zero-inflation component (Shapley avg of A-B, C-D)", "value": ((Am - Bm) + (C - D)) / 2},
        {"quantity": "MATCHED RMSE ratio causal/leaked, daylight",
         "value": cell("daylight", CR, "RMSE") / cell("daylight", MK, "RMSE")},
        {"quantity": "MATCHED MAE ratio causal/leaked, daylight",
         "value": cell("daylight", CR, "MAE") / cell("daylight", MK, "MAE")},
        {"quantity": "MATCHED RMSE ratio causal/leaked, all hours",
         "value": cell("all hours", CR, "RMSE") / cell("all hours", MK, "RMSE")},
    ]
    decomp = pd.DataFrame(decomp_matched + [
        {"quantity": "R2 leaked all-hours (A)", "value": A},
        {"quantity": "R2 leaked daylight (B)", "value": B},
        {"quantity": "R2 causal all-hours (C)", "value": C},
        {"quantity": "R2 causal daylight (D)", "value": D},
        {"quantity": "R2 gap A-D", "value": A - D},
        {"quantity": "leakage component (Shapley avg of B-D, A-C)", "value": leak_share},
        {"quantity": "zero-inflation component (Shapley avg of A-B, C-D)", "value": zero_share},
        {"quantity": "RMSE ratio causal/leaked, daylight (common set)",
         "value": cell("daylight", CR, "RMSE") / cell("daylight", LK, "RMSE")},
        {"quantity": "MAE ratio causal/leaked, daylight (common set)",
         "value": cell("daylight", CR, "MAE") / cell("daylight", LK, "MAE")},
        {"quantity": "RMSE ratio causal-daylight / leaked-all-hours (old definition)",
         "value": cell("daylight", CR, "RMSE") / cell("all hours", LK, "RMSE")},
        {"quantity": "RMSE ratio causal/leaked (leaked RF with forecaster settings), daylight",
         "value": cell("daylight", CR, "RMSE") / cell("daylight", "Leaked RF nowcast (forecaster settings)", "RMSE")},
        {"quantity": "MAE ratio causal/leaked (leaked RF with forecaster settings), daylight",
         "value": cell("daylight", CR, "MAE") / cell("daylight", "Leaked RF nowcast (forecaster settings)", "MAE")},
        {"quantity": "RMSE ratio causal/leaked round-2 version, daylight",
         "value": cell("daylight", CR, "RMSE") / cell("daylight", "Leaked RF nowcast (round-2, rolling mean incl. target)", "RMSE")},
    ])
    decomp.to_csv(os.path.join(OUT, "leakage_decomposition.csv"), index=False)

    # ------------------------------------------------- daylight sensitivity
    sens = []
    for fname, m in day_masks.items():
        s_ = causal[m]
        k = int(len(s_) * 0.80)
        a_tr, a_te = s_.iloc[:k], s_.iloc[k:]
        pr, _ = fit_causal(a_tr, a_te, a_tr.groupby("mod")["target"].mean())
        yy = a_te["target"].values
        for mdl in ("Persistence", "Linear Regression", "Random Forest (full)", "XGBoost"):
            s = scores(yy, pr[mdl], peak)
            s.update(filter=fname, model=mdl, n_rows=len(s_), n_train=len(a_tr))
            for loss in ("SE", "AE"):
                s["DM_p_%s_vs_pers" % loss] = dm_test(yy, pr["Persistence"], pr[mdl], loss)[1] if mdl != "Persistence" else np.nan
            s["DM_p_SE_LR_vs_RF"] = dm_test(yy, pr["Linear Regression"], pr["Random Forest (full)"], "SE")[1]
            s["DM_p_AE_LR_vs_RF"] = dm_test(yy, pr["Linear Regression"], pr["Random Forest (full)"], "AE")[1]
            sens.append(s)
    # latitude sensitivity for the elevation filter
    for lat in (12.0, 28.0):
        fl = build_features(df, lat=lat).dropna(subset=FULL + ["target"])
        s_ = fl[fl["elev_tau"] > ELEV_MIN]
        k = int(len(s_) * 0.80)
        a_tr, a_te = s_.iloc[:k], s_.iloc[k:]
        pr, _ = fit_causal(a_tr, a_te, a_tr.groupby("mod")["target"].mean())
        yy = a_te["target"].values
        for mdl in ("Persistence", "Linear Regression", "Random Forest (full)", "XGBoost"):
            s = scores(yy, pr[mdl], peak)
            s.update(filter="elev_tau>5 @ lat %.0f" % lat, model=mdl, n_rows=len(s_), n_train=len(a_tr))
            for loss in ("SE", "AE"):
                s["DM_p_%s_vs_pers" % loss] = dm_test(yy, pr["Persistence"], pr[mdl], loss)[1] if mdl != "Persistence" else np.nan
            s["DM_p_SE_LR_vs_RF"] = dm_test(yy, pr["Linear Regression"], pr["Random Forest (full)"], "SE")[1]
            s["DM_p_AE_LR_vs_RF"] = dm_test(yy, pr["Linear Regression"], pr["Random Forest (full)"], "AE")[1]
            sens.append(s)
    # partial-inverter sensitivity: stamps with < 22 inverters set missing
    _, dfp = load_grid(drop_partial=True)
    fp = build_features(dfp).dropna(subset=FULL + ["target"])
    s_ = fp[fp["elev_tau"] > ELEV_MIN]
    k = int(len(s_) * 0.80)
    a_tr, a_te = s_.iloc[:k], s_.iloc[k:]
    pr, _ = fit_causal(a_tr, a_te, a_tr.groupby("mod")["target"].mean())
    yy = a_te["target"].values
    for mdl in ("Persistence", "Linear Regression", "Random Forest (full)", "XGBoost"):
        s = scores(yy, pr[mdl], peak)
        s.update(filter="elev_tau>5, all 22 inverters", model=mdl, n_rows=len(s_), n_train=len(a_tr))
        for loss in ("SE", "AE"):
            s["DM_p_%s_vs_pers" % loss] = dm_test(yy, pr["Persistence"], pr[mdl], loss)[1] if mdl != "Persistence" else np.nan
        s["DM_p_SE_LR_vs_RF"] = dm_test(yy, pr["Linear Regression"], pr["Random Forest (full)"], "SE")[1]
        s["DM_p_AE_LR_vs_RF"] = dm_test(yy, pr["Linear Regression"], pr["Random Forest (full)"], "AE")[1]
        sens.append(s)
    pd.DataFrame(sens).to_csv(os.path.join(OUT, "sensitivity_daylight_filter.csv"), index=False)

    # ----------------------------------------------------- MBE train vs test
    nv = int(len(tr) * 0.75)
    tr_a, va = tr.iloc[:nv], tr.iloc[nv:]
    pv, mv = fit_causal(tr_a, va, tr_a.groupby("mod")["target"].mean())
    mbe = []
    for mdl in ("Persistence", "Linear Regression", "XGBoost", "Random Forest (full)"):
        if mdl == "Persistence":
            in_s = float(np.mean(tr["ac_now"] - tr["target"]))
        else:
            mm = {"Linear Regression": "lr", "XGBoost": "xgb", "Random Forest (full)": "rf"}[mdl]
            in_s = float(np.mean(np.clip(models[mm].predict(tr[FULL]), 0, None) - tr["target"]))
        mbe.append({"model": mdl, "MBE_train_in_sample": in_s,
                    "MBE_validation_last25pct_of_train": float(np.mean(pv[mdl] - va["target"].values)),
                    "MBE_test": float(table.loc[mdl, "MBE"])})
    kt = lambda s_: float((s_["IRRADIATION"] / np.maximum(clear_sky_ghi(s_["elev_t"]) / 1000.0, 1e-3)).clip(0, 2).median())
    regime = pd.DataFrame([
        {"block": "train", "mean_target_kW": tr["target"].mean(), "mean_G_t": tr["IRRADIATION"].mean(),
         "median_clear_sky_index_G": kt(tr), "mean_abs_ramp_kW": tr["target"].sub(tr["ac_now"]).abs().mean()},
        {"block": "validation (last 25% of train)", "mean_target_kW": va["target"].mean(), "mean_G_t": va["IRRADIATION"].mean(),
         "median_clear_sky_index_G": kt(va), "mean_abs_ramp_kW": va["target"].sub(va["ac_now"]).abs().mean()},
        {"block": "test", "mean_target_kW": te["target"].mean(), "mean_G_t": te["IRRADIATION"].mean(),
         "median_clear_sky_index_G": kt(te), "mean_abs_ramp_kW": te["target"].sub(te["ac_now"]).abs().mean()},
    ])
    pd.DataFrame(mbe).to_csv(os.path.join(OUT, "mbe_train_val_test.csv"), index=False)
    regime.to_csv(os.path.join(OUT, "regime_train_vs_test.csv"), index=False)

    # ------------------------------------------- permutation importance (50)
    # Computed on the validation block only (forest refitted on the first 75%
    # of the training block), so no feature screening touches the test block.
    rf = mv["rf"]
    Xte = va[FULL].reset_index(drop=True)
    yv = va["target"].values
    base = np.mean((yv - rf.predict(Xte)) ** 2)
    prng = np.random.default_rng(SEED)
    pi = {c: [] for c in FULL}
    gpi = {"lag stack (P_t-1,P_t-2,P_t-4,P_t-96,r_t,dP_t)": [], "P_t": [], "G_t + dG_t": []}
    groups = {"lag stack (P_t-1,P_t-2,P_t-4,P_t-96,r_t,dP_t)": LAG_GROUP,
              "P_t": ["ac_now"], "G_t + dG_t": ["IRRADIATION", "d_irr_15m"]}
    for _ in range(PI_REPEATS):
        for c in FULL:
            Xp = Xte.copy()
            Xp[c] = prng.permutation(Xp[c].values)
            pi[c].append(np.mean((yv - rf.predict(Xp)) ** 2) - base)
        for g, cols in groups.items():
            Xp = Xte.copy()
            perm = prng.permutation(len(Xp))
            Xp[cols] = Xp[cols].values[perm]
            gpi[g].append(np.mean((yv - rf.predict(Xp)) ** 2) - base)
    pim = pd.DataFrame({"feature": FULL,
                        "mean_dMSE": [np.mean(pi[c]) for c in FULL],
                        "sd_dMSE": [np.std(pi[c], ddof=1) for c in FULL]})
    pim["se_dMSE"] = pim["sd_dMSE"] / np.sqrt(PI_REPEATS)
    pim["z_mean_over_sd"] = pim["mean_dMSE"] / pim["sd_dMSE"]
    pim["share_of_positive_%"] = 100 * pim["mean_dMSE"].clip(lower=0) / pim["mean_dMSE"].clip(lower=0).sum()
    pim = pim.sort_values("mean_dMSE", ascending=False)
    pim.to_csv(os.path.join(OUT, "permutation_importance_50.csv"), index=False)
    pd.DataFrame({"group": list(gpi), "mean_dMSE": [np.mean(v) for v in gpi.values()],
                  "sd_dMSE": [np.std(v, ddof=1) for v in gpi.values()]}).to_csv(
        os.path.join(OUT, "grouped_permutation_importance.csv"), index=False)

    # ------------------------------------------------ per-day test diagnostics
    tdf = te[["target"]].copy()
    for k in ("Persistence", "Linear Regression", "Random Forest (full)"):
        tdf[k] = preds[k]
    tdf["date"] = tdf.index.date
    perday = tdf.groupby("date").apply(lambda g: pd.Series(
        {"n": len(g), **{f"RMSE {k}": np.sqrt(np.mean((g[k] - g["target"]) ** 2))
                         for k in ("Persistence", "Linear Regression", "Random Forest (full)")}}))
    perday.to_csv(os.path.join(OUT, "per_day_test_rmse.csv"))
    # lag of peak cross-correlation between persistence forecast and actual,
    # computed within contiguous daylight blocks on the regular grid
    act = df["AC_POWER"]
    xc = []
    for lag in range(-3, 4):
        a = act.loc[te.index].values
        pidx = te.index + pd.Timedelta(minutes=15 * lag)
        p = te["ac_now"].reindex(pidx).values
        ok = ~np.isnan(p)
        xc.append({"lag_intervals": lag, "corr": float(np.corrcoef(a[ok], p[ok])[0, 1])})
    pd.DataFrame(xc).to_csv(os.path.join(OUT, "persistence_xcorr.csv"), index=False)

    # --------------------------------------------------------------- figures
    plot_residuals(y, preds)
    plot_importance(pim)

    # ---------------------------------------------------------------- report
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    say("\nTABLE I (causal, daylight elev>5, test N=%d)" % len(te))
    say(table.round(4).to_string())
    say("\nDM TESTS (NW5 + HLN)")
    say(dm.round(4).to_string())
    say("\nBOOTSTRAP CIs")
    say(pd.DataFrame(brows).round(2).to_string())
    say("\n2x2 LEAKAGE")
    say(t2.round(4).to_string())
    say(decomp.round(4).to_string())
    say("\nSENSITIVITY")
    say(pd.DataFrame(sens)[["filter", "model", "n_rows", "n_train", "N", "RMSE", "MAE", "R2", "MBE",
                            "DM_p_SE_vs_pers", "DM_p_AE_vs_pers", "DM_p_SE_LR_vs_RF", "DM_p_AE_LR_vs_RF"]].round(4).to_string())
    say("\nMBE")
    say(pd.DataFrame(mbe).round(1).to_string())
    say(regime.round(3).to_string())
    say("\nPERMUTATION IMPORTANCE (50)")
    say(pim.round(1).to_string())
    say(pd.read_csv(os.path.join(OUT, "grouped_permutation_importance.csv")).round(1).to_string())
    say("\nOPERATIONAL")
    say(ops.round(1).to_string())
    say("\nPER-DAY TEST RMSE")
    say(perday.round(0).to_string())
    say(pd.DataFrame(xc).round(4).to_string())
    with open(os.path.join(OUT, "run_log.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")


def plot_residuals(y, preds):
    """Test residual densities as step outlines with distinct dash patterns,
    so the overlapping curves stay distinguishable in greyscale."""
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7})
    fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=300)
    bins = np.histogram_bin_edges(np.concatenate([y - preds[k] for k in
                                  ("Persistence", "Linear Regression", "XGBoost",
                                   "Random Forest (full)")]), bins=45)
    for k, col, ls, lab in (("Persistence", "0.45", (0, (1, 1)), "Persist."),
                            ("Linear Regression", "crimson", "--", "LR"),
                            ("XGBoost", "darkorange", "-.", "XGB"),
                            ("Random Forest (full)", "teal", "-", "RF")):
        r = y - preds[k]
        ax.hist(r, bins=bins, density=True, histtype="step", lw=1.2, color=col, ls=ls,
                label="%s ($\\sigma$ = %.0f)" % (lab, np.std(r, ddof=1)))
    ax.axvline(0, color="black", ls="--", lw=0.6)
    ax.set_title("Residual Error Density", fontweight="bold")
    ax.set_xlabel("Residual [Actual $-$ Forecast] (kW)")
    ax.set_ylabel("Probability Density")
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, ls=":", alpha=0.6, lw=0.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG_DIR, "fig2_residual_distribution.png"), dpi=300)
    plt.close(fig)


def plot_importance(pim):
    """Fig. 2(b); labels follow the paper's notation (lags in 15-min steps)."""
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    SHORT = {"IRRADIATION": "$G_t$", "MODULE_TEMPERATURE": "$T^{m}_t$",
             "AMBIENT_TEMPERATURE": "$T^{a}_t$", "time_sin": "$s_t$",
             "time_cos": "$c_t$", "ac_now": "$P_t$", "lag_15m": "$P_{t-1}$",
             "lag_30m": "$P_{t-2}$", "lag_1h": "$P_{t-4}$",
             "lag_24h": "$P_{t-96}$", "roll_1h": "$r_t$",
             "d_ac_15m": "$\\Delta P_t$", "d_irr_15m": "$\\Delta G_t$"}
    asc = pim.sort_values("mean_dMSE")
    fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=300)
    vals = asc["mean_dMSE"].values / 1e6
    ax.barh(range(len(asc)), vals, xerr=asc["sd_dMSE"].values / 1e6,
            color=["indianred" if v < 0 else "teal" for v in vals],
            error_kw={"elinewidth": 0.7, "ecolor": "0.25"})
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(asc)))
    ax.set_yticklabels([SHORT[c] for c in asc["feature"]])
    ax.set_title("Permutation Importance", fontweight="bold")
    ax.set_xlabel("Increase in validation MSE ($10^{6}$ kW$^{2}$)")
    ax.grid(True, axis="x", ls=":", alpha=0.6, lw=0.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG_DIR, "fig3_feature_importance.png"), dpi=300)
    plt.close(fig)


def horizons():
    """Persistence, CLIPER, LR and RF at 15/30/60 min (h = 1, 2, 4), each on
    its own daylight set (elevation at the target > 5 deg) and 80/20 split.
    The RF is tuned per horizon on the last 25% of its training block."""
    raw, df = load_grid()
    peak = float(raw["AC_POWER"][raw["N_INV"] == N_INVERTERS].max())
    rows, search = [], []
    for h in (1, 2, 4):
        f = build_features(df, h=h).dropna(subset=FULL + ["target"])
        s_ = f[f["elev_tau"] > ELEV_MIN]
        k = int(len(s_) * 0.80)
        tr, te = s_.iloc[:k], s_.iloc[k:]
        y, y_tr = te["target"].values, tr["target"].values
        nv = int(len(tr) * 0.75)
        fit, val = tr.iloc[:nv], tr.iloc[nv:]
        best, best_rmse = None, np.inf
        for depth in (6, 10, 14, None):
            for leaf in (1, 5, 10):
                for mf in (1.0, 0.5, "sqrt"):
                    p = dict(n_estimators=300, max_depth=depth, min_samples_leaf=leaf,
                             max_features=mf)
                    m = RandomForestRegressor(**p, random_state=SEED, n_jobs=-1).fit(
                        fit[FULL], fit["target"])
                    r = np.sqrt(np.mean((np.clip(m.predict(val[FULL]), 0, None) - val["target"]) ** 2))
                    search.append({"h": h, "params": repr(p), "val_RMSE": r})
                    if r < best_rmse:
                        best, best_rmse = p, r
        clim = tr.groupby("mod")["target"].mean()
        c_tr, c_te = clim_lookup(clim, tr["mod"]), clim_lookup(clim, te["mod"])
        alpha = float(np.clip(np.sum((tr["ac_now"].values - c_tr) * (y_tr - c_tr))
                              / np.sum((tr["ac_now"].values - c_tr) ** 2), 0, 1))
        preds = {
            "Persistence": te["ac_now"].values,
            "CLIPER": alpha * te["ac_now"].values + (1 - alpha) * c_te,
            "Linear Regression": np.clip(LinearRegression().fit(tr[FULL], y_tr).predict(te[FULL]), 0, None),
            "Random Forest (full)": np.clip(RandomForestRegressor(**best, random_state=SEED, n_jobs=-1)
                                            .fit(tr[FULL], y_tr).predict(te[FULL]), 0, None),
        }
        r_p = np.sqrt(np.mean((preds["Persistence"] - y) ** 2))
        r_c = np.sqrt(np.mean((preds["CLIPER"] - y) ** 2))
        for name, p in preds.items():
            s = scores(y, p, peak)
            s.update(horizon_min=15 * h, model=name, n_train=len(tr), cliper_alpha=alpha,
                     rf_params=repr(best) if name.startswith("Random") else "",
                     skill_vs_pers=1 - s["RMSE"] / r_p, skill_vs_cliper=1 - s["RMSE"] / r_c)
            for ref in ("Persistence", "CLIPER"):
                for loss in ("SE", "AE"):
                    s["DM_p_%s_vs_%s" % (loss, ref)] = (
                        dm_test(y, preds[ref], p, loss, h=h)[1] if name != ref else np.nan)
            rows.append(s)
    out = pd.DataFrame(rows)
    # Holm within each horizon: the four learner-vs-CLIPER tests (LR, RF x
    # SE, AE) form one family, the four learner-vs-persistence tests another.
    for hm in out["horizon_min"].unique():
        lrn = (out["horizon_min"] == hm) & out["model"].isin(["Linear Regression", "Random Forest (full)"])
        for ref in ("CLIPER", "Persistence"):
            cols = ["DM_p_SE_vs_%s" % ref, "DM_p_AE_vs_%s" % ref]
            adj = holm(out.loc[lrn, cols].values.ravel()).reshape(-1, 2)
            for j, c in enumerate(cols):
                out.loc[lrn, c + "_Holm"] = adj[:, j]
    out.to_csv(os.path.join(OUT, "horizons.csv"), index=False)
    pd.DataFrame(search).to_csv(os.path.join(OUT, "hyperparameter_search_horizons.csv"), index=False)
    pd.set_option("display.width", 250)
    print(out[["horizon_min", "model", "N", "RMSE", "MAE", "R2", "MBE", "skill_vs_pers",
               "skill_vs_cliper", "DM_p_SE_vs_Persistence", "DM_p_AE_vs_Persistence",
               "DM_p_SE_vs_CLIPER", "DM_p_AE_vs_CLIPER", "DM_p_SE_vs_CLIPER_Holm", "DM_p_AE_vs_CLIPER_Holm"]].round(4).to_string())


if __name__ == "__main__":
    if os.environ.get("HORIZONS") == "1":
        horizons()
    else:
        main()
