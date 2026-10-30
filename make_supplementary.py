"""Rebuild supplementary/table_S{1,2,3}_*.csv from results/ (selection and
rounding only; no values are computed here).
Usage: python make_supplementary.py"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
SUP = os.path.join(ROOT, "supplementary")
os.makedirs(SUP, exist_ok=True)


def both(rel1, rel2):
    """Plant 1 and Plant 2 versions of one result file, with a plant column."""
    parts = []
    for plant, rel in ((1, rel1), (2, rel2)):
        d = pd.read_csv(os.path.join(RES, rel))
        d.insert(0, "plant", plant)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


s1 = both("horizons.csv", os.path.join("plant2", "horizons.csv"))
s1 = s1[["plant", "horizon_min", "model", "N", "RMSE", "MAE", "MBE", "R2",
         "skill_vs_pers", "skill_vs_cliper", "DM_p_SE_vs_Persistence",
         "DM_p_AE_vs_Persistence", "DM_p_SE_vs_CLIPER", "DM_p_AE_vs_CLIPER",
         "DM_p_SE_vs_CLIPER_Holm", "DM_p_AE_vs_CLIPER_Holm"]].round(5)
s1.to_csv(os.path.join(SUP, "table_S1_horizons_15_30_60.csv"), index=False)

s2 = both(os.path.join("robustness", "plant1", "weather_only_leak.csv"),
          os.path.join("robustness", "plant2", "weather_only_leak.csv"))
s2 = s2.drop(columns=["params"]).round(4)
s2.to_csv(os.path.join(SUP, "table_S2_weather_only_comparator.csv"), index=False)

s3 = both("mbe_train_val_test.csv", os.path.join("plant2", "mbe_train_val_test.csv")).round(1)
s3.to_csv(os.path.join(SUP, "table_S3_mbe_train_val_test.csv"), index=False)
print("wrote", len(s1), len(s2), len(s3), "rows to supplementary/")
