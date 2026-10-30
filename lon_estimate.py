"""Longitude estimate from the daily irradiance onset/offset midpoint
(timestamps taken as IST). Same method as used for Plant 1 in round 3."""
import sys
import numpy as np
import pandas as pd

for k in (1, 2):
    w = pd.read_csv(f"data/Plant_{k}_Weather_Sensor_Data.csv")
    w["DATE_TIME"] = pd.to_datetime(w["DATE_TIME"])
    w["date"] = w.DATE_TIME.dt.date
    w["h"] = w.DATE_TIME.dt.hour + w.DATE_TIME.dt.minute / 60
    rows = []
    for d, gq in w.groupby("date"):
        gq = gq[gq.IRRADIATION > 0]
        if len(gq) < 30:
            continue
        on = gq[gq.IRRADIATION > 0.01]
        g = 2 * np.pi / 365 * (pd.Timestamp(d).dayofyear - 1)
        eot = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                        - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
        rows.append(((on.h.min() + on.h.max()) / 2, eot))
    r = pd.DataFrame(rows, columns=["mid", "eot"])
    lon = 82.5 - (r.mid * 60 - 720 + r.eot) / 4
    print("Plant %d: days %d, longitude median %.2f, IQR %.2f-%.2f"
          % (k, len(r), lon.median(), lon.quantile(.25), lon.quantile(.75)))
