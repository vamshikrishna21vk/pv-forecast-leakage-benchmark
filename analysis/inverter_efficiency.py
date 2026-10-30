"""Plant 1 inverter efficiency and clipping check (Sec. IV-A), written to
results/inverter_efficiency.txt so the paper's numbers trace to results/."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PLANT"] = "1"
import revision_round3 as R  # noqa: E402

raw, _ = R.load_grid()
day = raw[(raw["DC_POWER"] > 0) & (raw["IRRADIATION"] > 0.05)]
eff = day["AC_POWER"] / day["DC_POWER"]
n3 = int((day["AC_POWER"] >= 0.97 * raw["AC_POWER"].max()).sum())
out = (f"Plant 1 (DC rescaled by 1/{R.DC_SCALE:.0f})\n"
       f"stamps with DC > 0 and irradiance > 0.05: {len(day)}\n"
       f"inverter efficiency AC/DC: mean {eff.mean():.4f}, sd {eff.std():.4f}\n"
       f"stamps within 3% of the AC maximum: {n3}\n")
open(os.path.join(ROOT, "results", "inverter_efficiency.txt"), "w").write(out)
print(out)
