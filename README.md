# Information Leakage and Benchmark Diagnosis in 15-Minute-Ahead Photovoltaic Power Forecasting

Code and results for the paper of the same title by Vamshi Krishna G, Aishwarya M F Prabhakar and Dushyanth S.

The code builds causal 15-minute-ahead forecasts for two Indian PV plants and compares them with leaked (target-time) formulations. The reference forecasts are persistence, smart persistence and CLIPER, and the learners are Linear Regression, XGBoost and Random Forest. Each learner is compared with the references using Diebold–Mariano tests with Holm correction. The comparisons are run on the original single split and on a four-fold rolling-origin evaluation.

`results/` holds the outputs behind every number in the paper. `supplementary/` holds Tables S1–S3.

## 1. Get the data

The dataset is *Solar Power Generation Data* by Ani Kannal on Kaggle:
<https://www.kaggle.com/datasets/anikannal/solar-power-generation-data>

It is not included in this repository. Download it in one of two ways:

- **Browser:** sign in to Kaggle, click **Download** on the dataset page, and unzip the archive.
- **Kaggle CLI:** this needs an API token in `~/.kaggle/kaggle.json`.

  ```
  pip install kaggle
  kaggle datasets download -d anikannal/solar-power-generation-data -p data --unzip
  ```

Put the four CSV files in `data/` at the repository root:

```
data/Plant_1_Generation_Data.csv
data/Plant_1_Weather_Sensor_Data.csv
data/Plant_2_Generation_Data.csv
data/Plant_2_Weather_Sensor_Data.csv
```

`data/` is listed in `.gitignore`.

## 2. Install

The results were produced with Python 3.12.3 and the package versions in `requirements.txt`.

```
python -m venv venv
venv/Scripts/activate        # Windows (Git Bash); on Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

## 3. Reproduce

Run every command from the repository root, in the order given. Commands use POSIX shell syntax (`VAR=value command`). In PowerShell, set the variable first, for example `$env:PLANT = "1"; python revision_round3.py`.

Run times were measured on a 2-core Intel Core i3-10110U laptop with 8 GB RAM.

**Step 1. Single split, both plants (Table I, Table II, Figs. 2–3, Table S3)**

```
PLANT=1 python revision_round3.py                     # ~6 min
PLANT=2 python revision_round3.py                     # ~9 min
```

**Step 2. Longer horizons (Table S1; 30- and 60-minute values in the text)**

```
PLANT=1 HORIZONS=1 python revision_round3.py          # ~2.5 min
PLANT=2 HORIZONS=1 python revision_round3.py          # ~2 min
```

**Step 3. Rolling-origin evaluation (Table III)**

`leakage_check.txt` is appended to, so delete it first when re-running in place.

```
rm -f results/rolling_origin/leakage_check.txt results/rolling_origin_F1F2/leakage_check.txt
PLANT=1 python analysis/rolling_origin_v2.py                  # ~3 min
PLANT=2 python analysis/rolling_origin_v2.py                  # ~3 min
PLANT=1 RO_FOLDS=F1,F2 python analysis/rolling_origin_v2.py   # ~1.5 min (two earliest folds)
PLANT=2 RO_FOLDS=F1,F2 python analysis/rolling_origin_v2.py   # ~1.5 min
```

**Step 4. Robustness analyses (Table S2, DM bandwidths, day-block bootstrap, Holm families)**

These scripts read the tuned settings from step 1 and write to `analysis/out/`. Copy their outputs into `results/robustness/`.

```
PLANT=1 python analysis/day_dependence.py             # ~2.5 min
PLANT=2 python analysis/day_dependence.py             # ~2 min
PLANT=1 python analysis/dayblock_persistence.py       # <10 s
PLANT=2 python analysis/dayblock_persistence.py       # <10 s
mkdir -p results/robustness/plant1 results/robustness/plant2
cp analysis/out/plant1/*.csv results/robustness/plant1/
cp analysis/out/plant2/*.csv results/robustness/plant2/
{ python analysis/holm6.py; python analysis/holm6_bw.py; } > results/robustness/holm6.txt   # <5 s
```

**Step 5. Remaining numbers, Fig. 1 and the supplementary tables**

```
python analysis/inverter_efficiency.py                # <10 s
python eval_revision_round2.py                        # <10 s
python generate_fig1_timeline.py                      # <5 s  (needs step 4)
python make_supplementary.py                          # <5 s  (needs steps 1, 2 and 4)
```

**Optional robustness runs (not used in the paper's tables)**

```
PLANT=2 EXCL_OUTAGE=1 python revision_round3.py       # ~5.5 min; Plant 2 without outage periods
python lon_estimate.py                                # seconds; longitude estimate
PLANT=1 python analysis/rolling_origin.py             # superseded first rolling-origin version (not timed)
```

Total: about 33 minutes for steps 1–5.

## 4. Where each table and figure comes from

| Paper item | Produced by | File(s) |
|---|---|---|
| Table I | step 1 | `results/table1_causal_daylight.csv`, `results/dm_tests.csv`, and the `plant2/` copies; bold Holm values in `results/robustness/holm6.txt` (step 4) |
| Table II | step 1 | `results/leakage_2x2.csv`, `results/plant2/leakage_2x2.csv`; decomposition in `leakage_decomposition.csv` and `holm6.txt` |
| Table III | step 3 | `results/rolling_origin/plant{1,2}/pooled.csv`, `pooled_dm_holm.csv`, `folds.csv` |
| Folds 1–2 results (text) | step 3 | `results/rolling_origin_F1F2/plant1/` |
| Fig. 1 | step 5 | `figures/fig1_timeline.png` (annotations from `results/robustness/plant1/weather_only_leak.csv`) |
| Fig. 2 | step 1 | `figures/fig2_residual_distribution.png` |
| Fig. 3 | step 1 | `figures/fig3_feature_importance.png` |
| Table S1 | steps 2, 5 | `supplementary/table_S1_horizons_15_30_60.csv` from `results/horizons.csv` |
| Table S2 | steps 4, 5 | `supplementary/table_S2_weather_only_comparator.csv` from `results/robustness/plant*/weather_only_leak.csv` |
| Table S3 | steps 1, 5 | `supplementary/table_S3_mbe_train_val_test.csv` from `results/mbe_train_val_test.csv` |
| Inverter efficiency (Sec. IV-A) | step 5 | `results/inverter_efficiency.txt` |
| Early leaked R² = 0.997 (Sec. I) | step 5 | `results/revision_metrics_round2.txt` |

## 5. Determinism

Every random component is seeded with 42: the forests, XGBoost, the bootstraps and permutation importance. Results are deterministic under seed 42 at the precision reported. The last floating-point digits (around 10⁻¹² relative) can vary between runs, because the forests and XGBoost train on several threads (`n_jobs=-1`).

To check this, a fresh copy was set up from scratch with the four CSVs in `data/` and an environment installed from `requirements.txt`. `results/` was deleted and every step above was run. Tables I and III, and every other numeric result, matched the files in this repository to within 10⁻¹² relative. Figs. 1–3 and Tables S1–S3 were byte-identical. A few hyperparameter labels differ only where two settings are equivalent (e.g. `max_depth` 14 vs None when no tree reaches depth 14); they give the same predictions.

## 6. Script overview

- `revision_round3.py` — the main pipeline. It does feature construction on a regular 15-minute grid, a solar-elevation daylight filter, tuning on the last 25 % of training, the references and learners, DM/HLN tests, bootstrap CIs, the 2×2 leakage table, permutation importance, and Figs. 2–3.
  - Environment switches:
    - `PLANT=1|2`
    - `HORIZONS=1` for the 15/30/60-minute runs
    - `EXCL_OUTAGE=1` to exclude Plant 2 outage periods
- `analysis/rolling_origin_v2.py` — the rolling-origin evaluation, with four expanding-window folds and per-fold re-tuning. It also writes a per-fold leakage check. `RO_FOLDS=F1,F2` restricts the run to the two earliest folds.
- `analysis/day_dependence.py` — day-block bootstrap, alternative DM bandwidths, and the weather-only leaked comparator.
- `analysis/dayblock_persistence.py` — day-block bootstrap against persistence.
- `analysis/holm6.py`, `analysis/holm6_bw.py` — Holm adjustment over each plant's six learner-vs-reference tests, and the R² decomposition.
- `analysis/inverter_efficiency.py` — the Plant 1 inverter-efficiency and clipping check.
- `generate_fig1_timeline.py` — Fig. 1.
- `eval_revision_round2.py` — the earlier leaked pipeline whose R² = 0.997 is quoted in the introduction.
- `make_supplementary.py` — selects and rounds Tables S1–S3 from `results/`.
- `lon_estimate.py` — estimates longitude from irradiance onset and offset (used for the solar-elevation filter).
- `analysis/rolling_origin.py` — first version of the rolling-origin run. It is superseded by `rolling_origin_v2.py` and kept because its outputs are in `results/robustness/`.
- `analysis/regen_fig2.py` — redraws Fig. 2 from the tuned settings without rerunning the pipeline.

Earlier exploratory pipelines are not included. The scripts above produce every number and figure in the paper.

## License

MIT; see `LICENSE`.
