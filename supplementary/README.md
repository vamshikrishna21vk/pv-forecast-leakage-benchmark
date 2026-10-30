# Supplementary tables

This is a single-split evaluation (original chronological 80/20 split; test from 11 June). The values are copied from `results/` without editing. The CSV versions are in this folder.

## Table S1. Causal benchmarks at 15, 30 and 60 minutes

- Columns: RMSE, MAE and MBE in kW; skill = 1 − RMSE/RMSE_ref.
- p-values: DM, Newey–West with 5 lags plus the HLN factor.
- Holm correction is applied within each horizon over the four tests of LR and RF against CLIPER.
- Source: `results/horizons.csv` and `results/plant2/horizons.csv`.

| plant | horizon_min | model | N | RMSE | MAE | MBE | R2 | skill_vs_pers | skill_vs_cliper | DM_p_SE_vs_CLIPER | DM_p_AE_vs_CLIPER | DM_p_SE_vs_CLIPER_Holm | DM_p_AE_vs_CLIPER_Holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 15 | Persistence | 304 | 3092.2463 | 2243.8461 | 61.686 | 0.8234 | 0.0 | -0.0745 | 0.0087 | 0.0067 | nan | nan |
| 1 | 15 | CLIPER | 304 | 2877.7201 | 2060.8187 | 435.6238 | 0.847 | 0.0694 | 0.0 | nan | nan | nan | nan |
| 1 | 15 | Linear Regression | 304 | 2798.0644 | 1994.0887 | 341.0803 | 0.8554 | 0.0951 | 0.0277 | 0.1182 | 0.1908 | 0.2365 | 0.2365 |
| 1 | 15 | Random Forest (full) | 304 | 2790.7872 | 1911.8598 | 382.2388 | 0.8561 | 0.0975 | 0.0302 | 0.0544 | 0.0044 | 0.1631 | 0.0178 |
| 1 | 30 | Persistence | 302 | 4210.2227 | 3272.6648 | 123.3871 | 0.6729 | 0.0 | -0.1907 | 0.0 | 0.0 | nan | nan |
| 1 | 30 | CLIPER | 302 | 3535.8293 | 2647.5998 | 667.8952 | 0.7693 | 0.1602 | 0.0 | nan | nan | nan | nan |
| 1 | 30 | Linear Regression | 302 | 3334.5189 | 2460.6151 | 473.9112 | 0.7948 | 0.208 | 0.0569 | 0.0062 | 0.0106 | 0.0185 | 0.0213 |
| 1 | 30 | Random Forest (full) | 302 | 3398.0488 | 2387.3097 | 653.9085 | 0.787 | 0.1929 | 0.039 | 0.0268 | 0.0003 | 0.0268 | 0.0014 |
| 1 | 60 | Persistence | 299 | 5470.4202 | 4524.5618 | 167.714 | 0.4553 | 0.0 | -0.426 | 0.0 | 0.0 | nan | nan |
| 1 | 60 | CLIPER | 299 | 3836.1972 | 2853.8897 | 876.0719 | 0.7322 | 0.2987 | 0.0 | nan | nan | nan | nan |
| 1 | 60 | Linear Regression | 299 | 3622.0138 | 2696.6321 | 723.2137 | 0.7612 | 0.3379 | 0.0558 | 0.0134 | 0.0742 | 0.0534 | 0.0742 |
| 1 | 60 | Random Forest (full) | 299 | 3653.4086 | 2655.5944 | 860.5814 | 0.7571 | 0.3322 | 0.0476 | 0.0156 | 0.024 | 0.0534 | 0.0534 |
| 2 | 15 | Persistence | 319 | 3176.0658 | 2285.8792 | 12.1949 | 0.5712 | 0.0 | -0.1169 | 0.0 | 0.0 | nan | nan |
| 2 | 15 | CLIPER | 319 | 2843.753 | 2070.49 | 392.699 | 0.6562 | 0.1046 | 0.0 | nan | nan | nan | nan |
| 2 | 15 | Linear Regression | 319 | 2778.5941 | 2023.3454 | 27.884 | 0.6718 | 0.1251 | 0.0229 | 0.0835 | 0.2548 | 0.2504 | 0.3867 |
| 2 | 15 | Random Forest (full) | 319 | 2787.1269 | 1946.3125 | 224.697 | 0.6698 | 0.1225 | 0.0199 | 0.1933 | 0.0018 | 0.3867 | 0.0072 |
| 2 | 30 | Persistence | 319 | 3950.4187 | 2942.8687 | -18.0278 | 0.3366 | 0.0 | -0.2388 | 0.0 | 0.0 | nan | nan |
| 2 | 30 | CLIPER | 319 | 3188.8387 | 2326.3221 | 685.7261 | 0.5677 | 0.1928 | 0.0 | nan | nan | nan | nan |
| 2 | 30 | Linear Regression | 319 | 3126.8742 | 2264.7926 | 34.8471 | 0.5843 | 0.2085 | 0.0194 | 0.3634 | 0.4562 | 0.7267 | 0.7267 |
| 2 | 30 | Random Forest (full) | 319 | 3024.4676 | 2124.6011 | 454.7775 | 0.6111 | 0.2344 | 0.0515 | 0.007 | 0.0003 | 0.0211 | 0.0013 |
| 2 | 60 | Persistence | 320 | 4550.952 | 3638.544 | -67.3106 | 0.1172 | 0.0 | -0.3985 | 0.0 | 0.0 | nan | nan |
| 2 | 60 | CLIPER | 320 | 3254.2584 | 2382.6917 | 972.4907 | 0.5486 | 0.2849 | 0.0 | nan | nan | nan | nan |
| 2 | 60 | Linear Regression | 320 | 3234.0386 | 2344.8479 | 309.317 | 0.5542 | 0.2894 | 0.0062 | 0.8493 | 0.7072 | 1.0 | 1.0 |
| 2 | 60 | Random Forest (full) | 320 | 3116.9415 | 2291.0708 | 781.3581 | 0.5859 | 0.3151 | 0.0422 | 0.0389 | 0.1446 | 0.1554 | 0.4337 |

## Table S2. Weather-only leaked comparator

- Target: P at τ.
- Inputs: G, T^m, T^a (optionally s, c), read at τ, with no power history.
- Model: Random Forest tuned on the last 25% of training.
- Comparison: the causal 13-feature Random Forest on the same rows.
- Source: `results/robustness/plant*/weather_only_leak.csv`.

| plant | regime | model | n_train | N | RMSE | MAE | R2 | RMSE_ratio_causal_over_this | MAE_ratio_causal_over_this |
|---|---|---|---|---|---|---|---|---|---|
| 1 | all hours | Causal RF forecast (13 features, P_{t+1} from I_t) | 2301 | 608 | 1979.7005 | 962.8471 | 0.9383 | 1.0 | 1.0 |
| 1 | all hours | Weather-only leaked RF (G, Tm, Ta at target) | 2301 | 608 | 507.1534 | 250.179 | 0.996 | 3.9036 | 3.8486 |
| 1 | all hours | Weather-only leaked RF (G, Tm, Ta, s, c at target) | 2301 | 608 | 483.3346 | 233.7538 | 0.9963 | 4.0959 | 4.1191 |
| 1 | daylight | Causal RF forecast (13 features, P_{t+1} from I_t) | 1214 | 304 | 2790.7872 | 1911.8598 | 0.8561 | 1.0 | 1.0 |
| 1 | daylight | Weather-only leaked RF (G, Tm, Ta at target) | 1214 | 304 | 718.0037 | 500.4303 | 0.9905 | 3.8869 | 3.8204 |
| 1 | daylight | Weather-only leaked RF (G, Tm, Ta, s, c at target) | 1214 | 304 | 676.0463 | 456.199 | 0.9916 | 4.1281 | 4.1908 |
| 2 | all hours | Causal RF forecast (13 features, P_{t+1} from I_t) | 2515 | 621 | 1982.1071 | 991.63 | 0.8712 | 1.0 | 1.0 |
| 2 | all hours | Weather-only leaked RF (G, Tm, Ta at target) | 2515 | 621 | 1760.3447 | 771.1906 | 0.8984 | 1.126 | 1.2858 |
| 2 | all hours | Weather-only leaked RF (G, Tm, Ta, s, c at target) | 2515 | 621 | 1735.0888 | 746.3028 | 0.9013 | 1.1424 | 1.3287 |
| 2 | daylight | Causal RF forecast (13 features, P_{t+1} from I_t) | 1275 | 319 | 2787.1269 | 1946.3125 | 0.6698 | 1.0 | 1.0 |
| 2 | daylight | Weather-only leaked RF (G, Tm, Ta at target) | 1275 | 319 | 2549.4322 | 1612.3888 | 0.7237 | 1.0932 | 1.2071 |
| 2 | daylight | Weather-only leaked RF (G, Tm, Ta, s, c at target) | 1275 | 319 | 2393.1074 | 1425.4171 | 0.7565 | 1.1646 | 1.3654 |

## Table S3. Mean bias error (kW)

MBE on the training block (in sample), on the validation slice (last 25% of training, models refitted on the first 75%) and on the test block. Source: `results/mbe_train_val_test.csv` and `results/plant2/mbe_train_val_test.csv`.

| plant | model | MBE_train_in_sample | MBE_validation_last25pct_of_train | MBE_test |
|---|---|---|---|---|
| 1 | Persistence | -11.1 | 30.2 | 61.7 |
| 1 | Linear Regression | 19.9 | 169.7 | 341.1 |
| 1 | XGBoost | -0.6 | 180.6 | 338.0 |
| 1 | Random Forest (full) | -3.7 | 174.1 | 382.2 |
| 2 | Persistence | -25.5 | -45.1 | 12.2 |
| 2 | Linear Regression | 5.0 | -44.8 | 27.9 |
| 2 | XGBoost | 10.1 | 10.1 | 97.0 |
| 2 | Random Forest (full) | -2.8 | 79.9 | 224.7 |
