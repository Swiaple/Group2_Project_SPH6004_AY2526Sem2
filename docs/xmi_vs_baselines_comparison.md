# XMI vs Baseline Comparison

- Generated at: 2026-04-15T13:21:13
- XMI: `D:\NUS\SPH6004\Group2_Project_SPH6004_AY2526Sem2\result\xmi_icu_gpu_full_20260415_1246\metrics_summary.csv`
- XGBoost baseline: `D:\NUS\SPH6004\Group2_Project_SPH6004_AY2526Sem2\result\xgboostmultiresult_gpu_full_20260415_1115\metrics_summary.csv`
- Logistic baseline: `D:\NUS\SPH6004\Group2_Project_SPH6004_AY2526Sem2\result\logisticmultiresult\metrics_summary.csv`

## Raw Metrics

| task | model | n_eval | auc | pr_auc | recall | precision |
|---|---:|---:|---:|---:|---:|---:|
| task1_discharge | xmi | 84657 | 0.864548 | 0.614878 | 0.870165 | 0.469120 |
| task1_discharge | xgboost | 83538 | 0.893805 | 0.692537 | 0.859918 | 0.541341 |
| task1_discharge | logistic | 83538 | 0.865171 | 0.619849 | 0.480726 | 0.657323 |
| task2_no_return72_masked | xmi | 19440 | 0.636496 | 0.967540 | 0.860657 | 0.954759 |
| task2_no_return72_masked | xgboost | 19560 | 0.635624 | 0.967357 | 0.842369 | 0.957882 |
| task2_no_return72_masked | logistic | 19560 | 0.537940 | 0.956660 | 0.993970 | 0.949547 |
| joint_both_positive | xmi | 84657 | 0.849719 | 0.565310 | 0.767201 | 0.460961 |
| joint_both_positive | xgboost | 83538 | 0.882093 | 0.630337 | 0.763069 | 0.536447 |
| joint_both_positive | logistic | 83538 | 0.862872 | 0.594941 | 0.481184 | 0.628994 |

## Delta (XMI - Baseline)

| task | metric | vs_xgboost | vs_logistic |
|---|---:|---:|---:|
| joint_both_positive | auc | -0.032374 | -0.013153 |
| joint_both_positive | pr_auc | -0.065027 | -0.029631 |
| joint_both_positive | recall | +0.004132 | +0.286016 |
| joint_both_positive | precision | -0.075486 | -0.168032 |
| task1_discharge | auc | -0.029257 | -0.000623 |
| task1_discharge | pr_auc | -0.077659 | -0.004971 |
| task1_discharge | recall | +0.010246 | +0.389439 |
| task1_discharge | precision | -0.072221 | -0.188203 |
| task2_no_return72_masked | auc | +0.000872 | +0.098556 |
| task2_no_return72_masked | pr_auc | +0.000183 | +0.010880 |
| task2_no_return72_masked | recall | +0.018288 | -0.133313 |
| task2_no_return72_masked | precision | -0.003123 | +0.005212 |

## Notes

- `n_eval` differs between XMI and baseline XGBoost/Logistic for task1/joint, so this is a practical comparison, not a strict same-sample A/B.
- For task2(masked), XMI and baseline XGBoost are very close on AUC/PR-AUC; XMI has higher recall and slightly lower precision.
- For task1/joint, XMI emphasizes recall while sacrificing precision compared with baseline XGBoost.