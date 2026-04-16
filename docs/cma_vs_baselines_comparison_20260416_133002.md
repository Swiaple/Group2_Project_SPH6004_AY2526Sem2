# CMA-Surv (2GPU Unfrozen-BERT) vs Baselines

- Generated at: 2026-04-16 13:30:02
- CMA source: `D:/NUS/SPH6004/Group2_Project_SPH6004_AY2526Sem2/result/cma_surv_remote_20260416_132849_g2_unfreezebert_light/metrics_summary.csv`
- XGBoost baseline: `D:/NUS/SPH6004/Group2_Project_SPH6004_AY2526Sem2/result/xgboostmultiresult_gpu_full_20260415_1115/metrics_summary.csv`
- Logistic baseline: `D:/NUS/SPH6004/Group2_Project_SPH6004_AY2526Sem2/result/logisticmultiresult/metrics_summary.csv`

## Metric Delta Table

| task | metric | cma_2gpu_unfreezebert | xgboost | delta_vs_xgboost | logistic | delta_vs_logistic |
|---|---:|---:|---:|---:|---:|---:|
| task1_discharge | auc | 0.849703 | 0.893805 | -0.044102 | 0.865171 | -0.015469 |
| task1_discharge | pr_auc | 0.601271 | 0.692537 | -0.091266 | 0.619849 | -0.018578 |
| task1_discharge | recall | 0.478016 | 0.859918 | -0.381902 | 0.480726 | -0.002710 |
| task1_discharge | precision | 0.612713 | 0.541341 | +0.071372 | 0.657323 | -0.044610 |
| task2_no_return72_masked | auc | 0.652463 | 0.635624 | +0.016839 | 0.537940 | +0.114523 |
| task2_no_return72_masked | pr_auc | 0.970187 | 0.967357 | +0.002830 | 0.956660 | +0.013527 |
| task2_no_return72_masked | recall | 0.999300 | 0.842369 | +0.156931 | 0.993970 | +0.005330 |
| task2_no_return72_masked | precision | 0.949900 | 0.957882 | -0.007982 | 0.949547 | +0.000353 |
| joint_both_positive | auc | 0.849805 | 0.882093 | -0.032288 | 0.862872 | -0.013067 |
| joint_both_positive | pr_auc | 0.584882 | 0.630337 | -0.045454 | 0.594941 | -0.010058 |
| joint_both_positive | recall | 0.486891 | 0.763069 | -0.276178 | 0.481184 | +0.005707 |
| joint_both_positive | precision | 0.592661 | 0.536447 | +0.056214 | 0.628994 | -0.036333 |