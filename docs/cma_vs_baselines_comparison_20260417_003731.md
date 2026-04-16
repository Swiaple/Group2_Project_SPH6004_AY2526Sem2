# CMA-Surv vs Baselines

- Generated at: 2026-04-17T00:37:31
- CMA metrics: `D:\NUS\SPH6004\Group2_Project_SPH6004_AY2526Sem2\result\cma_surv_remote_20260417_003552_g2_legacy_fixbce\metrics_summary.csv`
- XGBoost metrics: `result/xgboostmultiresult_gpu_full_20260415_1115/metrics_summary.csv`
- Logistic metrics: `result/logisticmultiresult/metrics_summary.csv`

## Raw Metrics

| model | task | auc | pr_auc | recall | precision |
| --- | --- | --- | --- | --- | --- |
| cma_surv | task1_discharge | 0.892794 | 0.690976 | 0.620501 | 0.674953 |
| cma_surv | task2_no_return72_masked | 0.643824 | 0.969713 | 1.000000 | 0.949642 |
| cma_surv | joint_both_positive | 0.890302 | 0.667280 | 0.624872 | 0.645479 |
| xgboost | task1_discharge | 0.893805 | 0.692537 | 0.859918 | 0.541341 |
| xgboost | task2_no_return72_masked | 0.635624 | 0.967357 | 0.842369 | 0.957882 |
| xgboost | joint_both_positive | 0.882093 | 0.630337 | 0.763069 | 0.536447 |
| logistic | task1_discharge | 0.865171 | 0.619849 | 0.480726 | 0.657323 |
| logistic | task2_no_return72_masked | 0.537940 | 0.956660 | 0.993970 | 0.949547 |
| logistic | joint_both_positive | 0.862872 | 0.594941 | 0.481184 | 0.628994 |

## Delta (CMA-Surv - Baseline)

| task | metric | cma_surv | xgboost | logistic | delta_vs_xgboost | delta_vs_logistic |
| --- | --- | --- | --- | --- | --- | --- |
| task1_discharge | auc | 0.892794 | 0.893805 | 0.865171 | -0.001011 | 0.027623 |
| task1_discharge | pr_auc | 0.690976 | 0.692537 | 0.619849 | -0.001561 | 0.071127 |
| task1_discharge | recall | 0.620501 | 0.859918 | 0.480726 | -0.239417 | 0.139775 |
| task1_discharge | precision | 0.674953 | 0.541341 | 0.657323 | 0.133612 | 0.017630 |
| task2_no_return72_masked | auc | 0.643824 | 0.635624 | 0.537940 | 0.008201 | 0.105885 |
| task2_no_return72_masked | pr_auc | 0.969713 | 0.967357 | 0.956660 | 0.002356 | 0.013053 |
| task2_no_return72_masked | recall | 1.000000 | 0.842369 | 0.993970 | 0.157631 | 0.006030 |
| task2_no_return72_masked | precision | 0.949642 | 0.957882 | 0.949547 | -0.008240 | 0.000095 |
| joint_both_positive | auc | 0.890302 | 0.882093 | 0.862872 | 0.008209 | 0.027430 |
| joint_both_positive | pr_auc | 0.667280 | 0.630337 | 0.594941 | 0.036944 | 0.072340 |
| joint_both_positive | recall | 0.624872 | 0.763069 | 0.481184 | -0.138197 | 0.143688 |
| joint_both_positive | precision | 0.645479 | 0.536447 | 0.628994 | 0.109032 | 0.016485 |
