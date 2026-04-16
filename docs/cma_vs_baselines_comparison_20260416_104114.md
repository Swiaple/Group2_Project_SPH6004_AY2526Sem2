# CMA-Surv (2GPU freeze-BERT) vs Baselines

- Generated at: 2026-04-16 10:41:14
- CMA 2GPU metrics source: `D:/NUS/SPH6004/Group2_Project_SPH6004_AY2526Sem2/result/cma_surv_remote_20260416_104012/metrics_summary.csv`
- XGBoost baseline source: `D:/NUS/SPH6004/Group2_Project_SPH6004_AY2526Sem2/result/xgboostmultiresult_gpu_full_20260415_1115/metrics_summary.csv`
- Logistic baseline source: `D:/NUS/SPH6004/Group2_Project_SPH6004_AY2526Sem2/result/logisticmultiresult/metrics_summary.csv`
- 4GPU (unfrozen-BERT) run status: failed before metrics export (DDP unused-parameter reduction error, parameter indices 209/210).

## Metric Delta Table

| task | metric | cma_2gpu_freezebert | xgboost | delta_vs_xgboost | logistic | delta_vs_logistic |
|---|---:|---:|---:|---:|---:|---:|
| task1_discharge | auc | 0.893626 | 0.893805 | -0.000179 | 0.865171 | +0.028454 |
| task1_discharge | pr_auc | 0.691188 | 0.692537 | -0.001348 | 0.619849 | +0.071339 |
| task1_discharge | recall | 0.636605 | 0.859918 | -0.223313 | 0.480726 | +0.155879 |
| task1_discharge | precision | 0.666060 | 0.541341 | +0.124720 | 0.657323 | +0.008738 |
| task2_no_return72_masked | auc | 0.644435 | 0.635624 | +0.008811 | 0.537940 | +0.106495 |
| task2_no_return72_masked | pr_auc | 0.968848 | 0.967357 | +0.001491 | 0.956660 | +0.012189 |
| task2_no_return72_masked | recall | 0.999785 | 0.842369 | +0.157416 | 0.993970 | +0.005814 |
| task2_no_return72_masked | precision | 0.949826 | 0.957882 | -0.008056 | 0.949547 | +0.000279 |
| joint_both_positive | auc | 0.891430 | 0.882093 | +0.009336 | 0.862872 | +0.028557 |
| joint_both_positive | pr_auc | 0.667559 | 0.630337 | +0.037223 | 0.594941 | +0.072619 |
| joint_both_positive | recall | 0.640915 | 0.763069 | -0.122153 | 0.481184 | +0.159731 |
| joint_both_positive | precision | 0.636801 | 0.536447 | +0.100354 | 0.628994 | +0.007808 |

## Quick Read

### task1_discharge
- auc: vs XGBoost -0.000179; vs Logistic +0.028454
- pr_auc: vs XGBoost -0.001348; vs Logistic +0.071339
- recall: vs XGBoost -0.223313; vs Logistic +0.155879
- precision: vs XGBoost +0.124720; vs Logistic +0.008738
### task2_no_return72_masked
- auc: vs XGBoost +0.008811; vs Logistic +0.106495
- pr_auc: vs XGBoost +0.001491; vs Logistic +0.012189
- recall: vs XGBoost +0.157416; vs Logistic +0.005814
- precision: vs XGBoost -0.008056; vs Logistic +0.000279
### joint_both_positive
- auc: vs XGBoost +0.009336; vs Logistic +0.028557
- pr_auc: vs XGBoost +0.037223; vs Logistic +0.072619
- recall: vs XGBoost -0.122153; vs Logistic +0.159731
- precision: vs XGBoost +0.100354; vs Logistic +0.007808