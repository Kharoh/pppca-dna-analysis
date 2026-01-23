# PPPCA-CGR Scaling Law Report

## Experimental setup
- Train CSV: `./GUE_v2/GUE/prom/prom_core_tata/train.csv`
- Test CSV: `./GUE_v2/GUE/prom/prom_core_tata/test.csv`
- Train sizes: [250, 500, 1000]
- Test size: 500
- Jmax: 10
- Hidden dims list: [[16, 4], [64, 32], [128, 64]]
- Epochs: 80

## Summary table
|   train_size | mlp_label   | hidden_dims   |   param_count |   rf_accuracy |   rf_mcc |   nn_accuracy |   nn_mcc |
|-------------:|:------------|:--------------|--------------:|--------------:|---------:|--------------:|---------:|
|          250 | 16-4        | [16, 4]       |           254 |         0.678 | 0.356004 |         0.702 | 0.403987 |
|          250 | 64-32       | [64, 32]      |          2850 |         0.678 | 0.356004 |         0.67  | 0.34092  |
|          250 | 128-64      | [128, 64]     |          9794 |         0.678 | 0.356004 |         0.672 | 0.34647  |
|          500 | 16-4        | [16, 4]       |           254 |         0.68  | 0.361389 |         0.712 | 0.424231 |
|          500 | 64-32       | [64, 32]      |          2850 |         0.68  | 0.361389 |         0.684 | 0.370724 |
|          500 | 128-64      | [128, 64]     |          9794 |         0.68  | 0.361389 |         0.71  | 0.425524 |
|         1000 | 16-4        | [16, 4]       |           254 |         0.692 | 0.386893 |         0.722 | 0.444014 |
|         1000 | 64-32       | [64, 32]      |          2850 |         0.692 | 0.386893 |         0.694 | 0.389426 |
|         1000 | 128-64      | [128, 64]     |          9794 |         0.692 | 0.386893 |         0.692 | 0.387727 |

## Figures
- accuracy_vs_train_size.png
- error_vs_train_size_loglog.png
- accuracy_vs_params.png

## Notes
Accuracy and MCC are reported for each model-size/train-size combination. The log-log plot provides a scaling-law diagnostic for generalization error $1-\text{accuracy}$.
