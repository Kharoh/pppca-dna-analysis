# PPPCA-CGR Scaling Law Report

## Experimental setup
- Train CSV: `.\GUE_v2\GUE\prom\prom_core_tata\train.csv`
- Test CSV: `.\GUE_v2\GUE\prom\prom_core_tata\test.csv`
- Train sizes: [1000, 2000]
- Test size: 500
- Jmax list: [5, 10, 15, 20]
- Hidden dims list: [[16, 4]]
- Epochs: 80

## Summary table
|   train_size |   Jmax | mlp_label   | hidden_dims   |   param_count |   rf_accuracy |   rf_mcc |   nn_accuracy |   nn_mcc |
|-------------:|-------:|:------------|:--------------|--------------:|--------------:|---------:|--------------:|---------:|
|         1000 |      5 | 16-4        | [16, 4]       |           174 |         0.684 | 0.370365 |         0.714 | 0.432055 |
|         1000 |     10 | 16-4        | [16, 4]       |           254 |         0.692 | 0.386893 |         0.722 | 0.444014 |
|         1000 |     15 | 16-4        | [16, 4]       |           334 |         0.696 | 0.400395 |         0.706 | 0.415404 |
|         1000 |     20 | 16-4        | [16, 4]       |           414 |         0.702 | 0.411674 |         0.706 | 0.411951 |
|         2000 |      5 | 16-4        | [16, 4]       |           174 |         0.712 | 0.428764 |         0.702 | 0.407757 |
|         2000 |     10 | 16-4        | [16, 4]       |           254 |         0.702 | 0.410405 |         0.71  | 0.421615 |
|         2000 |     15 | 16-4        | [16, 4]       |           334 |         0.696 | 0.400395 |         0.706 | 0.413285 |
|         2000 |     20 | 16-4        | [16, 4]       |           414 |         0.702 | 0.411024 |         0.724 | 0.453688 |

## Figures
- accuracy_vs_train_size.png
- error_vs_train_size_loglog.png
- accuracy_vs_params.png

## Notes
Accuracy and MCC are reported for each model-size/train-size combination. The log-log plot provides a scaling-law diagnostic for generalization error $1-\text{accuracy}$.
