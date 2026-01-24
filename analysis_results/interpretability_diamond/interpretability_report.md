# PPPCA-CGR Interpretability Report

## Experimental setup
- Train CSV: `.\GUE_v2\GUE\prom\prom_core_tata\train.csv`
- Test CSV: `.\GUE_v2\GUE\prom\prom_core_tata\test.csv`
- Train size: 500
- Test size: 500
- Jmax: 10
- MLP hidden dims: [16, 4]
- Epochs: 100
- Batch size: 32
- Learning rate: 0.001

## Performance summary
- Random Forest accuracy: 0.6800
- Random Forest MCC: 0.3620
- Neural network accuracy: 0.6980
- Neural network MCC: 0.3961

## Eigenvalue table
|   component |   eigenvalue |   explained_variance_ratio |
|------------:|-------------:|---------------------------:|
|           1 |    29.7281   |                 0.753046   |
|           2 |     4.96772  |                 0.125838   |
|           3 |     1.70093  |                 0.0430866  |
|           4 |     1.17517  |                 0.0297683  |
|           5 |     0.588524 |                 0.014908   |
|           6 |     0.440725 |                 0.011164   |
|           7 |     0.389752 |                 0.00987286 |
|           8 |     0.197853 |                 0.00501184 |
|           9 |     0.165567 |                 0.00419399 |
|          10 |     0.12279  |                 0.0031104  |

## Model component usage
|   component |   rf_usage |   nn_usage |
|------------:|-----------:|-----------:|
|           1 |  0.253383  |  0.0778982 |
|           2 |  0.142564  |  0.0749318 |
|           3 |  0.140492  |  0.125191  |
|           4 |  0.062397  |  0.106634  |
|           5 |  0.0767531 |  0.0924761 |
|           6 |  0.0530756 |  0.0607498 |
|           7 |  0.0510162 |  0.109589  |
|           8 |  0.0724493 |  0.124322  |
|           9 |  0.084409  |  0.104665  |
|          10 |  0.0634614 |  0.123544  |

## Figures
- cgr_eigenfunctions.png
- eigenvalue_scree.png
- rf_hot_regions.png
- nn_hot_regions.png
- synthetic_single_promoter_cgr.png
- synthetic_repeated_promoter_cgr.png
- synthetic_noisy_promoters.png
- synthetic_noise_only.png
- synthetic_promoter_vs_nonpromoter.png
- real_promoter_individuals.png
- real_nonpromoter_individuals.png
- real_promoter_vs_nonpromoter.png
- interpretability_pca_scores_projection.png
- interpretability_confusion_matrices.png
- interpretability_nn_training_history.png

## Notes
The eigenfunctions are evaluated on a uniform CGR grid, producing a spatial map of the learned PPPCA components. The scree plot summarizes the energy distribution across components and provides an empirical variance explanation profile.
