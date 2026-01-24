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
- Random Forest accuracy: 0.6420
- Random Forest MCC: 0.2863
- Neural network accuracy: 0.6600
- Neural network MCC: 0.3271

## Eigenvalue table
|   component |   eigenvalue |   explained_variance_ratio |
|------------:|-------------:|---------------------------:|
|           1 |   23.2204    |                 0.709839   |
|           2 |    4.65294   |                 0.142238   |
|           3 |    2.78677   |                 0.0851905  |
|           4 |    0.650821  |                 0.0198953  |
|           5 |    0.54809   |                 0.0167549  |
|           6 |    0.308513  |                 0.00943111 |
|           7 |    0.169508  |                 0.00518179 |
|           8 |    0.166043  |                 0.00507587 |
|           9 |    0.124809  |                 0.00381537 |
|          10 |    0.0843278 |                 0.00257786 |

## Model component usage
|   component |   rf_usage |   nn_usage |
|------------:|-----------:|-----------:|
|           1 |  0.05688   |  0.0581244 |
|           2 |  0.0725853 |  0.0614868 |
|           3 |  0.241225  |  0.101383  |
|           4 |  0.143553  |  0.14678   |
|           5 |  0.144245  |  0.104366  |
|           6 |  0.0621132 |  0.0917764 |
|           7 |  0.0963683 |  0.137531  |
|           8 |  0.0655482 |  0.0758778 |
|           9 |  0.0657803 |  0.0837141 |
|          10 |  0.051703  |  0.138961  |

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
