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
- Random Forest MCC: 0.3614
- Neural network accuracy: 0.7020
- Neural network MCC: 0.4043

## Eigenvalue table
|   component |   eigenvalue |   explained_variance_ratio |
|------------:|-------------:|---------------------------:|
|           1 |   33.3984    |                 0.784039   |
|           2 |    4.56167   |                 0.107087   |
|           3 |    2.72827   |                 0.064047   |
|           4 |    0.57497   |                 0.0134976  |
|           5 |    0.507628  |                 0.0119167  |
|           6 |    0.311944  |                 0.00732298 |
|           7 |    0.160129  |                 0.00375907 |
|           8 |    0.153251  |                 0.00359762 |
|           9 |    0.121252  |                 0.00284642 |
|          10 |    0.0803776 |                 0.00188689 |

## Model component usage
|   component |   rf_usage |   nn_usage |
|------------:|-----------:|-----------:|
|           1 |  0.152545  |  0.0656695 |
|           2 |  0.301371  |  0.107185  |
|           3 |  0.0770093 |  0.0895773 |
|           4 |  0.156306  |  0.148095  |
|           5 |  0.0748444 |  0.0970913 |
|           6 |  0.0461686 |  0.0893458 |
|           7 |  0.0445269 |  0.0926672 |
|           8 |  0.0558307 |  0.0949233 |
|           9 |  0.0482994 |  0.0939508 |
|          10 |  0.0430978 |  0.121495  |

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
