# PPPCA-CGR Interpretability Report

## Experimental setup
- Train CSV: `.\GUE_v2\GUE\prom\prom_core_tata\train.csv`
- Test CSV: `.\GUE_v2\GUE\prom\prom_core_tata\test.csv`
- Train size: 1000
- Test size: 500
- Jmax: 20
- MLP hidden dims: [16, 4]
- Epochs: 100
- Batch size: 32
- Learning rate: 0.001

## Performance summary
- Random Forest accuracy: 0.6940
- Random Forest MCC: 0.4014
- Neural network accuracy: 0.7020
- Neural network MCC: 0.4041

## Eigenvalue table
|   component |   eigenvalue |   explained_variance_ratio |
|------------:|-------------:|---------------------------:|
|           1 |   30.9302    |                0.754147    |
|           2 |    4.79141   |                0.116825    |
|           3 |    1.769     |                0.0431322   |
|           4 |    1.18731   |                0.0289493   |
|           5 |    0.571072  |                0.013924    |
|           6 |    0.422105  |                0.0102919   |
|           7 |    0.392338  |                0.00956608  |
|           8 |    0.200403  |                0.00488627  |
|           9 |    0.162224  |                0.0039554   |
|          10 |    0.117211  |                0.00285787  |
|          11 |    0.106821  |                0.00260454  |
|          12 |    0.0655971 |                0.00159941  |
|          13 |    0.0469    |                0.00114353  |
|          14 |    0.0455335 |                0.00111021  |
|          15 |    0.0420069 |                0.00102422  |
|          16 |    0.0395415 |                0.000964111 |
|          17 |    0.0350533 |                0.000854678 |
|          18 |    0.0320124 |                0.000780535 |
|          19 |    0.0290455 |                0.000708194 |
|          20 |    0.0277034 |                0.00067547  |

## Model component usage
|   component |   rf_usage |   nn_usage |
|------------:|-----------:|-----------:|
|           1 |  0.236611  |  0.0233367 |
|           2 |  0.0734779 |  0.0172853 |
|           3 |  0.0796549 |  0.0368329 |
|           4 |  0.0773372 |  0.0483171 |
|           5 |  0.0383352 |  0.0234386 |
|           6 |  0.040697  |  0.0470131 |
|           7 |  0.0183602 |  0.0432616 |
|           8 |  0.046451  |  0.0403117 |
|           9 |  0.0541397 |  0.05627   |
|          10 |  0.0430412 |  0.0573178 |
|          11 |  0.0277731 |  0.0419657 |
|          12 |  0.0447886 |  0.0395891 |
|          13 |  0.0407633 |  0.0743771 |
|          14 |  0.0298226 |  0.0536761 |
|          15 |  0.0215741 |  0.0686648 |
|          16 |  0.0267336 |  0.0616969 |
|          17 |  0.0253145 |  0.0891406 |
|          18 |  0.0209359 |  0.0612539 |
|          19 |  0.031665  |  0.0744307 |
|          20 |  0.022524  |  0.0418203 |

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
