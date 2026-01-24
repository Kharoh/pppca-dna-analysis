# CapsNet vs PPPCA Baseline Report

## Experimental setup
- Train CSV: `.\GUE_v2\GUE\prom\prom_core_tata\train.csv`
- Test CSV: `.\GUE_v2\GUE\prom\prom_core_tata\test.csv`
- Train size: 100
- Test size: 100
- PPPCA Jmax: 10
- PPPCA kernel: linear
- MLP hidden dims: [16, 4]
- CapsNet grid resolution: 64
- CapsNet epochs: 5
- CapsNet batch size: 32
- CapsNet routing iterations: 3

## Performance summary
| model       |   accuracy |      mcc |
|:------------|-----------:|---------:|
| CapsNet     |       0.51 | 0        |
| PPPCA + RF  |       0.58 | 0.15954  |
| PPPCA + MLP |       0.64 | 0.281764 |

## Capsule Network classification report
|              |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| 0            |      0      |     0    |   0        |     49    |
| 1            |      0.51   |     1    |   0.675497 |     51    |
| accuracy     |      0.51   |     0.51 |   0.51     |      0.51 |
| macro avg    |      0.255  |     0.5  |   0.337748 |    100    |
| weighted avg |      0.2601 |     0.51 |   0.344503 |    100    |

## PPPCA + RF/MLP classification reports
### Random Forest

|              |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| 0            |    0.589744 | 0.469388 |   0.522727 |     49    |
| 1            |    0.57377  | 0.686275 |   0.625    |     51    |
| accuracy     |    0.58     | 0.58     |   0.58     |      0.58 |
| macro avg    |    0.581757 | 0.577831 |   0.573864 |    100    |
| weighted avg |    0.581597 | 0.58     |   0.574886 |    100    |

### MLP

|              |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| 0            |    0.622642 | 0.673469 |   0.647059 |     49    |
| 1            |    0.659574 | 0.607843 |   0.632653 |     51    |
| accuracy     |    0.64     | 0.64     |   0.64     |      0.64 |
| macro avg    |    0.641108 | 0.640656 |   0.639856 |    100    |
| weighted avg |    0.641477 | 0.64     |   0.639712 |    100    |

## Figures
- pppca_scores_projection.png
- pppca_confusion_matrices.png
- pppca_mlp_training_history.png
- cgr_eigenfunctions.png
- eigenvalue_scree.png
- pppca_rf_hot_regions.png
- pppca_mlp_hot_regions.png
- capsnet_training_history.png
- capsnet_confusion_matrix.png
- capsnet_embedding_pca.png
- capsnet_fcgr_mean_class_0.png
- capsnet_fcgr_mean_class_1.png
- capsnet_saliency_class_1.png
- capsnet_saliency_class_0.png

## Notes
CapsNet embeddings are projected to 2D using PCA on the digit capsule vectors. Saliency maps are computed from gradients of the predicted capsule length with respect to the FCGR input. PPPCA hot-region maps weigh eigenfunctions by model usage (RF feature importances and MLP effective weights).
