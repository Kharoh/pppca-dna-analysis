# CapsNet vs PPPCA Baseline Report

## Experimental setup
- Train CSV: `./GUE_v2/GUE/prom/prom_core_tata/train.csv`
- Test CSV: `./GUE_v2/GUE/prom/prom_core_tata/test.csv`
- Train size: 2000
- Test size: 500
- PPPCA Jmax: 10
- PPPCA kernel: linear
- MLP hidden dims: [16, 4]
- CapsNet grid resolution: 64
- CapsNet epochs: 10
- CapsNet batch size: 32
- CapsNet routing iterations: 3

## Performance summary
| model       |   accuracy |      mcc |
|:------------|-----------:|---------:|
| CapsNet     |      0.702 | 0.403987 |
| PPPCA + RF  |      0.672 | 0.346139 |
| PPPCA + MLP |      0.684 | 0.369182 |

## Capsule Network classification report
|              |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| 0            |    0.705394 | 0.685484 |   0.695297 |   248     |
| 1            |    0.698842 | 0.718254 |   0.708415 |   252     |
| accuracy     |    0.702    | 0.702    |   0.702    |     0.702 |
| macro avg    |    0.702118 | 0.701869 |   0.701856 |   500     |
| weighted avg |    0.702092 | 0.702    |   0.701908 |   500     |

## PPPCA + RF/MLP classification reports
### Random Forest

|              |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| 0            |    0.694444 | 0.604839 |   0.646552 |   248     |
| 1            |    0.65493  | 0.738095 |   0.69403  |   252     |
| accuracy     |    0.672    | 0.672    |   0.672    |     0.672 |
| macro avg    |    0.674687 | 0.671467 |   0.670291 |   500     |
| weighted avg |    0.674529 | 0.672    |   0.670481 |   500     |

### MLP

|              |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| 0            |    0.700893 | 0.633065 |   0.665254 |   248     |
| 1            |    0.67029  | 0.734127 |   0.700758 |   252     |
| accuracy     |    0.684    | 0.684    |   0.684    |     0.684 |
| macro avg    |    0.685591 | 0.683596 |   0.683006 |   500     |
| weighted avg |    0.685469 | 0.684    |   0.683148 |   500     |

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
