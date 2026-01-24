Running PPPCA on the promoter task of the GUE dataset (Genome Understanding Evaluation).

The task consists in fitting the PCA on train sequences (unsupervised) and then projecting test sequences into the learned PCA space.

The scores of the train sequences are used to train simple classifiers (e.g. Random Forest, Small MLP) on the reduced feature space.

The performance of these classifiers is then evaluated on the test sequences projected into the same PCA space.

## Interpretability analysis (CGR eigenfunctions)

Use `interpretability_analysis.py` to generate publication-ready CGR eigenfunction plots, eigenvalue scree plots, hot-region maps, and a markdown report. Hot regions are computed by weighting PPPCA eigenfunctions by model usage: Random Forest feature importances and MLP effective input weights (absolute Linear-layer weight products). The resulting map highlights CGR regions that most influence each classifier.

### Example

```powershell
python interpretability_analysis.py --train-path .\GUE_v2\GUE\prom\prom_core_tata\train.csv --test-path .\GUE_v2\GUE\prom\prom_core_tata\test.csv --n-train 1000 --n-test 500 --Jmax 10
```

Outputs are written to `analysis_results/interpretability/` by default, including:

- `cgr_eigenfunctions.png`
- `eigenvalue_scree.png`
- `rf_hot_regions.png`
- `nn_hot_regions.png`
- `cgr_point_process_scatter.png`
- `cgr_synthetic_promoter_patterns.png`
- `interpretability_report.md`

## CapsNet vs PPPCA comparison

Use `capsnet_comparison.py` to train a Capsule Network on FCGR images and compare it to PPPCA + Random Forest/MLP baselines. The script produces a markdown report, FCGR saliency maps, CapsNet embedding plots, and PPPCA hot-region maps.

### Example

```powershell
python capsnet_comparison.py --train-path .\GUE_v2\GUE\prom\prom_core_tata\train.csv --test-path .\GUE_v2\GUE\prom\prom_core_tata\test.csv --n-train 1000 --n-test 500 --Jmax 10 --caps-epochs 50
```

```bash
python capsnet_comparison.py --train-path ./GUE_v2/GUE/prom/prom_core_tata/train.csv --test-path ./GUE_v2/GUE/prom/prom_core_tata/test.csv --n-train 1000 --n-test 500 --Jmax 10 --caps-epochs 50
```

Outputs are written to `analysis_results/capsnet_comparison/` by default, including:

- `capsnet_comparison_report.md`
- `capsnet_embedding_pca.png`
- `capsnet_saliency_class_*.png`
- `pppca_rf_hot_regions.png`
- `pppca_mlp_hot_regions.png`

## Scaling-law experiments

Use `scaling_law_runner.py` to run the pipeline across multiple training set sizes and MLP sizes. The script saves a CSV summary and scaling-law plots.

### Example

```powershell
python scaling_law_runner.py --train-path .\GUE_v2\GUE\prom\prom_core_tata\train.csv --test-path .\GUE_v2\GUE\prom\prom_core_tata\test.csv --train-sizes 250,500,1000 --mlp-sizes 16-4;64-32;128-64 --Jmax-list 6,10,14
```

Outputs are written to `analysis_results/scaling_laws/` by default, including:

- `scaling_summary.csv`
- `accuracy_vs_train_size.png`
- `error_vs_train_size_loglog.png`
- `scaling_law_report.md`