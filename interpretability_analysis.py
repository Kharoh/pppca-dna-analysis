import argparse
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from main import run_dna_analysis_pipeline


@dataclass
class InterpretabilityConfig:
    train_path: str
    test_path: str
    n_train: int
    n_test: int
    Jmax: int
    kernel: str
    output_dir: str
    random_seed: int
    nn_hidden_dims: List[int]
    nn_epochs: int
    nn_batch_size: int
    nn_learning_rate: float
    num_eigenfunctions: int
    grid_size: int


def plot_cgr_eigenfunctions(pca_model, output_dir: str, num_eigenfunctions: int = 4, grid_size: int = 120) -> str:
    """
    Plot PPPCA eigenfunctions for 2D CGR representations.

    Returns
    -------
    str
        Path to the saved figure.
    """
    eigenfun = pca_model.pca_results["eigenfun"]
    eigenvals = np.array(pca_model.pca_results["eigenval"])

    num_eigenfunctions = min(num_eigenfunctions, len(eigenvals))
    grid = np.linspace(0.0, 1.0, grid_size)
    X, Y = np.meshgrid(grid, grid)
    X_flat = np.stack([X.ravel(), Y.ravel()], axis=1)

    eta_vals = eigenfun(X_flat)  # (grid_size^2, Jmax)

    n_cols = min(3, num_eigenfunctions)
    n_rows = int(np.ceil(num_eigenfunctions / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for idx in range(num_eigenfunctions):
        ax = axes[idx]
        values = eta_vals[:, idx].reshape(grid_size, grid_size)
        cf = ax.contourf(X, Y, values, levels=30, cmap="RdBu")
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Eigenfunction {idx + 1} (λ={eigenvals[idx]:.3e})")
        ax.set_xlabel("CGR x")
        ax.set_ylabel("CGR y")

    # Hide any unused axes
    for ax in axes[num_eigenfunctions:]:
        ax.axis("off")

    fig.suptitle("PPPCA Eigenfunctions in CGR Space", fontsize=14)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "cgr_eigenfunctions.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_eigenvalue_scree(eigenvalues: np.ndarray, output_dir: str) -> str:
    ratios = eigenvalues / np.sum(eigenvalues)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, marker="o")
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].set_title("PPPCA Eigenvalue Spectrum")
    axes[0].grid(alpha=0.3)

    axes[1].bar(np.arange(1, len(ratios) + 1), ratios)
    axes[1].set_xlabel("Component")
    axes[1].set_ylabel("Explained variance ratio")
    axes[1].set_title("Explained Variance")
    axes[1].set_ylim(0, max(ratios) * 1.1)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "eigenvalue_scree.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_interpretability_report(
    config: InterpretabilityConfig,
    metrics: dict,
    eigenvalues: np.ndarray,
    figure_paths: List[str],
    output_dir: str,
) -> str:
    report_path = os.path.join(output_dir, "interpretability_report.md")
    ratios = eigenvalues / np.sum(eigenvalues)
    table = pd.DataFrame({
        "component": np.arange(1, len(eigenvalues) + 1),
        "eigenvalue": eigenvalues,
        "explained_variance_ratio": ratios,
    })

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# PPPCA-CGR Interpretability Report\n\n")
        handle.write("## Experimental setup\n")
        handle.write(f"- Train CSV: `{config.train_path}`\n")
        handle.write(f"- Test CSV: `{config.test_path}`\n")
        handle.write(f"- Train size: {config.n_train}\n")
        handle.write(f"- Test size: {config.n_test}\n")
        handle.write(f"- Jmax: {config.Jmax}\n")
        handle.write(f"- MLP hidden dims: {config.nn_hidden_dims}\n")
        handle.write(f"- Epochs: {config.nn_epochs}\n")
        handle.write(f"- Batch size: {config.nn_batch_size}\n")
        handle.write(f"- Learning rate: {config.nn_learning_rate}\n")
        handle.write("\n")

        handle.write("## Performance summary\n")
        handle.write(f"- Random Forest accuracy: {metrics['rf_accuracy']:.4f}\n")
        handle.write(f"- Random Forest MCC: {metrics['rf_mcc']:.4f}\n")
        if metrics.get("nn_accuracy") is not None:
            handle.write(f"- Neural network accuracy: {metrics['nn_accuracy']:.4f}\n")
            handle.write(f"- Neural network MCC: {metrics['nn_mcc']:.4f}\n")
        handle.write("\n")

        handle.write("## Eigenvalue table\n")
        handle.write(table.to_markdown(index=False))
        handle.write("\n\n")

        handle.write("## Figures\n")
        for path in figure_paths:
            rel_path = os.path.relpath(path, output_dir)
            handle.write(f"- {rel_path}\n")
        handle.write("\n")

        handle.write("## Notes\n")
        handle.write("The eigenfunctions are evaluated on a uniform CGR grid, producing a spatial map of the learned PPPCA components. The scree plot summarizes the energy distribution across components and provides an empirical variance explanation profile.\n")

    return report_path


def run_interpretability_analysis(config: InterpretabilityConfig) -> str:
    os.makedirs(config.output_dir, exist_ok=True)

    results = run_dna_analysis_pipeline(
        train_path=config.train_path,
        test_path=config.test_path,
        n_train=config.n_train,
        n_test=config.n_test,
        Jmax=config.Jmax,
        kernel=config.kernel,
        output_dir=config.output_dir,
        random_seed=config.random_seed,
        use_nn=True,
        nn_hidden_dims=config.nn_hidden_dims,
        nn_epochs=config.nn_epochs,
        nn_batch_size=config.nn_batch_size,
        nn_learning_rate=config.nn_learning_rate,
        make_plots=True,
        plot_prefix="interpretability",
    )

    eigenvalues = np.array(results["eigenvalues"])
    eigen_plot = plot_cgr_eigenfunctions(
        results["pca_model"],
        output_dir=config.output_dir,
        num_eigenfunctions=config.num_eigenfunctions,
        grid_size=config.grid_size,
    )
    scree_plot = plot_eigenvalue_scree(eigenvalues, config.output_dir)

    report_path = write_interpretability_report(
        config,
        metrics=results,
        eigenvalues=eigenvalues,
        figure_paths=[
            eigen_plot,
            scree_plot,
            os.path.join(config.output_dir, "interpretability_pca_scores_projection.png"),
            os.path.join(config.output_dir, "interpretability_confusion_matrices.png"),
            os.path.join(config.output_dir, "interpretability_nn_training_history.png"),
        ],
        output_dir=config.output_dir,
    )

    return report_path


def parse_hidden_dims(value: str) -> List[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PPPCA-CGR interpretability analysis")
    parser.add_argument("--train-path", required=True, help="Path to training CSV")
    parser.add_argument("--test-path", required=True, help="Path to testing CSV")
    parser.add_argument("--n-train", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--n-test", type=int, default=500, help="Number of test samples")
    parser.add_argument("--Jmax", type=int, default=10, help="Number of PPPCA components")
    parser.add_argument("--kernel", default="linear", help="Kernel type for PCA")
    parser.add_argument("--output-dir", default="analysis_results/interpretability", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hidden-dims", default="16,4", help="Comma-separated hidden layer sizes")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-eigenfunctions", type=int, default=6, help="Number of eigenfunctions to plot")
    parser.add_argument("--grid-size", type=int, default=120, help="Grid resolution for eigenfunction plots")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = InterpretabilityConfig(
        train_path=args.train_path,
        test_path=args.test_path,
        n_train=args.n_train,
        n_test=args.n_test,
        Jmax=args.Jmax,
        kernel=args.kernel,
        output_dir=args.output_dir,
        random_seed=args.seed,
        nn_hidden_dims=parse_hidden_dims(args.hidden_dims),
        nn_epochs=args.epochs,
        nn_batch_size=args.batch_size,
        nn_learning_rate=args.learning_rate,
        num_eigenfunctions=args.num_eigenfunctions,
        grid_size=args.grid_size,
    )

    report_path = run_interpretability_analysis(config)
    print(f"Interpretability report saved to: {report_path}")


if __name__ == "__main__":
    main()
