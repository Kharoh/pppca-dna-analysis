"""
Interpretability analysis for PPPCA-CGR.

Hot-region maps are constructed by re-aggregating eigenfunctions using model
usage weights over the PPPCA score dimensions. For the random forest, the
feature importances provide a non-negative usage weight per component. For the
neural network, we compute an effective input-to-output weight by multiplying
absolute Linear layer weights across the MLP, then average across output
classes. The resulting weight vector is used to linearly combine eigenfunctions
in CGR space, yielding a spatial map of regions that most influence each model.

The script also plots CGR point-process scatters for real promoter vs
non-promoter samples and synthetic promoter-like patterns (e.g., noisy/offset
TATA-box insertions) so you can compare where motifs cluster in [0, 1]^2.
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from main import run_dna_analysis_pipeline
from utils.dna_to_point_processes_cgr import dna_to_point_processes
from utils.load_dna_sequences_from_csv import load_dna_sequences_from_csv


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


def _sample_points_from_processes(processes: List[np.ndarray], max_points: int, rng: np.random.RandomState) -> np.ndarray:
    if not processes:
        return np.zeros((0, 2))
    points = np.vstack([proc for proc in processes if proc.size])
    if points.shape[0] <= max_points:
        return points
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def _plot_cgr_points(points: np.ndarray, title: str, output_dir: str, filename: str, color: str) -> str:
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))
    if points.size:
        ax.scatter(points[:, 0], points[:, 1], s=10, alpha=0.5, color=color)
    ax.set_title(title)
    ax.set_xlabel("CGR x")
    ax.set_ylabel("CGR y")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_single_sequence_cgr(sequence: str, output_dir: str, filename: str, title: str) -> str:
    process = dna_to_point_processes([sequence])[0].numpy()
    return _plot_cgr_points(process, title=title, output_dir=output_dir, filename=filename, color="#E45756")


def plot_repeated_sequence_cgr(sequence: str, repeats: int, output_dir: str, filename: str, title: str) -> str:
    repeated = sequence * repeats
    process = dna_to_point_processes([repeated])[0].numpy()
    return _plot_cgr_points(process, title=title, output_dir=output_dir, filename=filename, color="#F58518")


def plot_comparison_scatter(
    promoter_sequences: List[str],
    non_promoter_sequences: List[str],
    output_dir: str,
    seed: int,
    filename: str,
    title: str,
    max_points: int = 6000,
) -> str:
    rng = np.random.RandomState(seed)
    promoter_processes = [p.numpy() for p in dna_to_point_processes(promoter_sequences)]
    non_promoter_processes = [p.numpy() for p in dna_to_point_processes(non_promoter_sequences)]

    promoter_points = _sample_points_from_processes(promoter_processes, max_points, rng)
    non_promoter_points = _sample_points_from_processes(non_promoter_processes, max_points, rng)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))
    if non_promoter_points.size:
        ax.scatter(non_promoter_points[:, 0], non_promoter_points[:, 1], s=8, alpha=0.25, color="#4C78A8", label="Non-promoter")
    if promoter_points.size:
        ax.scatter(promoter_points[:, 0], promoter_points[:, 1], s=10, alpha=0.45, color="#E45756", label="Promoter")
    ax.set_title(title)
    ax.set_xlabel("CGR x")
    ax.set_ylabel("CGR y")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=True)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _mutate_motif(motif: str, noise_rate: float, rng: np.random.RandomState) -> str:
    bases = np.array(["A", "C", "G", "T"])
    chars = []
    for char in motif:
        if rng.rand() < noise_rate:
            choices = bases[bases != char]
            chars.append(rng.choice(choices))
        else:
            chars.append(char)
    return "".join(chars)


def generate_synthetic_promoter_sequences(
    n_sequences: int,
    length: int,
    motif: str,
    noise_rate: float,
    rng: np.random.RandomState,
) -> List[str]:
    bases = np.array(["A", "C", "G", "T"])
    sequences = []
    for _ in range(n_sequences):
        background = rng.choice(bases, size=length)
        motif_variant = _mutate_motif(motif, noise_rate, rng)
        start = rng.randint(0, max(1, length - len(motif_variant) + 1))
        background[start:start + len(motif_variant)] = list(motif_variant)
        sequences.append("".join(background))
    return sequences


def plot_synthetic_noise_promoters(
    output_dir: str,
    seed: int,
    length: int,
    motif: str,
    noise_rate: float,
    filename: str,
    title: str,
) -> str:
    rng = np.random.RandomState(seed)
    sequences = generate_synthetic_promoter_sequences(40, length, motif, noise_rate=noise_rate, rng=rng)
    processes = [p.numpy() for p in dna_to_point_processes(sequences)]
    points = _sample_points_from_processes(processes, 6000, rng)
    return _plot_cgr_points(points, title=title, output_dir=output_dir, filename=filename, color="#E45756")


def plot_synthetic_noise_only(
    output_dir: str,
    seed: int,
    length: int,
    filename: str,
    title: str,
) -> str:
    rng = np.random.RandomState(seed)
    sequences = generate_synthetic_promoter_sequences(40, length, "", noise_rate=0.0, rng=rng)
    processes = [p.numpy() for p in dna_to_point_processes(sequences)]
    points = _sample_points_from_processes(processes, 6000, rng)
    return _plot_cgr_points(points, title=title, output_dir=output_dir, filename=filename, color="#72B7B2")


def plot_real_individuals_with_aggregate(
    sequences: List[str],
    output_dir: str,
    filename: str,
    title: str,
    seed: int,
    color: str,
) -> str:
    rng = np.random.RandomState(seed)
    processes = [p.numpy() for p in dna_to_point_processes(sequences)]
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.5))
    axes = axes.flatten()

    for idx, proc in enumerate(processes[:8]):
        ax = axes[idx]
        if proc.size:
            ax.scatter(proc[:, 0], proc[:, 1], s=10, alpha=0.6, color=color)
        ax.set_title(f"Sequence {idx + 1}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)
        ax.set_xticks([])
        ax.set_yticks([])

    aggregate_points = _sample_points_from_processes(processes, 8000, rng)
    ax = axes[8]
    if aggregate_points.size:
        ax.scatter(aggregate_points[:, 0], aggregate_points[:, 1], s=8, alpha=0.4, color=color)
    ax.set_title("Aggregate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        weights = weights.flatten()
    weights = np.abs(weights)
    total = weights.sum()
    if total <= 0:
        return np.ones_like(weights) / max(len(weights), 1)
    return weights / total


def compute_rf_component_usage(rf_model, n_components: int) -> np.ndarray:
    importances = np.asarray(getattr(rf_model, "feature_importances_", None))
    if importances is None or importances.size == 0:
        return np.ones(n_components) / max(n_components, 1)
    if importances.size != n_components:
        importances = np.resize(importances, n_components)
    return _normalize_weights(importances)


def compute_nn_component_usage(nn_model) -> np.ndarray:
    if nn_model is None:
        return np.array([])

    linear_layers = []
    for layer in nn_model.network:
        weight = getattr(layer, "weight", None)
        if weight is None:
            continue
        if weight.ndim != 2:
            continue
        linear_layers.append(layer)
    if not linear_layers:
        return np.array([])

    weights = [layer.weight.detach().cpu().numpy() for layer in linear_layers]
    effective = np.abs(weights[-1])
    for w in reversed(weights[:-1]):
        effective = effective @ np.abs(w)
    importance = effective.mean(axis=0)
    return _normalize_weights(importance)


def plot_model_hot_region(
    pca_model,
    component_weights: np.ndarray,
    output_dir: str,
    title: str,
    filename: str,
    grid_size: int = 120,
) -> str:
    eigenfun = pca_model.pca_results["eigenfun"]
    grid = np.linspace(0.0, 1.0, grid_size)
    X, Y = np.meshgrid(grid, grid)
    X_flat = np.stack([X.ravel(), Y.ravel()], axis=1)

    eta_vals = eigenfun(X_flat)
    weights = _normalize_weights(component_weights)
    if weights.size != eta_vals.shape[1]:
        weights = np.resize(weights, eta_vals.shape[1])
        weights = _normalize_weights(weights)

    hot_map = eta_vals @ weights
    hot_map = hot_map.reshape(grid_size, grid_size)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    cf = ax.contourf(X, Y, hot_map, levels=30, cmap="magma")
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("CGR x")
    ax.set_ylabel("CGR y")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _sample_sequences_by_label(
    sequences: List[str],
    labels: np.ndarray,
    label_value: int,
    n_samples: int,
    rng: np.random.RandomState,
) -> List[str]:
    indices = np.where(labels == label_value)[0]
    if indices.size == 0:
        return []
    n_samples = min(n_samples, indices.size)
    chosen = rng.choice(indices, size=n_samples, replace=False)
    return [sequences[i] for i in chosen]


def write_interpretability_report(
    config: InterpretabilityConfig,
    metrics: dict,
    eigenvalues: np.ndarray,
    component_usage: dict,
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

        if component_usage:
            handle.write("## Model component usage\n")
            n_components = len(eigenvalues)
            rf_usage = component_usage.get("rf")
            nn_usage = component_usage.get("nn")
            if rf_usage is None or len(rf_usage) != n_components:
                rf_usage = np.full(n_components, np.nan)
            if nn_usage is None or len(nn_usage) != n_components:
                nn_usage = np.full(n_components, np.nan)
            usage_table = pd.DataFrame({
                "component": np.arange(1, n_components + 1),
                "rf_usage": rf_usage,
                "nn_usage": nn_usage,
            })
            handle.write(usage_table.to_markdown(index=False))
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

    rng = np.random.RandomState(config.random_seed)
    train_sequences, train_labels = load_dna_sequences_from_csv(config.train_path)
    test_sequences, test_labels = load_dna_sequences_from_csv(config.test_path)
    all_sequences = train_sequences + test_sequences
    all_labels = np.concatenate([train_labels, test_labels])

    promoter_label = int(all_labels.max())
    non_promoter_label = int(all_labels.min())
    promoter_sequences = _sample_sequences_by_label(all_sequences, all_labels, promoter_label, 80, rng)
    non_promoter_sequences = _sample_sequences_by_label(all_sequences, all_labels, non_promoter_label, 80, rng)

    promoter_example = "TATAAATATAAA"
    synthetic_single = plot_single_sequence_cgr(
        promoter_example,
        output_dir=config.output_dir,
        filename="synthetic_single_promoter_cgr.png",
        title="Synthetic promoter sequence (single)",
    )
    synthetic_repeated = plot_repeated_sequence_cgr(
        promoter_example,
        repeats=40,
        output_dir=config.output_dir,
        filename="synthetic_repeated_promoter_cgr.png",
        title="Synthetic promoter sequence (repeated)",
    )
    synthetic_noisy = plot_synthetic_noise_promoters(
        output_dir=config.output_dir,
        seed=config.random_seed,
        length=200,
        motif=promoter_example,
        noise_rate=0.2,
        filename="synthetic_noisy_promoters.png",
        title="Synthetic promoters with noise",
    )
    synthetic_noise_only = plot_synthetic_noise_only(
        output_dir=config.output_dir,
        seed=config.random_seed,
        length=200,
        filename="synthetic_noise_only.png",
        title="Synthetic noise-only controls",
    )
    synthetic_promoters = generate_synthetic_promoter_sequences(40, 200, promoter_example, noise_rate=0.1, rng=rng)
    synthetic_non_promoters = generate_synthetic_promoter_sequences(40, 200, "", noise_rate=0.0, rng=rng)
    synthetic_comparison = plot_comparison_scatter(
        promoter_sequences=synthetic_promoters,
        non_promoter_sequences=synthetic_non_promoters,
        output_dir=config.output_dir,
        seed=config.random_seed,
        filename="synthetic_promoter_vs_nonpromoter.png",
        title="Synthetic promoter vs non-promoter (aggregated)",
    )

    real_promoter_individuals = plot_real_individuals_with_aggregate(
        sequences=_sample_sequences_by_label(all_sequences, all_labels, promoter_label, 8, rng),
        output_dir=config.output_dir,
        filename="real_promoter_individuals.png",
        title="Real promoter sequences (8) + aggregate",
        seed=config.random_seed,
        color="#E45756",
    )
    real_non_promoter_individuals = plot_real_individuals_with_aggregate(
        sequences=_sample_sequences_by_label(all_sequences, all_labels, non_promoter_label, 8, rng),
        output_dir=config.output_dir,
        filename="real_nonpromoter_individuals.png",
        title="Real non-promoter sequences (8) + aggregate",
        seed=config.random_seed,
        color="#4C78A8",
    )
    real_combined = plot_comparison_scatter(
        promoter_sequences=promoter_sequences,
        non_promoter_sequences=non_promoter_sequences,
        output_dir=config.output_dir,
        seed=config.random_seed,
        filename="real_promoter_vs_nonpromoter.png",
        title="Real promoter vs non-promoter (aggregated)",
    )

    rf_usage = compute_rf_component_usage(results["rf_classifier"], len(eigenvalues))
    rf_hot = plot_model_hot_region(
        results["pca_model"],
        rf_usage,
        output_dir=config.output_dir,
        title="Random Forest Hot Regions",
        filename="rf_hot_regions.png",
        grid_size=config.grid_size,
    )

    nn_usage = np.array([])
    nn_hot = None
    if results.get("nn_classifier") is not None:
        nn_usage = compute_nn_component_usage(results["nn_classifier"])
        nn_hot = plot_model_hot_region(
            results["pca_model"],
            nn_usage,
            output_dir=config.output_dir,
            title="Neural Network Hot Regions",
            filename="nn_hot_regions.png",
            grid_size=config.grid_size,
        )

    figure_paths = [
        eigen_plot,
        scree_plot,
        rf_hot,
        nn_hot,
        synthetic_single,
        synthetic_repeated,
        synthetic_noisy,
        synthetic_noise_only,
        synthetic_comparison,
        real_promoter_individuals,
        real_non_promoter_individuals,
        real_combined,
        os.path.join(config.output_dir, "interpretability_pca_scores_projection.png"),
        os.path.join(config.output_dir, "interpretability_confusion_matrices.png"),
        os.path.join(config.output_dir, "interpretability_nn_training_history.png"),
    ]
    figure_paths = [path for path in figure_paths if path]

    report_path = write_interpretability_report(
        config,
        metrics=results,
        eigenvalues=eigenvalues,
        component_usage={
            "rf": rf_usage,
            "nn": nn_usage if nn_usage.size else None,
        },
        figure_paths=figure_paths,
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
