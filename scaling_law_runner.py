import argparse
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from main import run_dna_analysis_pipeline
from utils.load_dna_sequences_from_csv import load_dna_sequences_from_csv


@dataclass
class ScalingConfig:
    train_path: str
    test_path: str
    train_sizes: List[int]
    n_test: int
    Jmax_list: List[int]
    output_dir: str
    random_seed: int
    nn_epochs: int
    nn_batch_size: int
    nn_learning_rate: float
    hidden_dims_list: List[List[int]]


def parse_int_list(value: str) -> List[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def parse_hidden_dims_list(value: str) -> List[List[int]]:
    configs: List[List[int]] = []
    if not value:
        return configs
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        normalized = chunk.replace(",", "-")
        dims = [int(part) for part in normalized.split("-") if part.strip()]
        configs.append(dims)
    return configs


def count_mlp_parameters(input_dim: int, hidden_dims: List[int], num_classes: int) -> int:
    dims = [input_dim] + hidden_dims + [num_classes]
    total = 0
    for i in range(len(dims) - 1):
        total += (dims[i] + 1) * dims[i + 1]
    return total


def plot_scaling_accuracy(df: pd.DataFrame, output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    for (jmax, model_name), group in df.groupby(["Jmax", "mlp_label"]):
        label = f"J={jmax} | {model_name}"
        ax.plot(group["train_size"], group["nn_accuracy"], marker="o", label=label)

    ax.set_xlabel("Training set size")
    ax.set_ylabel("Neural network accuracy")
    ax.set_title("Scaling of Accuracy with Training Size")
    ax.grid(alpha=0.3)
    ax.legend()

    out_path = os.path.join(output_dir, "accuracy_vs_train_size.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_scaling_error(df: pd.DataFrame, output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    for (jmax, model_name), group in df.groupby(["Jmax", "mlp_label"]):
        error = 1.0 - group["nn_accuracy"].clip(0, 1)
        label = f"J={jmax} | {model_name}"
        ax.plot(group["train_size"], error, marker="o", label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training set size (log)")
    ax.set_ylabel("Generalization error (log)")
    ax.set_title("Scaling Law: Error vs Training Size")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    out_path = os.path.join(output_dir, "error_vs_train_size_loglog.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_accuracy_vs_params(df: pd.DataFrame, output_dir: str) -> str:
    largest_train = df["train_size"].max()
    subset = df[df["train_size"] == largest_train]

    fig, ax = plt.subplots(figsize=(6, 4))
    for jmax, group in subset.groupby("Jmax"):
        ax.plot(group["param_count"], group["nn_accuracy"], marker="o", label=f"J={jmax}")
    for _, row in subset.iterrows():
        ax.annotate(
            f"J={row['Jmax']}|{row['mlp_label']}",
            (row["param_count"], row["nn_accuracy"]),
            textcoords="offset points",
            xytext=(4, 4),
        )

    ax.set_xscale("log")
    ax.set_xlabel("MLP parameter count (log)")
    ax.set_ylabel("Neural network accuracy")
    ax.set_title(f"Accuracy vs Model Size (train={largest_train})")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    out_path = os.path.join(output_dir, "accuracy_vs_params.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_scaling_report(config: ScalingConfig, df: pd.DataFrame, figure_paths: List[str], output_dir: str) -> str:
    report_path = os.path.join(output_dir, "scaling_law_report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# PPPCA-CGR Scaling Law Report\n\n")
        handle.write("## Experimental setup\n")
        handle.write(f"- Train CSV: `{config.train_path}`\n")
        handle.write(f"- Test CSV: `{config.test_path}`\n")
        handle.write(f"- Train sizes: {config.train_sizes}\n")
        handle.write(f"- Test size: {config.n_test}\n")
        handle.write(f"- Jmax list: {config.Jmax_list}\n")
        handle.write(f"- Hidden dims list: {config.hidden_dims_list}\n")
        handle.write(f"- Epochs: {config.nn_epochs}\n")
        handle.write("\n")

        handle.write("## Summary table\n")
        handle.write(df.to_markdown(index=False))
        handle.write("\n\n")

        handle.write("## Figures\n")
        for path in figure_paths:
            rel_path = os.path.relpath(path, output_dir)
            handle.write(f"- {rel_path}\n")
        handle.write("\n")

        handle.write("## Notes\n")
        handle.write("Accuracy and MCC are reported for each model-size/train-size combination. The log-log plot provides a scaling-law diagnostic for generalization error $1-\\text{accuracy}$.\n")

    return report_path


def run_scaling_experiments(config: ScalingConfig) -> str:
    os.makedirs(config.output_dir, exist_ok=True)

    _, labels = load_dna_sequences_from_csv(config.train_path)
    num_classes = len(np.unique(labels))

    rows = []
    for train_size in config.train_sizes:
        for jmax in config.Jmax_list:
            for hidden_dims in config.hidden_dims_list:
                label = "-".join(str(v) for v in hidden_dims)
                param_count = count_mlp_parameters(jmax, hidden_dims, num_classes)

                results = run_dna_analysis_pipeline(
                    train_path=config.train_path,
                    test_path=config.test_path,
                    n_train=train_size,
                    n_test=config.n_test,
                    Jmax=jmax,
                    output_dir=config.output_dir,
                    random_seed=config.random_seed,
                    use_nn=True,
                    nn_hidden_dims=hidden_dims,
                    nn_epochs=config.nn_epochs,
                    nn_batch_size=config.nn_batch_size,
                    nn_learning_rate=config.nn_learning_rate,
                    make_plots=False,
                    plot_prefix=f"scaling_n{train_size}_J{jmax}_mlp{label}",
                )

                rows.append({
                    "train_size": train_size,
                    "Jmax": jmax,
                    "mlp_label": label,
                    "hidden_dims": hidden_dims,
                    "param_count": param_count,
                    "rf_accuracy": results["rf_accuracy"],
                    "rf_mcc": results["rf_mcc"],
                    "nn_accuracy": results["nn_accuracy"],
                    "nn_mcc": results["nn_mcc"],
                })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(config.output_dir, "scaling_summary.csv")
    df.to_csv(csv_path, index=False)

    accuracy_plot = plot_scaling_accuracy(df, config.output_dir)
    error_plot = plot_scaling_error(df, config.output_dir)
    params_plot = plot_accuracy_vs_params(df, config.output_dir)

    report_path = write_scaling_report(config, df, [accuracy_plot, error_plot, params_plot], config.output_dir)
    return report_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scaling-law runner for PPPCA-CGR")
    parser.add_argument("--train-path", required=True, help="Path to training CSV")
    parser.add_argument("--test-path", required=True, help="Path to testing CSV")
    parser.add_argument("--train-sizes", default="250,500,1000", help="Comma-separated train sizes")
    parser.add_argument("--n-test", type=int, default=500, help="Number of test samples")
    parser.add_argument("--Jmax-list", default="6,10,14", help="Comma-separated PPPCA components to sweep")
    parser.add_argument("--output-dir", default="analysis_results/scaling_laws", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--mlp-sizes",
        default="16-4;64-32;128-64",
        help="Semicolon-separated MLP hidden sizes, each as dash- or comma-separated list",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = ScalingConfig(
        train_path=args.train_path,
        test_path=args.test_path,
        train_sizes=parse_int_list(args.train_sizes),
        n_test=args.n_test,
        Jmax_list=parse_int_list(args.Jmax_list),
        output_dir=args.output_dir,
        random_seed=args.seed,
        nn_epochs=args.epochs,
        nn_batch_size=args.batch_size,
        nn_learning_rate=args.learning_rate,
        hidden_dims_list=parse_hidden_dims_list(args.mlp_sizes),
    )

    report_path = run_scaling_experiments(config)
    print(f"Scaling-law report saved to: {report_path}")


if __name__ == "__main__":
    main()
