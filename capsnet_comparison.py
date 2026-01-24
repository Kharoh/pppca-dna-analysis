"""
Capsule Network vs PPPCA baselines on GUE promoter classification.

This script trains a 2D Capsule Network on FCGR images derived from CGR point
processes and compares performance/interpretability against PPPCA + Random
Forest/MLP baselines. It produces a markdown report with plots.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
)
from torch.utils.data import DataLoader, Dataset

from main import DNAPPCA, train_neural_network, predict_neural_network
from utils.dna_to_point_processes_cgr import dna_to_point_processes
from utils.load_dna_sequences_from_csv import load_dna_sequences_from_csv
from interpretability_analysis import (
    compute_nn_component_usage,
    compute_rf_component_usage,
    plot_cgr_eigenfunctions,
    plot_eigenvalue_scree,
    plot_model_hot_region,
)


@dataclass
class CapsNetComparisonConfig:
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
    caps_grid_res: int
    caps_epochs: int
    caps_batch_size: int
    caps_learning_rate: float
    caps_num_iterations: int
    caps_primary_caps: int
    caps_primary_dim: int
    caps_digit_dim: int
    num_eigenfunctions: int
    grid_size: int
    caps_saliency_samples: int


class CGRDataset(Dataset):
    def __init__(self, point_processes: List[torch.Tensor], labels: np.ndarray, grid_resolution: int = 64):
        self.point_processes = point_processes
        self.labels = labels
        self.res = grid_resolution

    def __len__(self) -> int:
        return len(self.point_processes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        points = self.point_processes[idx]
        label = int(self.labels[idx])

        if points.size(0) == 0:
            return torch.zeros((1, self.res, self.res), dtype=torch.float32), label

        coords = (points * self.res).long().clamp(0, self.res - 1)
        row_indices = coords[:, 1]
        col_indices = coords[:, 0]
        flat_indices = row_indices * self.res + col_indices

        counts = torch.bincount(flat_indices, minlength=self.res * self.res)
        image = counts.view(self.res, self.res).float()
        image = image / (image.max() + 1e-8)
        return image.unsqueeze(0), label


class CapsuleLayer(nn.Module):
    def __init__(
        self,
        num_capsules: int,
        num_route_nodes: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int | None] = None,
        stride: Union[int | None] = None,
        num_iterations: int = 3,
    ):
        super().__init__()
        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules
        self.out_channels = out_channels

        if num_route_nodes != -1:
            # Digit Caps (Routing)
            # Use 0.1 init to ensure signal flow
            self.route_weights = nn.Parameter(
                torch.randn(num_capsules, num_route_nodes, in_channels, out_channels) * 0.1
            )
        else:
            # Primary Caps (Conv)
            self.capsules = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
                for _ in range(num_capsules)
            ])


    @staticmethod
    def squash(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1.0 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- PRIMARY CAPSULES ---
        if self.num_route_nodes == -1:
            outputs = torch.stack([capsule(x) for capsule in self.capsules], dim=1)
            batch_size, num_caps, out_dim, height, width = outputs.shape
            
            # Flatten grid: (B, Caps, Dim, H, W) -> (B, Caps * H * W, Dim)
            outputs = outputs.permute(0, 1, 3, 4, 2).contiguous()
            outputs = outputs.view(batch_size, -1, out_dim)
            
            # CRITICAL FIX: Do NOT squash Primary Capsules!
            # Raw vectors allow gradients to flow back to Conv1.
            return outputs

        batch_size = x.size(0)
        u_hat = torch.einsum("bri,crio->bcro", x, self.route_weights)

        b_ij = torch.zeros(batch_size, self.num_capsules, self.num_route_nodes, device=x.device)

        for iteration in range(self.num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            s_j = (c_ij.unsqueeze(-1) * u_hat).sum(dim=2)
            v_j = self.squash(s_j)

            if iteration < self.num_iterations - 1:
                a_ij = (u_hat * v_j.unsqueeze(2)).sum(dim=-1)
                b_ij = b_ij + a_ij

        return v_j


class DNACapsNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        grid_res: int = 64,
        primary_capsules: int = 8,
        primary_dim: int = 32,
        digit_dim: int = 16,
        routing_iterations: int = 3,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_caps = CapsuleLayer(
            num_capsules=primary_capsules,
            num_route_nodes=-1,
            in_channels=256,
            out_channels=primary_dim,
            kernel_size=9,
            stride=2,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, grid_res, grid_res)
            conv_out = self.conv1(dummy)
            primary_out = self.primary_caps(conv_out)
            num_route_nodes = primary_out.size(1)

        self.digit_caps = CapsuleLayer(
            num_capsules=num_classes,
            num_route_nodes=num_route_nodes,
            in_channels=primary_dim,
            out_channels=digit_dim,
            num_iterations=routing_iterations,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = self.primary_caps(x)
        output_caps = self.digit_caps(x)
        classes_probs = torch.sqrt((output_caps ** 2).sum(dim=2) + 1e-8)
        return classes_probs, output_caps


class CapsNetLoss(nn.Module):
    def __init__(self, m_plus: float = 0.9, m_minus: float = 0.1, lambda_val: float = 0.5):
        super().__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_val = lambda_val

    def forward(self, output_probs: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
        batch_size = output_probs.size(0)
        num_classes = output_probs.size(1)
        targets = torch.zeros(batch_size, num_classes, device=output_probs.device)
        targets.scatter_(1, target_indices.view(-1, 1), 1.0)

        present_error = F.relu(self.m_plus - output_probs) ** 2
        absent_error = F.relu(output_probs - self.m_minus) ** 2
        loss = targets * present_error + self.lambda_val * (1.0 - targets) * absent_error
        return loss.sum(dim=1).mean()


def sample_sequences(
    sequences: List[str], labels: np.ndarray, n_samples: int, seed: int
) -> Tuple[List[str], np.ndarray]:
    if n_samples >= len(sequences):
        return sequences, labels
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(sequences), size=n_samples, replace=False)
    sampled_seqs = [sequences[i] for i in indices]
    return sampled_seqs, labels[indices]


def build_fcgr_dataset(seqs: List[str], labels: np.ndarray, grid_res: int) -> CGRDataset:
    processes = dna_to_point_processes(seqs)
    processes = [proc.float() for proc in processes]
    return CGRDataset(processes, labels, grid_resolution=grid_res)


def train_capsnet(
    model: DNACapsNet,
    train_loader: DataLoader,
    val_loader: Union[DataLoader | None],
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> Dict[str, List[float]]:
    model.to(device)
    criterion = CapsNetLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            if epoch == 0: 
                print(f"DEBUG: Input Max: {images.max().item():.4f}")
                print(f"DEBUG: Input Mean: {images.mean().item():.4f}")
                print(f"DEBUG: Input Non-Zeros: {(images > 0).float().sum().item()}")
                
                # Check if first layer is dead
                with torch.no_grad():
                    conv_out = F.relu(model.conv1(images))
                    print(f"DEBUG: Conv1 Max: {conv_out.max().item():.4f}")
                    
                    # Check Primary Caps (BEFORE routing)
                    prim_out = model.primary_caps(conv_out)
                    print(f"DEBUG: Primary Caps Mean Norm: {prim_out.norm(dim=-1).mean().item():.4f}")

            optimizer.zero_grad()
            probs, _ = model(images)
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(probs, dim=1)
            running_total += labels.size(0)
            running_correct += (preds == labels).sum().item()

        history["train_loss"].append(running_loss / max(running_total, 1))
        history["train_acc"].append(running_correct / max(running_total, 1))

        if val_loader is not None:
            val_metrics = evaluate_capsnet(model, val_loader, device)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])

        # if (epoch + 1) % max(1, epochs // 5) == 0:
        print(
            f"Epoch {epoch + 1:03d}/{epochs} - "
            f"train loss: {history['train_loss'][-1]:.4f}, "
            f"train acc: {history['train_acc'][-1]:.4f}"
        )

    return history


def evaluate_capsnet(model: DNACapsNet, loader: DataLoader, device: torch.device) -> Dict[str, float | np.ndarray]:
    model.eval()
    criterion = CapsNetLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            probs, _ = model(images)
            loss = criterion(probs, labels)

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(probs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds = np.argmax(probs, axis=1)

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
        "labels": labels,
        "preds": preds,
        "probs": probs,
    }


def plot_capsnet_history(history: Dict[str, List[float]], output_dir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    if history["val_loss"]:
        axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_title("CapsNet Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Margin loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train")
    if history["val_acc"]:
        axes[1].plot(epochs, history["val_acc"], label="Validation")
    axes[1].set_title("CapsNet Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "capsnet_training_history.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, output_path: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_pppca_scores(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    acc_rf: float,
    acc_nn: Union[float | None],
    output_dir: str,
) -> str:
    fig, axes = plt.subplots(1, 2 if acc_nn is not None else 1, figsize=(18 if acc_nn is not None else 9, 5))
    if acc_nn is None:
        axes = [axes]

    axes[0].scatter(
        train_scores[:, 0],
        train_scores[:, 1],
        c=y_train,
        cmap="coolwarm",
        alpha=0.3,
        label="Train",
        marker="o",
        s=30,
    )
    axes[0].scatter(
        test_scores[:, 0],
        test_scores[:, 1],
        c=y_test,
        cmap="coolwarm",
        alpha=0.9,
        edgecolors="black",
        label="Test",
        marker="s",
        s=50,
    )
    axes[0].set_xlabel("Principal Component 1")
    axes[0].set_ylabel("Principal Component 2")
    axes[0].set_title(f"PPPCA Scores\nRandom Forest Acc: {acc_rf:.2%}")
    axes[0].legend()

    if acc_nn is not None:
        axes[1].scatter(
            train_scores[:, 0],
            train_scores[:, 1],
            c=y_train,
            cmap="coolwarm",
            alpha=0.3,
            label="Train",
            marker="o",
            s=30,
        )
        axes[1].scatter(
            test_scores[:, 0],
            test_scores[:, 1],
            c=y_test,
            cmap="coolwarm",
            alpha=0.9,
            edgecolors="black",
            label="Test",
            marker="s",
            s=50,
        )
        axes[1].set_xlabel("Principal Component 1")
        axes[1].set_ylabel("Principal Component 2")
        axes[1].set_title(f"PPPCA Scores\nMLP Acc: {acc_nn:.2%}")
        axes[1].legend()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "pppca_scores_projection.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pppca_confusions(
    y_test: np.ndarray,
    y_pred_rf: np.ndarray,
    y_pred_nn: Union[np.ndarray | None],
    acc_rf: float,
    mcc_rf: float,
    acc_nn: Union[float | None],
    mcc_nn: Union[float | None],
    output_dir: str,
) -> str:
    fig, axes = plt.subplots(1, 2 if y_pred_nn is not None else 1, figsize=(12 if y_pred_nn is not None else 6, 5))
    if y_pred_nn is None:
        axes = [axes]

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title(f"Random Forest\nAcc: {acc_rf:.2%}, MCC: {mcc_rf:.3f}")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    if y_pred_nn is not None:
        cm_nn = confusion_matrix(y_test, y_pred_nn)
        sns.heatmap(cm_nn, annot=True, fmt="d", cmap="Greens", ax=axes[1])
        axes[1].set_title(f"MLP\nAcc: {acc_nn:.2%}, MCC: {mcc_nn:.3f}")
        axes[1].set_ylabel("True Label")
        axes[1].set_xlabel("Predicted Label")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "pppca_confusion_matrices.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_capsule_embedding(embeddings: np.ndarray, labels: np.ndarray, output_dir: str) -> str:
    reducer = PCA(n_components=2, random_state=42)
    reduced = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="coolwarm", alpha=0.8, s=30)
    ax.set_title("CapsNet Embeddings (PCA)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(alpha=0.3)
    ax.legend(*scatter.legend_elements(), title="Label")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "capsnet_embedding_pca.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_fcgr_means(images: np.ndarray, labels: np.ndarray, output_dir: str) -> List[str]:
    paths = []
    for label in np.unique(labels):
        class_images = images[labels == label]
        if class_images.size == 0:
            continue
        mean_image = class_images.mean(axis=0)
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(mean_image[0], cmap="viridis")
        ax.set_title(f"Mean FCGR - class {label}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")
        path = os.path.join(output_dir, f"capsnet_fcgr_mean_class_{label}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def compute_capsule_embeddings(
    model: DNACapsNet, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    embeddings = []
    labels_list = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probs, output_caps = model(images)
            embeddings.append(output_caps.view(output_caps.size(0), -1).cpu().numpy())
            labels_list.append(labels.numpy())

    return np.concatenate(embeddings, axis=0), np.concatenate(labels_list, axis=0)


def compute_capsule_saliency(
    model: DNACapsNet,
    loader: DataLoader,
    device: torch.device,
    max_samples_per_class: int,
) -> Dict[int, np.ndarray]:
    model.eval()
    saliency_sum: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        images.requires_grad_(True)

        probs, _ = model(images)
        selected = probs.gather(1, labels.view(-1, 1)).sum()

        model.zero_grad()
        if images.grad is not None:
            images.grad.zero_()
        selected.backward()

        grads = images.grad.detach().abs().cpu()
        labels_cpu = labels.cpu().numpy()

        for idx, label in enumerate(labels_cpu):
            label = int(label)
            if counts.get(label, 0) >= max_samples_per_class:
                continue
            saliency_sum[label] = saliency_sum.get(label, torch.zeros_like(grads[idx])) + grads[idx]
            counts[label] = counts.get(label, 0) + 1

        if counts and all(value >= max_samples_per_class for value in counts.values()):
            break

    saliency_maps = {label: (saliency_sum[label] / max(counts[label], 1)).numpy() for label in saliency_sum}
    return saliency_maps


def plot_saliency_maps(saliency_maps: Dict[int, np.ndarray], output_dir: str) -> List[str]:
    paths = []
    for label, saliency in saliency_maps.items():
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(saliency[0], cmap="magma")
        ax.set_title(f"CapsNet Saliency - class {label}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")
        path = os.path.join(output_dir, f"capsnet_saliency_class_{label}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def write_report(
    config: CapsNetComparisonConfig,
    metrics: Dict[str, float],
    capsnet_report: Dict[str, str],
    pppca_report: Dict[str, str],
    figure_paths: List[str],
    output_dir: str,
) -> str:
    report_path = os.path.join(output_dir, "capsnet_comparison_report.md")
    table = pd.DataFrame(
        {
            "model": ["CapsNet", "PPPCA + RF", "PPPCA + MLP"],
            "accuracy": [metrics["capsnet_accuracy"], metrics["rf_accuracy"], metrics.get("nn_accuracy")],
            "mcc": [metrics["capsnet_mcc"], metrics["rf_mcc"], metrics.get("nn_mcc")],
        }
    )

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# CapsNet vs PPPCA Baseline Report\n\n")
        handle.write("## Experimental setup\n")
        handle.write(f"- Train CSV: `{config.train_path}`\n")
        handle.write(f"- Test CSV: `{config.test_path}`\n")
        handle.write(f"- Train size: {config.n_train}\n")
        handle.write(f"- Test size: {config.n_test}\n")
        handle.write(f"- PPPCA Jmax: {config.Jmax}\n")
        handle.write(f"- PPPCA kernel: {config.kernel}\n")
        handle.write(f"- MLP hidden dims: {config.nn_hidden_dims}\n")
        handle.write(f"- CapsNet grid resolution: {config.caps_grid_res}\n")
        handle.write(f"- CapsNet epochs: {config.caps_epochs}\n")
        handle.write(f"- CapsNet batch size: {config.caps_batch_size}\n")
        handle.write(f"- CapsNet routing iterations: {config.caps_num_iterations}\n")
        handle.write("\n")

        handle.write("## Performance summary\n")
        handle.write(table.to_markdown(index=False))
        handle.write("\n\n")

        handle.write("## Capsule Network classification report\n")
        handle.write(capsnet_report["text"])
        handle.write("\n\n")

        if pppca_report.get("text"):
            handle.write("## PPPCA + RF/MLP classification reports\n")
            handle.write(pppca_report["text"])
            handle.write("\n\n")

        handle.write("## Figures\n")
        for path in figure_paths:
            rel_path = os.path.relpath(path, output_dir)
            handle.write(f"- {rel_path}\n")
        handle.write("\n")

        handle.write("## Notes\n")
        handle.write(
            "CapsNet embeddings are projected to 2D using PCA on the digit capsule vectors. "
            "Saliency maps are computed from gradients of the predicted capsule length with respect to the FCGR input. "
            "PPPCA hot-region maps weigh eigenfunctions by model usage (RF feature importances and MLP effective weights).\n"
        )

    return report_path


def run_pppca_baseline(
    train_seqs: List[str],
    test_seqs: List[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    config: CapsNetComparisonConfig,
) -> Tuple[Dict[str, float], Dict[str, str], List[str], Dict[str, object]]:
    pca_model = DNAPPCA(Jmax=config.Jmax, kernel=config.kernel)
    train_scores = pca_model.fit_transform(train_seqs)
    test_scores = pca_model.transform(test_seqs)
    eigenvalues = np.array(pca_model.pca_results["eigenval"])

    rf_model = RandomForestClassifier(n_estimators=200, random_state=config.random_seed, max_depth=5)
    rf_model.fit(train_scores, y_train)
    y_pred_rf = rf_model.predict(test_scores)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    mcc_rf = matthews_corrcoef(y_test, y_pred_rf)

    nn_model = None
    nn_history = None
    y_pred_nn = None
    acc_nn = None
    mcc_nn = None

    num_classes = len(np.unique(y_train))
    nn_model, nn_history = train_neural_network(
        X_train=train_scores,
        y_train=y_train,
        X_val=test_scores,
        y_val=y_test,
        hidden_dims=config.nn_hidden_dims,
        num_classes=num_classes,
        epochs=config.nn_epochs,
        batch_size=config.nn_batch_size,
        learning_rate=config.nn_learning_rate,
        random_seed=config.random_seed,
    )
    y_pred_nn, _ = predict_neural_network(nn_model, test_scores)
    acc_nn = accuracy_score(y_test, y_pred_nn)
    mcc_nn = matthews_corrcoef(y_test, y_pred_nn)

    figure_paths = []
    figure_paths.append(plot_pppca_scores(train_scores, test_scores, y_train, y_test, acc_rf, acc_nn, config.output_dir))
    figure_paths.append(
        plot_pppca_confusions(y_test, y_pred_rf, y_pred_nn, acc_rf, mcc_rf, acc_nn, mcc_nn, config.output_dir)
    )

    if nn_history is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs_range = range(1, len(nn_history["train_loss"]) + 1)
        axes[0].plot(epochs_range, nn_history["train_loss"], label="Train Loss", linewidth=2)
        if nn_history["val_loss"]:
            axes[0].plot(epochs_range, nn_history["val_loss"], label="Val Loss", linewidth=2)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("MLP Training Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(epochs_range, nn_history["train_acc"], label="Train Acc", linewidth=2)
        if nn_history["val_acc"]:
            axes[1].plot(epochs_range, nn_history["val_acc"], label="Val Acc", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("MLP Training Accuracy")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        history_path = os.path.join(config.output_dir, "pppca_mlp_training_history.png")
        plt.tight_layout()
        plt.savefig(history_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(history_path)

    eigen_plot = plot_cgr_eigenfunctions(
        pca_model,
        output_dir=config.output_dir,
        num_eigenfunctions=config.num_eigenfunctions,
        grid_size=config.grid_size,
    )
    scree_plot = plot_eigenvalue_scree(eigenvalues, config.output_dir)
    figure_paths.extend([eigen_plot, scree_plot])

    rf_usage = compute_rf_component_usage(rf_model, len(eigenvalues))
    rf_hot = plot_model_hot_region(
        pca_model,
        rf_usage,
        output_dir=config.output_dir,
        title="PPPCA + RF Hot Regions",
        filename="pppca_rf_hot_regions.png",
        grid_size=config.grid_size,
    )
    figure_paths.append(rf_hot)

    nn_usage = compute_nn_component_usage(nn_model)
    if nn_usage.size:
        nn_hot = plot_model_hot_region(
            pca_model,
            nn_usage,
            output_dir=config.output_dir,
            title="PPPCA + MLP Hot Regions",
            filename="pppca_mlp_hot_regions.png",
            grid_size=config.grid_size,
        )
        figure_paths.append(nn_hot)

    report_text = ""
    report_text += "### Random Forest\n\n"
    report_text += pd.DataFrame(
        classification_report(y_test, y_pred_rf, output_dict=True, zero_division=0)
    ).T.to_markdown()
    report_text += "\n\n### MLP\n\n"
    report_text += pd.DataFrame(
        classification_report(y_test, y_pred_nn, output_dict=True, zero_division=0)
    ).T.to_markdown()

    metrics = {
        "rf_accuracy": acc_rf,
        "rf_mcc": mcc_rf,
        "nn_accuracy": acc_nn,
        "nn_mcc": mcc_nn,
    }
    extra = {
        "pca_model": pca_model,
        "rf_classifier": rf_model,
        "nn_classifier": nn_model,
        "train_scores": train_scores,
        "test_scores": test_scores,
        "eigenvalues": eigenvalues,
    }
    return metrics, {"text": report_text}, figure_paths, extra


def run_capsnet_pipeline(
    train_seqs: List[str],
    test_seqs: List[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    config: CapsNetComparisonConfig,
) -> Tuple[Dict[str, float], Dict[str, str], List[str], Dict[str, object]]:
    train_dataset = build_fcgr_dataset(train_seqs, y_train, config.caps_grid_res)
    test_dataset = build_fcgr_dataset(test_seqs, y_test, config.caps_grid_res)

    train_loader = DataLoader(train_dataset, batch_size=config.caps_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.caps_batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNACapsNet(
        num_classes=len(np.unique(y_train)),
        grid_res=config.caps_grid_res,
        primary_capsules=config.caps_primary_caps,
        primary_dim=config.caps_primary_dim,
        digit_dim=config.caps_digit_dim,
        routing_iterations=config.caps_num_iterations,
    )

    history = train_capsnet(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=config.caps_epochs,
        learning_rate=config.caps_learning_rate,
        device=device,
    )

    eval_results = evaluate_capsnet(model, test_loader, device)
    acc_caps = eval_results["accuracy"]
    mcc_caps = matthews_corrcoef(eval_results["labels"], eval_results["preds"])

    figure_paths = []
    figure_paths.append(plot_capsnet_history(history, config.output_dir))

    confusion_path = os.path.join(config.output_dir, "capsnet_confusion_matrix.png")
    figure_paths.append(
        plot_confusion(eval_results["labels"], eval_results["preds"], "CapsNet Confusion Matrix", confusion_path)
    )

    embeddings, labels = compute_capsule_embeddings(model, test_loader, device)
    figure_paths.append(plot_capsule_embedding(embeddings, labels, config.output_dir))

    images = np.stack([train_dataset[i][0].numpy() for i in range(len(train_dataset))], axis=0)
    labels_train = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    figure_paths.extend(plot_fcgr_means(images, labels_train, config.output_dir))

    saliency_maps = compute_capsule_saliency(
        model,
        test_loader,
        device=device,
        max_samples_per_class=config.caps_saliency_samples,
    )
    figure_paths.extend(plot_saliency_maps(saliency_maps, config.output_dir))

    report_text = pd.DataFrame(
        classification_report(
            eval_results["labels"],
            eval_results["preds"],
            output_dict=True,
            zero_division=0,
        )
    ).T
    report_text = report_text.to_markdown()

    metrics = {
        "capsnet_accuracy": acc_caps,
        "capsnet_mcc": mcc_caps,
    }
    extra = {
        "model": model,
        "history": history,
        "eval": eval_results,
    }
    return metrics, {"text": report_text}, figure_paths, extra


def parse_hidden_dims(value: str) -> List[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CapsNet vs PPPCA comparison on promoter sequences")
    parser.add_argument("--train-path", required=True, help="Path to training CSV")
    parser.add_argument("--test-path", required=True, help="Path to testing CSV")
    parser.add_argument("--n-train", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--n-test", type=int, default=500, help="Number of test samples")
    parser.add_argument("--Jmax", type=int, default=10, help="Number of PPPCA components")
    parser.add_argument("--kernel", default="linear", help="Kernel type for PCA")
    parser.add_argument("--output-dir", default="analysis_results/capsnet_comparison", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--hidden-dims", default="16,4", help="Comma-separated MLP hidden layer sizes")
    parser.add_argument("--nn-epochs", type=int, default=100, help="MLP epochs")
    parser.add_argument("--nn-batch-size", type=int, default=32, help="MLP batch size")
    parser.add_argument("--nn-learning-rate", type=float, default=0.001, help="MLP learning rate")

    parser.add_argument("--caps-grid-res", type=int, default=64, help="FCGR grid resolution")
    parser.add_argument("--caps-epochs", type=int, default=50, help="CapsNet epochs")
    parser.add_argument("--caps-batch-size", type=int, default=32, help="CapsNet batch size")
    parser.add_argument("--caps-learning-rate", type=float, default=0.00001, help="CapsNet learning rate")
    parser.add_argument("--caps-iterations", type=int, default=3, help="Routing iterations")
    parser.add_argument("--caps-primary-caps", type=int, default=8, help="Primary capsule count")
    parser.add_argument("--caps-primary-dim", type=int, default=32, help="Primary capsule dimension")
    parser.add_argument("--caps-digit-dim", type=int, default=16, help="Digit capsule dimension")
    parser.add_argument("--caps-saliency-samples", type=int, default=40, help="Samples per class for saliency")

    parser.add_argument("--num-eigenfunctions", type=int, default=6, help="Number of eigenfunctions to plot")
    parser.add_argument("--grid-size", type=int, default=120, help="Grid resolution for eigenfunction plots")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = CapsNetComparisonConfig(
        train_path=args.train_path,
        test_path=args.test_path,
        n_train=args.n_train,
        n_test=args.n_test,
        Jmax=args.Jmax,
        kernel=args.kernel,
        output_dir=args.output_dir,
        random_seed=args.seed,
        nn_hidden_dims=parse_hidden_dims(args.hidden_dims),
        nn_epochs=args.nn_epochs,
        nn_batch_size=args.nn_batch_size,
        nn_learning_rate=args.nn_learning_rate,
        caps_grid_res=args.caps_grid_res,
        caps_epochs=args.caps_epochs,
        caps_batch_size=args.caps_batch_size,
        caps_learning_rate=args.caps_learning_rate,
        caps_num_iterations=args.caps_iterations,
        caps_primary_caps=args.caps_primary_caps,
        caps_primary_dim=args.caps_primary_dim,
        caps_digit_dim=args.caps_digit_dim,
        num_eigenfunctions=args.num_eigenfunctions,
        grid_size=args.grid_size,
        caps_saliency_samples=args.caps_saliency_samples,
    )

    os.makedirs(config.output_dir, exist_ok=True)

    train_sequences, train_labels = load_dna_sequences_from_csv(config.train_path)
    test_sequences, test_labels = load_dna_sequences_from_csv(config.test_path)

    train_seqs, y_train = sample_sequences(train_sequences, train_labels, config.n_train, config.random_seed)
    test_seqs, y_test = sample_sequences(test_sequences, test_labels, config.n_test, config.random_seed + 1)

    print("Running PPPCA baselines...")
    pppca_metrics, pppca_report, pppca_figs, _ = run_pppca_baseline(
        train_seqs, test_seqs, y_train, y_test, config
    )

    print("Running Capsule Network...")
    caps_metrics, caps_report, caps_figs, caps_extra = run_capsnet_pipeline(
        train_seqs, test_seqs, y_train, y_test, config
    )

    metrics = {**pppca_metrics, **caps_metrics}
    figures = pppca_figs + caps_figs

    report_path = write_report(
        config=config,
        metrics=metrics,
        capsnet_report=caps_report,
        pppca_report=pppca_report,
        figure_paths=figures,
        output_dir=config.output_dir,
    )

    summary_path = os.path.join(config.output_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"CapsNet report saved to: {report_path}")
    print(f"Metrics summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
