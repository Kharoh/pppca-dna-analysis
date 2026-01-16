from __future__ import annotations

from itertools import chain
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Types:
# - A single multivariate point process i is a float tensor of shape (k_i, d) with coordinates in [0,1]^d
# - point_processes is a list of n such tensors, potentially with different k_i
PointArray = torch.Tensor
PointProcessesND = List[PointArray]

# def _pairwise_integral_FiFj(points_i: PointArray, points_j: PointArray) -> torch.Tensor:
#     """
#     Compute S_ij = ∫_{[0,1]^d} F_i(x) F_j(x) dx in closed form, without grids.

#     Shapes:
#       - points_i: (k_i, d)
#       - points_j: (k_j, d)
#       - returns: scalar tensor []

#     Formula:
#       S_ij = Σ_{p in i} Σ_{q in j} ∏_{r=1..d} (1 - max(p_r, q_r))
#     """
#     if points_i.numel() == 0 or points_j.numel() == 0:
#         return torch.tensor(0.0, dtype=torch.float64)

#     # Broadcast pairwise max over all event pairs and dimensions
#     # max_vals: (k_i, k_j, d)
#     max_vals = torch.maximum(points_i[:, None, :], points_j[None, :, :])
#     contrib = (1.0 - max_vals).clamp(min=0.0)         # (k_i, k_j, d)
#     prod_contrib = contrib.prod(dim=-1)               # (k_i, k_j)
#     return prod_contrib.sum().to(dtype=torch.float64) # []

# def _build_S_matrix(point_processes: PointProcessesND) -> torch.Tensor:
#     """
#     Build S with entries S_ij = ∫ F_i F_j over [0,1]^d using the closed form.

#     Shapes:
#       - point_processes: list of length n with tensors (k_i, d)
#       - returns S: (n, n)
#     """
#     n = len(point_processes)
#     S = torch.zeros((n, n), dtype=torch.float64)
#     prog = tqdm(range(n * (n + 1) // 2), desc="Building S matrix")
#     for i in range(n):
#         Pi = point_processes[i].to(dtype=torch.float64)
#         for j in range(i, n):
#             Pj = point_processes[j].to(dtype=torch.float64)
#             val = _pairwise_integral_FiFj(Pi, Pj)
#             S[i, j] = val
#             if j != i:
#                 S[j, i] = val
#             prog.update(1)
#     return S

def _pairwise_integral_FiFj_outermin(points_i: torch.Tensor,
                                     points_j: torch.Tensor,
                                     *,
                                     block_cols: int = 8192,
                                     device: torch.device | None = None,
                                     work_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    # Precompute complements α = 1 - p, β = 1 - q
    Pi = points_i.to(dtype=work_dtype, copy=False)
    Pj = points_j.to(dtype=work_dtype, copy=False)
    if device is None:
        device = Pi.device
    Ai = (1.0 - Pi).to(device, non_blocking=True)
    Aj = (1.0 - Pj)  # keep on source device until block move

    ki, d = Ai.shape
    kj = Aj.shape[0]
    total = torch.zeros((), dtype=torch.float64, device=device)

    for c0 in range(0, kj, block_cols):
        c1 = min(c0 + block_cols, kj)
        Aj_blk = Aj[c0:c1].to(device, non_blocking=True)
        R = torch.ones((ki, Aj_blk.shape[0]), dtype=work_dtype, device=device)
        for r in range(d):
            ai = Ai[:, r].unsqueeze(1)           # (ki, 1)
            bj = Aj_blk[:, r].unsqueeze(0)       # (1, cb)
            R *= torch.minimum(ai, bj)           # broadcasted outer-min → (ki, cb)
        # Accumulate in float64 for summation accuracy
        total += R.to(torch.float64).sum()

    return total

def _build_S_matrix(point_processes: list[torch.Tensor],
                    *,
                    device: torch.device | None = None,
                    work_dtype: torch.dtype = torch.float32,
                    block_cols: int = 8192) -> torch.Tensor:
    n = len(point_processes)
    S = torch.zeros((n, n), dtype=torch.float64)
    
    # FIX: Do NOT precompute complements here. 
    # The inner function _pairwise_integral_FiFj_outermin does the (1-P) calculation.
    # We just ensure they are on the correct device/dtype if needed, or pass as is.
    
    prog = tqdm(range(n * (n + 1) // 2), desc="Building S matrix")

    for i in range(n):
        # Pass the raw point process P_i
        Pi = point_processes[i]
        
        for j in range(i, n):
            Pj = point_processes[j]
            
            val = _pairwise_integral_FiFj_outermin(Pi, Pj,
                                                   block_cols=block_cols,
                                                   device=device,
                                                   work_dtype=work_dtype)
            S[i, j] = val.item()
            if i != j:
                S[j, i] = S[i, j]
            prog.update(1)
            
    return S


def _center_gram_from_S(S: torch.Tensor) -> torch.Tensor:
    """
    K = H S H with H = I - 11^T/n, equivalent to elementwise centering:
      K_ij = S_ij - row_mean_i - col_mean_j + grand_mean
      K_ij = <F_{Delta i}, F_{Delta j}>

    Shapes:
      - S: (n, n)
      - returns K: (n, n)
    """
    n = S.size(0)
    row_mean = S.mean(dim=1, keepdim=True)           # (n, 1)
    col_mean = S.mean(dim=0, keepdim=True)           # (1, n)
    grand_mean = S.mean()                            # []
    K = S - row_mean - col_mean + grand_mean
    # Symmetrize for numerical stability
    return 0.5 * (K + K.T)

def pppca(
    point_processes: PointProcessesND,
    Jmax: int,
) -> Dict[str, object]:
    """
    Multivariate dual-Gram PCA for point processes on [0,1]^d.

    Input:
      - point_processes: list length n, each is a tensor of shape (k_i, d) with entries in [0,1]
      - Jmax: number of leading components

    Output:
      - 'eigenval': list[float], operator eigenvalues λ (length Jmax)
      - 'scores':  DataFrame (n x Jmax), scores s_{iℓ} = sqrt(nλ_ℓ) c_i^{(ℓ)}
      - 'coeff':   ndarray (n x Jmax), Gram eigenvectors c^{(ℓ)} (columns)
      - 'eigenfun': a callable eval(x) returning η at x, where x is (m_eval, d)
                    Note: evaluation is on demand via cumulative counts relative to x

    Shapes in a small 2D example:
      - Suppose n=2 processes:
          P1 = tensor([[0.2, 0.5], [0.7, 0.1]]) shape (2,2)
          P2 = tensor([[0.1, 0.4], [0.6, 0.8], [0.3, 0.2]]) shape (3,2)
        S is (2,2); K is (2,2); eigenval length Jmax; coeff is (2, Jmax); scores is (2, Jmax)
    """
    if Jmax < 1:
        raise ValueError("Jmax must be a positive integer")
    if not point_processes:
        raise ValueError("point_processes must not be empty")

    n = len(point_processes)

    # 1) Build uncentered second-moment matrix S via closed-form orthant integrals
    S = _build_S_matrix(point_processes)  # (n, n)

    # 2) Center to covariance Gram K = H S H
    K = _center_gram_from_S(S)            # (n, n)

    # 3) Eigendecomposition of K
    evals_K, evecs_K = torch.linalg.eigh(K)          # ascending
    order = torch.argsort(evals_K, descending=True)
    evals_K = evals_K[order]                         # (n,)
    evecs_K = evecs_K[:, order]                      # (n, n)

    # 4) Map to operator eigenvalues: λ = μ / n, keep positive ones
    op_evals = evals_K / float(n)                    # (n,)
    if Jmax > op_evals.numel():
        raise ValueError("Jmax exceeds the available number of components")
    pos = (op_evals > 0)
    if pos.sum().item() < Jmax:
        raise ValueError("Not enough positive eigenvalues; reduce Jmax or check data")
    idx = torch.nonzero(pos, as_tuple=False).flatten()[:Jmax]
    eigenval = op_evals[idx].contiguous()            # (Jmax,)
    C = evecs_K[:, idx].contiguous()                 # (n, Jmax), c^{(ℓ)} columns
    scale = torch.sqrt(float(n) * eigenval)          # (Jmax,)

    # 5) Scores: s_{iℓ} = sqrt(n λ_ℓ) c_i^{(ℓ)}
    scores = (C * scale.unsqueeze(0))                # (n, Jmax)

    # 6) Provide an evaluator for eigenfunctions on-demand
    #    η_ℓ(x) = (1/sqrt(nλ_ℓ)) Σ_i c_i^{(ℓ)} (F_i(x) - F̄(x)), with
    #    F_i(x) = # { p in P_i : p <= x coordwise }, and F̄(x) = (1/n) Σ_i F_i(x)
    processes_fp64 = [Pi.to(dtype=torch.float64) for Pi in point_processes]
    def eigenfun_eval(X: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Evaluate all Jmax eigenfunctions at query locations X.

        Input:
          - X: (m_eval, d) array/tensor with entries in [0,1]
        Output:
          - values: (m_eval, Jmax) numpy array with η_ℓ(X_r) per column ℓ
        """
        X_t = torch.as_tensor(X, dtype=torch.float64)           # (m_eval, d)
        m_eval, d = X_t.shape
        # Compute F_i(X) for all i: counts of points <= X (coordwise)
        # Vectorized per i; complexity ~ sum_i k_i * m_eval
        Fi_list = []
        for Pi in processes_fp64:
            if Pi.numel() == 0:
                Fi = torch.zeros((m_eval,), dtype=torch.float64)
            else:
                # Pi: (k_i, d), X_t: (m_eval, d) -> comp: (k_i, m_eval, d)
                comp = (Pi[:, None, :] <= X_t[None, :, :])      # boolean
                le_all = comp.all(dim=-1)                       # (k_i, m_eval)
                Fi = le_all.sum(dim=0).to(dtype=torch.float64)  # (m_eval,)
            Fi_list.append(Fi)
        F_stack = torch.stack(Fi_list, dim=1)                   # (m_eval, n)
        Fbar = F_stack.mean(dim=1, keepdim=True)                # (m_eval, 1)
        FDelta = F_stack - Fbar                                 # (m_eval, n)

        # η(X): (m_eval, Jmax) = (1/sqrt(nλ)) * FDelta @ C
        Eta = (FDelta @ C) / scale.unsqueeze(0)                 # (m_eval, Jmax)
        return Eta.cpu().numpy()

    # Prepare outputs
    eigenval_np = eigenval.cpu().numpy()
    scores_df = pd.DataFrame(
        scores.cpu().numpy(),
        columns=[f"axis{i}" for i in range(1, idx.numel() + 1)],
    )
    coeff_np = C.cpu().numpy()  # c^{(ℓ)} columns

    return {
        "eigenval": eigenval_np.tolist(),
        "scores": scores_df,
        "coeff": coeff_np,
        "eigenfun": eigenfun_eval,  # call with X (m_eval, d)
    }

# Plotting examples
import matplotlib.pyplot as plt

# Adjust dimension here (d≥1)
d = 4

def _sample_point_processes(
    num_processes: int = 100,
    base_events: int = 20,
    jitter: float = 0.05,
    seed: int = 1234,
) -> List[torch.Tensor]:
    """Generate synthetic *d*-dim point processes for demonstration.
    Output: list of (num_points, d) tensors; each entry is a point process."""
    torch.manual_seed(seed)
    base_grid = torch.linspace(0.1, 0.9, base_events, dtype=torch.float64)
    processes: List[torch.Tensor] = []
    for _ in range(num_processes):
        # Sample number of events (stochastically drop some events)
        mask = torch.rand(base_events) > 0.25
        if not mask.any():
            mask[torch.randint(0, base_events, (1,), dtype=torch.long)] = True
        selected = base_grid[mask]
        # Make each event a random [0,1]^d point with base grid in first coordinate
        coords = [selected]
        for dd in range(1, d):
            coords.append(torch.rand_like(selected))
        events = torch.stack(coords, dim=1)  # (num_events, d)
        noise = (torch.rand_like(events) - 0.5) * jitter
        events = torch.clamp(events + noise, min=0.0, max=1.0)
        processes.append(events)
    return processes

def plot_sample_processes(processes, d, num_plot=6):
    import matplotlib.pyplot as plt
    if d == 3:
        from mpl_toolkits.mplot3d import Axes3D  # Import in function for 3D plot
    plt.figure(figsize=(6, 4))
    if d == 1:
        for i in range(min(num_plot, len(processes))):
            X = processes[i].cpu().numpy()
            plt.plot(X[:, 0], i + np.zeros_like(X[:, 0]), 'o-', label=f"Process {i+1}")
        plt.xlabel("x")
        plt.ylabel("Process Index")
        plt.title("Sample 1D Point Processes")
        plt.yticks(range(num_plot))
        plt.legend(loc="best")
    elif d == 2:
        for i in range(min(num_plot, len(processes))):
            X = processes[i].cpu().numpy()
            plt.plot(X[:, 0], X[:, 1], 'o-', label=f"Process {i+1}")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Sample 2D Point Processes")
        plt.legend(loc="best")
    elif d == 3:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(min(num_plot, len(processes))):
            X = processes[i].cpu().numpy()
            ax.plot(X[:, 0], X[:, 1], X[:, 2], 'o-', label=f"Process {i+1}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        ax.set_title("Sample 3D Point Processes")
        ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

def main():
    """Run PPPCA_dual_multivariate on synthetic data and display diagnostic plots (first components)."""

    Jmax = 3
    sample_processes = _sample_point_processes(num_processes=25, base_events=6, seed=42)

    # Plot some sample point processes
    plot_sample_processes(sample_processes, d, num_plot=6)

    results = pppca(sample_processes, Jmax=Jmax)
    print("Eigenvalues:", results["eigenval"])
    print("Scores head:\n", results["scores"].head())
    print("Gram eigenvector coeff (first few):\n", results["coeff"][:5, :])

    # Plot leading eigenfunctions for d = 1, 2, or 3
    if d == 1:
        grid_lin = np.linspace(0, 1, 200).reshape(-1, 1)  # (200, 1)
        eta_vals = results["eigenfun"](grid_lin)  # shape (200, Jmax)
        plt.figure(figsize=(10, 4))
        for j in range(min(Jmax, 3)):
            plt.plot(grid_lin[:, 0], eta_vals[:, j], label=f"Eigenfunction {j+1}")
        plt.xlabel("x")
        plt.ylabel("Eigenfun value")
        plt.title("Leading Empirical Eigenfunctions (1D)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    elif d == 2:
        grid_lin = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(grid_lin, grid_lin)
        X_flat = np.stack([X.ravel(), Y.ravel()], axis=1)  # shape (10000, 2)
        eta_vals = results["eigenfun"](X_flat)  # (10000, Jmax)
        plt.figure(figsize=(12, 4))
        for j in range(min(Jmax, 3)):
            plt.subplot(1, min(Jmax, 3), j + 1)
            plt.contourf(X, Y, eta_vals[:, j].reshape(100, 100), levels=20, cmap='RdBu')
            plt.colorbar()
            plt.title(f"Eigenfunction {j+1}")
            plt.xlabel("x1")
            plt.ylabel("x2")
        plt.suptitle("Leading Empirical Eigenfunctions (2D contour)")
        plt.tight_layout()
        plt.show()
    elif d == 3:
        from mpl_toolkits.mplot3d import Axes3D
        grid_lin = np.linspace(0, 1, 30)
        X, Y, Z = np.meshgrid(grid_lin, grid_lin, grid_lin)
        XYZ_flat = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (27000, 3)
        eta_vals = results["eigenfun"](XYZ_flat)  # (27000, Jmax)
        # For 3D, show scatter for leading component
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        val = eta_vals[:, 0]
        sel = np.abs(val) > np.percentile(np.abs(val), 95)
        p = ax.scatter(XYZ_flat[sel, 0], XYZ_flat[sel, 1], XYZ_flat[sel, 2], c=val[sel], cmap='RdBu', s=8)
        fig.colorbar(p)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        ax.set_title("Leading Eigenfunction (top values, 3D)")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()