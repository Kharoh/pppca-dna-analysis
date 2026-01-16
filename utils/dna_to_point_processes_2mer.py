import torch
from typing import List


def dna_to_point_processes(sequences: List[str]) -> List[torch.Tensor]:
    """
    Converts DNA sequences into 16D point processes using overlapping 2-mers (dinucleotides).

    Representation:
      - For each overlapping pair seq[i:i+2] (stride 1), create one event (row).
      - The event time/position t is written into exactly one of 16 dimensions,
        corresponding to that dinucleotide; all other dims are 0.

    Dimension indexing:
      - Map bases A,C,G,T -> 0,1,2,3
      - For pair XY, dim_idx = 4 * idx(X) + idx(Y)
        so the order is: AA, AC, AG, AT, CA, CC, ..., TT (16 total).

    Time normalization:
      - If a sequence has L characters, it has N = L-1 overlapping 2-mers.
      - The i-th 2-mer (i in [0, N-1]) gets t = (i+1)/N, so the last event is at 1.0.

    Unknown characters (e.g., 'N') in either position cause that event to be all-zeros.

    Parameters
    ----------
    sequences : List[str]
        List of DNA strings.

    Returns
    -------
    List[torch.Tensor]
        List of tensors, each of shape (max(L-1, 0), 16).
    """
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    processes = []

    for seq in sequences:
        L = len(seq)
        n_pairs = max(L - 1, 0)

        # Handle sequences too short to form a pair
        if n_pairs == 0:
            processes.append(torch.zeros((0, 16), dtype=torch.float32))
            continue

        process = torch.zeros((n_pairs, 16), dtype=torch.float32)

        for i in range(n_pairs):
            pair = seq[i:i + 2].upper()
            b0, b1 = pair[0], pair[1]

            # Normalized position in [0, 1], last event exactly at 1.0
            t = (i + 1) / n_pairs
            # Optional if you want to avoid boundary issues:
            # t = t * 0.9999

            if b0 in base_map and b1 in base_map:
                dim_idx = 4 * base_map[b0] + base_map[b1]
                process[i, dim_idx] = t
            # else: leave as all-zeros (no event)

        processes.append(process)

    return processes
