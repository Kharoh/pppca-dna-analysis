import torch
from typing import List

def dna_to_point_processes(sequences: List[str]) -> List[torch.Tensor]:
    """
    Converts DNA sequences into 4D point processes.
    
    Instead of a 5D representation (time, one_hot_A, ...), this uses a 4D representation
    where the time/position value is placed directly into the dimension corresponding
    to the nucleotide base.
    
    Mapping:
      A -> Dimension 0
      C -> Dimension 1
      G -> Dimension 2
      T -> Dimension 3
      
    Example:
      Sequence: "AC" (Length 2)
      t_0 = 1/2 = 0.5 (Base A) -> Point: [0.5, 0.0, 0.0, 0.0]
      t_1 = 2/2 = 1.0 (Base C) -> Point: [0.0, 1.0, 0.0, 0.0]
      
    Parameters
    ----------
    sequences : List[str]
        List of DNA strings.
        
    Returns
    -------
    List[torch.Tensor]
        List of tensors, each shape (L, 4).
    """
    # Mapping for base to dimension index
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    processes = []
    
    for seq in sequences:
        L = len(seq)
        
        # Handle empty sequences
        if L == 0:
            processes.append(torch.zeros((0, 4), dtype=torch.float32))
            continue
            
        # Initialize tensor of zeros (L, 4)
        # We use float32 for storage, pppca will cast to float64 if needed
        process = torch.zeros((L, 4), dtype=torch.float32)
        
        for i, char in enumerate(seq):
            # Calculate normalized position (time)
            # Using (i+1)/L puts the last base exactly at 1.0
            # To avoid the 1.0 integration issue in pppca, we can use a tiny epsilon scaling
            # or simply rely on the fact that only ONE dimension is 1.0, 
            # so the product (1 - x_j) will not be zero for the zero-dims.
            # However, safer to map to [0, 0.9999] if pppca kernel is strict.
            
            t = (i + 1) / L 
            
            # Optional: Clamp t slightly to avoid boundary issues if strictly needed
            # t = t * 0.9999 
            
            upper_char = char.upper()
            if upper_char in base_map:
                dim_idx = base_map[upper_char]
                process[i, dim_idx] = t
            # Characters like 'N' are left as all-zeros (effectively no event)
            
        processes.append(process)
        
    return processes


# Example usage:
if __name__ == "__main__":
    # Sample DNA sequences (all length 4)
    dna_sequences = [
        "ACGT",
        "ACGT",
        "TTTT",
        "AAAA",
        "CGCG"
    ]
    
    # Convert to point processes
    processes = dna_to_point_processes(dna_sequences)
    
    print(f"Number of processes: {len(processes)}")
    print(f"Shape of each process: {processes[0].shape}")
    print("\nFirst process (ACGT):")
    print(processes[0])
    print("\nThird process (TTTT):")
    print(processes[2])
