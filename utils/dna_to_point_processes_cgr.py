# https://www.perplexity.ai/search/i-developed-a-framework-for-do-JRMXHoKVSg6IA9sehIT3BQ#1

import torch
from typing import List

def dna_to_point_processes(sequences: List[str]) -> List[torch.Tensor]:
    """
    Converts DNA sequences into 2D point processes using Chaos Game Representation (CGR).
    
    Instead of a Time-vs-Type mapping, this projects the sequence into a 
    spatial cloud in the unit square [0,1]^2. This provides shift-invariance:
    a specific motif (like a promoter) will always generate points in the same 
    spatial sub-region, regardless of where it appears in the sequence.

    Mapping (Standard CGR):
      A -> (0, 0)
      C -> (0, 1)
      G -> (1, 1)
      T -> (1, 0)
      
    Parameters
    ----------
    sequences : List[str]
        List of DNA strings.
        
    Returns
    -------
    List[torch.Tensor]
        List of tensors. Each tensor has shape (N_events, 2).
        Column 0 is the CGR X-coordinate.
        Column 1 is the CGR Y-coordinate.
    """
    # Define corner coordinates for the CGR unit square
    # Using tuples for faster lookup and unpacking in the loop
    corners = {
        # 'A': (0.0, 0.0),
        # 'C': (0.0, 1.0),
        # # 'G': (1.0, 1.0), # acgt clockwise
        # # 'T': (1.0, 0.0)
        # 'T': (1.0, 1.0), # actg clockwise
        # 'G': (1.0, 0.0)
        'A': (0.0, 0.5),
        'C': (0.5, 1.0),
        'G': (1.0, 0.5),
        'T': (0.5, 0.0)
    }
    
    processes = []
    
    for seq in sequences:
        # Filter strictly for valid bases to avoid breaking the trajectory
        # In CGR, the position of point N depends on points 0...N-1.
        
        # Start at the center of the square (0.5, 0.5)
        current_x, current_y = 0.5, 0.5
        points = []
        
        for char in seq:
            upper_char = char.upper()
            
            if upper_char in corners:
                corner_x, corner_y = corners[upper_char]
                
                # CGR Update Rule: P_new = 0.5 * (P_old + Corner)
                current_x = 0.5 * (current_x + corner_x)
                current_y = 0.5 * (current_y + corner_y)
                
                points.append([current_x, current_y])
        
        # Stack points into a tensor of shape (N_events, 2)
        if points:
            process_tensor = torch.tensor(points, dtype=torch.float32)
        else:
            # Handle empty or fully invalid sequences
            process_tensor = torch.zeros((0, 2), dtype=torch.float32)
            
        processes.append(process_tensor)
        
    return processes

# Example usage:
if __name__ == "__main__":
    # Sample DNA sequences
    dna_sequences = [
        "ACGT",
        "AAAA",
        "AC N GT", # Gap is skipped, trajectory continues from previous valid point
        "T"
    ]
    
    # Convert to point processes
    processes = dna_to_point_processes(dna_sequences)
    
    print(f"Number of processes: {len(processes)}")
    
    # Validation
    for i, (seq, proc) in enumerate(zip(dna_sequences, processes)):
        print(f"\nSequence {i} ('{seq}'):")
        print(f"Shape: {proc.shape}")
        # Show first few points to verify CGR logic
        print(proc)
