import torch
from typing import List

def dna_to_point_processes(sequences: List[str]) -> List[torch.Tensor]:
    """
    Converts DNA sequences into 2D point processes (Time, Space).
    
    Each nucleotide is represented as a point (t, x) where:
      - t: Normalized position in the sequence (0 < t <= 1)
      - x: Spatial coordinate representing the nucleotide identity.
    
    Spatial Mapping:
      A -> 0.0
      C -> 0.25
      G -> 0.50
      T -> 0.75
      
    Parameters
    ----------
    sequences : List[str]
        List of DNA strings.
        
    Returns
    -------
    List[torch.Tensor]
        List of tensors. Each tensor has shape (N_events, 2), where N_events
        is the number of valid nucleotides in the sequence.
        Column 0 is Time (t), Column 1 is Space (x).
    """
    # Mapping for base to spatial coordinate
    base_map = {'A': 0.0, 'C': 0.25, 'G': 0.50, 'T': 0.75}
    
    processes = []
    
    for seq in sequences:
        L = len(seq)
        
        # Handle empty sequences
        if L == 0:
            processes.append(torch.zeros((0, 2), dtype=torch.float32))
            continue
            
        points = []
        
        for i, char in enumerate(seq):
            upper_char = char.upper()
            
            if upper_char in base_map:
                # 1. Time coordinate: Normalized position
                t = (i + 1) / L
                
                # 2. Space coordinate: Nucleotide identity
                x = base_map[upper_char]
                
                points.append([t, x])
            
            # Note: 'N' or other characters are skipped entirely. 
            # In a 2D dense representation, we cannot leave them as "zeros" 
            # because (t, 0.0) would be interpreted as an Adenine.
        
        # Stack points into a tensor of shape (N_events, 2)
        if points:
            process_tensor = torch.tensor(points, dtype=torch.float32)
        else:
            process_tensor = torch.zeros((0, 2), dtype=torch.float32)
            
        processes.append(process_tensor)
        
    return processes

# Example usage:
if __name__ == "__main__":
    # Sample DNA sequences
    dna_sequences = [
        "ACGT",
        "AAAA",
        "AC N GT", # Sequence with a gap/unknown base
        "T"
    ]
    
    # Convert to point processes
    processes = dna_to_point_processes(dna_sequences)
    
    print(f"Number of processes: {len(processes)}")
    
    # Validation
    for i, (seq, proc) in enumerate(zip(dna_sequences, processes)):
        print(f"\nSequence {i} ('{seq}'):")
        print(f"Shape: {proc.shape}")
        print(proc)
