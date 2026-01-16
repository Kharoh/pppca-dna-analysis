import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

def load_dna_sequences_from_csv(
    filepath: str,
    sequence_col: str = 'sequence',
    label_col: str = 'label',
    validate: bool = True
) -> Tuple[List[str], np.ndarray]:
    """
    Load DNA sequences and labels from a CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file
    sequence_col : str, default='sequence'
        Name of the column containing DNA sequences
    label_col : str, default='label'
        Name of the column containing labels
    validate : bool, default=True
        If True, validate that sequences contain only valid DNA bases (A, C, G, T)
        and that all sequences have the same length
    
    Returns
    -------
    sequences : List[str]
        List of DNA sequences (uppercase)
    labels : np.ndarray
        Array of labels (integer encoded if necessary)
    
    Raises
    ------
    FileNotFoundError
        If the CSV file doesn't exist
    ValueError
        If required columns are missing or validation fails
    
    Examples
    --------
    >>> sequences, labels = load_dna_sequences_from_csv('data.csv')
    >>> print(f"Loaded {len(sequences)} sequences")
    >>> print(f"Label distribution: {np.bincount(labels)}")
    """
    # Read CSV file
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    # Check if required columns exist
    if sequence_col not in df.columns:
        raise ValueError(
            f"Column '{sequence_col}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )
    if label_col not in df.columns:
        raise ValueError(
            f"Column '{label_col}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Extract sequences and labels
    sequences = df[sequence_col].astype(str).str.upper().tolist()
    labels_raw = df[label_col].values
    
    # Remove any NaN values
    valid_mask = pd.notna(df[sequence_col]) & pd.notna(df[label_col])
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        print(f"Warning: Removing {n_invalid} rows with missing values")
        sequences = [seq for seq, valid in zip(sequences, valid_mask) if valid]
        labels_raw = labels_raw[valid_mask]
    
    # Validate sequences if requested
    if validate:
        valid_bases = set('ACGT')
        seq_lengths = []
        
        for i, seq in enumerate(sequences):
            # Check for invalid characters
            invalid_chars = set(seq) - valid_bases
            if invalid_chars:
                raise ValueError(
                    f"Sequence {i} contains invalid bases: {invalid_chars}. "
                    f"Valid bases are: A, C, G, T"
                )
            seq_lengths.append(len(seq))
        
        # Check uniform length
        if len(set(seq_lengths)) > 1:
            raise ValueError(
                f"All sequences must have the same length. "
                f"Found lengths: {sorted(set(seq_lengths))}"
            )
        
        print(f"✓ Validated {len(sequences)} sequences of length {seq_lengths[0]}")
    
    # Convert labels to integers if they're not already
    if labels_raw.dtype == object or labels_raw.dtype.kind == 'U':
        # String labels - encode them
        unique_labels = sorted(set(labels_raw))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        labels = np.array([label_map[label] for label in labels_raw])
        print(f"Encoded labels: {label_map}")
    else:
        labels = labels_raw.astype(int)
    
    print(f"Loaded {len(sequences)} sequences with {len(np.unique(labels))} unique labels")
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    return sequences, labels


def save_dna_sequences_to_csv(
    filepath: str,
    sequences: List[str],
    labels: np.ndarray,
    sequence_col: str = 'sequence',
    label_col: str = 'label'
) -> None:
    """
    Save DNA sequences and labels to a CSV file.
    
    Parameters
    ----------
    filepath : str
        Path where the CSV file will be saved
    sequences : List[str]
        List of DNA sequences
    labels : np.ndarray
        Array of labels
    sequence_col : str, default='sequence'
        Name for the sequence column
    label_col : str, default='label'
        Name for the label column
    
    Examples
    --------
    >>> save_dna_sequences_to_csv('output.csv', sequences, labels)
    """
    df = pd.DataFrame({
        sequence_col: sequences,
        label_col: labels
    })
    df.to_csv(filepath, index=False)
    print(f"Saved {len(sequences)} sequences to {filepath}")


# Example usage
if __name__ == "__main__":
    # Create example CSV file
    example_data = pd.DataFrame({
        'sequence': [
            'ACGTACGT',
            'GGGGCCCC',
            'ATATATATAT',  # Different length - will fail validation
            'TTTTAAAA',
            'CGCGCGCG'
        ],
        'label': [0, 1, 0, 1, 1]
    })
    example_data.to_csv('example_dna.csv', index=False)
    print("Created example_dna.csv")
    
    # Load with validation (will fail due to different lengths)
    try:
        sequences, labels = load_dna_sequences_from_csv(
            'example_dna.csv',
            validate=True
        )
    except ValueError as e:
        print(f"Validation error: {e}")
    
    # Create valid example
    valid_data = pd.DataFrame({
        'sequence': ['ACGTACGT', 'GGGGCCCC', 'TTTTAAAA', 'CGCGCGCG'],
        'label': [0, 1, 0, 1]
    })
    valid_data.to_csv('valid_dna.csv', index=False)
    
    # Load successfully
    sequences, labels = load_dna_sequences_from_csv('valid_dna.csv')
    print(f"\nFirst sequence: {sequences[0]}")
    print(f"Labels: {labels}")
