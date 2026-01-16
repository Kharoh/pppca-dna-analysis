from typing import List, Dict
from utils.dna_to_point_processes_2d import dna_to_point_processes
from pppca import pppca, _pairwise_integral_FiFj_outermin

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from utils.load_dna_sequences_from_csv import load_dna_sequences_from_csv

class DNAPPCA:
    """
    DNA Point Process PCA wrapper using EXACT analytical integration.
    """
    
    def __init__(self, Jmax: int):
        self.Jmax = Jmax
        self.pca_results = None
        self.train_processes = None
        
        # Statistics for centering new data
        self.S_row_means = None  # Mean inner product of each train sample vs all train
        self.S_grand_mean = None # Global mean inner product
        
    def fit(self, train_sequences: List[str]):
        """
        Fit PCA on training DNA sequences.
        """
        # 1. Convert DNA to point processes
        self.train_processes = dna_to_point_processes(train_sequences)
        
        # Ensure float32 for speed, or float64 for precision if needed
        self.train_processes = [p.float() for p in self.train_processes]

        # 2. Fit PPPCA
        # This computes the eigenvalues and coefficients
        self.pca_results = pppca(self.train_processes, Jmax=self.Jmax)
        
        # 3. Compute statistics needed for projecting NEW data.
        # We need the S matrix (uncentered Gram matrix) statistics.
        # The pppca function computes S internally but might not return it directly.
        # We can reconstruct the necessary means efficiently.
        
        # Note: If your pppca library returns 'S' or 'K', we could use that.
        # Assuming standard output, we re-compute the row means of S roughly or
        # ideally, we modify pppca to return S. 
        # For now, let's re-calculate the S statistics to be safe and independent.
        
        n = len(self.train_processes)
        S_row_sums = torch.zeros(n, dtype=torch.float64)
        
        print("Precomputing training statistics for projection...")
        # We only need row sums to center future test data
        for i in tqdm(range(n), desc="Training Integrals"):
            Pi = self.train_processes[i]
            row_sum = 0.0
            for j in range(n):
                Pj = self.train_processes[j]
                # Optimization: S is symmetric, but loop overhead is low compared to calc
                val = _pairwise_integral_FiFj_outermin(Pi, Pj)
                row_sum += val
            S_row_sums[i] = row_sum
            
        self.S_row_means = S_row_sums / n
        self.S_grand_mean = S_row_sums.sum() / (n * n)
        
        return self

    def transform(self, test_sequences: List[str]) -> np.ndarray:
        """
        Project test sequences using analytical inner products.
        """
        if self.pca_results is None:
            raise ValueError("Must call fit() before transform()")
            
        test_processes = dna_to_point_processes(test_sequences)
        test_processes = [p.float() for p in test_processes]
        
        n_test = len(test_processes)
        n_train = len(self.train_processes)
        
        # Retrieve PCA components
        # C: eigenvectors of K (n_train x Jmax)
        C = torch.tensor(self.pca_results['coeff'], dtype=torch.float64) 
        eigenvals = torch.tensor(self.pca_results['eigenval'], dtype=torch.float64)
        
        # Scale factor for projection: 1 / sqrt(n * lambda)
        scale = 1.0 / torch.sqrt(n_train * eigenvals)
        
        scores = np.zeros((n_test, self.Jmax))
        
        print(f"Projecting {n_test} test sequences...")
        
        # For each test sequence, we compute its vector of inner products 
        # against all training sequences: k_new = [ <F_new, F_train_1>, ... ]
        for t_idx, P_test in enumerate(tqdm(test_processes, desc="Test Projection")):
            
            k_new = torch.zeros(n_train, dtype=torch.float64)
            
            # Compute integrals against all training samples
            for i, P_train in enumerate(self.train_processes):
                val = _pairwise_integral_FiFj_outermin(P_test, P_train)
                k_new[i] = val
                
            # Center the kernel vector (Double Centering)
            # k_centered = k_new - mean(k_new) - S_row_means + S_grand_mean
            mean_k_new = k_new.mean()
            k_centered = k_new - mean_k_new - self.S_row_means + self.S_grand_mean
            
            # Project onto eigenvectors
            # score = k_centered @ C * scale
            proj = torch.matmul(k_centered, C) * scale
            scores[t_idx] = proj.numpy()
            
        return scores

    def fit_transform(self, train_sequences: List[str]) -> np.ndarray:
        self.fit(train_sequences)
        return self.pca_results['scores'].values


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, matthews_corrcoef
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class DNAClassifierNN(nn.Module):
    """
    Simple feedforward neural network for DNA sequence classification.
    """
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32], num_classes: int = 2, dropout: float = 0.3):
        """
        Parameters
        ----------
        input_dim : int
            Number of input features (PCA components)
        hidden_dims : list, default=[64, 32]
            List of hidden layer dimensions
        num_classes : int, default=2
            Number of output classes
        dropout : float, default=0.3
            Dropout probability for regularization
        """
        super(DNAClassifierNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_neural_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    hidden_dims: list = [64, 32],
    num_classes: int = 2,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    dropout: float = 0.3,
    random_seed: int = 42,
    device: str = None
):
    """
    Train a neural network classifier on PCA scores.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features (PCA scores)
    y_train : np.ndarray
        Training labels
    X_val : np.ndarray, optional
        Validation features
    y_val : np.ndarray, optional
        Validation labels
    hidden_dims : list, default=[64, 32]
        Hidden layer dimensions
    num_classes : int, default=2
        Number of classes
    epochs : int, default=100
        Number of training epochs
    batch_size : int, default=32
        Batch size for training
    learning_rate : float, default=0.001
        Learning rate for optimizer
    dropout : float, default=0.3
        Dropout probability
    random_seed : int, default=42
        Random seed for reproducibility
    device : str, optional
        Device to use ('cuda' or 'cpu'). If None, auto-detect.
    
    Returns
    -------
    model : DNAClassifierNN
        Trained neural network model
    history : dict
        Training history with losses and accuracies
    """
    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"      Using device: {device}")
    
    # Prepare data
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if X_val is not None and y_val is not None:
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = DNAClassifierNN(input_dim, hidden_dims, num_classes, dropout).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item() * batch_X.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_loss /= val_total
            val_acc = val_correct / val_total
            
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if (epoch + 1) % 20 == 0:
                print(f"      Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            if (epoch + 1) % 20 == 0:
                print(f"      Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    
    return model, history

def predict_neural_network(model, X: np.ndarray, device: str = None):
    """
    Make predictions using trained neural network.
    
    Parameters
    ----------
    model : DNAClassifierNN
        Trained model
    X : np.ndarray
        Input features
    device : str, optional
        Device to use
    
    Returns
    -------
    predictions : np.ndarray
        Predicted class labels
    probabilities : np.ndarray
        Class probabilities
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    
    return predictions, probabilities



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef


def run_dna_analysis_pipeline(
    train_path: str,
    test_path: str,
    n_train: int,
    n_test: int,
    Jmax: int = 5,
    output_dir: str = '.',
    random_seed: int = 42,
    use_nn: bool = True,
    nn_hidden_dims: list = [64, 32],
    nn_epochs: int = 100,
    nn_batch_size: int = 32,
    nn_learning_rate: float = 0.001
):
    """
    Complete workflow to load data, fit PPPCA, project test data, and train classifiers.

    Parameters
    ----------
    train_path : str
        Path to training CSV (must have 'sequence' and 'label' columns).
    test_path : str
        Path to testing CSV (must have 'sequence' and 'label' columns).
    n_train : int
        Number of sequences to sample from the training set.
    n_test : int
        Number of sequences to sample from the test set.
    Jmax : int, default=5
        Number of principal components to compute.
    output_dir : str, default='.'
        Directory to save plots and results.
    random_seed : int, default=42
        Seed for reproducibility.
    use_nn : bool, default=True
        Whether to train neural network classifier in addition to Random Forest.
    nn_hidden_dims : list, default=[64, 32]
        Hidden layer dimensions for neural network.
    nn_epochs : int, default=100
        Number of epochs for neural network training.
    nn_batch_size : int, default=32
        Batch size for neural network training.
    nn_learning_rate : float, default=0.001
        Learning rate for neural network.
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Starting DNA Analysis Pipeline ---")
    print(f"Train Source: {train_path}")
    print(f"Test Source:  {test_path}")
    
    # 1. Load and Subsample Training Data
    print(f"\\n[1/6] Loading Training Data...")
    full_train_seqs, full_train_labels = load_dna_sequences_from_csv(train_path)
    
    # Subsample training data
    if n_train < len(full_train_seqs):
        print(f"      Subsampling {n_train} sequences from {len(full_train_seqs)} total.")
        rng = np.random.RandomState(random_seed)
        indices = rng.choice(len(full_train_seqs), size=n_train, replace=False)
        train_seqs = [full_train_seqs[i] for i in indices]
        y_train = full_train_labels[indices]
    else:
        print(f"      Using all {len(full_train_seqs)} sequences (requested {n_train}).")
        train_seqs = full_train_seqs
        y_train = full_train_labels

    # 2. Load and Subsample Test Data
    print(f"\\n[2/6] Loading Test Data...")
    full_test_seqs, full_test_labels = load_dna_sequences_from_csv(test_path)
    
    # Subsample test data
    if n_test < len(full_test_seqs):
        print(f"      Subsampling {n_test} sequences from {len(full_test_seqs)} total.")
        rng = np.random.RandomState(random_seed + 1)
        indices = rng.choice(len(full_test_seqs), size=n_test, replace=False)
        test_seqs = [full_test_seqs[i] for i in indices]
        y_test = full_test_labels[indices]
    else:
        print(f"      Using all {len(full_test_seqs)} sequences (requested {n_test}).")
        test_seqs = full_test_seqs
        y_test = full_test_labels

    # 3. Fit PPPCA on Training Data
    print(f"\\n[3/6] Fitting PPPCA (Jmax={Jmax})...")
    pca_model = DNAPPCA(Jmax=Jmax)
    
    train_scores = pca_model.fit_transform(train_seqs)
    eigenvals = np.array(pca_model.pca_results['eigenval'])
    print(f"      Eigenvalues: {np.array2string(eigenvals, precision=4, separator=', ')}")

    # 4. Project Test Data
    print(f"\\n[4/6] Projecting Test Data...")
    test_scores = pca_model.transform(test_seqs)

    # 5. Train Random Forest Classifier
    print(f"\\n[5/6] Training Random Forest and Evaluating...")
    
    rf_model = RandomForestClassifier(n_estimators=200, random_state=random_seed, max_depth=5)
    rf_model.fit(train_scores, y_train)
    
    y_test_pred_rf = rf_model.predict(test_scores)
    acc_rf = accuracy_score(y_test, y_test_pred_rf)
    mcc_rf = matthews_corrcoef(y_test, y_test_pred_rf)
    
    print(f"\\n--- Random Forest Results ---")
    print(f"Test Set Accuracy: {acc_rf:.4f}")
    print(f"Matthews Correlation Coefficient: {mcc_rf:.4f}")
    print("\\nClassification Report:")
    print(classification_report(y_test, y_test_pred_rf))

    # 6. Train Neural Network Classifier
    nn_model = None
    nn_history = None
    y_test_pred_nn = None
    acc_nn = None
    mcc_nn = None
    
    if use_nn:
        print(f"\\n[6/6] Training Neural Network Classifier...")
        num_classes = len(np.unique(y_train))
        
        nn_model, nn_history = train_neural_network(
            X_train=train_scores,
            y_train=y_train,
            X_val=test_scores,
            y_val=y_test,
            hidden_dims=nn_hidden_dims,
            num_classes=num_classes,
            epochs=nn_epochs,
            batch_size=nn_batch_size,
            learning_rate=nn_learning_rate,
            random_seed=random_seed
        )
        
        y_test_pred_nn, _ = predict_neural_network(nn_model, test_scores)
        acc_nn = accuracy_score(y_test, y_test_pred_nn)
        mcc_nn = matthews_corrcoef(y_test, y_test_pred_nn)
        
        print(f"\\n--- Neural Network Results ---")
        print(f"Test Set Accuracy: {acc_nn:.4f}")
        print(f"Matthews Correlation Coefficient: {mcc_nn:.4f}")
        print("\\nClassification Report:")
        print(classification_report(y_test, y_test_pred_nn))

    # --- Visualization ---
    
    # 1. Score Plot (PC1 vs PC2) - now with both RF and NN predictions if available
    fig, axes = plt.subplots(1, 2 if use_nn else 1, figsize=(18 if use_nn else 10, 5))
    if not use_nn:
        axes = [axes]
    
    # RF predictions
    axes[0].scatter(train_scores[:, 0], train_scores[:, 1], 
                    c=y_train, cmap='coolwarm', alpha=0.3, label='Train', marker='o', s=30)
    axes[0].scatter(test_scores[:, 0], test_scores[:, 1], 
                    c=y_test, cmap='coolwarm', alpha=0.9, edgecolors='black', label='Test', marker='s', s=50)
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].set_title(f'PPPCA Scores\\nRandom Forest Acc: {acc_rf:.2%}')
    axes[0].legend()
    
    if use_nn:
        axes[1].scatter(train_scores[:, 0], train_scores[:, 1], 
                        c=y_train, cmap='coolwarm', alpha=0.3, label='Train', marker='o', s=30)
        axes[1].scatter(test_scores[:, 0], test_scores[:, 1], 
                        c=y_test, cmap='coolwarm', alpha=0.9, edgecolors='black', label='Test', marker='s', s=50)
        axes[1].set_xlabel('Principal Component 1')
        axes[1].set_ylabel('Principal Component 2')
        axes[1].set_title(f'PPPCA Scores\\nNeural Network Acc: {acc_nn:.2%}')
        axes[1].legend()
    
    save_path = os.path.join(output_dir, 'pca_scores_projection.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\\nSaved score plot to: {save_path}")
    plt.close()

    # 2. Confusion Matrices
    fig, axes = plt.subplots(1, 2 if use_nn else 1, figsize=(12 if use_nn else 6, 5))
    if not use_nn:
        axes = [axes]
    
    cm_rf = confusion_matrix(y_test, y_test_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'Random Forest\\nAcc: {acc_rf:.2%}, MCC: {mcc_rf:.3f}')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    if use_nn:
        cm_nn = confusion_matrix(y_test, y_test_pred_nn)
        sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Greens', ax=axes[1])
        axes[1].set_title(f'Neural Network\\nAcc: {acc_nn:.2%}, MCC: {mcc_nn:.3f}')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
    
    cm_path = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrices to: {cm_path}")
    plt.close()
    
    # 3. Training History (if NN was trained)
    if use_nn and nn_history is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs_range = range(1, len(nn_history['train_loss']) + 1)
        
        # Loss plot
        axes[0].plot(epochs_range, nn_history['train_loss'], label='Train Loss', linewidth=2)
        if nn_history['val_loss']:
            axes[0].plot(epochs_range, nn_history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Neural Network Training Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(epochs_range, nn_history['train_acc'], label='Train Acc', linewidth=2)
        if nn_history['val_acc']:
            axes[1].plot(epochs_range, nn_history['val_acc'], label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Neural Network Training Accuracy')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        history_path = os.path.join(output_dir, 'nn_training_history.png')
        plt.savefig(history_path, dpi=300, bbox_inches='tight')
        print(f"Saved NN training history to: {history_path}")
        plt.close()

    return {
        'pca_model': pca_model,
        'rf_classifier': rf_model,
        'nn_classifier': nn_model,
        'nn_history': nn_history,
        'train_scores': train_scores,
        'test_scores': test_scores,
        'rf_accuracy': acc_rf,
        'rf_mcc': mcc_rf,
        'nn_accuracy': acc_nn,
        'nn_mcc': mcc_nn
    }


# --- Example Usage ---
if __name__ == "__main__":
    TRAIN_CSV = "./GUE_v2/GUE/prom/prom_core_tata/train.csv" 
    TEST_CSV = "./GUE_v2/GUE/prom/prom_core_tata/test.csv"

    if not os.path.exists(TRAIN_CSV):
        print("Note: Example CSVs not found. Please provide valid paths.")
    else:
        results = run_dna_analysis_pipeline(
            train_path=TRAIN_CSV,
            test_path=TEST_CSV,
            n_train=1000,
            n_test=500,
            Jmax=10,
            output_dir='analysis_results',
            use_nn=True,
            nn_hidden_dims=[16, 4],
            nn_epochs=100,
            nn_batch_size=32,
            nn_learning_rate=0.001
        )
