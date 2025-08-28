"""
Train CNN-LSTM Model for ECG Cardiovascular Disease Detection

This script trains the CNN-LSTM model on ECG data for cardiovascular disease detection.
It can be run from the command line with various parameters to customize the training process.

Example usage:
    python train_model.py --data_path data/ecg_dataset --epochs 50 --batch_size 32
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
import joblib
import datetime
import zipfile
import glob

# Add the parent directory to the path to import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_lstm_model import ECGModel, create_sample_data

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CNN-LSTM model for ECG analysis')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to the ECG dataset directory or ZIP file')
    parser.add_argument('--use_sample_data', action='store_true',
                        help='Use generated sample data instead of loading from file')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    
    # Model parameters
    parser.add_argument('--seq_length', type=int, default=1000,
                        help='ECG sequence length (default: 1000)')
    parser.add_argument('--n_features', type=int, default=1,
                        help='Number of ECG features/leads (default: 1)')
    parser.add_argument('--cnn_layers', type=int, default=3,
                        help='Number of CNN layers (default: 3)')
    parser.add_argument('--lstm_layers', type=int, default=2,
                        help='Number of LSTM layers (default: 2)')
    parser.add_argument('--dense_layers', type=int, default=2,
                        help='Number of dense layers (default: 2)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (default: 0.3)')
    parser.add_argument('--data_augmentation', action='store_true',
                        help='Apply data augmentation during training')
    
    # Output parameters
    parser.add_argument('--model_name', type=str, default='cnn_lstm_model',
                        help='Name for the saved model (default: cnn_lstm_model)')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save the model and results (default: models)')
    
    return parser.parse_args()

def load_data(args):
    """
    Load ECG data from the specified path or generate sample data.
    
    Args:
        args: Command line arguments
        
    Returns:
        X_train, X_val, y_train, y_val: Training and validation data
    """
    if args.use_sample_data:
        print("Generating sample ECG data...")
        X, y = create_sample_data(n_samples=1000, seq_length=args.seq_length, n_features=args.n_features)
    elif args.data_path is None:
        print("No data path provided and not using sample data. Exiting.")
        sys.exit(1)
    else:
        print(f"Loading ECG data from {args.data_path}...")
        
        # Check if the path is a ZIP file
        if args.data_path.endswith('.zip'):
            # Create a temporary directory to extract files
            temp_dir = os.path.join(os.path.dirname(args.data_path), 'temp_extracted')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Extract the ZIP file
            with zipfile.ZipFile(args.data_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Use the extracted directory as the data path
            data_path = temp_dir
        else:
            data_path = args.data_path
        
        # Load data based on the format
        # This is a simplified example - in a real application, you would need to handle
        # various data formats and structures
        
        # Try to find CSV files
        csv_files = glob.glob(os.path.join(data_path, '*.csv'))
        if csv_files:
            # Assume the first CSV file contains the data
            data = pd.read_csv(csv_files[0])
            
            # Assume the last column is the label and the rest are features
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
            
            # Reshape X to match the expected input shape for the model
            X = X.reshape(-1, args.seq_length, args.n_features)
        else:
            # Try to find NPY files
            npy_files = glob.glob(os.path.join(data_path, '*.npy'))
            if len(npy_files) >= 2:
                # Assume there are separate files for X and y
                X = np.load(npy_files[0])
                y = np.load(npy_files[1])
            else:
                print("Could not find suitable data files. Using sample data instead.")
                X, y = create_sample_data(n_samples=1000, seq_length=args.seq_length, n_features=args.n_features)
        
        # Clean up temporary directory if it was created
        if args.data_path.endswith('.zip'):
            import shutil
            shutil.rmtree(temp_dir)
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, random_state=42, stratify=y
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    return X_train, X_val, y_train, y_val

def apply_data_augmentation(X_train, y_train, augmentation_factor=2):
    """
    Apply data augmentation to the training data.
    
    Args:
        X_train: Training data
        y_train: Training labels
        augmentation_factor: Factor by which to increase the dataset size
        
    Returns:
        Augmented training data and labels
    """
    print("Applying data augmentation...")
    
    n_samples, seq_length, n_features = X_train.shape
    X_aug = np.zeros((n_samples * augmentation_factor, seq_length, n_features))
    y_aug = np.zeros(n_samples * augmentation_factor)
    
    # Copy original data
    X_aug[:n_samples] = X_train
    y_aug[:n_samples] = y_train
    
    # Apply augmentation techniques
    for i in range(1, augmentation_factor):
        for j in range(n_samples):
            idx = n_samples * i + j
            
            # Choose a random augmentation technique
            aug_type = np.random.randint(0, 4)
            
            if aug_type == 0:
                # Time shifting
                shift = np.random.randint(-seq_length // 10, seq_length // 10)
                X_aug[idx] = np.roll(X_train[j], shift, axis=0)
            
            elif aug_type == 1:
                # Amplitude scaling
                scale = np.random.uniform(0.8, 1.2)
                X_aug[idx] = X_train[j] * scale
            
            elif aug_type == 2:
                # Add noise
                noise_level = np.random.uniform(0.01, 0.05)
                noise = np.random.normal(0, noise_level, (seq_length, n_features))
                X_aug[idx] = X_train[j] + noise
            
            else:
                # Time warping (simplified)
                warp_factor = np.random.uniform(0.9, 1.1)
                indices = np.round(np.arange(0, seq_length - 1, warp_factor)).astype(int)
                indices = indices[indices < seq_length]
                X_aug[idx, :len(indices)] = X_train[j, indices]
                if len(indices) < seq_length:
                    X_aug[idx, len(indices):] = X_train[j, -1]
            
            # Keep the same label
            y_aug[idx] = y_train[j]
    
    print(f"Augmented data shape: {X_aug.shape}")
    
    return X_aug, y_aug

def train_model(args, X_train, X_val, y_train, y_val):
    """
    Train the CNN-LSTM model.
    
    Args:
        args: Command line arguments
        X_train, X_val, y_train, y_val: Training and validation data
        
    Returns:
        Trained model and training history
    """
    print("Building and training CNN-LSTM model...")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Apply data augmentation if specified
    if args.data_augmentation:
        X_train, y_train = apply_data_augmentation(X_train, y_train)
    
    # Create and build the model
    ecg_model = ECGModel(seq_length=args.seq_length, n_features=args.n_features, n_classes=1)
    model = ecg_model.build_model(
        cnn_layers=args.cnn_layers,
        lstm_layers=args.lstm_layers,
        dense_layers=args.dense_layers,
        dropout_rate=args.dropout
    )
    
    # Set custom learning rate
    tf.keras.backend.set_value(model.optimizer.learning_rate, args.learning_rate)
    
    # Print model summary
    model.summary()
    
    # Train the model
    history = ecg_model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    # Save the model
    model_path = os.path.join(args.output_dir, f"{args.model_name}.h5")
    scaler_path = os.path.join(args.output_dir, f"{args.model_name}_scaler.pkl")
    ecg_model.save(model_path=model_path, scaler_path=scaler_path)
    
    return ecg_model, history

def evaluate_model(ecg_model, X_val, y_val, args):
    """
    Evaluate the trained model and generate performance metrics and visualizations.
    
    Args:
        ecg_model: Trained ECG model
        X_val, y_val: Validation data
        args: Command line arguments
    """
    print("Evaluating model performance...")
    
    # Evaluate the model
    loss, accuracy = ecg_model.evaluate(X_val, y_val)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # Get predictions
    y_pred_prob = ecg_model.predict(X_val).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Save the ROC curve
    roc_path = os.path.join(args.output_dir, f"{args.model_name}_roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    
    # Plot training history
    history_fig = ecg_model.plot_training_history()
    
    # Save the training history plot
    history_path = os.path.join(args.output_dir, f"{args.model_name}_training_history.png")
    history_fig.savefig(history_path)
    plt.close(history_fig)
    
    # Save evaluation metrics to a file
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'training_params': vars(args)
    }
    
    metrics_path = os.path.join(args.output_dir, f"{args.model_name}_metrics.json")
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Evaluation results saved to {args.output_dir}")

def main():
    """Main function to train and evaluate the model."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load data
    X_train, X_val, y_train, y_val = load_data(args)
    
    # Train the model
    ecg_model, history = train_model(args, X_train, X_val, y_train, y_val)
    
    # Evaluate the model
    evaluate_model(ecg_model, X_val, y_val, args)
    
    print("Training and evaluation completed successfully!")

if __name__ == "__main__":
    main() 