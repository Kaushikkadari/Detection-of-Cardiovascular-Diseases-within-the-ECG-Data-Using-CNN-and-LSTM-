"""
CNN-LSTM Model for ECG Cardiovascular Disease Detection
This module implements a hybrid CNN-LSTM architecture for analyzing ECG signals
and detecting patterns associated with cardiovascular diseases.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization,
    LSTM, Bidirectional, Dense, Flatten, TimeDistributed, Attention,
    GlobalAveragePooling1D, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import datetime

class ECGModel:
    """
    CNN-LSTM model for ECG analysis and cardiovascular disease detection.
    """
    
    def __init__(self, seq_length=1000, n_features=1, n_classes=1):
        """
        Initialize the ECG model.
        
        Args:
            seq_length (int): Length of the ECG sequence
            n_features (int): Number of features (ECG leads)
            n_classes (int): Number of output classes (1 for binary classification)
        """
        self.seq_length = seq_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def build_model(self, cnn_layers=3, lstm_layers=2, dense_layers=2, 
                   filters_base=32, lstm_units_base=64, dropout_rate=0.3):
        """
        Build the CNN-LSTM model architecture.
        
        Args:
            cnn_layers (int): Number of CNN layers
            lstm_layers (int): Number of LSTM layers
            dense_layers (int): Number of dense layers
            filters_base (int): Base number of filters for CNN layers
            lstm_units_base (int): Base number of units for LSTM layers
            dropout_rate (float): Dropout rate for regularization
            
        Returns:
            The compiled Keras model
        """
        # Input layer
        inputs = Input(shape=(self.seq_length, self.n_features))
        
        # CNN Block
        x = inputs
        for i in range(cnn_layers):
            filters = filters_base * (2 ** i)  # Double filters in each layer
            x = Conv1D(filters=filters, kernel_size=5, padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(dropout_rate)(x)
        
        # LSTM Block
        for i in range(lstm_layers - 1):
            units = lstm_units_base * (2 ** i)  # Double units in each layer
            x = Bidirectional(LSTM(units, return_sequences=True))(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
        
        # Final LSTM layer
        x = Bidirectional(LSTM(lstm_units_base * (2 ** (lstm_layers - 1))))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Dense Block
        for i in range(dense_layers - 1):
            units = lstm_units_base // (2 ** i)  # Halve units in each layer
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
        
        # Output layer
        if self.n_classes == 1:
            # Binary classification
            outputs = Dense(1, activation='sigmoid')(x)
        else:
            # Multi-class classification
            outputs = Dense(self.n_classes, activation='softmax')(x)
        
        # Create and compile model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Compile with appropriate loss function
        if self.n_classes == 1:
            loss = 'binary_crossentropy'
        else:
            loss = 'categorical_crossentropy'
            
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=loss,
            metrics=['accuracy']
        )
        
        return self.model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            batch_size=32, epochs=50, callbacks=None, class_weights=None):
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            batch_size: Batch size for training
            epochs: Number of training epochs
            callbacks: List of Keras callbacks
            class_weights: Class weights for imbalanced datasets
            
        Returns:
            Training history
        """
        # Fit the scaler on training data
        if X_train.shape[1] == self.seq_length and X_train.shape[2] == self.n_features:
            # Reshape for scaler
            X_train_2d = X_train.reshape(-1, self.n_features)
            self.scaler.fit(X_train_2d)
            
            # Transform the data
            X_train_scaled = self.scaler.transform(X_train_2d).reshape(X_train.shape)
            
            if X_val is not None:
                X_val_2d = X_val.reshape(-1, self.n_features)
                X_val_scaled = self.scaler.transform(X_val_2d).reshape(X_val.shape)
            else:
                X_val_scaled = None
        else:
            # If data is already in the right shape, just fit the scaler
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        
        # Default callbacks if none provided
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        # Train the model
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val) if X_val is not None else None,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        return self.history
    
    def _get_default_callbacks(self):
        """
        Create default callbacks for training.
        
        Returns:
            List of Keras callbacks
        """
        # Create models directory if it doesn't exist
        models_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(models_dir, exist_ok=True)
        
        # Model checkpoint to save best model
        checkpoint = ModelCheckpoint(
            os.path.join(models_dir, 'cnn_lstm_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            mode='min',
            verbose=1
        )
        
        # Reduce learning rate when plateau is reached
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            mode='min',
            verbose=1
        )
        
        # TensorBoard for visualization
        log_dir = os.path.join(models_dir, 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        )
        
        return [checkpoint, early_stopping, reduce_lr, tensorboard]
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Input data
            
        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model has not been built or loaded yet.")
        
        # Reshape and scale the input data
        if X.shape[1] == self.seq_length and X.shape[2] == self.n_features:
            X_2d = X.reshape(-1, self.n_features)
            X_scaled = self.scaler.transform(X_2d).reshape(X.shape)
        else:
            X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    def evaluate(self, X, y):
        """
        Evaluate the model on test data.
        
        Args:
            X: Test data
            y: Test labels
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been built or loaded yet.")
        
        # Reshape and scale the input data
        if X.shape[1] == self.seq_length and X.shape[2] == self.n_features:
            X_2d = X.reshape(-1, self.n_features)
            X_scaled = self.scaler.transform(X_2d).reshape(X.shape)
        else:
            X_scaled = self.scaler.transform(X)
        
        return self.model.evaluate(X_scaled, y)
    
    def save(self, model_path=None, scaler_path=None):
        """
        Save the model and scaler to disk.
        
        Args:
            model_path: Path to save the model
            scaler_path: Path to save the scaler
        """
        if self.model is None:
            raise ValueError("Model has not been built or loaded yet.")
        
        # Default paths
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cnn_lstm_model.h5')
        
        if scaler_path is None:
            scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scaler.pkl')
        
        # Save model and scaler
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load(self, model_path=None, scaler_path=None):
        """
        Load the model and scaler from disk.
        
        Args:
            model_path: Path to the saved model
            scaler_path: Path to the saved scaler
        """
        # Default paths
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cnn_lstm_model.h5')
        
        if scaler_path is None:
            scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scaler.pkl')
        
        # Load model and scaler
        if os.path.exists(model_path):
            self.model = load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
        else:
            print(f"Scaler file not found at {scaler_path}. Using default scaler.")
            self.scaler = StandardScaler()
    
    def plot_training_history(self):
        """
        Plot the training history.
        
        Returns:
            Matplotlib figure
        """
        if self.history is None:
            raise ValueError("Model has not been trained yet.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'])
        if 'val_accuracy' in self.history.history:
            ax1.plot(self.history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Validation'], loc='lower right')
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'])
        if 'val_loss' in self.history.history:
            ax2.plot(self.history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Validation'], loc='upper right')
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    
    def summary(self):
        """
        Print the model summary.
        """
        if self.model is None:
            raise ValueError("Model has not been built or loaded yet.")
        
        return self.model.summary()


def create_sample_data(n_samples=1000, seq_length=1000, n_features=1):
    """
    Create sample ECG data for testing.
    
    Args:
        n_samples: Number of samples to generate
        seq_length: Length of each ECG sequence
        n_features: Number of features (ECG leads)
        
    Returns:
        X: Generated ECG data
        y: Generated labels
    """
    # Generate time points
    t = np.linspace(0, 10, seq_length)
    
    # Initialize arrays
    X = np.zeros((n_samples, seq_length, n_features))
    y = np.zeros(n_samples)
    
    # Generate normal and abnormal ECG patterns
    for i in range(n_samples):
        # Base ECG signal (normal pattern)
        base_signal = np.sin(2 * np.pi * 1.1 * t)  # Basic sine wave
        
        # Add heartbeat-like peaks
        peaks = np.zeros_like(t)
        peak_positions = np.arange(0.5, 10, 1.0)  # Regular heartbeats
        
        for pos in peak_positions:
            # Create QRS complex
            idx = np.argmin(np.abs(t - pos))
            width = int(seq_length * 0.03)  # Width of the QRS complex
            
            # Create a peak
            for j in range(max(0, idx - width), min(seq_length, idx + width)):
                dist = abs(j - idx) / width
                peaks[j] += 1.5 * np.exp(-dist ** 2)
        
        # Combine base signal and peaks
        signal = base_signal + peaks
        
        # Add noise
        noise = np.random.normal(0, 0.1, seq_length)
        signal += noise
        
        # For abnormal patterns (50% of samples)
        if i % 2 == 1:
            # Introduce abnormalities
            abnormality_type = np.random.randint(0, 3)
            
            if abnormality_type == 0:
                # Irregular heartbeat (arrhythmia)
                irregular_idx = np.random.randint(seq_length // 4, 3 * seq_length // 4)
                signal[irregular_idx:irregular_idx + width * 2] += 2.0 * np.random.random(width * 2)
            
            elif abnormality_type == 1:
                # ST segment elevation (myocardial infarction)
                for pos in peak_positions:
                    idx = np.argmin(np.abs(t - pos))
                    st_start = idx + width
                    st_end = st_start + int(seq_length * 0.1)
                    if st_end < seq_length:
                        signal[st_start:st_end] += 0.5
            
            else:
                # Low amplitude (cardiomyopathy)
                signal *= 0.6
            
            # Label as abnormal
            y[i] = 1
        
        # Store in array
        for j in range(n_features):
            # Add slight variations for different leads
            lead_variation = np.random.normal(0, 0.05, seq_length)
            X[i, :, j] = signal + lead_variation
    
    return X, y


if __name__ == "__main__":
    # Example usage
    print("Generating sample ECG data...")
    X, y = create_sample_data(n_samples=100, seq_length=1000, n_features=1)
    
    # Split into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    print("Building and training CNN-LSTM model...")
    ecg_model = ECGModel(seq_length=1000, n_features=1, n_classes=1)
    model = ecg_model.build_model()
    print(model.summary())
    
    # Train for just a few epochs as an example
    ecg_model.fit(X_train, y_train, X_val=X_test, y_val=y_test, epochs=5, batch_size=16)
    
    # Evaluate
    loss, accuracy = ecg_model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Save model
    ecg_model.save()
    
    # Plot training history
    ecg_model.plot_training_history()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_history.png'))
    plt.close()
    
    print("Done!") 