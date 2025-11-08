import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, RepeatVector, TimeDistributed

class EncoderDecoderPredictor:
    def __init__(self, num_features, sequence_length=9, latent_dim=64):
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Build an encoder-decoder model for binary classification (0 or 1)
        """
        # Input: sequence of traffic states
        encoder_inputs = Input(shape=(self.sequence_length, self.num_features), name='encoder_input')
        
        # Encoder: Process the sequence
        encoder_lstm1 = LSTM(128, return_sequences=True, name='encoder_lstm1')(encoder_inputs)
        encoder_dropout1 = Dropout(0.2, name='encoder_dropout1')(encoder_lstm1)
        encoder_lstm2 = LSTM(self.latent_dim, name='encoder_lstm2')(encoder_dropout1)
        encoder_dropout2 = Dropout(0.2, name='encoder_dropout2')(encoder_lstm2)
        
        # Decoder: Predict binary future state (0 or 1)
        decoder_dense1 = Dense(64, activation='relu', name='decoder_dense1')(encoder_dropout2)
        decoder_dropout1 = Dropout(0.2, name='decoder_dropout1')(decoder_dense1)
        decoder_dense2 = Dense(32, activation='relu', name='decoder_dense2')(decoder_dropout1)
        
        # 🔥 CHANGE: Use sigmoid activation for binary output
        decoder_outputs = Dense(self.num_features, activation='sigmoid', name='decoder_output')(decoder_dense2)
        
        # Create model
        model = Model(encoder_inputs, decoder_outputs, name='encoder_decoder_model')
        
        # 🔥 CHANGE: Use binary_crossentropy loss for binary classification
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        return model
    
    def predict(self, sequence, threshold=0.5):
        """
        Predict binary traffic state (0 or 1)
        threshold: Values above this become 1, below become 0
        """
        if len(sequence) < self.sequence_length:
            return np.zeros(self.num_features, dtype=int)
            
        input_seq = np.array(sequence[-self.sequence_length:]).reshape(1, self.sequence_length, self.num_features)
        prediction = self.model.predict(input_seq, verbose=0)[0]
        
        # 🔥 Convert probabilities to binary predictions (0 or 1)
        binary_prediction = (prediction > threshold).astype(int)
        
        return binary_prediction
    
    def predict_sequence(self, sequence):
        """
        Predict a full sequence of future traffic states
        Returns the complete predicted sequence
        """
        # Ensure input shape: (1, sequence_length, num_features)
        if len(sequence) < self.sequence_length:
            # Pad sequence if needed
            padded_sequence = np.zeros((self.sequence_length, self.num_features))
            padded_sequence[-len(sequence):] = sequence
            input_seq = padded_sequence.reshape(1, self.sequence_length, self.num_features)
        else:
            # Get the last sequence_length states
            input_seq = np.array(sequence[-self.sequence_length:]).reshape(1, self.sequence_length, self.num_features)
        
        # Get prediction sequence
        prediction_sequence = self.model.predict(input_seq, verbose=0)[0]
        
        return prediction_sequence
    
    def train_batch(self, input_sequences, target_sequences):
        """
        Train the model on a batch of sequences
        input_sequences: shape (batch_size, sequence_length, num_features)
        target_sequences: shape (batch_size, sequence_length, num_features)
        """
        return self.model.fit(input_sequences, target_sequences, epochs=1, verbose=0)
    
    def save_model(self, path):
        """
        Save the trained model
        """
        self.model.save(path)
        
    def load_model(self, path):
        """
        Load a pre-trained model
        """
        self.model = tf.keras.models.load_model(path)
    
    def get_model_summary(self):
        """
        Get model architecture summary
        """
        return self.model.summary()


class EncoderDecoderAutoencoder:
    """
    Alternative encoder-decoder implementation for anomaly detection
    This version reconstructs the input sequence (autoencoder approach)
    """
    def __init__(self, num_features, sequence_length=9, latent_dim=64):
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.model = self._build_autoencoder()
        self.anomaly_threshold = None  # Will be set during training
    
    def _build_autoencoder(self):
        """
        Build an autoencoder for traffic state reconstruction and anomaly detection
        """
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.num_features), name='input')
        
        # Encoder
        encoded = LSTM(128, return_sequences=True, name='encoder_lstm1')(inputs)
        encoded = Dropout(0.2)(encoded)
        encoded = LSTM(self.latent_dim, name='encoder_lstm2')(encoded)
        encoded = Dropout(0.2)(encoded)
        
        # Bottleneck
        bottleneck = Dense(32, activation='relu', name='bottleneck')(encoded)
        
        # Decoder
        decoded = RepeatVector(self.sequence_length, name='repeat_vector')(bottleneck)
        decoded = LSTM(self.latent_dim, return_sequences=True, name='decoder_lstm1')(decoded)
        decoded = Dropout(0.2)(decoded)
        decoded = LSTM(128, return_sequences=True, name='decoder_lstm2')(decoded)
        decoded = Dropout(0.2)(decoded)
        decoded = TimeDistributed(Dense(self.num_features, activation='linear'), name='output')(decoded)
        
        # Create autoencoder model
        autoencoder = Model(inputs, decoded, name='autoencoder')
        
        # Compile
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        autoencoder.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        
        return autoencoder
    
    def train_on_normal_traffic(self, normal_sequences, epochs=10, batch_size=32):
        """
        Train the autoencoder on normal traffic patterns
        normal_sequences: array of shape (num_samples, sequence_length, num_features)
        """
        print(f"Training autoencoder on {len(normal_sequences)} normal traffic sequences...")
        
        # Train the autoencoder to reconstruct normal traffic patterns
        history = self.model.fit(
            normal_sequences, 
            normal_sequences,  # Autoencoder: input = target
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        # Calculate anomaly threshold based on reconstruction errors of normal data
        reconstruction_errors = []
        for seq in normal_sequences:
            error = self.get_reconstruction_error(seq)
            reconstruction_errors.append(error)
        
        # Set threshold as mean + 2*std of reconstruction errors
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        self.anomaly_threshold = mean_error + 2 * std_error
        
        print(f"Autoencoder training completed!")
        print(f"Anomaly threshold set to: {self.anomaly_threshold:.4f}")
        
        return history
    
    def is_anomaly(self, sequence, threshold=None):
        """
        Detect if a traffic sequence is anomalous
        Returns True if anomaly detected, False otherwise
        """
        if threshold is None:
            threshold = self.anomaly_threshold
            
        if threshold is None:
            print("Warning: No anomaly threshold set. Call train_on_normal_traffic first.")
            return False
            
        error = self.get_reconstruction_error(sequence)
        return error > threshold
    
    def predict(self, sequence):
        """
        Reconstruct the input sequence
        """
        if len(sequence) < self.sequence_length:
            return np.zeros((self.sequence_length, self.num_features))
            
        input_seq = np.array(sequence[-self.sequence_length:]).reshape(1, self.sequence_length, self.num_features)
        reconstructed = self.model.predict(input_seq, verbose=0)[0]
        
        return reconstructed
    
    def get_reconstruction_error(self, sequence):
        """
        Calculate reconstruction error for anomaly detection
        """
        if len(sequence) < self.sequence_length:
            return 0.0
            
        reconstructed = self.predict(sequence)
        original = np.array(sequence[-self.sequence_length:])
        
        # Calculate MSE between original and reconstructed
        mse = np.mean((original - reconstructed) ** 2)
        return mse
    
    def train_batch(self, input_sequences):
        """
        Train on a batch of sequences (for incremental training)
        """
        if len(input_sequences) == 0:
            return None
            
        # Convert to numpy array
        X = np.array(input_sequences)
        
        # Train for one epoch
        return self.model.fit(X, X, epochs=1, verbose=0)
    
    def save_model(self, path):
        """
        Save the trained autoencoder model and threshold
        """
        self.model.save(path)
        
        # Save threshold separately
        threshold_path = path.replace('.h5', '_threshold.npy')
        if self.anomaly_threshold is not None:
            np.save(threshold_path, self.anomaly_threshold)
        
    def load_model(self, path):
        """
        Load a pre-trained autoencoder model and threshold
        """
        self.model = tf.keras.models.load_model(path)
        
        # Load threshold
        threshold_path = path.replace('.h5', '_threshold.npy')
        try:
            self.anomaly_threshold = np.load(threshold_path)
        except FileNotFoundError:
            print("Warning: Anomaly threshold file not found. You may need to retrain.")
            self.anomaly_threshold = None
    
    def get_model_summary(self):
        """
        Get model architecture summary
        """
        return self.model.summary()

# Example usage and testing
if __name__ == "__main__":
    # Example parameters for traffic signal control
    num_features = 80  # Assuming 80 features as in typical traffic control scenarios
    sequence_length = 9
    
    # Create encoder-decoder model
    print("Creating Encoder-Decoder Predictor...")
    ed_predictor = EncoderDecoderPredictor(num_features, sequence_length)
    print("Model created successfully!")
    print("\nModel Summary:")
    ed_predictor.get_model_summary()
    
    # Test with dummy data
    dummy_sequence = np.random.rand(sequence_length, num_features)
    prediction = ed_predictor.predict(dummy_sequence)
    print(f"\nPrediction shape: {prediction.shape}")
    
    # Create autoencoder model
    print("\nCreating Autoencoder...")
    autoencoder = EncoderDecoderAutoencoder(num_features, sequence_length)
    print("Autoencoder created successfully!")
    
    # Test reconstruction
    reconstructed = autoencoder.predict(dummy_sequence)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Test reconstruction error
    error = autoencoder.get_reconstruction_error(dummy_sequence)
    print(f"Reconstruction error: {error:.4f}")