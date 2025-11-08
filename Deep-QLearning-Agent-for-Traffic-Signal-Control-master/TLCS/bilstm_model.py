import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

class BiLSTMPredictor:
    def __init__(self, num_features, sequence_length=9):
        """
        Initialize the Bidirectional LSTM predictor model
        
        Args:
            num_features: Number of features in the state representation
            sequence_length: Number of time steps in the input sequence
        """
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Build and compile the Bi-LSTM model architecture
        
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        # First Bidirectional LSTM layer with return sequences for stacking
        model.add(Bidirectional(LSTM(128, return_sequences=True), 
                                input_shape=(self.sequence_length, self.num_features)))
        model.add(Dropout(0.25))
        
        # Second Bidirectional LSTM layer
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.25))
        
        # Dense hidden layer
        model.add(Dense(32, activation='relu'))
        
        # Output layer with linear activation for regression
        model.add(Dense(self.num_features, activation='linear'))
        
        # Compile with Adam optimizer and MSE loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        
        return model
    
    def predict(self, sequence):
        """
        Make a prediction based on a sequence of states
        
        Args:
            sequence: Array of state vectors
            
        Returns:
            Predicted next state
        """
        # Ensure input shape: (1, sequence_length, num_features)
        if len(sequence) < self.sequence_length:
            # Pad sequence if needed (shouldn't happen with proper implementation)
            return np.zeros(self.num_features)
            
        # Get the last sequence_length states
        input_seq = np.array(sequence[-self.sequence_length:]).reshape(1, self.sequence_length, self.num_features)
        return self.model.predict(input_seq, verbose=0)[0]
    
    def save_model(self, path):
        """Save the model to disk"""
        self.model.save(path)
        
    def load_model(self, path):
        """Load a pre-trained model from disk"""
        self.model = tf.keras.models.load_model(path)