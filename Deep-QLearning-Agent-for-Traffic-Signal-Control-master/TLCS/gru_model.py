import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

class GRUPredictor:
    def __init__(self, num_features, sequence_length=9):
        """
        Initialize the GRU predictor model
        
        Args:
            num_features: Number of features in the state representation
            sequence_length: Number of time steps in the input sequence
        """
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Build and compile the GRU model architecture
        
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        # First GRU layer with return sequences for stacking
        model.add(GRU(128, input_shape=(self.sequence_length, self.num_features), 
                      return_sequences=True))
        model.add(Dropout(0.2))
        # Second GRU layer
        model.add(GRU(64))
        model.add(Dropout(0.2))
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