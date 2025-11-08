import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class LSTMPredictor:
    def __init__(self, num_features, sequence_length=9):
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential()
        # More robust architecture
        model.add(LSTM(128, input_shape=(self.sequence_length, self.num_features), 
                      return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.num_features, activation='linear'))
        
        # Use Adam optimizer with a slightly lower learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        return model
        
    def predict(self, sequence):
        # Ensure input shape: (1, sequence_length, num_features)
        if len(sequence) < self.sequence_length:
            # Pad sequence if needed (shouldn't happen with proper implementation)
            return np.zeros(self.num_features)
            
        # Get the last sequence_length states
        input_seq = np.array(sequence[-self.sequence_length:]).reshape(1, self.sequence_length, self.num_features)
        return self.model.predict(input_seq, verbose=0)[0]
    
    def save_model(self, path):
        self.model.save(path)
        
    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)