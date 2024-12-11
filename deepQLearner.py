import tensorflow as tf
import os

# Hyperparameters
ALPHA = 0.01   # Learning rate
GAMMA = 0.5    # Discount factor
EPSILON = 0.8  # Exploration rate
EPSILON_DECAY = 0.95
EPSILON_MIN = 0.3
BATCH_SIZE = 32
MEMORY_SIZE = 10000

class DQN:
    def __init__(self, input_shape, action_size):
        self.model = self.build_model(input_shape, action_size)

    def build_model(self, input_shape, action_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=ALPHA), loss='mse')
        return model

    def predict(self, state):
        return self.model.predict(state)

    def train(self, x, y):
        self.model.fit(x, y, verbose=0)

    def save_model(self, file_path):
        self.model.save(file_path)

    def load_model(self, file_path):
        if os.path.exists(file_path):
            self.model = tf.keras.models.load_model(file_path)