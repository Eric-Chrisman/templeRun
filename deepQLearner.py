import tensorflow as tf
import os

# Hyperparameters
ALPHA = 0.3   # Learning rate
GAMMA = 0.5    # Discount factor
EPSILON = 0.9 # Exploration rate
EPSILON_DECAY = 0.97
EPSILON_MIN = 0.3
BATCH_SIZE = 32
MEMORY_SIZE = 10000

class DQN:
    def __init__(self, input_shape, action_size):
        self.model = self.build_model(input_shape, action_size)

    def build_model(self, input_shape, action_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(action_size, activation='linear')  # Output layer
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
