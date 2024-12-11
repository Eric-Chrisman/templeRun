import threading
from deepQLearner import *
from templeRunHandler import *
from frameCapture import *
import time
import random
import numpy as np
from collections import deque
import cv2

# Define the monitor region (customize this for your game window)
monitor = {"top": 50, "left": 0, "width": 500, "height": 900}

# Define actions
actions = ['jump', 'slide', 'go_left', 'go_right', 'lean_left', 'lean_center', 'lean_right']
action_size = len(actions)
input_shape = (224, 224, 3)

# Initialize model, memory, and locks
dqn = DQN(input_shape, action_size)
memory = deque(maxlen=MEMORY_SIZE)
memory_lock = threading.Lock()
model_file_path = 'dqn_model.keras'  # Change to Keras format
dqn.load_model(model_file_path)

# Video writer for saving gameplay footage
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (800, 600))

# Flag for dead state
is_dead = False
dead_lock = threading.Lock()

# New parameter to limit the number of deaths
max_deaths = 1000
current_deaths = 0
new_experience_counter = 0

TRAIN_AFTER_NEW_EXPERIENCES = 50  # Adjust this value based on your requirements

# Training thread
def train_network():
    global new_experience_counter  # Counter to track new experiences

    while True:
        if len(memory) >= BATCH_SIZE and new_experience_counter >= TRAIN_AFTER_NEW_EXPERIENCES:
            with memory_lock:
                batch = random.sample(memory, BATCH_SIZE)

            for state, action_idx, reward, next_state, done in batch:
                target = reward
                if not done:
                    target += GAMMA * np.max(dqn.predict(next_state)[0])
                target_f = dqn.predict(state)
                target_f[0][action_idx] = target
                dqn.train(state, target_f)

            new_experience_counter = 0  # Reset the counter after training

        time.sleep(0.01)  # Small delay to avoid overloading CPU

# Check dead state in its own thread
def check_if_dead():
    global is_dead
    while True:
        with dead_lock:
            is_dead = tryReset()  # Update is_dead based on the dead check
        time.sleep(0.1)  # Adjust the frequency of checks as needed

# Main gameplay loop
try:
    # Start threads for training and dead checking
    training_thread = threading.Thread(target=train_network, daemon=True)
    training_thread.start()

    dead_check_thread = threading.Thread(target=check_if_dead, daemon=True)
    dead_check_thread.start()

    startMacro()
    while True:
        frame = capture_screen(monitor)
        state = preprocess_image(frame)

        # Optional: Display the gameplay footage
        cv2.imshow("Recording", frame)
        out.write(frame)
        
        # Prepare the state for prediction
        state = prepare_input(state)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Action selection using epsilon-greedy strategy
        if random.random() <= EPSILON:
            action_idx = random.randrange(action_size)  # Explore
        else:
            q_values = dqn.predict(state)
            action_idx = np.argmax(q_values[0])  # Exploit

        action = actions[action_idx]
        perform_action(action)

        # Reward calculation
        with dead_lock:
            reward = 1 if not is_dead else -1000  # Reward for staying alive or penalize heavily for dying

        # Store experience in memory
        if not is_dead:
            next_frame = capture_screen(monitor)
            next_state = preprocess_image(next_frame)
            next_state = prepare_input(next_state)

        with memory_lock:
            memory.append((state, action_idx, reward, next_state, is_dead))
            new_experience_counter += 1  # Increment the counter when adding new experiences

        # Reset game if dead
        if is_dead:
            current_deaths += 1  # Increment the death counter
            resetMacro()
            startMacro()

        # Exit the program if max deaths have been reached
        if current_deaths >= max_deaths:
            print(f"Max deaths reached: {current_deaths}. Exiting program.")
            break

        # Adjust epsilon after each action
        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY

finally:
    # Save the model before exiting
    dqn.save_model(model_file_path)  # Use the Keras format
    out.release()
    cv2.destroyAllWindows()