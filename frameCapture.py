import numpy as np
import cv2
from mss import mss
import tensorflow as tf

# takes a region you want captured. converts into a numpy array
def capture_screen(region=None):
    with mss() as sct:
        screenshot = sct.grab(region)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB) 
        return img

# reduces image size for the nerual network
def preprocess_image(img):
    img = cv2.resize(img, (224, 224)) 
    img = img / 255.0          
    return img

# after data is processed, convert to tensor and then process it in network
def prepare_input(img):
    tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    tensor = tf.expand_dims(tensor, axis=0)
    return tensor