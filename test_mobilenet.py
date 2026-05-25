import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
from PIL import Image

# Load MobileNetV2
model = MobileNetV2(weights='imagenet')
print("Model loaded successfully")
