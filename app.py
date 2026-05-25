import os
import numpy as np
from io import BytesIO
from PIL import Image
import streamlit as st

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detection")
st.markdown("Upload a photo of a plant leaf to identify potential diseases.")

# Load Model
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "models")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

MODEL = load_model()

CLASS_NAMES = [
    "Apple Scab", "Apple Black Rot", "Apple Rust", "Apple Healthy",
    "Blueberry Healthy",
    "Cherry Powdery Mildew", "Cherry Healthy",
    "Corn Cercospora Gray Leaf Spot", "Corn Common Rust", "Corn Northern Leaf Blight", "Corn Healthy",
    "Grape Black Rot", "Grape Esca", "Grape Leaf Blight", "Grape Healthy",
    "Orange Haunglongbing",
    "Peach Bacterial Spot", "Peach Healthy",
    "Pepper Bacterial Spot", "Pepper Healthy",
    "Potato Early Blight", "Potato Late Blight", "Potato Healthy",
    "Raspberry Healthy",
    "Soybean Healthy",
    "Squash Powdery Mildew",
    "Strawberry Leaf Scorch", "Strawberry Healthy",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight",
    "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Spider Mites",
    "Tomato Target Spot", "Tomato Yellow Leaf Curl Virus", "Tomato Mosaic Virus", "Tomato Healthy",
]

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    
    st.write("Classifying...")
    
    # Preprocess image
    image_array = np.array(image)
    pil_image = Image.fromarray(image_array).resize((224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(pil_image)
    input_arr = np.array([input_arr])
    preprocessed = tf.keras.applications.mobilenet.preprocess_input(input_arr)
    
    # Predict
    predictions = MODEL.predict(preprocessed)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    
    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2%}")
