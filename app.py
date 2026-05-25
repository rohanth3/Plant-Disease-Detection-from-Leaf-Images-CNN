import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# Function to load images as base64 for CSS
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Paths to the React frontend images
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BG_IMG_PATH = os.path.join(BASE_DIR, "frontend", "src", "background", "background.jpg")

# Inject Custom CSS for Glassmorphism and Background
if os.path.exists(BG_IMG_PATH):
    bg_base64 = get_base64_of_bin_file(BG_IMG_PATH)
    page_bg_img = f'''
    <style>
    /* Background Image */
    .stApp {{
        background-image: url("data:image/jpeg;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* Glassmorphism container for the main content area */
    .block-container {{
        background-color: rgba(0, 0, 0, 0.45) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0px 8px 32px 0px rgba(0, 0, 0, 0.37) !important;
        border-radius: 20px !important;
        padding: 40px !important;
        margin-top: 50px !important;
        color: white !important;
    }}

    /* Title and text colors */
    h1, h2, h3, p, label {{
        color: white !important;
    }}
    
    /* Upload Box Styling to match Dropzone */
    .stFileUploader > div > div {{
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 2px dashed rgba(255, 255, 255, 0.4) !important;
        border-radius: 16px !important;
    }}

    /* Header matching the Appbar Green */
    header[data-testid="stHeader"] {{
        background-color: rgba(77, 153, 0, 0.75) !important;
        backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.15) !important;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


st.title("🌿 Plant Care")
st.markdown("### Upload a photo of a plant leaf to identify potential diseases.")

# Load Model
@st.cache_resource
def load_model():
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

uploaded_file = st.file_uploader("Click here or Drag and Drop to select an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='', use_container_width=True)
    
    with st.spinner("Processing Image..."):
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
        
        st.markdown(f"<h3 style='text-align: center;'>{predicted_class}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center; font-weight: normal; color: #e0e0e0;'>Confidence: <b style='color: white;'>{confidence:.2%}</b></h4>", unsafe_allow_html=True)
