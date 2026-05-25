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
    layout="centered",
    initial_sidebar_state="collapsed"
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

# Inject Custom CSS for Premium Earthy Glassmorphism
if os.path.exists(BG_IMG_PATH):
    bg_base64 = get_base64_of_bin_file(BG_IMG_PATH)
    page_bg_img = f'''
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {{
        font-family: 'Outfit', sans-serif !important;
    }}

    /* Background Image */
    .stApp {{
        background-image: url("data:image/jpeg;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* Hide Streamlit Branding for cleaner look */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{background-color: transparent !important;}}

    /* Main Container Glassmorphism - Earthy Dark Green */
    .block-container {{
        background-color: rgba(20, 35, 20, 0.75) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(120, 200, 120, 0.25) !important;
        box-shadow: 0px 12px 40px 0px rgba(0, 0, 0, 0.6) !important;
        border-radius: 24px !important;
        padding: 3rem 4rem 4rem 4rem !important;
        margin-top: 3rem !important;
        margin-bottom: 3rem !important;
        max-width: 750px !important;
    }}

    /* Text Colors */
    h1, h2, h3, p, label, span, li {{
        color: #e8f5e9 !important; /* Soft earthy mint/white */
    }}
    
    /* Titles alignment */
    h1 {{
        text-align: center;
        font-weight: 700 !important;
        text-shadow: 0px 2px 10px rgba(0,0,0,0.5);
        margin-bottom: 5px !important;
    }}
    .subtitle {{
        text-align: center;
        font-weight: 300;
        font-size: 1.1rem;
        margin-bottom: 30px;
        color: #a5d6a7 !important;
    }}
    
    /* Upload Box Styling */
    [data-testid="stFileUploadDropzone"] {{
        background-color: rgba(0, 0, 0, 0.4) !important;
        border: 2px dashed rgba(165, 214, 167, 0.6) !important;
        border-radius: 16px !important;
        transition: all 0.3s ease-in-out;
        padding: 20px !important;
    }}
    [data-testid="stFileUploadDropzone"]:hover {{
        background-color: rgba(77, 153, 0, 0.3) !important;
        border-color: rgba(165, 214, 167, 1) !important;
        transform: translateY(-2px);
    }}
    
    /* Upload Box Button and Text */
    [data-testid="stFileUploadDropzone"] button {{
        background-color: rgba(77, 153, 0, 0.9) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
    }}
    [data-testid="stFileUploadDropzone"] button:hover {{
        background-color: rgba(40, 100, 20, 1) !important;
    }}
    [data-testid="stFileUploadDropzone"] div, [data-testid="stFileUploadDropzone"] span {{
        color: #e8f5e9 !important;
    }}

    /* Expander Styling */
    div[data-testid="stExpander"] {{
        background-color: rgba(0, 0, 0, 0.4) !important;
        border: 1px solid rgba(165, 214, 167, 0.2) !important;
        border-radius: 12px !important;
        margin-bottom: 20px;
    }}
    /* Fix Expander Header to never turn white */
    div[data-testid="stExpander"] details summary {{
        background-color: transparent !important;
        color: #e8f5e9 !important;
    }}
    div[data-testid="stExpander"] details summary:hover {{
        background-color: rgba(255, 255, 255, 0.05) !important;
    }}
    div[data-testid="stExpander"] details summary svg {{
        fill: #e8f5e9 !important;
        stroke: #e8f5e9 !important;
    }}
    
    /* Result Card Styling */
    .result-card {{
        background: linear-gradient(135deg, rgba(67, 160, 71, 0.85) 0%, rgba(27, 94, 32, 0.9) 100%);
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        margin-top: 20px;
        animation: fadeIn 0.5s ease-in;
    }}
    .result-card h2 {{
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.3);
    }}
    .result-card h4 {{
        margin: 10px 0 0 0;
        font-weight: 400;
        color: #c8e6c9 !important;
    }}
    
    /* Image display rounding */
    [data-testid="stImage"] img {{
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


st.markdown("<h1>🌿 Plant Care AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload a photo of a plant leaf to detect potential diseases.</p>", unsafe_allow_html=True)

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

# Extract unique species for UI display
SPECIES_LIST = sorted(list(set([name.split(' ')[0] for name in CLASS_NAMES])))

with st.expander("ℹ️ View Supported Plants & Species"):
    st.markdown("Our AI model is specially trained to detect diseases on the following plants. **Please only upload images of leaves from these species:**")
    st.write(" • " + " • ".join(SPECIES_LIST))

# -----------------
# GATEKEEPER LISTS
# -----------------
BLACKLIST = [
    # Vehicles & Transport
    "car", "truck", "bus", "boat", "ship", "plane", "aircraft", "bike", 
    "bicycle", "train", "motorcycle", "cart", "wagon", "scooter", "submarine", "jet",
    
    # Electronics & Tech
    "computer", "laptop", "phone", "monitor", "television", "tv", "camera", 
    "keyboard", "mouse", "printer", "speaker", "remote", "radio", "clock", "player",
    
    # Animals, Insects & Biology
    "dog", "cat", "bird", "fish", "spider", "snake", "frog", "monkey", 
    "horse", "cow", "sheep", "elephant", "lion", "tiger", "bear", "insect", 
    "bug", "beetle", "fly", "bee", "ant", "butterfly", "moth", "worm", 
    "mammal", "reptile", "amphibian", "crab", "lobster", "snail", "shell",
    
    # Humans & Anatomy
    "person", "man", "woman", "boy", "girl", "people", "face", "hand", "foot",
    
    # Clothing & Wearables
    "shirt", "shoe", "hat", "bag", "glasses", "watch", "coat", "jacket", 
    "pants", "dress", "skirt", "glove", "sock", "boot", "helmet", "backpack", 
    "purse", "wallet", "umbrella", "sunglasses",
    
    # Furniture & Decor
    "chair", "table", "bed", "sofa", "couch", "desk", "lamp", "rug", "carpet", 
    "curtain", "mirror", "cabinet", "shelf", "pillow", "blanket", "shade",
    
    # Household & Kitchen (Removed 'pot' as it can mean plant pot)
    "bottle", "cup", "plate", "sink", "refrigerator", "oven", "stove", 
    "microwave", "bowl", "fork", "spoon", "knife", "pan", "glass", "mug", 
    "toaster", "blender", "washer", "dryer", "tub", "toilet",
    
    # Buildings & Infrastructure
    "building", "house", "bridge", "street", "road", "church", "tower", 
    "castle", "barn", "fence", "wall", "tent", "roof", "window", "door", "store",
    
    # Backgrounds & Materials
    "fabric", "cloth", "velvet", "paper", "plastic", "metal", "wood", 
    "leather", "rubber", "screen", "mat",
    
    # Nature (Non-Plant)
    "mountain", "ocean", "beach", "desert", "rock", "stone", "sand", "water", 
    "sea", "lake", "river", "snow", "ice", "cloud", "sky", "dirt", "mud",
    
    # Sports & Toys
    "ball", "bat", "racket", "net", "toy", "game", "doll", "puzzle", "frisbee",
    
    # Random Objects & Tools
    "book", "box", "weapon", "instrument", "tool", "machine", "guitar", 
    "piano", "drum", "hammer", "drill", "saw", "sword", "gun", "scissors", 
    "pen", "pencil", "carton", "sign", "flag", "engine", "robot", "match"
]

PLANT_WORDS = [
    "leaf", "plant", "tree", "flower", "grass",
    "forest", "fern", "moss", "cactus", "vine",
    "herb", "bamboo", "palm", "oak", "maple",
    "pine", "sunflower", "daisy", "rose",
    "corn", "wheat", "vegetable", "fruit"
]

# -----------------
# LOAD MODELS
# -----------------
@st.cache_resource
def load_disease_model():
    MODEL_PATH = os.path.join(BASE_DIR, "models")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

@st.cache_resource
def load_gatekeeper_model():
    return tf.keras.applications.MobileNetV2(weights='imagenet')

DISEASE_MODEL = load_disease_model()
GATEKEEPER_MODEL = load_gatekeeper_model()

# -----------------
# VALIDATION LOGIC
# -----------------
def is_plant_image(image):
    """
    Advanced heuristic and AI gatekeeper to check if an image is a natural plant.
    1. Fast Color Heuristic
    2. Deep Learning Semantic Check (MobileNetV2)
    """
    # ==========================
    # STEP 1: FAST COLOR CHECK
    # ==========================
    hsv_image = image.convert('HSV')
    hsv_array = np.array(hsv_image)
    
    h = hsv_array[:, :, 0]
    s = hsv_array[:, :, 1]
    v = hsv_array[:, :, 2]
    
    plant_mask = (h >= 15) & (h <= 95) & (s > 20) & (v > 20)
    plant_ratio = np.sum(plant_mask) / (h.shape[0] * h.shape[1])
    
    green_mask = (h >= 45) & (h <= 85) & (s > 25) & (v > 25)
    green_ratio = np.sum(green_mask) / (h.shape[0] * h.shape[1])
    
    # Reject obvious non-plants based on color
    if green_ratio < 0.025:
        return False, "This doesn't look like a natural plant leaf (Missing distinct green colors)."
    if np.percentile(s, 75) < 25:
        return False, "This doesn't look like a natural plant leaf (Image is mostly colorless/grayscale)."
    if plant_ratio > 0:
        hue_std = np.std(h[plant_mask])
    else:
        hue_std = 0
    if hue_std < 2.0:
        return False, "This doesn't look like a natural plant leaf (Colors are flat and synthetic)."

    # ==========================
    # STEP 2: AI SEMANTIC CHECK
    # ==========================
    # Preprocess for MobileNetV2
    img_resized = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Predict using ImageNet gatekeeper
    preds = GATEKEEPER_MODEL.predict(img_preprocessed)
    top_5 = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]

    # Extract words from the predictions (e.g. "sports_car" -> ["sports", "car"])
    predicted_words = []
    for _, label, _ in top_5:
        words = label.lower().replace('_', ' ').replace('-', ' ').split()
        predicted_words.extend(words)

    # Check against our lists
    blacklist_found = any(word in predicted_words for word in BLACKLIST)
    plant_word_found = any(word in predicted_words for word in PLANT_WORDS)

    # Logic: Reject if it contains a blacklisted word AND doesn't explicitly contain a plant word
    if blacklist_found and not plant_word_found:
        # Get the top prediction for the error message
        top_prediction = top_5[0][1].replace('_', ' ').title()
        return False, f"This looks like a **{top_prediction}**, not a plant leaf!"

    return True, ""


# -----------------
# UI LOGIC
# -----------------
uploaded_file = st.file_uploader("Click here or drag and drop to select an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display the uploaded image centered
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(image, use_container_width=True)
    
    with st.spinner("Analyzing Leaf Health..."):
        # Validate if image is a plant
        is_valid, error_msg = is_plant_image(image)
        
        if not is_valid:
            st.error(f"🚫 **Image Rejected:** {error_msg}")
        else:
            # Preprocess image for Disease Model (MobileNet V1 typically uses the same preprocess, 
            # but we explicitly use the generic mobilenet preprocess_input as originally used)
            image_array = np.array(image)
            pil_image = Image.fromarray(image_array).resize((224, 224))
            input_arr = tf.keras.preprocessing.image.img_to_array(pil_image)
            input_arr = np.array([input_arr])
            preprocessed = tf.keras.applications.mobilenet.preprocess_input(input_arr)
            
            # Predict
            predictions = DISEASE_MODEL.predict(preprocessed)
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0]))
            
            # Display stunning result card
            result_html = f"""
            <div class="result-card">
                <h2>{predicted_class}</h2>
                <h4>Confidence: <strong style="color: white;">{confidence:.2%}</strong></h4>
            </div>
            """
            st.markdown(result_html, unsafe_allow_html=True)
