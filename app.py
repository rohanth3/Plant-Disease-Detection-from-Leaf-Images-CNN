import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Paths to models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------
# INJECT CUSTOM CSS
# -----------------
page_css = '''
<style>
/* Import modern font matching the Ankur aesthetic */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
}

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {background-color: transparent !important;}

/* Main Container Layout */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 650px !important;
}

/* Hero Section Typography */
.hero-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    background: -webkit-linear-gradient(45deg, #2E5A27, #4CAF50);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px !important;
    padding-bottom: 0px !important;
}
.hero-subtitle {
    text-align: center;
    font-size: 1.2rem;
    font-weight: 700;
    color: #2E5A27;
    margin-top: 10px;
    margin-bottom: 5px;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.hero-desc {
    text-align: center;
    font-size: 1rem;
    font-weight: 400;
    color: #5C765C;
    margin-bottom: 40px;
}

/* Upload Box Styling (Drag & Drop Card) */
[data-testid="stFileUploadDropzone"] {
    background-color: #FFFFFF !important;
    border: 2px dashed #A5D6A7 !important;
    border-radius: 20px !important;
    padding: 40px 20px !important;
    transition: all 0.3s ease-in-out;
    box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.03);
}
[data-testid="stFileUploadDropzone"]:hover {
    background-color: #F1F8E9 !important;
    border-color: #4CAF50 !important;
    transform: translateY(-4px) scale(1.01);
    box-shadow: 0px 10px 25px rgba(46, 90, 39, 0.1);
}

/* Upload Button and Text */
[data-testid="stFileUploadDropzone"] button {
    background: linear-gradient(90deg, #2E5A27 0%, #4CAF50 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 30px !important;
    padding: 10px 25px !important;
    transition: transform 0.2s;
}
[data-testid="stFileUploadDropzone"] button:hover {
    transform: scale(1.05);
}
[data-testid="stFileUploadDropzone"] div, [data-testid="stFileUploadDropzone"] span {
    color: #2E5A27 !important;
}

/* Expander Styling */
div[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E0E0E0 !important;
    border-radius: 16px !important;
    margin-bottom: 30px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.02);
}
div[data-testid="stExpander"] details summary {
    background-color: transparent !important;
    color: #2E5A27 !important;
    font-weight: 600 !important;
}
div[data-testid="stExpander"] details summary svg {
    fill: #2E5A27 !important;
    stroke: #2E5A27 !important;
}

/* Result Card */
.result-container {
    background-color: #FFFFFF;
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0px 8px 30px rgba(46, 90, 39, 0.08);
    margin-top: 20px;
    text-align: center;
    animation: slideUp 0.5s ease-out;
}
.result-class {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1F3622;
    margin-bottom: 20px;
}

/* Custom Progress Bar */
.progress-bg {
    width: 100%;
    background-color: #E8F5E9;
    border-radius: 10px;
    height: 12px;
    overflow: hidden;
    margin-bottom: 10px;
}
.progress-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 1s ease-in-out;
}
.progress-text {
    font-size: 1.1rem;
    font-weight: 600;
    display: flex;
    justify-content: space-between;
}

/* Image rounding */
[data-testid="stImage"] img {
    border-radius: 20px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.1);
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* 📱 Mobile Responsiveness (Phones & Small Tablets) */
@media screen and (max-width: 768px) {
    .block-container {
        padding-top: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .hero-title {
        font-size: 2.2rem !important;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1rem !important;
    }
    .hero-desc {
        font-size: 0.9rem !important;
        margin-bottom: 20px !important;
    }
    [data-testid="stFileUploadDropzone"] {
        padding: 30px 10px !important;
    }
    .result-class {
        font-size: 1.6rem !important;
    }
    .result-container {
        padding: 20px !important;
    }
}
</style>
'''
st.markdown(page_css, unsafe_allow_html=True)


# -----------------
# HERO SECTION
# -----------------
st.markdown("<h1 class='hero-title'>🌿 Plant Care AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Scan To Discover</p>", unsafe_allow_html=True)
st.markdown("<p class='hero-desc'>Open up a world of possibilities. Upload a leaf to instantly identify its health and diseases.</p>", unsafe_allow_html=True)

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

SPECIES_LIST = sorted(list(set([name.split(' ')[0] for name in CLASS_NAMES])))

with st.expander("🌱 View Supported Plants & Species"):
    st.markdown("Our AI model is specially trained to detect diseases on the following plants. **Please only upload images of leaves from these species:**")
    st.write(" • " + " • ".join(SPECIES_LIST))

# -----------------
# GATEKEEPER LISTS
# -----------------
BLACKLIST = [
    "car", "truck", "bus", "boat", "ship", "plane", "aircraft", "bike", 
    "bicycle", "train", "motorcycle", "cart", "wagon", "scooter", "submarine", "jet",
    "computer", "laptop", "phone", "monitor", "television", "tv", "camera", 
    "keyboard", "mouse", "printer", "speaker", "remote", "radio", "clock", "player",
    "dog", "cat", "bird", "fish", "spider", "snake", "frog", "monkey", 
    "horse", "cow", "sheep", "elephant", "lion", "tiger", "bear", "insect", 
    "bug", "beetle", "fly", "bee", "ant", "butterfly", "moth", "worm", 
    "mammal", "reptile", "amphibian", "crab", "lobster", "snail", "shell",
    "person", "man", "woman", "boy", "girl", "people", "face", "hand", "foot",
    "shirt", "shoe", "hat", "bag", "glasses", "watch", "coat", "jacket", 
    "pants", "dress", "skirt", "glove", "sock", "boot", "helmet", "backpack", 
    "purse", "wallet", "umbrella", "sunglasses",
    "chair", "table", "bed", "sofa", "couch", "desk", "lamp", "rug", "carpet", 
    "curtain", "mirror", "cabinet", "shelf", "pillow", "blanket", "shade",
    "bottle", "cup", "plate", "sink", "refrigerator", "oven", "stove", 
    "microwave", "bowl", "fork", "spoon", "knife", "pan", "glass", "mug", 
    "toaster", "blender", "washer", "dryer", "tub", "toilet",
    "building", "house", "bridge", "street", "road", "church", "tower", 
    "castle", "barn", "fence", "wall", "tent", "roof", "window", "door", "store",
    "fabric", "cloth", "velvet", "paper", "plastic", "metal", "wood", 
    "leather", "rubber", "screen", "mat",
    "mountain", "ocean", "beach", "desert", "rock", "stone", "sand", "water", 
    "sea", "lake", "river", "snow", "ice", "cloud", "sky", "dirt", "mud",
    "ball", "bat", "racket", "net", "toy", "game", "doll", "puzzle", "frisbee",
    "book", "box", "weapon", "instrument", "tool", "machine", "guitar", 
    "piano", "drum", "hammer", "drill", "saw", "sword", "gun", "scissors", 
    "pen", "pencil", "carton", "sign", "flag", "engine", "robot", "match"
]

PLANT_WORDS = [
    "leaf", "plant", "tree", "flower", "grass", "forest", "fern", "moss", 
    "cactus", "vine", "herb", "bamboo", "palm", "oak", "maple", "pine", 
    "sunflower", "daisy", "rose", "corn", "wheat", "vegetable", "fruit"
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
    hsv_image = image.convert('HSV')
    hsv_array = np.array(hsv_image)
    
    h = hsv_array[:, :, 0]
    s = hsv_array[:, :, 1]
    v = hsv_array[:, :, 2]
    
    plant_mask = (h >= 15) & (h <= 95) & (s > 20) & (v > 20)
    plant_ratio = np.sum(plant_mask) / (h.shape[0] * h.shape[1])
    
    green_mask = (h >= 45) & (h <= 85) & (s > 25) & (v > 25)
    green_ratio = np.sum(green_mask) / (h.shape[0] * h.shape[1])
    
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

    img_resized = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    preds = GATEKEEPER_MODEL.predict(img_preprocessed)
    top_5 = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]

    predicted_words = []
    for _, label, _ in top_5:
        words = label.lower().replace('_', ' ').replace('-', ' ').split()
        predicted_words.extend(words)

    blacklist_found = any(word in predicted_words for word in BLACKLIST)
    plant_word_found = any(word in predicted_words for word in PLANT_WORDS)

    if blacklist_found and not plant_word_found:
        top_prediction = top_5[0][1].replace('_', ' ').title()
        return False, f"This looks like a **{top_prediction}**, not a plant leaf!"

    return True, ""


# -----------------
# UI LOGIC
# -----------------
uploaded_file = st.file_uploader("Click to upload or drag and drop a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(image, use_container_width=True)
    
    with st.spinner("Analyzing Leaf Health..."):
        is_valid, error_msg = is_plant_image(image)
        
        if not is_valid:
            st.error(f"🚫 **Image Rejected:** {error_msg}")
        else:
            image_array = np.array(image)
            pil_image = Image.fromarray(image_array).resize((224, 224))
            input_arr = tf.keras.preprocessing.image.img_to_array(pil_image)
            input_arr = np.array([input_arr])
            preprocessed = tf.keras.applications.mobilenet.preprocess_input(input_arr)
            
            predictions = DISEASE_MODEL.predict(preprocessed)
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0]))
            
            # Progress bar color logic
            if confidence >= 0.80:
                bar_color = "#4CAF50" # Green
                text_color = "#2E5A27"
            elif confidence >= 0.50:
                bar_color = "#FFC107" # Yellow
                text_color = "#F57F17"
            else:
                bar_color = "#F44336" # Red
                text_color = "#B71C1C"

            # Dynamic Result Card with Custom Progress Bar
            result_html = f"""
            <div class="result-container">
                <div class="result-class">{predicted_class}</div>
                <div class="progress-text" style="color: {text_color};">
                    <span>Confidence Score</span>
                    <span>{confidence:.1%}</span>
                </div>
                <div class="progress-bg">
                    <div class="progress-fill" style="width: {confidence:.1%}; background-color: {bar_color};"></div>
                </div>
            </div>
            """
            st.markdown(result_html, unsafe_allow_html=True)
