import os
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models")
# Load model using legacy Keras (tf-keras) with compile=False to support the SavedModel format
MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)

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


@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "Plant Disease Detection API is running"}


def read_file_as_image(data: bytes) -> np.ndarray:
    """Read raw bytes, force RGB, and return as a numpy array."""
    image = Image.open(BytesIO(data)).convert("RGB")
    return np.array(image)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Receive an uploaded image and return the predicted disease class and confidence."""
    raw = await file.read()
    image_array = read_file_as_image(raw)

    pil_image = Image.fromarray(image_array).resize((224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(pil_image)
    input_arr = np.array([input_arr])

    preprocessed = tf.keras.applications.mobilenet.preprocess_input(input_arr)
    predictions = MODEL.predict(preprocessed)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {
        "class": predicted_class,
        "confidence": confidence,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
