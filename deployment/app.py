import os
import sys
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import gdown
import cv2
import numpy as np

# Ensure parent directory is in path to import model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.model import load_model, AlexNet

# ------------------ Directories & Device ------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)  # Ensure models folder exists

MODEL_PATH = os.path.join(MODELS_DIR, "best_model_alexnet.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ Download Model ------------------
FILE_ID = "1a6z8daFXuPl6oANsTteKsMSmbXdu2lMf"
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# ------------------ Load Model ------------------
model = load_model(weights_path=MODEL_PATH, device=DEVICE)

# ------------------ Image Transform ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ------------------ Garbage Classes ------------------
CLASSES = [
    'metal', 'battery', 'cardboard', 'glass', 'trash',
    'clothes', 'paper', 'shoes', 'biological', 'plastic'
]

# ------------------ Streamlit Page Setup ------------------
st.set_page_config(
    page_title="Garbage Classifier ♻️",
    page_icon="♻️",
    layout="wide"
)

st.sidebar.title("Garbage Classifier Info")
st.sidebar.markdown("""
**Detectable types:**  
- metal, battery, cardboard, glass, trash  
- clothes, paper, shoes, biological, plastic  

**How to use:**  
1. Upload an image or use webcam.  
2. Click Predict for uploaded images.  
3. For live webcam, click Start Webcam to see real-time predictions.
""")

st.title("♻️ Garbage Classifier")
st.markdown("Detect and classify types of garbage using AI.")

# ------------------ Prediction Function ------------------
def predict_image(image: Image.Image):
    """Predict the garbage class of an image."""
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        if predicted.item() < len(CLASSES):
            return CLASSES[predicted.item()]
        else:
            return None

# ------------------ Input Selection ------------------
option = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

# ------------------ Image Upload ------------------
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image",  use_container_width=True)

        if st.button("Predict"):
            result = predict_image(image)
            if result:
                st.success(f"Garbage detected: **{result}**")
            else:
                st.warning("I don't know this type of garbage")

# ------------------ Live Webcam ------------------
elif option == "Use Webcam":
    start_webcam = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])

    if start_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam.")
        else:
            try:
                while start_webcam:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Failed to grab frame.")
                        break

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb_frame)

                    # Predict
                    result = predict_image(img)
                    label_text = f"Garbage: {result}" if result else "I don't know this type"

                    # Draw label on frame
                    cv2.putText(frame, label_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            finally:
                cap.release()
