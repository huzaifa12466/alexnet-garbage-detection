import os
import sys
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import gdown

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.model import load_alexnet, load_resnet

# ------------------ Directories & Device ------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ Download Weights ------------------
alexnet_path = os.path.join(MODELS_DIR, "best_model_alexnet.pth")
resnet_path = os.path.join(MODELS_DIR, "best_model_resnet.pth")

FILE_ID_ALEX = "1a6z8daFXuPl6oANsTteKsMSmbXdu2lMf"
if not os.path.exists(alexnet_path):
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID_ALEX}", alexnet_path, quiet=False)

FILE_ID_RESNET = "1kqzn-CTBNQY-3pKOX0xFILbk9McvJ1dp"  # replace with your actual ID
if not os.path.exists(resnet_path):
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID_RESNET}", resnet_path, quiet=False)

# ------------------ Load Models ------------------
models_dict = {
    "ResNet-50": load_resnet(resnet_path, device=DEVICE),
    "AlexNet": load_alexnet(alexnet_path, device=DEVICE)
}

# ------------------ Image Transform ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ------------------ Garbage Classes ------------------
CLASSES = [
    'metal',
    'biological',
    'trash',
    'glass',
    'paper',
    'clothes',
    'cardboard',
    'shoes',
    'battery',
    'plastic'
]

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Garbage Classifier ♻️", page_icon="♻️", layout="wide")
st.title("♻️ Garbage Classifier")

# ------------------ Model Selection ------------------
selected_model_name = st.selectbox("Choose Model", ["ResNet-50", "AlexNet"])
model = models_dict[selected_model_name]

# ------------------ Prediction Function ------------------
def predict_image(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        return CLASSES[predicted.item()] if predicted.item() < len(CLASSES) else None

# ------------------ Image Upload ------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        result = predict_image(image)
        if result:
            st.success(f"Garbage detected: **{result}**")
        else:
            st.warning("I don't know this type of garbage")
