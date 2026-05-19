app_code = """
import os
import torch
import torch.nn as nn
import streamlit as st
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16

st.set_page_config(page_title="Kvasir ViT Classifier", page_icon="🩺", layout="wide")

CLASS_NAMES = [
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis",
    "normal-cecum",
    "normal-pylorus",
    "normal-z-line",
    "polyps",
    "ulcerative-colitis",
]

MODEL_PATH = os.getenv("MODEL_PATH", "vit_kvasir_best.pth")
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

@st.cache_resource
def load_model():
    model = vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(CLASS_NAMES))
    )
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

def predict(image, model):
    image = image.convert("RGB")
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].numpy()
    order = np.argsort(probs)[::-1]
    return [(CLASS_NAMES[i], float(probs[i])) for i in order]

st.title("🩺 Vision Transformer Gastro Classification")
st.write("Upload an endoscopy image and get the predicted Kvasir class.")

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    model = load_model()
    preds = predict(image, model)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with col2:
        st.subheader("Top prediction")
        st.success(f"{preds[0][0]} ({preds[0][1]:.2%})")
        st.bar_chart({k: v for k, v in preds})
        st.write("Probabilities")
        for cls, p in preds:
            st.write(f"- {cls}: {p:.2%}")
else:
    st.info("Upload a JPG or PNG image to start prediction.")
"""
