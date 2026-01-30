import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import torch
from utils.safety_rules import check_violations

# Prevent CPU overload on Streamlit Cloud
torch.set_num_threads(1)

st.set_page_config(
    page_title="AI Construction Safety Check",
    layout="wide"
)

st.title("🏗️ AI Construction Safety Checker")
st.write("Detects helmet, vest & harness safety violations")

# Load YOLO model (cached)
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Upload construction site image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Run inference
    results = model(img_array, device="cpu", imgsz=640)

    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        detections.append({
            "class": model.names[cls_id],
            "confidence": round(conf, 2)
        })

    annotated_img = results[0].plot()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🧠 AI Detection")
        st.image(annotated_img, use_container_width=True)

    # Safety violations
    st.subheader("🚨 Safety Violations")
    violations = check_violations(detections)

    if violations:
        for v in violations:
            st.error(v)
    else:
        st.success("✅ No safety violations detected")
