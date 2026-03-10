import streamlit as st
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import tempfile
import os
import cv2
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Custom styling
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    color: #F8F9FA;
}
.custom-subtext {
    font-size: 20px;
    color: #C9D1D9;
    margin-bottom: 25px;
}
.info-box {
    background-color: #161B22;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #30363D;
    margin-bottom: 10px;
}
.result-box {
    background-color: #111827;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #374151;
    margin-top: 10px;
}
.footer-note {
    font-size: 14px;
    color: #9CA3AF;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>🧠 Brain Tumor Detection with YOLOv8</h1>
    <p class='custom-subtext' style='text-align: center;'>
        Upload an MRI image and the model will detect and classify possible brain tumors.
    </p>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Project Info")
st.sidebar.markdown("""
**Model:** YOLOv8  
**Task:** Brain Tumor Detection  
**Classes:**
- Glioma
- Meningioma
- Pituitary
""")

st.sidebar.warning(
    "This system is for research and educational purposes only. "
    "It should not replace professional medical diagnosis."
)

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    runs_dir = Path("runs/detect")

    if not runs_dir.exists():
        return None, "No training runs found. Train the model first."

    train_dirs = sorted(
        [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("train")],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    if not train_dirs:
        return None, "No train folders found in runs/detect."

    for train_dir in train_dirs:
        best_path = train_dir / "weights" / "best.pt"
        if best_path.exists():
            model = YOLO(str(best_path))
            return model, f"Loaded model: {best_path}"

    return None, "No best.pt found in any training run."

# -----------------------------
# Prediction summary
# -----------------------------
def draw_prediction_summary(result, model):
    if len(result.boxes) == 0:
        st.warning("No tumor detected in this image.")
        return

    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)

    best_idx = int(np.argmax(confs))
    best_class = clss[best_idx]
    best_conf = float(confs[best_idx])
    best_label = model.names[best_class]

    tumor_icons = {
        "glioma": "🧬",
        "meningioma": "🧠",
        "pituitary": "⚡"
    }

    icon = tumor_icons.get(best_label.lower(), "🔬")

    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.success(f"{icon} Top Prediction: {best_label} ({best_conf:.3f})")
    st.subheader("Confidence Score")
    st.progress(min(max(best_conf, 0.0), 1.0))
    st.write(f"Model confidence: **{best_conf:.3f}**")
    st.markdown("</div>", unsafe_allow_html=True)

    rows = []
    for i, (cls_id, conf) in enumerate(zip(clss, confs), start=1):
        rows.append({
            "Detection": i,
            "Class": model.names[int(cls_id)],
            "Confidence": round(float(conf), 3),
        })

    df = pd.DataFrame(rows)

    st.subheader("Detected Objects")
    st.dataframe(df, use_container_width=True)

# -----------------------------
# Load model status
# -----------------------------
model, status_msg = load_model()

if model is None:
    st.error(status_msg)
else:
    st.info(status_msg)

# -----------------------------
# Inputs
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload MRI image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)

# -----------------------------
# Main prediction flow
# -----------------------------
if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original MRI Image")
        st.image(image, use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)
        temp_path = tmp_file.name

    try:
        results = model.predict(
            source=temp_path,
            imgsz=640,
            conf=confidence,
            device="cpu"   # change to 0 later if your GPU environment is fully working
        )

        result = results[0]
        plotted = result.plot()
        plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader("Detection Output")
            st.image(plotted_rgb, use_container_width=True)

        draw_prediction_summary(result, model)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <p class='footer-note'>
    Built with Streamlit + YOLOv8 for brain tumor detection from MRI images.
    </p>
    """,
    unsafe_allow_html=True
)