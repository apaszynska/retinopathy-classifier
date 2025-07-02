# app.py

import sys
from pathlib import Path
import streamlit as st
import torch
from transformers import ViTForImageClassification, AutoImageProcessor
from PIL import Image
import pandas as pd

# --- Path Setup ---

home_dir = Path.home()
project_dir = home_dir / 'Documents' / "retinopathy-classifier"

# project_dir = r"D:\DataScienceKurs\retinopathy-classifier"
if str(project_dir) not in sys.path:
    sys.path.append(str(project_dir))

try:
    from notebooks import utils
    from notebooks import config
except ImportError:
    st.error("Could not import 'utils' and 'config'. Make sure they are in a 'notebooks' subfolder inside your project directory.")
    st.stop()

# ======================================================================
# --- APPLICATION CONFIGURATION ---
# ======================================================================
cfg = config.get_pytorch_vit_config()
MODEL_PATH = home_dir / 'Documents' / "retinopathy-classifier" / "models_pytorch_vit" / "vit_v3_advaug_0.2split_20epoch" / "final_model" # Użyj finalnego modelu
CLASS_NAMES = cfg['class_names']

# ======================================================================
# --- APPLICATION LOGIC ---
# ======================================================================
@st.cache_resource
def load_model_and_processor(path):
    try:
        model = ViTForImageClassification.from_pretrained(path)
        processor = AutoImageProcessor.from_pretrained(path)
        device = torch.device("cpu")
        model.to(device)
        model.eval()
        return model, processor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def predict(image, model, processor):
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    top_prob, top_class_idx = torch.max(probabilities, 1)
    predicted_class_name = CLASS_NAMES[top_class_idx.item()]
    confidence = top_prob.item()
    all_probs = {CLASS_NAMES[i]: prob.item() for i, prob in enumerate(probabilities[0])}
    return predicted_class_name, confidence, all_probs, inputs['pixel_values']

# ======================================================================
# --- USER INTERFACE (UI) ---
# ======================================================================
st.set_page_config(layout="wide")

with st.sidebar:
    st.image("../assets/eye_logo.png", width=100) 
    st.title("Control Panel")
    uploaded_file = st.file_uploader(
        "1. Upload a fundus image", 
        type=["png", "jpg", "jpeg", "tif"]
    )
    st.info("After uploading an image, the results will appear on the main screen.")

st.title("Retina Image Analyzer")
st.warning("DISCLAIMER: This tool is a demonstration project. The results are not a medical diagnosis and cannot replace a consultation with a qualified doctor.")

model, processor = load_model_and_processor(MODEL_PATH)

if model and processor:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # === KROK 1: KRYTYCZNA POPRAWKA - UJEDNOLICENIE ROZMIARU ===
        # Definiujemy standardowy rozmiar wejściowy dla modelu
        MODEL_INPUT_SIZE = (224, 224)
        
        # Tworzymy wersję obrazu w rozmiarze, którego oczekuje model.
        # Tej wersji będziemy używać do PREDYKCJI i WIZUALIZACJI HEATMAPY.
        resized_image = image.resize(MODEL_INPUT_SIZE)
        # ==========================================================
        
        st.subheader("Analysis Results")
        
        with st.spinner('Model is analyzing the image...'):
            # === KROK 2: PREYDKCJA NA OBRAZIE O POPRAWNYM ROZMIARZE ===
            predicted_class, confidence, all_probs, image_tensor = predict(resized_image, model, processor)
            
            col1, col2 = st.columns(2)
            with col1:
                # Dla użytkownika wciąż wyświetlamy oryginalny obraz w wysokiej rozdzielczości
                st.image(image, caption="Uploaded retina image.", use_container_width=True)
            with col2:
                # --- ZMIANA: Wyświetlamy teraz estetyczny wykres z Plotly ---
                st.metric("Top Prediction", predicted_class.replace('_', ' ').title(), f"{confidence*100:.1f}%")

                # Wywołaj nową funkcję z utils.py, aby stworzyć wykres
                fig = utils.create_probability_bar_chart(all_probs)
                
                # Wyświetl wykres za pomocą Streamlit
                st.plotly_chart(fig, use_container_width=True)
                # ----------------------------------------------------

        st.write("---")
        st.subheader("Model's Decision Explanation")

        with st.spinner("Generating heatmap..."):
            # Call the function and get the dictionary of results
            cam_results = utils.generate_grad_cam(model, image_tensor, resized_image)
            
            # Create two columns to display the heatmaps side-by-side
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(cam_results["raw_cam"], caption="Raw Heatmap (from model)", use_container_width=True)
                
            with col2:
                st.image(cam_results["final_image"], caption="Final Overlay", use_container_width=True)
    else:
        st.info("Please upload an image using the control panel on the left to start the analysis.")
else:
    st.error("Failed to load the model. Check the path in the configuration at the top of the app.py file.")