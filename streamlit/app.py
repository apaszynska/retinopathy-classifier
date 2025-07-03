import sys
from pathlib import Path
import streamlit as st
from PIL import Image
import torch
from transformers import ViTForImageClassification, AutoImageProcessor
import pandas as pd
import plotly.graph_objects as go
import io

# --- Konfiguracja Strony ---
st.set_page_config(
    page_title="ARISE",
    page_icon="👁️",
    layout="wide",
)

# --- CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background-color: #f0f2f6;
    }

    /* Ukrycie paska bocznego i domyślnych elementów */
    section[data-testid="stSidebar"], #MainMenu, footer, header {
        display: none !important;
    }
    
    .block-container {
        padding: 1rem 3rem !important;
        max-width: 1000px !important;
    }
    
    h1 {
        color: #0d2c6e;
        text-align: center;
        font-weight: 800;
        font-size: 6.5rem;
    }
    h2 {
        color: #5a6a85;
        text-align: center;
        font-weight: 400;
        margin-bottom: 2rem;
        font-size: 1.0rem;
    }
    
    [data-testid="stMetric"] {
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    .stButton>button {
        background-color: #0d2c6e;
        color: #ffffff;
        border: none;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #1a4a9c;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #5a6a85;
    }
    .stTabs [aria-selected="true"] {
        color: #0d2c6e !important;
        border-bottom: 2px solid #0d2c6e !important;
    }
    
    .stImage > img {
        border-radius: 12px;
        border: 1px solid #e0e4eb;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
        width: auto;
        margin: auto;
        display: block;
    }
</style>
""", unsafe_allow_html=True)

# --- Konfiguracja Ścieżek i Aplikacji ---
home_dir = Path.home()
project_dir = home_dir / 'Documents' / "retinopathy-classifier"
if str(project_dir) not in sys.path:
    sys.path.append(str(project_dir))

try:
    from notebooks import utils, config
except ImportError:
    st.error("Import Error: Could not find 'utils' and 'config'.")
    st.stop()

cfg = config.get_pytorch_vit_config()
MODEL_PATH = project_dir / "models_pytorch_vit" / "vit_v3_advaug_0.2split_20epoch" / "final_model"
CLASS_NAMES = cfg['class_names']

# --- Logika Aplikacji ---
@st.cache_resource
def load_model_and_processor(path):
    try:
        model = ViTForImageClassification.from_pretrained(path)
        processor = AutoImageProcessor.from_pretrained(path)
        model.to("cpu").eval()
        return model, processor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

@st.cache_data
def get_prediction(_model, _processor, image_bytes, top_k_percent):
    original_image = Image.open(io.BytesIO(image_bytes))
    
    MODEL_INPUT_SIZE = (224, 224)
    resized_image = original_image.resize(MODEL_INPUT_SIZE)
    
    if resized_image.mode != "RGB":
        resized_image = resized_image.convert("RGB")
        
    inputs = _processor(images=resized_image, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        outputs = _model(**inputs)
    logits = outputs.logits
        
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    top_prob, top_class_idx = torch.max(probabilities, 1)
    predicted_class = CLASS_NAMES[top_class_idx.item()]
    confidence = top_prob.item()
    all_probs = {CLASS_NAMES[i]: prob.item() for i, prob in enumerate(probabilities[0])}
    
    cam_results_small = utils.generate_grad_cam(_model, inputs['pixel_values'], resized_image, top_k_percent=top_k_percent)
    
    low_res_heatmap_image = cam_results_small["final_image"]
    
    high_res_heatmap_image = low_res_heatmap_image.resize(original_image.size, Image.Resampling.BICUBIC)
    
    return predicted_class, confidence, all_probs, high_res_heatmap_image, original_image

def get_display_name(class_name):
    if class_name == "grade_0":
        return "No Retinopathy"
    return class_name.replace('_', ' ').title()

def create_bar_chart(all_probs):
    display_probs = {get_display_name(k): v for k, v in all_probs.items()}
    df = pd.DataFrame(list(display_probs.items()), columns=['Class', 'Probability'])
    df = df.sort_values(by='Probability', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df['Probability'],
        y=df['Class'],
        orientation='h',
        marker_color='#0d2c6e',
        text=df['Probability'].apply(lambda x: f' {x:.1%}'),
        textposition='outside'
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#263238"),
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=10, t=10, b=10),
        height=250,
    )
    return fig

# --- Interfejs Użytkownika (UI) ---

st.markdown("<h1>ARISE</h1>", unsafe_allow_html=True)
st.markdown("<h2>Automated Retinal Image Scoring and Evaluation</h2>", unsafe_allow_html=True)

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

model, processor = load_model_and_processor(MODEL_PATH)

if not model or not processor:
    st.error("Application cannot run without a loaded model.")
    st.stop()

# Widok 1: Upload pliku
if st.session_state.uploaded_file is None:
    uploaded_file = st.file_uploader(
        "Upload a retinal image for analysis",
        type=["png", "jpg", "jpeg", "tif"],
    )
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file.getvalue()
        st.rerun()

# Widok 2: Wyniki analizy
else:
    # Pobranie danych do wyświetlenia
    predicted_class, confidence, all_probs, _, original_image = get_prediction(
        model,
        processor,
        st.session_state.uploaded_file,
        top_k_percent=10 # Domyślna wartość, nie ma znaczenia dla tej zakładki
    )

    main_tab1, main_tab2 = st.tabs(["📊 Analysis Results", "🧠 AI Decision Process"])

    with main_tab1:
        _, col_metric, _ = st.columns([1, 2, 1])
        with col_metric:
            st.metric(
                label="Predicted Grade",
                value=get_display_name(predicted_class),
                delta=f"{confidence*100:.1f}% Confidence",
                delta_color="normal"
            )

        with st.expander("Show Classification Details"):
            fig = create_bar_chart(all_probs)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.image(original_image, caption="Uploaded retinal image", use_container_width=True)

    with main_tab2:
        st.info(
            "Use the slider to adjust the heatmap's sensitivity. A lower percentage shows only the most critical areas.",
            icon="💡"
        )
        percentile = st.slider("Show top % of activations", 1, 100, 10, key="heatmap_slider")
        
        # Ponowne uruchomienie z nową wartością suwaka tylko dla tej zakładki
        _, _, _, final_heatmap_interactive, _ = get_prediction(
            model,
            processor,
            st.session_state.uploaded_file,
            top_k_percent=percentile
        )
        
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.image(
                original_image,
                caption="Original Image",
                use_container_width=True
            )
        with col2:
            st.image(
                final_heatmap_interactive,
                caption=f"Heatmap showing top {percentile}% of model's attention",
                use_container_width=True
            )
            
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅️ Analyze another image"):
        st.session_state.uploaded_file = None
        st.rerun()