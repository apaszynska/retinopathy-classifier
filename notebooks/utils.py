# utils.py

import torch
from torch.utils.data import Dataset
from transformers import ViTForImageClassification, AutoImageProcessor
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
import cv2
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import plotly.graph_objects as go
from pytorch_grad_cam import LayerCAM



# --- Funkcje dla Eksperymentu PyTorch ViT ---

def load_paths_and_labels(config):
    """
    Wyszukuje ścieżki do obrazów i przypisuje im etykiety.
    """
    data_dir = config["data_dir"]
    class_to_label = config["class_to_label"]

    extensions = ['*.tif', '*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in extensions:
        image_files.extend(data_dir.rglob(ext))
        
    tif_files = [str(p) for p in image_files]
    labels = [class_to_label[Path(p).parent.name] for p in tif_files]
    
    print(f"Znaleziono {len(tif_files)} obrazów.")
    return tif_files, labels

class RetinopathyDataset(Dataset):
    """
    Niestandardowy Dataset dla PyTorch do wczytywania obrazów siatkówki.
    """
    def __init__(self, paths, labels, processor):
        self.paths = paths
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0) 
        label = torch.tensor(self.labels[idx])
        return {"pixel_values": pixel_values, "labels": label}

def create_pytorch_vit_model(config, class_weights_tensor=None):
    """
    Tworzy model ViT w wersji dla PyTorch i opcjonalnie dołącza wagi klas.
    """
    model = ViTForImageClassification.from_pretrained(
        config["hf_model_name"],
        num_labels=config["num_classes"],
        id2label=config["label_to_class"],
        label2id=config["class_to_label"],
        ignore_mismatched_sizes=True
    )
    
    if class_weights_tensor is not None:
        model.class_weights = class_weights_tensor
    
    return model

def compute_metrics(eval_pred):
    """
    Funkcja do obliczania metryk podczas ewaluacji, wymagana przez Trainer.
    """
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

def create_probability_bar_chart(probabilities):
    """
    Twrzy estetyczny, poziomy wykres słupkowy w Plotly,
    z ustaloną, logiczną kolejnością klas.
    """
    # === POPRAWKA: Zaktualizowano listy klas do zakresu 0-3 ===
    display_order = ['grade_0', 'grade_1', 'grade_2', 'grade_3']
    display_names = {
        "grade_0": "No Retinopathy",
        "grade_1": "Grade 1",
        "grade_2": "Grade 2",
        "grade_3": "Grade 3"
    }
    # ==========================================================
    
    final_class_names = []
    final_probs_values = []
    
    for class_key in display_order:
        if class_key in probabilities:
            name = display_names.get(class_key, class_key)
            prob = probabilities[class_key] * 100
            
            final_class_names.append(name)
            final_probs_values.append(prob)
    
    fig = go.Figure(go.Bar(
        x=final_probs_values,
        y=final_class_names,
        orientation='h',
        text=[f'{p:.1f}%' for p in final_probs_values],
        textposition='inside',
        marker_color='rgba(26, 118, 255, 0.7)',
        insidetextanchor='middle'
    ))

    fig.update_layout(
        title_text='Detailed Probabilities',
        title_font_size=20,
        xaxis_title="Probability (%)",
        yaxis_title="Class",
        height=300,
        margin={'t':40, 'b':40, 'l':10, 'r':10},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=14),
            autorange="reversed"
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.3)',
            range=[0, 100]
        )
    )
    return fig

def generate_grad_cam(model, image_tensor, original_pil_image):
    """
    Generates a heatmap using EigenCAM. Transparency is controlled
    by the 'image_weight' parameter only.
    """
    def reshape_transform(tensor, height=14, width=14):
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    class HuggingFaceModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x).logits

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    # We will continue to use the masking technique
    img_for_mask = np.array(original_pil_image.convert("L"))
    _, mask = cv2.threshold(img_for_mask, 15, 255, cv2.THRESH_BINARY)
    mask_tensor = torch.from_numpy(mask).to(image_tensor.device).float() / 255.0
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    masked_input_tensor = image_tensor * mask_tensor

    wrapped_model = HuggingFaceModelWrapper(model)
    target_layers = [model.vit.encoder.layer[-1].layernorm_before]

    cam = EigenCAM(
        model=wrapped_model,
        target_layers=target_layers,
        reshape_transform=reshape_transform
    )

    grayscale_cam = cam(input_tensor=masked_input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    # The problematic np.clip line has been removed.
    
    rgb_img_numpy = np.array(original_pil_image.convert("RGB"), dtype=np.float32) / 255.0

    visualization = show_cam_on_image(
        rgb_img_numpy,
        mask=grayscale_cam,
        use_rgb=True,
        # You can now control the transparency ONLY with this value.
        # Try a value like 0.5 for more color, or 0.7 for more transparency.
        image_weight=0.8
    )

    final_image = Image.fromarray(visualization)
    
    return {
        "final_image": final_image,
        "raw_cam": grayscale_cam
    }