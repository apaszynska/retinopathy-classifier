# config.py
import os
from pathlib import Path

def get_pytorch_vit_config():
    """
    Zwraca konfigurację dla eksperymentu z modelem ViT w PyTorch.
    """
    # --- Ścieżki Projektu ---
    # project_root = Path(r"D:\DataScienceKurs\retinopathy-classifier")
    # data_dir = project_root / "data" / "grouped"
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    data_dir = Path(os.path.join(project_root, "data", "grouped"))
    model_name = "vit_v3_high_resolution" 

    # --- Dynamiczne Parametry Danych ---
    if data_dir.exists():
        class_names = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
        num_classes = len(class_names)
    else:
        class_names = ['0', '1', '2', '3'] 
        num_classes = 4

    # --- Główna Konfiguracja ---
    config = {
        # --- Ścieżki ---
        "data_dir": data_dir,
        # "output_dir": project_root / "models_pytorch_vit" / model_name, # Nowy folder na wyniki
        "output_dir": Path(os.path.join(project_root, "models_pytorch_vit", model_name)),

        # --- Parametry Danych ---
        "image_size": (384,384, 3),
        "num_classes": num_classes,
        "class_names": class_names,
        "class_to_label": {name: i for i, name in enumerate(class_names)},
        "label_to_class": {i: name for i, name in enumerate(class_names)},

        # --- Parametry Podziału Danych & Reprodukowalność ---
        "test_split_size": 0.1,
        "validation_split_size": 0.1,
        "random_seed": 42,
        
        # --- Parametry Modelu Hugging Face ---
        "hf_model_name": "rafalosa/diabetic-retinopathy-224-procnorm-vit",
        "hf_processor_name": "google/vit-base-patch16-224",

        # --- Parametry Treningu ---
        "epochs": 15, # Zacznij od mniejszej liczby epok
        "batch_size": 16,
        "learning_rate": 2e-5,
        "early_stopping_patience": 3 # Typowe dla fine-tuningu Transformerów
    }
    
    os.makedirs(config["output_dir"], exist_ok=True)
    return config