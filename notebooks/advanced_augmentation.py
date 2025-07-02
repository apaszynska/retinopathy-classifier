# advanced_augmentation.py

import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import os
import random
import cv2
import numpy as np

# --- POMOCNICZA KLASA DO CLAHE ---
class ApplyCLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    def __call__(self, pil_img):
        img_cv = np.array(pil_img)
        img_lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2Lab)
        l_channel, a_channel, b_channel = cv2.split(img_lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        cl_channel = clahe.apply(l_channel)
        merged_lab = cv2.merge((cl_channel, a_channel, b_channel))
        final_img_cv = cv2.cvtColor(merged_lab, cv2.COLOR_Lab2RGB)
        return Image.fromarray(final_img_cv)

# ==============================================================================
# --- GŁÓWNA KONFIGURACJA STRATEGII AUGMENTACJI ---
# Zdefiniuj tutaj, co chcesz zrobić z każdą klasą.
# ==============================================================================


AUGMENTATION_STRATEGY = {
    "grade_0": {
        "target_count": 700,
        "transforms": T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ])
    },
    "grade_1": {
        "target_count": 700,
        "transforms": T.Compose([
            ApplyCLAHE(clip_limit=3.0),
            T.ElasticTransform(alpha=50.0, sigma=5.0),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            T.RandomHorizontalFlip(p=0.5),
        ])
    },
    "grade_2": {
        "target_count": 700,
        # Używamy tej samej, silnej strategii co dla grade_1
        "transforms": T.Compose([
            ApplyCLAHE(clip_limit=3.0),
            T.ElasticTransform(alpha=50.0, sigma=5.0),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            T.RandomHorizontalFlip(p=0.5),
        ])
    },
    "grade_3": {
        "target_count": 700,
        "transforms": T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=20),
        ])
    }
}

# Główny folder z danymi
DATA_DIR = Path.home() / 'Documents' / "retinopathy-classifier" / "data" / "grouped"

# --- GŁÓWNA FUNKCJA ---
def run_augmentations():
    for class_name, params in AUGMENTATION_STRATEGY.items():
        target_dir = DATA_DIR / class_name
        target_count = params["target_count"]
        transform = params["transforms"]

        print("="*50)
        print(f"Przetwarzanie klasy: {class_name}")
        print("="*50)

        if not target_dir.is_dir():
            print(f"Folder {target_dir} nie istnieje. Pomijanie.")
            continue

        image_paths = list(target_dir.glob("*.tif"))
        num_original = len(image_paths)

        if num_original == 0:
            print(f"Brak obrazów .tif w folderze {target_dir}. Pomijanie.")
            continue

        num_to_create = target_count - num_original
        
        if num_to_create <= 0:
            print(f"Liczba obrazów ({num_original}) jest wystarczająca. Nie potrzeba augmentacji.")
            continue

        print(f"Znaleziono {num_original} obrazów. Do stworzenia: {num_to_create}.")

        for i in range(num_to_create):
            source_path = random.choice(image_paths)
            try:
                original_image = Image.open(source_path).convert("RGB")
                
                # Zastosuj odpowiedni potok transformacji dla tej klasy
                # UWAGA: niektóre transformacje wymagają wejścia jako Tensor
                # Jeśli transformacja działa na obrazie PIL, upewnij się, że nie ma ToTensor() na początku
                # W przykładzie powyżej transformacje są mieszane, co wymaga konwersji
                
                # Prostsze podejście to mieć wszystkie transformacje działające na PIL
                augmented_image = transform(original_image)

                original_stem = source_path.stem
                random_suffix = random.randint(10000, 99999)
                new_filename = f"{original_stem}_aug_{random_suffix}.tif"
                new_filepath = target_dir / new_filename
                
                augmented_image.save(new_filepath, format='TIFF')
                print(f"  Stworzono: {new_filename} ({i+1}/{num_to_create})")

            except Exception as e:
                print(f"  Błąd przy przetwarzaniu {source_path.name}: {e}")
        
        print(f"✅ Zakończono dla klasy {class_name}. Nowa liczba obrazów: {len(list(target_dir.glob('*.tif')))}")

if __name__ == "__main__":
    run_augmentations()