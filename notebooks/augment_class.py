# augment_class.py

import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import os
import random
import cv2
import numpy as np


# ==============================================================================
# --- KONFIGURACJA ---
# Zmień te wartości, aby dostosować skrypt do swoich potrzeb
# ==============================================================================
home_dir = Path.home()

# 1. Ścieżka do folderu z klasą, którą chcesz zaugmentować
TARGET_CLASS_FOLDER = home_dir / 'Documents' / "retinopathy-classifier" / "data" / "grouped" / "grade_0"

# 2. Ile nowych, zaugmentowanych wersji chcesz stworzyć DLA KAŻDEGO oryginalnego obrazu?
# Jeśli masz 15 obrazów i ustawisz tę wartość na 9, stworzysz 15 * 9 = 135 nowych obrazów.
NUM_AUGMENTATIONS_PER_IMAGE = 4
TOTAL_IMAGES_DESIRED = 700


# ==============================================================================

def augment_and_save_images():
    """
    Wczytuje obrazy z folderu docelowego, augmentuje je i zapisuje w tym samym folderze.
    """
    # Zamień ścieżkę w stringu na obiekt Path
    target_dir = Path(TARGET_CLASS_FOLDER)

    if not target_dir.exists():
        print(f"BŁĄD: Folder {target_dir} nie istnieje!")
        return

    # Zdefiniuj potok transformacji (augmentacji)
    # Możesz tu dodawać lub usuwać transformacje, aby uzyskać różne efekty
    transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5), # Obrót w poziomie z 50% prawdopodobieństwem
        T.RandomVerticalFlip(p=0.5),   # Obrót w pionie z 50% prawdopodobieństwem
        T.RandomRotation(degrees=15),  # Losowy obrót o maksymalnie 15 stopni
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Lekkie zmiany kolorów
    ])

    # Znajdź wszystkie pliki .tif w folderze
    image_paths = list(target_dir.glob("*.tif"))
    num_original_images = len(image_paths)

    if num_original_images == 0:
        print(f"W folderze {target_dir} nie znaleziono żadnych obrazów .tif.")
        return
    

    num_to_create = TOTAL_IMAGES_DESIRED - num_original_images

    if num_to_create <= 0:
        print(f"Liczba obrazów ({num_original_images}) jest już równa lub większa niż docelowa ({TOTAL_IMAGES_DESIRED}).")
        print("✅ Nie ma potrzeby augmentacji. Zakończono.")
        return

    print(f"Znaleziono {num_original_images} obrazów w folderze '{target_dir.name}'.")
    print(f"Docelowa liczba obrazów: {TOTAL_IMAGES_DESIRED}.")
    print(f"Do stworzenia pozostało: {num_to_create} nowych obrazów.")
    print("-" * 30)


    print(f"Znaleziono {num_original_images} oryginalnych obrazów w folderze '{target_dir.name}'.")
    print(f"Dla każdego obrazu zostanie stworzonych {NUM_AUGMENTATIONS_PER_IMAGE} nowych, zaugmentowanych wersji.")
    print("-" * 30)

    # # Przejdź przez każdy oryginalny obraz
    # for i, image_path in enumerate(image_paths):
    #     print(f"Augmentowanie obrazu {i+1}/{num_original_images}: {image_path.name}")
    #     try:
    #         original_image = Image.open(image_path).convert("RGB")

    #         # Stwórz i zapisz N zaugmentowanych wersji
    #         for j in range(NUM_AUGMENTATIONS_PER_IMAGE):
    #             augmented_image = transform(original_image)
                
    #             # Stwórz nową, unikalną nazwę pliku
    #             original_stem = image_path.stem # nazwa pliku bez rozszerzenia
    #             new_filename = f"{original_stem}_aug_{j+1}.tif"
    #             new_filepath = target_dir / new_filename
                
    #             # Zapisz zaugmentowany obraz
    #             augmented_image.save(new_filepath, format='TIFF')

    #     except Exception as e:
    #         print(f"  Nie udało się przetworzyć obrazu {image_path.name}. Błąd: {e}")

            # --- NOWA LOGIKA: Pętla tworząca brakującą liczbę obrazów ---
    for i in range(num_to_create):
        # Wybierz losowy obraz z puli oryginalnych
        source_image_path = random.choice(image_paths)
        
        print(f"Tworzenie obrazu {i+1}/{num_to_create} (na podstawie: {source_image_path.name})")
        try:
            original_image = Image.open(source_image_path).convert("RGB")
            augmented_image = transform(original_image)
            
            # Stwórz nową, unikalną nazwę pliku, aby uniknąć nadpisywania
            original_stem = source_image_path.stem
            # Dodajemy losowy numer, aby nazwa była unikalna nawet przy wielokrotnym uruchomieniu
            random_suffix = random.randint(1000, 9999)
            new_filename = f"{original_stem}_aug_{random_suffix}.tif"
            new_filepath = target_dir / new_filename
            
            augmented_image.save(new_filepath, format='TIFF')

        except Exception as e:
            print(f"  Nie udało się przetworzyć obrazu {source_image_path.name}. Błąd: {e}")

    # total_new_images = num_original_images * NUM_AUGMENTATIONS_PER_IMAGE
    # print("-" * 30)
    # print(f"✅ Zakończono! Stworzono {total_new_images} nowych obrazów w folderze {target_dir}.")

    print("-" * 30)
    print(f"✅ Zakończono! Stworzono {num_to_create} nowych obrazów w folderze {target_dir}.")
    print(f"Łączna liczba obrazów w folderze: {len(list(target_dir.glob('*.tif')))}")

if __name__ == "__main__":
    augment_and_save_images()