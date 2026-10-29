import os
import cv2
import numpy as np
from pathlib import Path

class LumbarRegionCropper:
    """
    Clase para realizar el recorte vertical de la zona lumbar-sacra (L1-S1)
    sobre imágenes en formato PNG.
    """
    def __init__(self, top_crop_ratio: float = 0.15, bottom_crop_ratio: float = 0.88):
        # top_crop_ratio = 0.15: Conserva la vértebra L1 (elimina columna torácica)
        # bottom_crop_ratio = 0.88: Conserva el disco L5-S1 y la plataforma sacra
        self.top_crop_ratio = top_crop_ratio
        self.bottom_crop_ratio = bottom_crop_ratio

    def crop_image(self, image: np.ndarray) -> np.ndarray:
        """Aplica el recorte matricial sobre el eje vertical Y."""
        height, width = image.shape[:2]
        y_start = int(height * self.top_crop_ratio)
        y_end = int(height * self.bottom_crop_ratio)
        return image[y_start:y_end, :]

    def process_dataset(self, input_folder: str, output_folder: str):
        input_path = Path(input_folder)
        output_path = Path(output_folder)

        if not input_path.exists():
            print(f"Error: La ruta de origen '{input_path.resolve()}' no existe.")
            return

        output_path.mkdir(parents=True, exist_ok=True)
        cropped_count = 0
        error_count = 0

        # Buscar todos los archivos PNG en el dataset
        png_files = list(input_path.rglob("*.png"))
        print(f"Total de imágenes PNG a recortar: {len(png_files)}")

        for file_path in png_files:
            try:
                # 1. Leer imagen PNG
                img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # 2. Aplicar recorte de la zona L1-S1
                cropped_img = self.crop_image(img)

                # 3. Mantenimiento de la estructura de subcarpetas
                relative_path = file_path.relative_to(input_path)
                output_file = output_path / relative_path
                output_file.parent.mkdir(parents=True, exist_ok=True)

                # 4. Guardar la imagen recortada
                cv2.imwrite(str(output_file), cropped_img)
                cropped_count += 1

                if cropped_count % 500 == 0:
                    print(f"Recortadas {cropped_count} imágenes...")

            except Exception as e:
                error_count += 1
                continue

        print("\n--- RESUMEN DE RECORTE LUMBAR L1-S1 ---")
        print(f"Imágenes recortadas L1-S1 con éxito: {cropped_count}")
        print(f"Errores u omisiones: {error_count}")
        print(f"Ubicación del dataset recortado: {output_path.resolve()}")

if __name__ == "__main__":
    # Nombres claros de entrada y salida
    CARPETA_PNG_ENTRADA = "./t2_midsagittal_png"
    CARPETA_LUMBAR_SALIDA = "./t2_midsagittal_l1s1_cropped"

    cropper = LumbarRegionCropper(top_crop_ratio=0.15, bottom_crop_ratio=0.88)
    cropper.process_dataset(
        input_folder=CARPETA_PNG_ENTRADA,
        output_folder=CARPETA_LUMBAR_SALIDA
    )