import os
import cv2
import numpy as np
from pathlib import Path

class DiscROIExtractor:
    """
    Clase encargada de extraer los 5 parches de Región de Interés (ROI)
    correspondientes a los niveles intervertebrales L1-L2, L2-L3, L3-L4, L4-L5 y L5-S1
    a partir de una imagen recortada L1-S1.
    """
    def __init__(self, patch_size: tuple = (128, 128)):
        self.patch_size = patch_size
        self.levels = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
        
        # Ratios Y recalibrados para ubicar exactamente el núcleo de cada disco
        self.y_ratios = [0.18, 0.35, 0.52, 0.69, 0.86]

    def extract_patches(self, image: np.ndarray) -> dict:
        height, width = image.shape[:2]
        
        # Centrado sobre el canal vertebral/disco
        center_x = int(width * 0.46) 
        
        patches = {}
        
        # Para evitar capturar discos adyacentes arriba/abajo,
        # definimos un alto de caja más estrecho (ej. 64px) antes de normalizar
        crop_h = 64
        crop_w = 96
        
        half_h = crop_h // 2
        half_w = crop_w // 2

        for level, ratio in zip(self.levels, self.y_ratios):
            center_y = int(height * ratio)
            
            y1 = max(0, center_y - half_h)
            y2 = min(height, center_y + half_h)
            x1 = max(0, center_x - half_w)
            x2 = min(width, center_x + half_w)
            
            # Recorte enfocado únicamente en el disco específico
            patch = image[y1:y2, x1:x2]
            
            # Redimensionar al tamaño estándar deseado (128x128) para el modelo
            patch_resized = cv2.resize(patch, self.patch_size, interpolation=cv2.INTER_CUBIC)
                
            patches[level] = patch_resized

        return patches

    def process_dataset_to_rois(self, input_folder: str, output_folder: str):
        """
        Lee todas las imágenes PNG recortadas y guarda cada parche en su subcarpeta correspondiente.
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        # Crear estructura de carpetas de salida por nivel
        for level in self.levels:
            (output_path / level).mkdir(parents=True, exist_ok=True)
            
        png_files = list(input_path.rglob("*.png"))
        print(f"Total de imágenes L1-S1 a procesar para ROIs: {len(png_files)}")
        
        total_parches = 0
        for file_path in png_files:
            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            patches = self.extract_patches(img)
            
            base_name = file_path.stem
            for level_name, patch_img in patches.items():
                out_file = output_path / level_name / f"{base_name}_{level_name}.png"
                cv2.imwrite(str(out_file), patch_img)
                total_parches += 1

        print("\n--- RESUMEN EXTRACCIÓN DE PARCHES ROI ---")
        print(f"Imágenes L1-S1 procesadas: {len(png_files)}")
        print(f"Total de parches de ROI generados: {total_parches}")
        print(f"Ubicación del dataset de ROIs: {output_path.resolve()}")


if __name__ == "__main__":
    CARPETA_ENTRADA_L1S1 = "./prueba_roi"
    CARPETA_SALIDA_ROIS = "./t2_disc_roi_patches"

    extractor = DiscROIExtractor(patch_size=(128, 128))
    extractor.process_dataset_to_rois(
        input_folder=CARPETA_ENTRADA_L1S1,
        output_folder=CARPETA_SALIDA_ROIS
    )