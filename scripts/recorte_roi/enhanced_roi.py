import cv2
import numpy as np
from pathlib import Path

def enhance_mri_patch(image_path: str, output_path: str):
    """
    Aplica recorte de percentiles, normalización Min-Max y CLAHE
    para homogeneizar el tono y resaltar estructuras en parches T2.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return

    # 1. Recorte de percentiles (1%-99%) para eliminar picos de ruido/grasa
    p_low, p_high = np.percentile(img, (1, 99))
    clipped = np.clip(img, p_low, p_high)

    # 2. Normalización Min-Max al rango [0, 255]
    if p_high > p_low:
        norm = ((clipped - p_low) / (p_high - p_low) * 255.0).astype(np.uint8)
    else:
        norm = clipped.astype(np.uint8)

    # 3. Ecualización adaptativa de contraste (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(norm)

    cv2.imwrite(output_path, equalized)

def process_seed_dataset(input_dir: str, output_dir: str):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    png_files = list(in_path.rglob("*.png"))
    print(f"Normalizando y ecualizando {len(png_files)} parches semilla...")

    for img_file in png_files:
        dest_file = out_path / img_file.name
        enhance_mri_patch(str(img_file), str(dest_file))

    print(f"Proceso completado. Parches listos para etiquetar en: {out_path.resolve()}")

if __name__ == "__main__":
    process_seed_dataset(
        input_dir="./seed_dataset_unlabeled",
        output_dir="./seed_dataset_enhanced"
    )