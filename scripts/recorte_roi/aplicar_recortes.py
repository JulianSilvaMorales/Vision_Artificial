import csv
import time
from pathlib import Path
import cv2
import numpy as np

# Rutas base
BASE_DIR = Path(__file__).resolve().parent

# 1. Carpeta con los recortes manuales de referencia (L1-S1)
REF_CROPPED_DIR = BASE_DIR / "t2_midsagittal_l1s1_cropped"

# 2. Carpeta con las imágenes completas v2 (sin recortar)
SRC_UNCROPPED_DIR = BASE_DIR / "t2_midsagittal_v2_curated"

# 3. Carpeta de destino para los nuevos recortes v2
DST_DIR = BASE_DIR / "t2_midsagittal_v2_l1s1_cropped_ext"

# Cuántos píxeles añadir por debajo del recorte manual.
# None = hasta el borde inferior de la imagen completa.
EXTRA_BAJO_PX = None

# Cuántos píxeles añadir por ENCIMA del recorte manual (limitado por el borde
# superior de la imagen completa). None = hasta el borde superior.
# El desplazamiento realmente aplicado se guarda como offset_arriba en el CSV.
EXTRA_ARRIBA_PX = 60


def cv2_imread_unicode(path: Path, flags=cv2.IMREAD_GRAYSCALE) -> np.ndarray:
    """Lectura segura de imágenes en Windows soportando caracteres especiales/tildes en la ruta."""
    with open(path, "rb") as f:
        arr = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(arr, flags)


def cv2_imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    """Escritura segura de imágenes en Windows soportando caracteres especiales/tildes en la ruta."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    success, buffer = cv2.imencode(ext, img)
    if success:
        with open(path, "wb") as f:
            f.write(buffer)
        return True
    return False


def replicar_recortes(ref_dir: Path, src_dir: Path, dst_dir: Path):
    """
    Localiza la posición exacta (x, y, w, h) de cada imagen recortada en ref_dir
    dentro de la imagen completa en src_dir usando correlación normalizada (Template Matching),
    y aplica exactamente el mismo recorte sobre la nueva versión v2.
    """
    ref_files = sorted(list(ref_dir.rglob("*.png")))
    total = len(ref_files)
    print(f"Iniciando replicación de recortes para {total} imágenes...")
    print(f"Referencia (recortes): {ref_dir.resolve()}")
    print(f"Origen (completas v2): {src_dir.resolve()}")
    print(f"Destino:               {dst_dir.resolve()}\n")

    dst_dir.mkdir(parents=True, exist_ok=True)
    log_rows = []
    exitos = 0
    faltantes = 0
    errores = 0

    t0 = time.time()

    for i, ref_file in enumerate(ref_files, 1):
        rel_path = ref_file.relative_to(ref_dir)
        src_file = src_dir / rel_path
        dst_file = dst_dir / rel_path

        if not src_file.exists():
            print(f"[!] No existe en v2: {rel_path}")
            faltantes += 1
            continue

        try:
            # Leer imagen de referencia recortada e imagen completa v2
            img_crop_ref = cv2_imread_unicode(ref_file, cv2.IMREAD_GRAYSCALE)
            img_full_v2 = cv2_imread_unicode(src_file, cv2.IMREAD_GRAYSCALE)

            if img_crop_ref is None or img_full_v2 is None:
                errores += 1
                continue

            h_crop, w_crop = img_crop_ref.shape[:2]
            h_full, w_full = img_full_v2.shape[:2]

            # Si el recorte ya es del mismo tamaño que la imagen completa
            if h_crop == h_full and w_crop == w_full:
                x, y = 0, 0
                max_val = 1.0
            else:
                # Template matching por correlación normalizada (invariante a brillo/contraste)
                res = cv2.matchTemplate(img_full_v2, img_crop_ref, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, (x, y) = cv2.minMaxLoc(res)

            # Aplicar el recorte sobre la imagen v2
            # El borde superior e inferior se definen según EXTRA_BAJO_PX:
            #   None  → llega hasta el borde inferior de la imagen completa
            #   int   → añade ese número de píxeles al recorte manual
            offset_arriba = y if EXTRA_ARRIBA_PX is None else min(y, EXTRA_ARRIBA_PX)
            y_ini = y - offset_arriba
            y_fin = h_full if EXTRA_BAJO_PX is None else min(h_full, y + h_crop + EXTRA_BAJO_PX)
            cropped_v2 = img_full_v2[y_ini : y_fin, x : x + w_crop]

            # Guardar manteniendo estructura de carpetas
            cv2_imwrite_unicode(dst_file, cropped_v2)

            log_rows.append([str(rel_path), x, y, w_crop, h_crop, y_fin - y_ini, offset_arriba, f"{max_val:.4f}"])
            exitos += 1

        except Exception as e:
            print(f"[ERROR] {rel_path}: {e}")
            errores += 1

        if i % 200 == 0 or i == total:
            print(f"  Progreso: {i}/{total} procesadas ({exitos} exitosas)")

    # Guardar log con coordenadas y confianza del matching
    log_file = dst_dir / "coordenadas_recortes.csv"
    with open(log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["archivo", "x", "y", "width", "height", "height_nuevo", "offset_arriba", "matching_score"])
        writer.writerows(log_rows)

    elapsed = time.time() - t0
    print("\n--- RESUMEN ---")
    print(f"Total procesadas:   {total}")
    print(f"Recortes exitosos:  {exitos}")
    print(f"Faltantes en v2:    {faltantes}")
    print(f"Errores:            {errores}")
    print(f"Tiempo total:       {elapsed:.2f}s ({elapsed/total*1000:.1f} ms/imagen)")
    print(f"Dataset generado:   {dst_dir.resolve()}")
    print(f"Log de coordenadas: {log_file.resolve()}")


if __name__ == "__main__":
    replicar_recortes(REF_CROPPED_DIR, SRC_UNCROPPED_DIR, DST_DIR)
