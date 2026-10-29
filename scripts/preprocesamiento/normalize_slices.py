import csv
import cv2
import numpy as np
from pathlib import Path

def normalize_slice(img, p_low=1.0, p_high=99.0):
    """
    Recorte de percentiles + min-max sobre el CORTE COMPLETO.
    Los percentiles se calculan solo sobre píxeles de tejido (>0), para que
    la cantidad de fondo negro no cambie el estiramiento entre pacientes.
    Devuelve (imagen uint8, p_low, p_high).
    """
    img = img.astype(np.float32)
    fg = img[img > 0]
    ref = fg if fg.size > 1000 else img.ravel()
    lo, hi = np.percentile(ref, (p_low, p_high))
    if hi <= lo:
        return np.clip(img, 0, 255).astype(np.uint8), float(lo), float(hi)
    out = (np.clip(img, lo, hi) - lo) / (hi - lo) * 255.0
    return np.round(out).astype(np.uint8), float(lo), float(hi)

def process_dataset(input_dir, output_dir, p_low=1.0, p_high=99.0):
    in_path, out_path = Path(input_dir), Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    files = sorted(in_path.rglob("*.png"))
    print(f"Normalizando {len(files)} cortes completos...")

    log, fallos = [], []
    for i, f in enumerate(files, 1):
        try:
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)  # conserva 8/16 bits
            if img is None:
                fallos.append((str(f), "no se pudo leer")); continue
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            norm, lo, hi = normalize_slice(img, p_low, p_high)

            dest = out_path / f.relative_to(in_path)      # misma estructura por paciente
            dest.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest), norm)
            log.append((str(f.relative_to(in_path)), img.dtype, lo, hi))
        except Exception as e:
            fallos.append((str(f), repr(e)))
        if i % 500 == 0:
            print(f"  {i}/{len(files)}")

    with open(out_path / "normalizacion_log.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["archivo", "dtype_original", "p_low", "p_high"])
        w.writerows(log)
    if fallos:
        (out_path / "fallos.txt").write_text("\n".join(f"{a}\t{b}" for a, b in fallos), encoding="utf-8")

    print(f"Listo: {len(log)} normalizados, {len(fallos)} fallos -> {out_path.resolve()}")

if __name__ == "__main__":
    process_dataset(
        input_dir="./t2_midsagittal_l1s1_cropped",
        output_dir="./t2_midsagittal_l1s1_normalized",
    )