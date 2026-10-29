import shutil
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# Ruta hacia el archivo normalizacion_log.csv
csv_path = BASE_DIR / "t2_midsagittal_l1s1_normalized" / "normalizacion_log.csv"
if not csv_path.exists():
    csv_path = Path("t2_midsagittal_l1s1_normalized/normalizacion_log.csv")

# Se usa sep=None (engine='python') para detectar automáticamente si el separador es ';' o ','
df = pd.read_csv(csv_path, sep=None, engine="python")
keep = df["archivo"].astype(str).str.replace("\\", "/", regex=False)

src = BASE_DIR / "t2_midsagittal_png_v2"
dst = BASE_DIR / "t2_midsagittal_v2_curated"

copiados = 0
faltantes = 0

for rel in keep:
    s = src / rel
    if s.exists():
        d = dst / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        copiados += 1
    else:
        print("Falta:", rel)
        faltantes += 1

print(f"\nProceso finalizado: {copiados} imágenes copiadas, {faltantes} faltantes en '{dst.name}'.")