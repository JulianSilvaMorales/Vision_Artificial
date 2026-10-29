import csv, random
from pathlib import Path
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
REF = BASE / "t2_midsagittal_l1s1_cropped"
NEW = BASE / "t2_midsagittal_v2_l1s1_cropped"
LOG = NEW / "coordenadas_recortes.csv"
UMBRAL = 0.95
H = 220

def leer(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

def a_altura(img):
    return cv2.resize(img, (max(1, round(img.shape[1] * H / img.shape[0])), H))

def pad_ancho(img, w):
    out = np.zeros((img.shape[0], w), np.uint8)
    out[:, :img.shape[1]] = img
    return out

rows = list(csv.DictReader(open(LOG, encoding="utf-8")))
sc = np.array([float(r["matching_score"]) for r in rows])
print(f"{len(rows)} recortes | score mediana {np.median(sc):.4f} | min {sc.min():.4f}")
for t in (0.99, 0.95, 0.90):
    print(f"  score < {t}: {(sc < t).sum()}")

bajos = sorted(rows, key=lambda r: float(r["matching_score"]))
dudosos = [r for r in bajos if float(r["matching_score"]) < UMBRAL]
if dudosos:
    (BASE / "recortes_dudosos.txt").write_text(
        "\n".join(f'{r["archivo"]}\t{r["matching_score"]}' for r in dudosos), encoding="utf-8")
    print(f"Lista de dudosos guardada: recortes_dudosos.txt ({len(dudosos)})")

random.seed(42)
muestra = bajos[:6] + random.sample(bajos[6:], min(6, max(0, len(bajos) - 6)))
pares = []
for r in muestra:
    a, b = leer(REF / r["archivo"]), leer(NEW / r["archivo"])
    if a is None or b is None:
        continue
    b = b[:a.shape[0], :a.shape[1]]   # recortar a la región común (ref puede ser más baja que v2)
    pares.append(np.hstack([a_altura(a), np.full((H, 6), 255, np.uint8), a_altura(b)]))
if pares:
    W = max(p.shape[1] for p in pares)
    pares = [pad_ancho(p, W) for p in pares]
    while len(pares) % 3:
        pares.append(np.zeros_like(pares[0]))
    filas = [np.hstack(pares[i:i + 3]) for i in range(0, len(pares), 3)]
    cv2.imencode(".png", np.vstack(filas))[1].tofile(str(BASE / "comparacion_ref_vs_v2.png"))
    print("Hoja de contacto: comparacion_ref_vs_v2.png (izquierda=referencia, derecha=v2)")