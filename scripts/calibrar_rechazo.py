"""Calibra el umbral con el que la webapp rechaza imagenes que no parecen un ROI de disco lumbar.

Para cada ROI de referencia calcula la distancia de Mahalanobis al cuadrado de su embedding (ResNet-50) en el espacio PCA
del SVM, y toma el percentil indicado (99 por defecto) como umbral. Guarda models/umbral_rechazo.json junto con la huella
del PCA usado: si el modelo cambia, la webapp avisa de que el umbral esta obsoleto.

Uso (desde la raiz del repo, con los 4 modelos en models/):
    python scripts/calibrar_rechazo.py --rois <carpeta con ROI de validacion> [--negativas <carpeta con imagenes que
        NO son discos>] [--percentil 99]

--rois busca PNG/JPG de forma recursiva; por defecto usa solo los originales (nombre con "__orig"), no las copias aumentadas.
Con --negativas informa cuantas de esas imagenes se rechazan.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.inference.predictor import Predictor, abrir_imagen      # noqa: E402
from src.utils.config import MODELOS_DIR                         # noqa: E402
from webapp.backend.services.validacion_service import huella_pca  # noqa: E402

EXTENSIONES = {".png", ".jpg", ".jpeg"}


def listar(carpeta, solo_originales=False):
    rutas = sorted(p for p in Path(carpeta).rglob("*") if p.suffix.lower() in EXTENSIONES)
    return [p for p in rutas if "__orig" in p.name] if solo_originales else rutas


def distancias(pred, rutas):
    d2 = []
    for r in rutas:
        emb, _ = pred._resnet(abrir_imagen(r))
        d2.append(pred.distancia_ood(emb)[0])
    return np.array(d2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rois", required=True, help="carpeta con ROI de validacion (recursivo)")
    ap.add_argument("--todas", action="store_true", help="usar tambien las copias aumentadas (por defecto solo __orig)")
    ap.add_argument("--negativas", help="carpeta con imagenes que NO son discos, para medir el rechazo")
    ap.add_argument("--percentil", type=float, default=99.0)
    ap.add_argument("--salida", default=str(MODELOS_DIR / "umbral_rechazo.json"))
    a = ap.parse_args()

    pred = Predictor(dispositivo="cpu")
    rutas = listar(a.rois, solo_originales=not a.todas)
    if len(rutas) < 100:
        sys.exit(f"Solo {len(rutas)} ROI en {a.rois}: se necesitan al menos 100 para un percentil fiable.")
    d2 = distancias(pred, rutas)
    umbral = float(np.percentile(d2, a.percentil))
    print(f"{len(rutas)} ROI de referencia: d2 mediana {np.median(d2):.1f}, p{a.percentil:g} = {umbral:.1f}, max {d2.max():.1f}")

    Path(a.salida).write_text(json.dumps(dict(
        d2_umbral=round(umbral, 3), percentil=a.percentil, n_referencia=len(rutas), origen=str(a.rois),
        pca_huella=huella_pca(pred._cargar("svm")["pca"]), generado=date.today().isoformat()), indent=2), encoding="utf-8")
    print(f"Umbral guardado en {a.salida}")

    if a.negativas:
        neg = listar(a.negativas)
        d_neg = distancias(pred, neg)
        print(f"Imagenes negativas rechazadas: {int((d_neg > umbral).sum())} / {len(neg)}")
        for r, v in sorted(zip(neg, d_neg), key=lambda t: t[1])[:10]:
            print(f"  aceptada o la mas cercana: {r.name}  d2={v:.1f}")


if __name__ == "__main__":
    main()
