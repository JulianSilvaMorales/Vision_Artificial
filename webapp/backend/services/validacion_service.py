"""Rechazo de imagenes que no son un ROI de disco lumbar ni un corte sagital de columna.

Los 4 modelos siempre eligen una de las 3 clases, sea cual sea la imagen. Este filtro corre antes de predecir:
1. Formato: la imagen debe ser (casi) en escala de grises y de un tamano razonable.
2. Tipo: casi cuadrada y pequena = ROI; grande o alargada = corte completo (se manda al detector de discos).
3. Fuera de distribucion: distancia de Mahalanobis del embedding de la ResNet-50 en el espacio PCA del SVM, contra un
   umbral calibrado con ROI de validacion (models/umbral_rechazo.json, ver scripts/demo/calibrar_rechazo.py).
   Si el archivo del umbral no existe, esta capa no se aplica y la respuesta lo indica.
"""
import hashlib
import io
import json
from functools import lru_cache

import numpy as np
from PIL import Image

from webapp.backend.core.config import UMBRAL_RECHAZO_JSON

LADO_MINIMO = 32
LADO_MAXIMO_ROI = 256
PROPORCION_MAXIMA_ROI = 1.3
PROPORCION_MAXIMA = 4.0
DIFERENCIA_COLOR_MAXIMA = 8.0      # media de |R-G|, |G-B|, |R-B| en niveles de gris


class ImagenRechazada(Exception):
    def __init__(self, motivo, mensaje, capa, valor=None, umbral=None):
        super().__init__(mensaje)
        self.detalle = dict(motivo=motivo, mensaje=mensaje, capa=capa, valor=valor, umbral=umbral)


def abrir_original(contenido):
    """Abre los bytes de una imagen en RGB, sin convertirla a gris. Lanza OSError / UnidentifiedImageError si es invalida."""
    img = Image.open(io.BytesIO(contenido))
    img.load()
    return img.convert("RGB")


def diferencia_color(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return float((np.abs(r - g).mean() + np.abs(g - b).mean() + np.abs(r - b).mean()) / 3)


def validar_formato(img):
    """Capa 1: rechaza fotos en color, imagenes diminutas o con proporciones raras."""
    w, h = img.size
    if min(w, h) < LADO_MINIMO:
        raise ImagenRechazada("muy_pequena", f"La imagen mide {w}x{h} px; es demasiado pequena.", 1, min(w, h), LADO_MINIMO)
    prop = max(w, h) / min(w, h)
    if prop > PROPORCION_MAXIMA:
        raise ImagenRechazada("proporcion", f"Proporcion {prop:.1f}:1; no parece un ROI ni un corte de resonancia.", 1,
                              round(prop, 2), PROPORCION_MAXIMA)
    d = diferencia_color(img)
    if d > DIFERENCIA_COLOR_MAXIMA:
        raise ImagenRechazada("color", "La imagen es en color; una resonancia o su ROI es en escala de grises.", 1,
                              round(d, 2), DIFERENCIA_COLOR_MAXIMA)


def tipo_imagen(img):
    """Capa 2: 'roi' si es chica y casi cuadrada (como los 128x128 del dataset); si no, 'corte' completo."""
    w, h = img.size
    if max(w, h) <= LADO_MAXIMO_ROI and max(w, h) / min(w, h) <= PROPORCION_MAXIMA_ROI:
        return "roi"
    return "corte"


@lru_cache(maxsize=1)
def umbral():
    """Umbral de distancia al cuadrado, o None si no se calibro. Se invalida si cambia el PCA del SVM."""
    if not UMBRAL_RECHAZO_JSON.exists():
        return None
    return json.loads(UMBRAL_RECHAZO_JSON.read_text(encoding="utf-8"))


def distancia2(pca, emb_escalado):
    """d^2 = suma(z_i^2 / lambda_i) del embedding ya escalado, proyectado con el PCA del SVM."""
    z = pca.transform(emb_escalado[None])[0]
    return float(np.sum(z ** 2 / pca.explained_variance_))


def huella_pca(pca):
    return hashlib.sha256(np.ascontiguousarray(pca.components_).tobytes()).hexdigest()[:16]


def validar_fuera_de_distribucion(d2, pca):
    """Capa 3. Devuelve el estado de la capa: 'aplicada' o 'sin_umbral'."""
    cfg = umbral()
    if cfg is None:
        return "sin_umbral"
    if cfg.get("pca_huella") and cfg["pca_huella"] != huella_pca(pca):
        return "umbral_obsoleto"
    if d2 > cfg["d2_umbral"]:
        raise ImagenRechazada("fuera_de_distribucion",
                              "La imagen no se parece a un disco lumbar de resonancia sagital T2.", 3,
                              round(d2, 1), round(cfg["d2_umbral"], 1))
    return "aplicada"
