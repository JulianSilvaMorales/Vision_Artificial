"""Orquesta la prediccion: carga unica del Predictor y traza de los 4 modelos."""
import time
from functools import lru_cache

from src.inference.predictor import Predictor
from webapp.backend.core.config import DISPOSITIVO, MODELOS_DIR
from webapp.backend.services import validacion_service
from webapp.backend.utils.imagenes import a_base64_png


@lru_cache(maxsize=1)
def predictor():
    return Predictor(MODELOS_DIR, dispositivo=DISPOSITIVO)


def precargar():
    """Carga los 4 modelos antes de recibir peticiones; devuelve los segundos que tardo."""
    t0 = time.perf_counter()
    predictor().cargar_todos()
    return round(time.perf_counter() - t0, 1)


def _traza(imagen, validar=False):
    """Traza de los 4 modelos. Con `validar`, rechaza (ImagenRechazada) lo que no se parece a un disco, antes de clasificar."""
    estado = {}

    def chequeo(emb):
        d2, pca = predictor().distancia_ood(emb)
        estado.update(capa3=validacion_service.validar_fuera_de_distribucion(d2, pca), d2=round(d2, 1))

    tr = predictor().trazar(imagen, validar=chequeo if validar else None)
    if validar:
        tr["pasos"]["validacion"] = estado
    tr["imagen_224"] = a_base64_png(tr["imagen_224"])
    return tr


def trazar_roi(roi_gris):
    """ROI en escala de grises (np.ndarray) ya recortado por el detector: solo se aplica el rechazo fuera de distribucion."""
    return _traza(roi_gris, validar=True)


def predecir_imagen(img):
    """Imagen subida (PIL, ya con el formato validado): devuelve la traza.
    Rechaza lo que no es un ROI de disco: los cortes completos van por estudios_service.analizar_corte_subido."""
    if validacion_service.tipo_imagen(img) == "corte":
        raise validacion_service.ImagenRechazada(
            "corte_completo", "Parece un corte completo de columna, no un ROI de un disco.", 2, list(img.size))
    return _traza(img, validar=True)
