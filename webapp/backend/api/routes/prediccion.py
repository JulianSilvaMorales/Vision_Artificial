"""Prediccion sobre una imagen subida por el usuario."""
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from webapp.backend.core.config import TAMANO_MAXIMO_IMAGEN
from webapp.backend.schemas.prediccion import Traza
from webapp.backend.services import modelos_service, prediccion_service, validacion_service

router = APIRouter(prefix="/api", tags=["prediccion"])


def _exigir_modelos():
    falta = modelos_service.faltantes()
    if falta:
        raise HTTPException(503, f"Faltan pesos en models/: {falta}. Ejecuta: python scripts/descargar_modelos.py")


async def _leer_imagen(imagen: UploadFile):
    """Lee la imagen subida con tope de tamano y la abre en RGB (sin pasarla a gris)."""
    contenido = await imagen.read(TAMANO_MAXIMO_IMAGEN + 1)
    if len(contenido) > TAMANO_MAXIMO_IMAGEN:
        raise HTTPException(413, "La imagen supera 5 MB")
    try:
        return validacion_service.abrir_original(contenido)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError):
        raise HTTPException(400, "El archivo no es una imagen valida")


@router.post("/predecir", response_model=Traza)
async def predecir(imagen: UploadFile = File(...)):
    """Recibe el recorte (ROI) de un disco y devuelve la prediccion de los 4 modelos con su traza.

    Responde 422 si la imagen no parece un ROI de disco lumbar (foto en color, imagen fuera de distribucion...). Si es un
    corte completo de columna responde 422 con motivo `corte_completo`."""
    img = await _leer_imagen(imagen)
    validacion_service.validar_formato(img)
    _exigir_modelos()
    return await run_in_threadpool(prediccion_service.predecir_imagen, img)

