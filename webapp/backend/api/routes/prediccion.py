"""Prediccion sobre una imagen subida por el usuario."""
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from webapp.backend.core.config import TAMANO_MAXIMO_IMAGEN
from webapp.backend.schemas.prediccion import Traza
from webapp.backend.services import modelos_service, prediccion_service

router = APIRouter(prefix="/api", tags=["prediccion"])


@router.post("/predecir", response_model=Traza)
async def predecir(imagen: UploadFile = File(...)):
    """Recibe el recorte (ROI) de un disco y devuelve la prediccion de los 4 modelos con su traza."""
    falta = modelos_service.faltantes()
    if falta:
        raise HTTPException(503, f"Faltan pesos en models/: {falta}. Ejecuta: python scripts/descargar_modelos.py")
    contenido = await imagen.read()
    if len(contenido) > TAMANO_MAXIMO_IMAGEN:
        raise HTTPException(413, "La imagen supera 5 MB")
    try:
        return prediccion_service.predecir_imagen(contenido)
    except UnidentifiedImageError:
        raise HTTPException(400, "El archivo no es una imagen valida")
