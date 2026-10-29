"""Estado del sistema y de los modelos."""
from typing import List

from fastapi import APIRouter

from webapp.backend.schemas.prediccion import EstadoSistema, ModeloInfo
from webapp.backend.services import modelos_service

router = APIRouter(prefix="/api", tags=["sistema"])


@router.get("/salud", response_model=EstadoSistema)
def salud():
    return modelos_service.estado()


@router.get("/modelos", response_model=List[ModeloInfo])
def modelos():
    """Los 4 modelos: disponibilidad, SHA-256 del archivo, configuracion y metricas oficiales."""
    return modelos_service.info_modelos()
