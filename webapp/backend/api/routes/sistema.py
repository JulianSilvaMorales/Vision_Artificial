"""Estado del sistema y de los modelos."""
from fastapi import APIRouter

from webapp.backend.schemas.prediccion import EstadoSistema
from webapp.backend.services import modelos_service

router = APIRouter(prefix="/api", tags=["sistema"])


@router.get("/salud", response_model=EstadoSistema)
def salud():
    return modelos_service.estado()
