"""Casos de demostracion: lista, imagen y prediccion trazada."""
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from webapp.backend.schemas.prediccion import Caso, Simulacion, Traza
from webapp.backend.services import casos_service, modelos_service, prediccion_service

router = APIRouter(prefix="/api/casos", tags=["casos"])


def _caso_o_404(id_caso):
    caso = casos_service.obtener(id_caso)
    if caso is None:
        raise HTTPException(404, f"No existe el caso {id_caso}")
    return caso


def _exigir_modelos():
    falta = modelos_service.faltantes()
    if falta:
        raise HTTPException(503, f"Faltan pesos en models/: {falta}. Ejecuta: python scripts/descargar_modelos.py")


@router.get("", response_model=List[Caso])
def listar(clase: Optional[str] = None, nivel: Optional[str] = None):
    return casos_service.listar(clase, nivel)


@router.get("/{id_caso}/imagen")
def imagen(id_caso: str):
    return FileResponse(casos_service.ruta_imagen(_caso_o_404(id_caso)), media_type="image/png")


@router.post("/{id_caso}/predecir", response_model=Traza)
def predecir(id_caso: str):
    """Corre los 4 modelos sobre el caso y lo compara con la etiqueta real y con la prediccion registrada."""
    caso = _caso_o_404(id_caso)
    _exigir_modelos()
    return prediccion_service.predecir_caso(caso)


@router.post("/simulacion", response_model=Simulacion)
def simulacion(clase: Optional[str] = None, nivel: Optional[str] = None):
    """Corre los 4 modelos sobre todos los casos (o los filtrados) y devuelve exactitud y matriz de confusion."""
    _exigir_modelos()
    casos = casos_service.listar(clase, nivel)
    if not casos:
        raise HTTPException(404, "No hay casos con ese filtro")
    return prediccion_service.simular(casos)
