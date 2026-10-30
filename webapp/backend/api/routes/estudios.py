"""Estudios de resonancia (.IMA / .dcm / .zip): conversion a PNG, deteccion de los 5 discos y prediccion."""
from typing import List

from fastapi import APIRouter, File, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from webapp.backend.core.config import ESTUDIO_MAX_BYTES
from webapp.backend.services import estudios_service, modelos_service

router = APIRouter(prefix="/api/estudios", tags=["estudios"])


class SeleccionCortes(BaseModel):
    serie: str
    cortes: List[int] = Field(..., min_length=1, max_length=50)
    recortar_zona_lumbar: bool = True    # False si el corte ya esta recortado a L1-S1


def _exigir_modelos():
    falta = modelos_service.faltantes()
    if falta:
        raise HTTPException(503, f"Faltan pesos en models/: {falta}. Ejecuta: python scripts/descargar_modelos.py")


@router.post("")
async def subir(archivos: List[UploadFile] = File(...)):
    """Sube los .IMA / .dcm (o un .zip con ellos), los convierte a PNG y devuelve las series con sus cortes.
    Los DICOM se borran al convertirlos; el estudio completo se borra solo a los 30 minutos."""
    leidos = [(a.filename or "archivo", await a.read(ESTUDIO_MAX_BYTES + 1)) for a in archivos]
    return await run_in_threadpool(estudios_service.crear, leidos)


@router.get("/{id_estudio}")
def ver(id_estudio: str):
    return estudios_service.ficha(id_estudio)


@router.get("/{id_estudio}/series/{serie}/cortes/{n}/imagen")
def imagen_corte(id_estudio: str, serie: str, n: int):
    return FileResponse(estudios_service.ruta_corte(id_estudio, serie, n), media_type="image/png")


@router.post("/{id_estudio}/recortar")
async def recortar(id_estudio: str, sel: SeleccionCortes):
    """Detector de discos y recorte de los 5 ROI de cada corte elegido, sin clasificar."""
    return await run_in_threadpool(estudios_service.recortar, id_estudio, sel.serie, sel.cortes, sel.recortar_zona_lumbar)


@router.post("/{id_estudio}/predecir")
async def predecir(id_estudio: str, sel: SeleccionCortes):
    """Recorta y corre los 4 modelos sobre cada ROI. Un ROI rechazado trae `rechazado` en vez de la traza."""
    _exigir_modelos()
    return await run_in_threadpool(estudios_service.predecir_cortes, id_estudio, sel.serie, sel.cortes,
                                   sel.recortar_zona_lumbar)


@router.post("/{id_estudio}/png.zip")
def descargar(id_estudio: str, sel: SeleccionCortes):
    """Descarga los PNG de los cortes elegidos."""
    datos = estudios_service.descargar_zip(id_estudio, sel.serie, sel.cortes)
    return Response(datos, media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="cortes.zip"'})


@router.delete("/{id_estudio}", status_code=204)
def borrar(id_estudio: str):
    estudios_service.borrar(id_estudio)
