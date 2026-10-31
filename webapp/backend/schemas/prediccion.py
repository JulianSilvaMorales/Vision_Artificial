"""Esquemas de entrada y salida de la API."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Prediccion(BaseModel):
    modelo: str
    nombre: str
    clase: str
    confianza: float
    puntajes: Dict[str, float]
    es_probabilidad: bool


class Traza(BaseModel):
    predicciones: List[Prediccion]
    pasos: Dict[str, Any]
    ms_total: float
    coinciden: bool
    ensamble: Dict[str, Any]
    imagen_224: str


class EstadoSistema(BaseModel):
    estado: str
    dispositivo: str
    commit: Optional[str]
    detector_ok: bool = False
    modelos_disponibles: List[str]
    faltan: List[str]
