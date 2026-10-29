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


class Caso(BaseModel):
    id: str
    archivo: str
    paciente: str
    nivel: str
    clase_real: str
    particion: str
    predicciones_registradas: Dict[str, str]


class Comparacion(BaseModel):
    modelo: str
    clase_predicha: str
    acierta: bool
    coincide_con_registrada: bool


class Traza(BaseModel):
    predicciones: List[Prediccion]
    pasos: Dict[str, Any]
    ms_total: float
    coinciden: bool
    ensamble: Dict[str, Any]
    imagen_224: str
    caso: Optional[Caso] = None
    comparacion: Optional[List[Comparacion]] = None


class ModeloInfo(BaseModel):
    modelo: str
    nombre: str
    archivo: str
    sha256: Optional[str]
    disponible: bool
    configuracion: Dict[str, Any]
    metricas_test: Dict[str, Dict[str, float]]


class EstadoSistema(BaseModel):
    estado: str
    dispositivo: str
    commit: Optional[str]
    imagenes_demo_ok: bool
    modelos_disponibles: List[str]
    faltan: List[str]
    casos_demo: int


class ResultadoModeloSimulacion(BaseModel):
    modelo: str
    nombre: str
    aciertos: int
    exactitud: float
    reproducidas: int
    matriz_confusion: List[List[int]]


class Simulacion(BaseModel):
    n: int
    clases: List[str]
    por_modelo: List[ResultadoModeloSimulacion]
    ensamble_aciertos: int
    segundos: float
    nota: str
