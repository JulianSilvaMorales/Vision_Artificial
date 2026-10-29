"""Orquesta la prediccion: carga unica del Predictor, traza de los 4 modelos y comparacion con la etiqueta real."""
import time
from functools import lru_cache

import numpy as np

from src.inference.predictor import Predictor
from src.utils.config import CLASES
from webapp.backend.core.config import DISPOSITIVO, MODELOS, MODELOS_DIR
from webapp.backend.services import casos_service
from webapp.backend.utils.imagenes import a_base64_png


@lru_cache(maxsize=1)
def predictor():
    return Predictor(MODELOS_DIR, dispositivo=DISPOSITIVO)


def precargar():
    """Carga los 4 modelos antes de recibir peticiones; devuelve los segundos que tardo."""
    t0 = time.perf_counter()
    predictor().cargar_todos()
    return round(time.perf_counter() - t0, 1)


def _traza(imagen):
    tr = predictor().trazar(imagen)
    tr["imagen_224"] = a_base64_png(tr["imagen_224"])
    return tr


def predecir_imagen(contenido):
    """Imagen subida por el usuario: no tiene etiqueta real, solo la traza."""
    return _traza(contenido)


def predecir_caso(caso):
    tr = _traza(casos_service.ruta_imagen(caso))
    tr["caso"] = caso
    tr["comparacion"] = [dict(modelo=p["modelo"], clase_predicha=p["clase"], acierta=p["clase"] == caso["clase_real"],
                              coincide_con_registrada=p["clase"] == caso["predicciones_registradas"][p["modelo"]])
                         for p in tr["predicciones"]]
    return tr


def simular(casos):
    """Corre los 4 modelos sobre los casos y arma exactitud, matriz de confusion y reproducibilidad por modelo."""
    t0 = time.perf_counter()
    idx = {c: i for i, c in enumerate(CLASES)}
    mats = {m: np.zeros((len(CLASES), len(CLASES)), int) for m in MODELOS}
    reprod = {m: 0 for m in MODELOS}
    nombres, ens_ok = {}, 0
    for caso in casos:
        tr = predictor().trazar(casos_service.ruta_imagen(caso))
        real = idx[caso["clase_real"]]
        for p in tr["predicciones"]:
            mats[p["modelo"]][real, idx[p["clase"]]] += 1
            reprod[p["modelo"]] += p["clase"] == caso["predicciones_registradas"][p["modelo"]]
            nombres[p["modelo"]] = p["nombre"]
        ens_ok += tr["ensamble"]["clase"] == caso["clase_real"]
    n = len(casos)
    por_modelo = [dict(modelo=m, nombre=nombres.get(m, m), aciertos=int(np.trace(mats[m])),
                       exactitud=round(float(np.trace(mats[m]) / n), 4) if n else 0.0, reproducidas=int(reprod[m]),
                       matriz_confusion=mats[m].tolist()) for m in MODELOS]
    return dict(n=n, clases=CLASES, por_modelo=por_modelo, ensamble_aciertos=int(ens_ok),
                segundos=round(time.perf_counter() - t0, 1),
                nota="Casos de demostracion balanceados (15 por clase); su exactitud no equivale a la del test oficial.")
