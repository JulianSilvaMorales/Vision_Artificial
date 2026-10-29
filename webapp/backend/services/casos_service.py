"""Casos de demostracion (data/demo): originales de test con su etiqueta real y la prediccion registrada."""
from functools import lru_cache

import pandas as pd

from webapp.backend.core.config import CASOS_CSV, IMAGENES_DEMO, MODELOS


@lru_cache(maxsize=1)
def _tabla():
    return pd.read_csv(CASOS_CSV, dtype={"paciente": str}).set_index("id", drop=False)


def _a_dict(fila):
    return dict(id=fila.id, archivo=fila.archivo, paciente=fila.paciente, nivel=fila.nivel, clase_real=fila.clase_real,
                particion=fila.particion, predicciones_registradas={m: fila[f"pred_registrada_{m}"] for m in MODELOS})


def listar(clase=None, nivel=None):
    t = _tabla()
    if clase:
        t = t[t.clase_real == clase]
    if nivel:
        t = t[t.nivel == nivel]
    return [_a_dict(f) for _, f in t.iterrows()]


def obtener(id_caso):
    t = _tabla()
    if id_caso not in t.index:
        return None
    return _a_dict(t.loc[id_caso])


def ruta_imagen(caso):
    return IMAGENES_DEMO / caso["archivo"]
