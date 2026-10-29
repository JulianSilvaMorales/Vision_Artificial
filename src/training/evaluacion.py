"""Cierre comun de cada entrenamiento: metricas, reportes, figuras y metadatos en docs/resultados/<slug>/."""
import datetime
import platform

import numpy as np

from src.utils.config import CLASES, RESULTADOS_DIR, SEMILLA
from src.utils.graficas import curvas, matriz_confusion
from src.utils.metricas import guardar_resultados, imprimir


def versiones():
    v = dict(python=platform.python_version())
    for mod in ("torch", "torchvision", "ultralytics", "sklearn", "numpy"):
        try:
            v[mod] = __import__(mod).__version__
        except ImportError:
            pass
    return v


def cerrar_modelo(slug, nombre, S, es_prob, df, conj, config, hist=None):
    meta = dict(modelo=nombre, fecha=datetime.datetime.now().isoformat(timespec="seconds"), semilla=SEMILLA,
                clases=CLASES, config=config, versiones=versiones(),
                seleccion="hiperparametros y epoca elegidos con validacion; el test se evalua una sola vez al final")
    tabla = guardar_resultados(slug, nombre, S, es_prob, df, conj, extra=meta)
    out = RESULTADOS_DIR / slug
    y = df.y.values
    for cj, titulo in (("test_orig", "test originales"), ("test_completo", "test completo")):
        m = conj[cj]
        matriz_confusion(y[m], S[m].argmax(1), f"{nombre} · {titulo} (n={m.sum()})",
                         out / f"matriz_confusion_{cj}.png", normalizar=True)
    if hist is not None:
        hist.to_csv(out / "historial.csv", index=False)
        curvas(hist, out / "curvas_entrenamiento.png", nombre)
    print(f"\n== {nombre} ==")
    imprimir(tabla)
    print("Resultados en:", out)
    return tabla
