"""Casos de demostracion para el MVP: imagenes originales de TEST que se incluyen en el repo.

Elige N casos por clase (semilla 42), repartidos entre los 5 niveles, solo de la particion de test y solo
originales (sin copias aumentadas), para que cualquiera que clone el repo pueda correr la demo sin el dataset
de 35k. Copia las imagenes a data/demo/imagenes/ y escribe data/demo/casos.csv con los datos de cada caso y la
prediccion que registro la evaluacion de cada modelo (docs/resultados/<modelo>/puntajes.npz), para comprobar
en la demo que el resultado se reproduce.

Uso (con el dataset de 35k descomprimido, desde la raiz del repo):
    python scripts/demo/seleccionar_casos_demo.py [--por-clase 15]
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.data.dataset import cargar_indice, verificar
from src.utils.config import CLASES, RAIZ_REPO, SEMILLA
from src.utils.metricas import cargar_puntajes

MODELOS = {"yolov8s_cls": "modelo1_yolov8s_cls", "resnet50": "modelo2_resnet50",
           "svm": "modelo3_svm", "random_forest": "modelo4_random_forest"}
OUT = RAIZ_REPO / "data" / "demo"

ap = argparse.ArgumentParser()
ap.add_argument("--por-clase", type=int, default=15)
args = ap.parse_args()

df = cargar_indice()
verificar(df)
rng = np.random.default_rng(SEMILLA)
test = df[(df.split == "test") & df.es_original]
elegidos = []
for clase in CLASES:
    sub = test[test.clase == clase]
    # ronda por niveles en orden aleatorio: reparte los casos entre L1-L2 ... L5-S1 cuando la clase lo permite
    grupos = {n: list(rng.permutation(g.index)) for n, g in sub.groupby("nivel")}
    orden = []
    while len(orden) < min(args.por_clase, len(sub)):
        for n in rng.permutation(sorted(grupos)):
            if grupos[n] and len(orden) < args.por_clase:
                orden.append(grupos[n].pop())
    elegidos += orden

casos = df.loc[elegidos].copy()
for corto, slug in MODELOS.items():
    _, S, _ = cargar_puntajes(slug, df)
    casos[f"pred_registrada_{corto}"] = [CLASES[i] for i in S[elegidos].argmax(1)]

(OUT / "imagenes").mkdir(parents=True, exist_ok=True)
for r in casos.itertuples():
    shutil.copy(r.ruta, OUT / "imagenes" / r.archivo)
casos = casos.rename(columns={"clase": "clase_real", "split": "particion"})
casos.insert(0, "id", [f"caso_{i:02d}" for i in range(1, len(casos) + 1)])
cols = ["id", "archivo", "paciente", "nivel", "clase_real", "particion", "es_original"] + [c for c in casos if c.startswith("pred_")]
casos[cols].to_csv(OUT / "casos.csv", index=False)
print(casos.groupby(["clase_real", "nivel"]).size().unstack(fill_value=0))
print(f"{len(casos)} casos de {casos.paciente.nunique()} pacientes -> {OUT}")
