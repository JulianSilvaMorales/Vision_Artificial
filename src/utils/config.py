"""Rutas y constantes compartidas por el entrenamiento y la inferencia.

Las rutas de datos y de corridas se pueden cambiar con variables de entorno:
- DEEPSPINE_DATA: carpeta con train/, val/ y test/ del dataset de 35k (por defecto ~/deepspine_data).
- DEEPSPINE_RUNS: carpeta para checkpoints y embeddings, fuera del repo (por defecto ~/deepspine_runs).
"""
import os
import random
from pathlib import Path

import numpy as np

RAIZ_REPO = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DEEPSPINE_DATA", Path.home() / "deepspine_data"))
RUNS_DIR = Path(os.environ.get("DEEPSPINE_RUNS", Path.home() / "deepspine_runs"))
RESULTADOS_DIR = RAIZ_REPO / "docs" / "resultados"
MODELOS_DIR = RAIZ_REPO / "models"
MANIFIESTO = RAIZ_REPO / "data" / "processed" / "dataset_35k" / "manifest_35k.csv"

SEMILLA = 42
IMGSZ = 224
CLASES = ["Degeneracion_Discal", "Normal", "Protrusion"]   # orden alfabetico, igual que ImageFolder y ultralytics
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # normalizacion de ImageNet


def fijar_semilla(s=SEMILLA):
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    except ImportError:
        pass
