"""Embeddings de ResNet-50 (2048 valores por imagen, salida antes de la capa final) para los modelos 3 y 4.

Dos extractores:
- ajustada (oficial): la ResNet-50 del modelo 2 (models/resnet50.pt), ajustada solo con train.
- imagenet (comparacion): ResNet-50 con pesos de ImageNet, sin ajustar con nuestras etiquetas.
Se guardan fuera del repo (~290 MB) en DEEPSPINE_RUNS/embeddings_resnet50_<extractor>.npz, en el orden del indice.

Uso (en Ubuntu, desde la raiz del repo; ajustada requiere haber corrido 02_resnet50.py):
    python scripts/entrenamiento/03_embeddings.py [--extractor ajustada|imagenet]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from src.data.dataset import cargar_indice, verificar
from src.models.resnet import cargador, extractor_ajustado, extractor_congelado, inferir
from src.utils.config import RUNS_DIR, SEMILLA, fijar_semilla

ap = argparse.ArgumentParser()
ap.add_argument("--extractor", choices=("ajustada", "imagenet"), default="ajustada")
args = ap.parse_args()
EMB_F = RUNS_DIR / f"embeddings_resnet50_{args.extractor}.npz"

fijar_semilla(SEMILLA)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
df = cargar_indice()
print("Pacientes por particion:", verificar(df))
t0 = time.time()
extractor = extractor_ajustado() if args.extractor == "ajustada" else extractor_congelado()
EMB = inferir(extractor.to(DEV), cargador(df.ruta.values, batch=128), DEV, softmax=False)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
np.savez(EMB_F, emb=EMB.astype(np.float32), archivos=np.array(df.archivo.tolist()))
print(f"Embeddings {EMB.shape} en {time.time() - t0:.0f} s -> {EMB_F}")
