"""Embeddings de una ResNet-50 congelada (ImageNet, sin ajustar con nuestras etiquetas) para los modelos 3 y 4.

Cada imagen se convierte en un vector de 2048 valores (salida antes de la capa final). Se guardan fuera del repo
(~290 MB) en DEEPSPINE_RUNS/embeddings_resnet50_imagenet.npz, en el mismo orden que el indice del dataset.

Uso (en Ubuntu, desde la raiz del repo):
    python scripts/entrenamiento/03_embeddings.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from src.data.dataset import cargar_indice, verificar
from src.models.resnet import cargador, extractor_congelado, inferir
from src.utils.config import RUNS_DIR, SEMILLA, fijar_semilla

EMB_F = RUNS_DIR / "embeddings_resnet50_imagenet.npz"

fijar_semilla(SEMILLA)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
df = cargar_indice()
print("Pacientes por particion:", verificar(df))
t0 = time.time()
EMB = inferir(extractor_congelado().to(DEV), cargador(df.ruta.values, batch=128), DEV, softmax=False)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
np.savez(EMB_F, emb=EMB.astype(np.float32), archivos=np.array(df.archivo.tolist()))
print(f"Embeddings {EMB.shape} en {time.time() - t0:.0f} s -> {EMB_F}")
