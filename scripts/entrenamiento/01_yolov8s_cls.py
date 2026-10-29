"""Modelo 1 · YOLOv8s-cls con ajuste fino sobre el dataset de 35k.

Repite la configuracion v2 de Jhon Jairo (AdamW, congelar 4 capas, dropout 0.3, coseno, sin aumento en linea:
el aumento ya esta en los archivos). Ultralytics guarda como best.pt la epoca con mayor exactitud top-1 en val
(val esta balanceado, asi que equivale a la exactitud balanceada). El test se evalua una sola vez al final.

Uso (en Ubuntu, desde la raiz del repo):
    python scripts/entrenamiento/01_yolov8s_cls.py [--epocas 15] [--lr0 0.0005]
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.data.dataset import cargar_indice, conjuntos, resumen, verificar
from src.training.evaluacion import cerrar_modelo
from src.utils.config import CLASES, DATA_DIR, IMGSZ, MODELOS_DIR, PIN_MEMORY, RUNS_DIR, SEMILLA, fijar_semilla

SLUG, NOMBRE = "modelo1_yolov8s_cls", "1 · YOLOv8s-cls"

ap = argparse.ArgumentParser()
ap.add_argument("--epocas", type=int, default=15)
ap.add_argument("--paciencia", type=int, default=4)
ap.add_argument("--lr0", type=float, default=5e-4)
ap.add_argument("--lrf", type=float, default=0.1)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--congelar", type=int, default=4)
ap.add_argument("--dropout", type=float, default=0.3)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--workers", type=int, default=8)
args = ap.parse_args()

fijar_semilla(SEMILLA)
df = cargar_indice()
print(resumen(df))
print("Pacientes por particion:", verificar(df), "· sin fuga · coincide con manifest_35k.csv")
conj = conjuntos(df)

from ultralytics import YOLO
from ultralytics.data import build as yolo_build

# build_dataloader no expone pin_memory como argumento de train(); se cambia su valor por defecto (ver PIN_MEMORY)
_d = list(yolo_build.build_dataloader.__defaults__)
_d[3] = PIN_MEMORY                      # (shuffle, rank, drop_last, pin_memory, device)
yolo_build.build_dataloader.__defaults__ = tuple(_d)

RUNS_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(RUNS_DIR)                      # yolov8s-cls.pt se descarga aqui, fuera del repo
modelo = YOLO("yolov8s-cls.pt")
modelo.train(data=str(DATA_DIR), imgsz=IMGSZ, epochs=args.epocas, patience=args.paciencia, batch=args.batch,
             seed=SEMILLA, deterministic=True, workers=args.workers,
             optimizer="AdamW", lr0=args.lr0, lrf=args.lrf, weight_decay=args.wd, freeze=args.congelar,
             dropout=args.dropout, cos_lr=True, warmup_epochs=1.0,
             project=str(RUNS_DIR / "yolo"), name=SLUG, exist_ok=True,
             fliplr=0.0, flipud=0.0, auto_augment=None, erasing=0.0, degrees=0.0, translate=0.0, scale=0.0,
             shear=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.05, plots=True)
run_dir = RUNS_DIR / "yolo" / SLUG

mejor = YOLO(run_dir / "weights" / "best.pt")
assert [mejor.names[i] for i in sorted(mejor.names)] == CLASES, "El orden de clases del YOLO no coincide"
rutas, P = df.ruta.tolist(), []
for i in range(0, len(rutas), 256):
    for r in mejor.predict(rutas[i:i + 256], imgsz=IMGSZ, verbose=False):
        P.append(r.probs.data.cpu().numpy())
S = np.vstack(P)

res = pd.read_csv(run_dir / "results.csv")
res.columns = [c.strip() for c in res.columns]
# Ultralytics no registra la exactitud de train ni el F1; se deja la exactitud de val y se marca train como NaN
hist = pd.DataFrame(dict(epoca=res.epoch, train_loss=res["train/loss"], val_loss=res["val/loss"],
                         train_acc=np.nan, val_acc=res["metrics/accuracy_top1"], val_f1_macro=np.nan))

config = dict(base="yolov8s-cls.pt (ImageNet)", **vars(args), cos_lr=True, warmup_epochs=1.0, imgsz=IMGSZ,
              aumento_en_linea="ninguno (solo hsv_v=0.05)", epocas_corridas=int(res.epoch.max()),
              criterio_best="exactitud top-1 en val completo (balanceado)")
cerrar_modelo(SLUG, NOMBRE, S, True, df, conj, config, hist)
MODELOS_DIR.mkdir(exist_ok=True)
shutil.copy(run_dir / "weights" / "best.pt", MODELOS_DIR / "yolov8s_cls.pt")
print("Pesos copiados a", MODELOS_DIR / "yolov8s_cls.pt")
