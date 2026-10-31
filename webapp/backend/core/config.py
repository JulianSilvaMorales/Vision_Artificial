"""Configuracion del backend (rutas y dispositivo).

DEEPSPINE_DISPOSITIVO: "cpu" (por defecto) o "cuda". En CPU las predicciones reproducen exactamente las de la
evaluacion oficial; en GPU la ResNet usa otra precision numerica y puede cambiar algun caso limite.
"""
import os
import tempfile
from pathlib import Path

from src.utils.config import MODELOS_DIR, RAIZ_REPO

DISPOSITIVO = os.environ.get("DEEPSPINE_DISPOSITIVO", "cpu")
FRONTEND_APP_DIR = RAIZ_REPO / "webapp" / "frontend" / "clinica"     # aplicacion para el radiologo
TAMANO_MAXIMO_IMAGEN = 5 * 1024 * 1024        # bytes aceptados en /api/predecir

# Estudios de resonancia subidos por el radiologo (se guardan en una carpeta temporal y se borran solos)
ESTUDIOS_DIR = Path(os.environ.get("DEEPSPINE_ESTUDIOS", Path(tempfile.gettempdir()) / "deepspine_estudios"))
ESTUDIO_MAX_BYTES = 300 * 1024 * 1024
ESTUDIO_MAX_ARCHIVOS = 500
ESTUDIO_TTL_MIN = 30
ESTUDIO_LIMPIEZA_S = 60        # cada cuanto se buscan estudios vencidos
SCRIPTS_DIR = RAIZ_REPO / "scripts"
DETECTOR_PT = MODELOS_DIR / "best_disc_detector.pt"
UMBRAL_RECHAZO_JSON = MODELOS_DIR / "umbral_rechazo.json"      # lo escribe scripts/calibrar_rechazo.py

# modelo -> archivo de pesos
MODELOS = {"yolov8s_cls": "yolov8s_cls.pt", "resnet50": "resnet50.pt", "svm": "svm.joblib", "random_forest": "random_forest.joblib"}

__all__ = ["DISPOSITIVO", "FRONTEND_APP_DIR", "TAMANO_MAXIMO_IMAGEN", "MODELOS", "MODELOS_DIR", "RAIZ_REPO", "ESTUDIOS_DIR",
           "ESTUDIO_MAX_BYTES", "ESTUDIO_MAX_ARCHIVOS", "ESTUDIO_TTL_MIN", "ESTUDIO_LIMPIEZA_S", "SCRIPTS_DIR", "DETECTOR_PT",
           "UMBRAL_RECHAZO_JSON"]
