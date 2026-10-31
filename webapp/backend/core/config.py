"""Configuracion del backend del MVP (rutas y dispositivo).

DEEPSPINE_DISPOSITIVO: "cpu" (por defecto) o "cuda". En CPU las predicciones reproducen exactamente las de la
evaluacion oficial; en GPU la ResNet usa otra precision numerica y puede cambiar algun caso limite.
"""
import os
import tempfile
from pathlib import Path

from src.utils.config import MODELOS_DIR, RAIZ_REPO, RESULTADOS_DIR

DISPOSITIVO = os.environ.get("DEEPSPINE_DISPOSITIVO", "cpu")
DEMO_DIR = RAIZ_REPO / "data" / "demo"
CASOS_CSV = DEMO_DIR / "casos.csv"
IMAGENES_DEMO = DEMO_DIR / "imagenes"
FRONTEND_DIR = RAIZ_REPO / "webapp" / "frontend" / "mvp"              # demo del MVP, servida en /mvp
FRONTEND_APP_DIR = RAIZ_REPO / "webapp" / "frontend" / "clinica"     # aplicacion para el radiologo, servida en /
TAMANO_MAXIMO_IMAGEN = 5 * 1024 * 1024        # bytes aceptados en /api/predecir

# Estudios de resonancia subidos por el radiologo (se guardan en una carpeta temporal y se borran solos)
ESTUDIOS_DIR = Path(os.environ.get("DEEPSPINE_ESTUDIOS", Path(tempfile.gettempdir()) / "deepspine_estudios"))
ESTUDIO_MAX_BYTES = 300 * 1024 * 1024
ESTUDIO_MAX_ARCHIVOS = 500
ESTUDIO_TTL_MIN = 30
SCRIPTS_DIR = RAIZ_REPO / "scripts"
DETECTOR_PT = MODELOS_DIR / "best_disc_detector.pt"
UMBRAL_RECHAZO_JSON = MODELOS_DIR / "umbral_rechazo.json"      # lo escribe scripts/demo/calibrar_rechazo.py
CORS_ORIGENES = [o for o in os.environ.get("DEEPSPINE_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o]

# modelo -> carpeta de resultados y archivo de pesos
MODELOS = {
    "yolov8s_cls": dict(slug="modelo1_yolov8s_cls", archivo="yolov8s_cls.pt"),
    "resnet50": dict(slug="modelo2_resnet50", archivo="resnet50.pt"),
    "svm": dict(slug="modelo3_svm", archivo="svm.joblib"),
    "random_forest": dict(slug="modelo4_random_forest", archivo="random_forest.joblib"),
}

__all__ = ["DISPOSITIVO", "DEMO_DIR", "CASOS_CSV", "IMAGENES_DEMO", "FRONTEND_DIR", "FRONTEND_APP_DIR", "TAMANO_MAXIMO_IMAGEN",
           "MODELOS", "MODELOS_DIR", "RAIZ_REPO", "RESULTADOS_DIR", "ESTUDIOS_DIR", "ESTUDIO_MAX_BYTES",
           "ESTUDIO_MAX_ARCHIVOS", "ESTUDIO_TTL_MIN", "SCRIPTS_DIR", "DETECTOR_PT", "UMBRAL_RECHAZO_JSON", "CORS_ORIGENES"]
