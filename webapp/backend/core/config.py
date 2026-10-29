"""Configuracion del backend del MVP (rutas y dispositivo).

DEEPSPINE_DISPOSITIVO: "cpu" (por defecto) o "cuda". En CPU las predicciones reproducen exactamente las de la
evaluacion oficial; en GPU la ResNet usa otra precision numerica y puede cambiar algun caso limite.
"""
import os

from src.utils.config import MODELOS_DIR, RAIZ_REPO, RESULTADOS_DIR

DISPOSITIVO = os.environ.get("DEEPSPINE_DISPOSITIVO", "cpu")
DEMO_DIR = RAIZ_REPO / "data" / "demo"
CASOS_CSV = DEMO_DIR / "casos.csv"
IMAGENES_DEMO = DEMO_DIR / "imagenes"
FRONTEND_DIR = RAIZ_REPO / "webapp" / "frontend" / "mvp"
TAMANO_MAXIMO_IMAGEN = 5 * 1024 * 1024        # bytes aceptados en /api/predecir

# modelo -> carpeta de resultados y archivo de pesos
MODELOS = {
    "yolov8s_cls": dict(slug="modelo1_yolov8s_cls", archivo="yolov8s_cls.pt"),
    "resnet50": dict(slug="modelo2_resnet50", archivo="resnet50.pt"),
    "svm": dict(slug="modelo3_svm", archivo="svm.joblib"),
    "random_forest": dict(slug="modelo4_random_forest", archivo="random_forest.joblib"),
}

__all__ = ["DISPOSITIVO", "DEMO_DIR", "CASOS_CSV", "IMAGENES_DEMO", "FRONTEND_DIR", "TAMANO_MAXIMO_IMAGEN",
           "MODELOS", "MODELOS_DIR", "RAIZ_REPO", "RESULTADOS_DIR"]
