"""Prediccion con los 4 modelos entrenados, para la aplicacion web.

Uso desde el backend (capa services/):

    from src.inference.predictor import Predictor
    pred = Predictor()                                   # lee los pesos de models/
    pred.predecir("roi.png", modelo="resnet50")          # un modelo
    pred.predecir_todos(imagen_pil)                      # los 4 modelos sobre la misma imagen

La imagen de entrada es el recorte (ROI) de un disco, como los del dataset: ruta, bytes, PIL.Image o arreglo numpy.
Pesos esperados en models/ (se descargan del GitHub Release): yolov8s_cls.pt, resnet50.pt, svm.joblib,
random_forest.joblib. Los modelos se cargan la primera vez que se usan.

El SVM no entrega probabilidades sino distancias al hiperplano (decision_function); aqui se convierten con softmax
para mostrarlas en la misma escala, pero NO son probabilidades calibradas (`es_probabilidad=False`).
Herramienta academica, no diagnostica.
"""
import io
from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.config import CLASES, MODELOS_DIR

MODELOS = {
    "yolov8s_cls": "1 · YOLOv8s-cls",
    "resnet50": "2 · ResNet-50 ajustada",
    "svm": "3 · SVM (RBF) sobre embeddings",
    "random_forest": "4 · Random Forest sobre embeddings",
}


def abrir_imagen(imagen):
    if isinstance(imagen, Image.Image):
        img = imagen
    elif isinstance(imagen, (bytes, bytearray)):
        img = Image.open(io.BytesIO(imagen))
    elif isinstance(imagen, np.ndarray):
        img = Image.fromarray(imagen)
    else:
        img = Image.open(Path(imagen))
    return img.convert("RGB")


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


class Predictor:
    def __init__(self, modelos_dir=MODELOS_DIR, dispositivo=None):
        import torch

        self.dir = Path(modelos_dir)
        self.dev = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache = {}

    def _cargar(self, nombre):
        if nombre in self._cache:
            return self._cache[nombre]
        if nombre == "yolov8s_cls":
            from ultralytics import YOLO

            m = YOLO(self.dir / "yolov8s_cls.pt")
            assert [m.names[i] for i in sorted(m.names)] == CLASES
        elif nombre == "resnet50":
            from src.models.resnet import cargar_resnet_ajustada

            m = cargar_resnet_ajustada(self.dir / "resnet50.pt", self.dev).to(self.dev)
        elif nombre == "extractor":
            from src.models.resnet import extractor_ajustado

            m = extractor_ajustado(self.dir / "resnet50.pt").to(self.dev)
        elif nombre in ("svm", "random_forest"):
            import joblib

            m = joblib.load(self.dir / f"{nombre}.joblib")
            assert m["clases"] == CLASES
        else:
            raise ValueError(f"Modelo desconocido: {nombre}. Opciones: {list(MODELOS)}")
        self._cache[nombre] = m
        return m

    def _tensor(self, img):
        from src.models.resnet import TF

        return TF(img).unsqueeze(0).to(self.dev)

    def _puntajes(self, img, nombre):
        import torch

        if nombre == "yolov8s_cls":
            r = self._cargar(nombre).predict(img, imgsz=224, verbose=False, device=self.dev)[0]
            return r.probs.data.cpu().numpy(), True
        with torch.no_grad():
            if nombre == "resnet50":
                return torch.softmax(self._cargar(nombre)(self._tensor(img)), 1)[0].cpu().numpy(), True
            emb = self._cargar("extractor")(self._tensor(img)).cpu().numpy()
        b = self._cargar(nombre)
        z = b["pca"].transform(b["escalador"].transform(emb))
        if b["es_prob"]:
            return b["modelo"].predict_proba(z)[0], True
        return _softmax(b["modelo"].decision_function(z)[0]), False

    def predecir(self, imagen, modelo="resnet50"):
        """Devuelve la clase predicha, la confianza y el puntaje de cada clase para un modelo."""
        if modelo not in MODELOS:
            raise ValueError(f"Modelo desconocido: {modelo}. Opciones: {list(MODELOS)}")
        p, es_prob = self._puntajes(abrir_imagen(imagen), modelo)
        k = int(np.argmax(p))
        return dict(modelo=modelo, nombre=MODELOS[modelo], clase=CLASES[k], confianza=float(p[k]),
                    puntajes={c: float(v) for c, v in zip(CLASES, p)}, es_probabilidad=es_prob)

    def predecir_todos(self, imagen):
        """Corre los 4 modelos sobre la misma imagen."""
        img = abrir_imagen(imagen)
        return [self.predecir(img, m) for m in MODELOS]
