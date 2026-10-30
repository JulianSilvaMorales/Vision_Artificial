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

`trazar(imagen)` corre los 4 modelos y devuelve cada paso intermedio (preprocesamiento, caracteristicas de la
ResNet-50, escalador y PCA, puntajes del SVM, votos de los arboles, YOLO) con su tiempo, para la demo de
trazabilidad. Las caracteristicas de la ResNet-50 se calculan una sola vez y las comparten los modelos 2, 3 y 4.
Herramienta academica, no diagnostica.
"""
import io
import time
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
    # Los modelos se entrenaron con ROI en escala de grises replicada en 3 canales: una imagen en color se pasa a gris.
    return img.convert("L").convert("RGB")


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
        elif nombre in ("svm", "random_forest"):
            import joblib

            m = joblib.load(self.dir / f"{nombre}.joblib")
            assert m["clases"] == CLASES
        else:
            raise ValueError(f"Modelo desconocido: {nombre}. Opciones: {list(MODELOS)}")
        self._cache[nombre] = m
        return m

    def cargar_todos(self):
        """Carga los 4 modelos de una vez (por ejemplo, al arrancar el servidor) para que la primera prediccion sea rapida."""
        for nombre in MODELOS:
            self._cargar(nombre)

    def _tensor(self, img):
        from src.models.resnet import TF

        return TF(img).unsqueeze(0).to(self.dev)

    def _resnet(self, img):
        """Una sola pasada por la ResNet-50 del modelo 2: caracteristicas (2048) y probabilidades de su capa final."""
        import torch

        m = self._cargar("resnet50")
        cuerpo = torch.nn.Sequential(*list(m.children())[:-1])     # todo menos la capa final (fc)
        with torch.no_grad():
            feats = cuerpo(self._tensor(img)).flatten(1)
            prob = torch.softmax(m.fc(feats), 1)
        return feats[0].cpu().numpy(), prob[0].cpu().numpy()

    def _clasico(self, nombre, emb):
        b = self._cargar(nombre)
        s = b["escalador"].transform(emb[None])
        z = b["pca"].transform(s)
        if b["es_prob"]:
            return b["modelo"].predict_proba(z)[0], True, s, z
        return _softmax(b["modelo"].decision_function(z)[0]), False, s, z

    def _puntajes(self, img, nombre):
        if nombre == "yolov8s_cls":
            r = self._cargar(nombre).predict(img, imgsz=224, verbose=False, device=self.dev)[0]
            return r.probs.data.cpu().numpy(), True
        emb, prob = self._resnet(img)
        if nombre == "resnet50":
            return prob, True
        return self._clasico(nombre, emb)[:2]

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

    def distancia_ood(self, emb):
        """Distancia de Mahalanobis al cuadrado del embedding de la ResNet-50 en el espacio PCA del SVM (ver validacion)."""
        b = self._cargar("svm")
        z = b["pca"].transform(b["escalador"].transform(emb[None]))[0]
        return float(np.sum(z ** 2 / b["pca"].explained_variance_)), b["pca"]

    def trazar(self, imagen, validar=None):
        """Corre los 4 modelos sobre una imagen y devuelve cada paso intermedio con su tiempo (ms).

        `validar(emb)`, si se pasa, se llama justo despues de calcular el embedding y antes de clasificar; si lanza una
        excepcion, no se corre ningun modelo mas (rechazo de imagenes que no son un disco)."""
        img = abrir_imagen(imagen)
        pasos, t0 = {}, time.perf_counter()

        def ms(desde):
            return round((time.perf_counter() - desde) * 1000, 1)

        x = self._tensor(img)
        gris = np.asarray(img.convert("L"))
        pasos["entrada"] = dict(tamano_original=list(img.size), rango_intensidad=[int(gris.min()), int(gris.max())],
                                tamano_modelo=[224, 224], normalizacion="ImageNet (media 0,485/0,456/0,406)",
                                media_normalizada=round(float(x.mean()), 3), desv_normalizada=round(float(x.std()), 3))

        t = time.perf_counter()
        emb, prob_r = self._resnet(img)
        if validar is not None:
            validar(emb)
        pasos["resnet50"] = dict(ms=ms(t), dimension=int(emb.size), norma=round(float(np.linalg.norm(emb)), 2),
                                 fraccion_ceros=round(float((emb == 0).mean()), 3),
                                 resumen_64=[round(float(v), 4) for v in emb.reshape(64, -1).mean(1)])
        pasos["modelo2_capa_final"] = dict(probabilidades=dict(zip(CLASES, map(float, prob_r))))

        t = time.perf_counter()
        p_svm, _, s, z = self._clasico("svm", emb)
        svm = self._cargar("svm")
        pasos["escalador_pca"] = dict(ms=ms(t), dimension_entrada=int(s.shape[1]), dimension_salida=int(z.shape[1]),
                                      primeras_componentes=[round(float(v), 3) for v in z[0, :8]],
                                      varianza_explicada=round(float(svm["pca"].explained_variance_ratio_.sum()), 4))
        pasos["svm"] = dict(decision=dict(zip(CLASES, map(float, svm["modelo"].decision_function(z)[0]))),
                            vectores_soporte=int(svm["modelo"].n_support_.sum()), C=svm["params"]["C"])

        t = time.perf_counter()
        p_rf, _, _, z_rf = self._clasico("random_forest", emb)
        bosque = self._cargar("random_forest")["modelo"]
        z32 = np.ascontiguousarray(z_rf, dtype=np.float32)     # los arboles de sklearn trabajan en float32
        votos = np.bincount([int(a.tree_.predict(z32)[0].argmax()) for a in bosque.estimators_], minlength=len(CLASES))
        pasos["random_forest"] = dict(ms=ms(t), arboles=len(bosque.estimators_), votos=dict(zip(CLASES, map(int, votos))))

        t = time.perf_counter()
        p_yolo, _ = self._puntajes(img, "yolov8s_cls")
        pasos["yolov8s_cls"] = dict(ms=ms(t))

        res = []
        for nombre, p, es_prob in (("yolov8s_cls", p_yolo, True), ("resnet50", prob_r, True),
                                   ("svm", p_svm, False), ("random_forest", p_rf, True)):
            k = int(np.argmax(p))
            res.append(dict(modelo=nombre, nombre=MODELOS[nombre], clase=CLASES[k], confianza=float(p[k]),
                            puntajes={c: float(v) for c, v in zip(CLASES, p)}, es_probabilidad=es_prob))
        ens = np.mean([p_yolo, prob_r, p_svm, p_rf], 0)
        clases = [r["clase"] for r in res]
        return dict(predicciones=res, pasos=pasos, ms_total=ms(t0), coinciden=len(set(clases)) == 1,
                    ensamble=dict(clase=CLASES[int(ens.argmax())], puntajes=dict(zip(CLASES, map(float, ens))),
                                  nota="Promedio simple; el puntaje del SVM es un softmax de distancias, no una probabilidad calibrada."),
                    imagen_224=img.resize((224, 224)))
