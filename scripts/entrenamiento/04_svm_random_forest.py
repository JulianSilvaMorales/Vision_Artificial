"""Modelos 3 y 4 · SVM (RBF) y Random Forest sobre los embeddings congelados (+ regresion logistica y KNN como extras).

StandardScaler y PCA se ajustan solo con train (regla de data leakage). Cada clasificador elige sus hiperparametros
por F1 macro en val completo; el test se evalua una sola vez al final. Requiere haber corrido 03_embeddings.py.

Uso (en Ubuntu, desde la raiz del repo):
    python scripts/entrenamiento/04_svm_random_forest.py [--pca 256] [--sin-extras]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data.dataset import cargar_indice, conjuntos, verificar
from src.training.evaluacion import cerrar_modelo
from src.utils.config import CLASES, MODELOS_DIR, RUNS_DIR, SEMILLA, fijar_semilla

ap = argparse.ArgumentParser()
ap.add_argument("--pca", type=int, default=256)
ap.add_argument("--sin-extras", action="store_true")
args = ap.parse_args()

fijar_semilla(SEMILLA)
df = cargar_indice()
print("Pacientes por particion:", verificar(df))
conj = conjuntos(df)
TR, VA = (df.split == "train").values, conj["val_completo"]
Y = df.y.values

z = np.load(RUNS_DIR / "embeddings_resnet50_imagenet.npz")
assert list(z["archivos"]) == df.archivo.tolist(), "Los embeddings no estan en el orden del indice; corre 03_embeddings.py"
EMB = z["emb"]

escalador = StandardScaler().fit(EMB[TR])
pca = PCA(n_components=args.pca, random_state=SEMILLA).fit(escalador.transform(EMB[TR]))
Z = pca.transform(escalador.transform(EMB))
var_pca = float(pca.explained_variance_ratio_.sum())
print(f"PCA: {Z.shape[1]} componentes explican {var_pca:.1%} de la varianza (ajustado solo con train)")


def elegir(nombre, fabrica, rejilla, es_prob):
    busqueda, mejor = [], None
    for pr in rejilla:
        t0 = time.time()
        m = fabrica(**pr).fit(Z[TR], Y[TR])
        S = m.predict_proba(Z[VA]) if es_prob else m.decision_function(Z[VA])
        f1 = f1_score(Y[VA], S.argmax(1), average="macro")
        busqueda.append(dict(**pr, f1_macro_val=round(f1, 4), seg=round(time.time() - t0)))
        print(f"  {nombre} {pr} -> F1 macro val {f1:.3f} ({time.time() - t0:.0f} s)", flush=True)
        if mejor is None or f1 > mejor[0]:
            mejor = (f1, pr, m)
    print(f"  elegido: {mejor[1]}")
    S_all = mejor[2].predict_proba(Z) if es_prob else mejor[2].decision_function(Z)
    return mejor[2], mejor[1], S_all, busqueda


def correr(slug, nombre, archivo, fabrica, rejilla, es_prob, descripcion):
    print(f"\n{nombre}")
    m, pr, S, busqueda = elegir(nombre, fabrica, rejilla, es_prob)
    config = dict(entrada="embeddings ResNet-50 ImageNet congelada (2048) -> StandardScaler -> PCA",
                  pca_componentes=args.pca, pca_varianza_explicada=round(var_pca, 4), clasificador=descripcion,
                  rejilla=rejilla, busqueda_val=busqueda, elegido=pr, criterio="F1 macro en val completo",
                  salida="probabilidades" if es_prob else "decision_function (no son probabilidades)")
    cerrar_modelo(slug, nombre, S, es_prob, df, conj, config)
    destino = (MODELOS_DIR if archivo.startswith(("svm", "random")) else RUNS_DIR) / f"{archivo}.joblib"
    joblib.dump(dict(modelo=m, params=pr, escalador=escalador, pca=pca, clases=CLASES, es_prob=es_prob,
                     extractor="torchvision resnet50 IMAGENET1K_V2, fc=Identity, Resize 224, normalizacion ImageNet"),
                destino, compress=3)
    print("Guardado:", destino)


correr("modelo3_svm", "3 · SVM (RBF) sobre embeddings", "svm",
       lambda **k: SVC(kernel="rbf", gamma="scale", decision_function_shape="ovr", random_state=SEMILLA, **k),
       [dict(C=0.5), dict(C=2), dict(C=8)], False, "SVC kernel RBF, gamma='scale', uno contra el resto")
correr("modelo4_random_forest", "4 · Random Forest sobre embeddings", "random_forest",
       lambda **k: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEMILLA, **k),
       [dict(max_depth=10), dict(max_depth=20), dict(max_depth=None)], True, "RandomForestClassifier, 300 arboles")
if not args.sin_extras:
    correr("extra_regresion_logistica", "extra · Regresion logistica", "logistica",
           lambda **k: LogisticRegression(max_iter=2000, random_state=SEMILLA, **k),
           [dict(C=0.01), dict(C=0.1), dict(C=1)], True, "LogisticRegression multinomial (modelo base del curso)")
    correr("extra_knn", "extra · KNN", "knn", lambda **k: KNeighborsClassifier(n_jobs=-1, **k),
           [dict(n_neighbors=5), dict(n_neighbors=15), dict(n_neighbors=31)], True, "KNeighborsClassifier, distancia euclidea")
