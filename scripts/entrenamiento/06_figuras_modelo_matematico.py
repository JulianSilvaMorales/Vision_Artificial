"""Figuras y valores del modelo matematico (docs/modelo_matematico.md).

Usa los modelos y resultados ya entrenados (models/ y docs/resultados/) y los embeddings de DEEPSPINE_RUNS.
Escribe las figuras en docs/modelo_matematico/figuras/ y los valores citados en el documento en
docs/modelo_matematico/valores.json, para que cada cifra del documento sea trazable.

Uso (en Ubuntu, desde la raiz del repo, despues de 01 a 05):
    python scripts/entrenamiento/06_figuras_modelo_matematico.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data.dataset import cargar_indice, conjuntos
from src.utils.config import CLASES, MODELOS_DIR, RAIZ_REPO, RESULTADOS_DIR, RUNS_DIR, SEMILLA
from src.utils.graficas import ETIQUETAS

OUT = RAIZ_REPO / "docs" / "modelo_matematico"
FIG = OUT / "figuras"
FIG.mkdir(parents=True, exist_ok=True)
COLORES = ["#d62728", "#2ca02c", "#1f77b4"]   # Degeneracion, Normal, Protrusion
V = {}


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(FIG / nombre, dpi=150)
    plt.close(fig)
    print("figura:", FIG / nombre)


df = cargar_indice()
conj = conjuntos(df)
Y = df.y.values
TR = (df.split == "train").values

# 1. Funciones de activacion y softmax ------------------------------------------------------------
z = np.linspace(-6, 6, 400)
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
axs[0].plot(z, np.maximum(z, 0), label="ReLU(z) = max(0, z)  (ResNet-50)")
axs[0].plot(z, z / (1 + np.exp(-z)), label="SiLU(z) = z·σ(z)  (YOLOv8)")
axs[0].set_title("Activaciones de las capas ocultas")
axs[0].legend(fontsize=8)
axs[0].set_xlabel("z")
axs[1].plot(z, 1 / (1 + np.exp(-z)), color="k")
axs[1].axhline(0.5, ls=":", color="gray")
axs[1].set_title("Sigmoide σ(z) = 1 / (1 + e^(−z))  (semana 9, 2 clases)")
axs[1].set_xlabel("z")
logits = np.array([2.0, 0.5, -1.0])
p = np.exp(logits) / np.exp(logits).sum()
axs[2].bar(np.arange(3) - 0.2, logits, 0.4, label="logits z_k", color="gray")
axs[2].bar(np.arange(3) + 0.2, p, 0.4, label="softmax p_k", color=COLORES)
for i, (a, b) in enumerate(zip(logits, p)):
    axs[2].text(i - 0.2, a + (0.08 if a >= 0 else -0.3), f"{a:.1f}", ha="center", fontsize=8)
    axs[2].text(i + 0.2, b + 0.05, f"{b:.2f}", ha="center", fontsize=8)
axs[2].set_xticks(range(3), ETIQUETAS, fontsize=8)
axs[2].axhline(0, color="k", lw=0.6)
axs[2].set_title("Softmax: 3 logits → 3 probabilidades (suman 1)")
axs[2].legend(fontsize=8)
guardar(fig, "01_activaciones_softmax.png")
V["ejemplo_softmax"] = dict(logits=logits.tolist(), probabilidades=p.round(4).tolist())

# 2. Funcion de costo: entropia cruzada y suavizado de etiquetas --------------------------------
eps, K = 0.1, len(CLASES)
q_ok, q_otro = 1 - eps + eps / K, eps / K
piso = -(q_ok * np.log(q_ok) + (K - 1) * q_otro * np.log(q_otro))
V["suavizado"] = dict(epsilon=eps, objetivo_clase_real=round(q_ok, 4), objetivo_otras=round(q_otro, 4),
                      perdida_minima=round(float(piso), 4))
pp = np.linspace(0.005, 0.995, 300)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
axs[0].plot(pp, -np.log(pp), label="−log p  (one-hot, YOLOv8)")
# con suavizado y las otras dos clases repartiendo (1 − p) por igual
axs[0].plot(pp, -(q_ok * np.log(pp) + (K - 1) * q_otro * np.log((1 - pp) / (K - 1))), label="con suavizado ε = 0,1 (ResNet-50)")
axs[0].axvline(q_ok, ls=":", color="gray")
axs[0].set_xlabel("probabilidad asignada a la clase real, p")
axs[0].set_ylabel("pérdida de una imagen")
axs[0].set_title("Entropía cruzada: castiga la seguridad equivocada")
axs[0].set_ylim(0, 5)
axs[0].legend(fontsize=8)
h = pd.read_csv(RESULTADOS_DIR / "modelo2_resnet50" / "historial.csv")
axs[1].plot(h.epoca, h.train_loss, "o-", label="J(θ) train")
axs[1].plot(h.epoca, h.val_loss, "o-", label="log loss val")
axs[1].axhline(piso, ls="--", color="gray", label=f"mínimo posible con suavizado ({piso:.3f})")
axs[1].axvline(int(h.loc[h.val_f1_macro.idxmax(), "epoca"]), ls=":", color="k", label="época elegida (mejor F1 val)")
axs[1].set_xlabel("época")
axs[1].set_title("ResNet-50: costo durante el entrenamiento")
axs[1].legend(fontsize=7)
guardar(fig, "02_funcion_costo.png")

# 3. Descenso del gradiente: tasa de aprendizaje en coseno ----------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.6))
T = 10
t = np.arange(0, T + 1)
ax.step(t[:-1] + 1, 1e-4 * 0.5 * (1 + np.cos(np.pi * t[:-1] / T)), where="post", label="ResNet-50: α₀ = 1e-4, T = 10 (paró en la época 6)")
T2 = 15
t2 = np.linspace(0, T2, 300)
lr_yolo = 5e-4 * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * t2 / T2)))
ax.plot(t2 + 1, lr_yolo, label="YOLOv8s-cls: α₀ = 5e-4, α_min = 5e-5, T = 15 (paró en la 6)")
ax.set_xlabel("época")
ax.set_ylabel("tasa de aprendizaje α")
ax.set_yscale("log")
ax.set_title("Tasa de aprendizaje con decaimiento en coseno")
ax.legend(fontsize=7)
guardar(fig, "03_tasa_aprendizaje.png")

# 4. Curvas de aprendizaje de las dos redes -------------------------------------------------------
hy = pd.read_csv(RESULTADOS_DIR / "modelo1_yolov8s_cls" / "historial.csv")
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, hh, nom in ((axs[0], hy, "1 · YOLOv8s-cls"), (axs[1], h, "2 · ResNet-50")):
    ax.plot(hh.epoca, hh.train_loss, "o-", label="pérdida train")
    ax.plot(hh.epoca, hh.val_loss, "o-", label="pérdida val")
    ax2 = ax.twinx()
    ax2.plot(hh.epoca, hh.val_acc, "s--", color="gray", label="exactitud val")
    ax2.set_ylim(0.6, 1)
    ax.set_title(nom)
    ax.set_xlabel("época")
    ax.legend(loc="upper center", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)
fig.suptitle("Sobreajuste: la pérdida de train baja y la de validación sube desde la época 2", fontsize=10)
guardar(fig, "04_curvas_aprendizaje.png")

# 5. PCA: varianza explicada ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.8))
V["pca"] = {}
for nombre, archivo in (("ResNet-50 ajustada (oficial)", "ajustada"), ("ImageNet sin ajustar", "imagenet")):
    E = np.load(RUNS_DIR / f"embeddings_resnet50_{archivo}.npz")["emb"]
    Zs = StandardScaler().fit(E[TR]).transform(E[TR])
    acum = np.cumsum(PCA(n_components=512, random_state=SEMILLA).fit(Zs).explained_variance_ratio_)
    ax.plot(np.arange(1, 513), acum, label=nombre)
    V["pca"][archivo] = {f"k={k}": round(float(acum[k - 1]), 4) for k in (2, 10, 50, 256, 512)}
ax.axvline(256, ls=":", color="k", label="k = 256 (usado)")
ax.set_xlabel("número de componentes k")
ax.set_ylabel("varianza explicada acumulada")
ax.set_title("PCA sobre los embeddings de train (2048 dimensiones)")
ax.legend(fontsize=8)
guardar(fig, "05_pca_varianza.png")

# 6. SVM con kernel RBF: ilustracion en 2 dimensiones ---------------------------------------------
E = np.load(RUNS_DIR / "embeddings_resnet50_ajustada.npz")["emb"]
b = joblib.load(MODELOS_DIR / "svm.joblib")
Z = b["pca"].transform(b["escalador"].transform(E))
idx = np.random.default_rng(SEMILLA).choice(np.where(conj["train_orig"])[0], 1200, replace=False)
Z2 = Z[idx, :2]
svm2 = SVC(kernel="rbf", C=b["params"]["C"], gamma="scale").fit(Z2, Y[idx])
xx, yy = np.meshgrid(np.linspace(*np.percentile(Z2[:, 0], [0.5, 99.5]), 300),
                     np.linspace(*np.percentile(Z2[:, 1], [0.5, 99.5]), 300))
reg = svm2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
axs[0].contourf(xx, yy, reg, levels=[-0.5, 0.5, 1.5, 2.5], colors=COLORES, alpha=0.18)
for k in range(3):
    m = Y[idx] == k
    axs[0].scatter(Z2[m, 0], Z2[m, 1], s=6, color=COLORES[k], label=f"{ETIQUETAS[k]} (n={m.sum()})")
sv = svm2.support_
axs[0].scatter(Z2[sv, 0], Z2[sv, 1], s=14, facecolors="none", edgecolors="k", lw=0.4, label=f"vectores de soporte ({len(sv)})")
axs[0].set_xlabel("componente principal 1")
axs[0].set_ylabel("componente principal 2")
axs[0].set_title("Ilustración en 2D (1.200 originales de train); el modelo real usa 256 dimensiones", fontsize=9)
axs[0].legend(fontsize=7, markerscale=2)
gamma_real = b["modelo"]._gamma
# distancias reales entre pares de imagenes de train en el espacio de 256 dimensiones
par = np.random.default_rng(SEMILLA).choice(np.where(TR)[0], (3000, 2))
dist = np.linalg.norm(Z[par[:, 0]] - Z[par[:, 1]], axis=1)
d = np.linspace(0, np.percentile(dist, 99), 300)
for g, etiqueta in ((gamma_real * 10, "γ × 10"), (gamma_real, f"γ = {gamma_real:.5f} (usado, 'scale')"), (gamma_real / 10, "γ / 10")):
    axs[1].plot(d, np.exp(-g * d ** 2), label=etiqueta)
axs[1].axvline(np.median(dist), ls=":", color="k", label=f"distancia mediana entre imágenes ({np.median(dist):.0f})")
axs[1].set_xlabel("distancia euclídea ‖u − v‖ en el espacio de 256 componentes")
axs[1].set_ylabel("K(u, v) = exp(−γ‖u − v‖²)")
axs[1].set_title("Kernel RBF: similitud entre dos imágenes según su distancia", fontsize=9)
axs[1].legend(fontsize=8)
guardar(fig, "06_svm_rbf.png")
V["svm"] = dict(C=b["params"]["C"], gamma=float(gamma_real), distancia_mediana=round(float(np.median(dist)), 2), n_vectores_soporte=int(b["modelo"].n_support_.sum()),
                vectores_soporte_por_clase=dict(zip(CLASES, map(int, b["modelo"].n_support_))), n_train=int(TR.sum()),
                pca_varianza_256=round(float(b["pca"].explained_variance_ratio_.sum()), 4))

# 7. Random Forest: impureza de Gini y estructura del bosque --------------------------------------
rf = joblib.load(MODELOS_DIR / "random_forest.joblib")["modelo"]
prof = np.array([e.get_depth() for e in rf.estimators_])
hojas = np.array([e.get_n_leaves() for e in rf.estimators_])
V["random_forest"] = dict(n_arboles=len(rf.estimators_), max_depth=rf.max_depth, max_features=rf.max_features,
                          variables_por_division=int(rf.estimators_[0].max_features_),
                          profundidad_media=round(float(prof.mean()), 1), hojas_media=round(float(hojas.mean()), 1))
fig, axs = plt.subplots(1, 3, figsize=(14, 3.7))
pg = np.linspace(0, 1, 200)
axs[0].plot(pg, 1 - pg ** 2 - (1 - pg) ** 2, color="k")
axs[0].set_xlabel("proporción de una clase en el nodo, p")
axs[0].set_ylabel("Gini = 1 − Σ p_k²")
axs[0].set_title("Impureza de Gini (2 clases): 0 = nodo puro", fontsize=9)
axs[0].annotate("máximo 0,5\n(mitad y mitad)", (0.5, 0.5), (0.6, 0.3), fontsize=8, arrowprops=dict(arrowstyle="->"))
imp = rf.feature_importances_
axs[1].bar(np.arange(1, 31), imp[:30])
axs[1].set_xlabel("componente principal")
axs[1].set_ylabel("disminución media de Gini")
axs[1].set_title(f"Importancia de las primeras 30 de 256 variables\n(las 10 primeras suman {imp[:10].sum():.0%})", fontsize=9)
axs[2].hist(hojas, bins=25, color="gray")
axs[2].set_xlabel("número de hojas del árbol")
axs[2].set_title(f"{len(prof)} árboles distintos (bootstrap + 16 variables al azar)\n"
                 f"todos llegan a la profundidad máxima ({rf.max_depth}); {hojas.mean():.0f} hojas en promedio", fontsize=9)
guardar(fig, "07_random_forest.png")
V["random_forest"]["importancia_10_primeras_componentes"] = round(float(imp[:10].sum()), 4)

# 8. Busqueda de hiperparametros en validacion ----------------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
for ax, (slug, hp, titulo) in zip(axs, (("modelo3_svm", "C", "3 · SVM: C"), ("modelo4_random_forest", "max_depth", "4 · Random Forest: profundidad"),
                                         ("extra_regresion_logistica", "C", "extra · Logística: C"), ("extra_knn", "n_neighbors", "extra · KNN: K"))):
    bus = json.loads((RESULTADOS_DIR / slug / "metadata.json").read_text(encoding="utf-8"))["config"]["busqueda_val"]
    xs = [str(r[hp]) for r in bus]
    ys = [r["f1_macro_val"] for r in bus]
    ax.plot(xs, ys, "o-")
    k = int(np.argmax(ys))
    ax.plot(xs[k], ys[k], "r*", ms=13)
    ax.set_title(titulo, fontsize=9)
    ax.set_xlabel(hp)
axs[0].set_ylabel("F1 macro en validación")
fig.suptitle("Selección de hiperparámetros con validación (estrella = elegido)", fontsize=10)
guardar(fig, "08_busqueda_hiperparametros.png")

# 9. Metricas a partir de la matriz de confusion del modelo 2 -------------------------------------
cm = pd.read_csv(RESULTADOS_DIR / "modelo2_resnet50" / "confusion_test_orig.csv", index_col=0).values
n = cm.sum()
por_clase = {}
for k, c in enumerate(CLASES):
    TP = cm[k, k]
    FP = cm[:, k].sum() - TP
    FN = cm[k, :].sum() - TP
    TN = n - TP - FP - FN
    prec, rec = TP / (TP + FP), TP / (TP + FN)
    por_clase[c] = dict(TP=int(TP), FP=int(FP), FN=int(FN), TN=int(TN), precision=round(prec, 4), recall=round(rec, 4),
                        f1=round(2 * prec * rec / (prec + rec), 4))
acc = np.trace(cm) / n
z95 = 1.96
centro = (acc + z95 ** 2 / (2 * n)) / (1 + z95 ** 2 / n)
radio = z95 * np.sqrt(acc * (1 - acc) / n + z95 ** 2 / (4 * n ** 2)) / (1 + z95 ** 2 / n)
V["metricas_resnet_test_orig"] = dict(matriz=cm.tolist(), n=int(n), aciertos=int(np.trace(cm)), exactitud=round(acc, 4),
                                      ic95_wilson=[round(centro - radio, 4), round(centro + radio, 4)], por_clase=por_clase,
                                      f1_macro=round(np.mean([v["f1"] for v in por_clase.values()]), 4),
                                      exactitud_siempre_protrusion=round(cm[2].sum() / n, 4))

# parametros de las redes
try:
    import torch
    from ultralytics import YOLO
    from src.models.resnet import cargar_resnet_ajustada

    r = cargar_resnet_ajustada(MODELOS_DIR / "resnet50.pt")
    yolo = YOLO(MODELOS_DIR / "yolov8s_cls.pt").model
    congelados = sum(p.numel() for nm, p in r.named_parameters() if nm.startswith(("conv1", "bn1", "layer1")))
    V["parametros"] = dict(resnet50_total=sum(p.numel() for p in r.parameters()), resnet50_congelados=congelados,
                           resnet50_capa_final=sum(p.numel() for p in r.fc.parameters()),
                           yolov8s_cls_total=sum(p.numel() for p in yolo.parameters()))
except Exception as e:      # no impide el resto de figuras
    print("Aviso: no se contaron los parametros:", e)

(OUT / "valores.json").write_text(json.dumps(V, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print(json.dumps(V, ensure_ascii=False, indent=1, default=float))
