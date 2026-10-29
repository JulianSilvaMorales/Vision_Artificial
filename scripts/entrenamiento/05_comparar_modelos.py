"""Comparacion de los 4 modelos con el mismo test y las mismas metricas (Registro de desempeno del Hito 2).

Lee docs/resultados/<modelo>/puntajes.npz de cada modelo y escribe en docs/resultados/comparacion/:
tabla larga de metricas, tablas por metrica (train/val/test), matrices de confusion lado a lado, curvas ROC
y una grafica de barras. El mejor modelo se elige con validacion (F1 macro en val completo); el test solo se informa.

Uso (en Ubuntu o Windows, desde la raiz del repo):
    python scripts/entrenamiento/05_comparar_modelos.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from src.data.dataset import cargar_indice, conjuntos, verificar
from src.utils.config import CLASES, RESULTADOS_DIR
from src.utils.graficas import ETIQUETAS
from src.utils.metricas import cargar_puntajes, tabla_modelo

MODELOS = ["modelo1_yolov8s_cls", "modelo2_resnet50", "modelo3_svm", "modelo4_random_forest"]
EXTRAS = ["extra_regresion_logistica", "extra_knn", "ablacion_imagenet_svm", "ablacion_imagenet_random_forest",
          "ablacion_imagenet_regresion_logistica", "ablacion_imagenet_knn"]
VISTA = ["train_orig", "val_orig", "val_completo", "test_orig", "test_completo"]

df = cargar_indice()
verificar(df)
conj = conjuntos(df)
y = df.y.values
out = RESULTADOS_DIR / "comparacion"
out.mkdir(parents=True, exist_ok=True)

puntajes, tablas = {}, []
for slug in MODELOS + EXTRAS:
    if not (RESULTADOS_DIR / slug / "puntajes.npz").exists():
        print("Falta:", slug)
        continue
    nombre, S, es_prob = cargar_puntajes(slug, df)
    puntajes[nombre] = (S, es_prob, slug in MODELOS)
    tablas.append(tabla_modelo(nombre, S, es_prob, df, conj))
# ensamble de los 4 modelos oficiales: promedio de puntajes (el SVM pasa por softmax; no son probabilidades calibradas)
if all(s in [p for p in MODELOS if (RESULTADOS_DIR / p / "puntajes.npz").exists()] for s in MODELOS):
    def a_prob(S, es_prob):
        if es_prob:
            return S
        e = np.exp(S - S.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    S_ens = np.mean([a_prob(S, es_prob) for S, es_prob, ppal in puntajes.values() if ppal], 0)
    puntajes["extra · Ensamble de los 4 (promedio)"] = (S_ens, True, False)
    tablas.append(tabla_modelo("extra · Ensamble de los 4 (promedio)", S_ens, True, df, conj))
T = pd.concat(tablas, ignore_index=True)
T.round(4).to_csv(out / "tabla_metricas_larga.csv", index=False)

for met in ("exactitud", "precision_macro", "recall_macro", "f1_macro", "exactitud_balanceada", "roc_auc_macro"):
    tab = T.pivot(index="modelo", columns="conjunto", values=met)[VISTA]
    tab["brecha_train_test_orig"] = tab.train_orig - tab.test_orig
    tab.round(4).to_csv(out / f"tabla_{met}.csv")
    print(f"\n{met.upper()}\n{tab.round(3).to_string()}")

principales = [n for n, v in puntajes.items() if v[2]]
val = T[(T.conjunto == "val_completo") & T.modelo.isin(principales)].sort_values("f1_macro", ascending=False)
mejor = val.iloc[0].modelo
print(f"\nMejor modelo segun VALIDACION (F1 macro, val completo): {mejor}")

# recall por clase en test de originales: la exactitud sola es enganosa (73% de este test es Protrusion)
rec = []
for nombre, (S, _, _) in puntajes.items():
    m = conj["test_orig"]
    yp = S[m].argmax(1)
    rec.append(dict(modelo=nombre, **{f"recall_{c}": float((yp[y[m] == i] == i).mean()) for i, c in enumerate(CLASES)}))
pd.DataFrame(rec).round(4).to_csv(out / "recall_por_clase_test_orig.csv", index=False)
print(pd.DataFrame(rec).round(3).to_string(index=False))

# matrices de confusion (test originales), normalizadas por fila
nombres = principales
fig, axs = plt.subplots(1, len(nombres), figsize=(3.9 * len(nombres), 3.9))
for ax, nm in zip(np.atleast_1d(axs), nombres):
    S = puntajes[nm][0]
    m = conj["test_orig"]
    cm = confusion_matrix(y[m], S[m].argmax(1), labels=range(len(CLASES)))
    cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(nm, fontsize=9)
    ax.set_xticks(range(len(CLASES)), ETIQUETAS, fontsize=7)
    ax.set_yticks(range(len(CLASES)), ETIQUETAS, fontsize=7)
    ax.set_xlabel("Predicha", fontsize=8)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.5 else "black")
np.atleast_1d(axs)[0].set_ylabel("Real")
fig.suptitle("Matrices de confusion · test originales (n=768)", fontsize=10)
fig.tight_layout()
fig.savefig(out / "matrices_confusion_test_orig.png", dpi=150)
plt.close(fig)

# ROC macro (promedio de las curvas uno contra el resto) en test originales
fig, ax = plt.subplots(figsize=(5.6, 4.8))
grid = np.linspace(0, 1, 200)
for nm in nombres:
    S = puntajes[nm][0]
    m = conj["test_orig"]
    tprs = [np.interp(grid, *roc_curve((y[m] == i).astype(int), S[m][:, i])[:2]) for i in range(len(CLASES))]
    auc = T[(T.modelo == nm) & (T.conjunto == "test_orig")].roc_auc_macro.iloc[0]
    ax.plot(grid, np.mean(tprs, 0), label=f"{nm} (AUC {auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("Tasa de falsos positivos")
ax.set_ylabel("Tasa de verdaderos positivos")
ax.set_title("ROC macro · test originales", fontsize=10)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(out / "roc_test_orig.png", dpi=150)
plt.close(fig)

# barras: exactitud y F1 macro en test originales y test completo
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, cj in zip(axs, ("test_orig", "test_completo")):
    sub = T[(T.conjunto == cj) & T.modelo.isin(nombres)].set_index("modelo").loc[nombres]
    x = np.arange(len(nombres))
    ax.bar(x - 0.2, sub.exactitud, 0.4, label="Exactitud")
    ax.bar(x + 0.2, sub.f1_macro, 0.4, label="F1 macro")
    for xi, (a, f) in enumerate(zip(sub.exactitud, sub.f1_macro)):
        ax.text(xi - 0.2, a + 0.01, f"{a:.2f}", ha="center", fontsize=7)
        ax.text(xi + 0.2, f + 0.01, f"{f:.2f}", ha="center", fontsize=7)
    ax.axhline(0.89, color="gray", ls=":", lw=1)
    ax.set_xticks(x, [n.split("·")[1].strip()[:18] for n in nombres], fontsize=7)
    ax.set_title(cj.replace("_", " "), fontsize=9)
    ax.set_ylim(0, 1.05)
axs[0].legend(fontsize=8)
fig.suptitle("Exactitud y F1 macro por modelo (linea punteada: 89 %)", fontsize=10)
fig.tight_layout()
fig.savefig(out / "barras_test.png", dpi=150)
plt.close(fig)

(out / "resumen.json").write_text(json.dumps(dict(
    mejor_por_validacion=mejor, criterio="F1 macro en val completo", modelos=nombres, extras=[n for n in puntajes if n not in nombres],
    advertencias=["seleccion por validacion; test solo informado", "se reportan originales y test completo",
                  "Degeneracion: 42 originales en test", "etiquetas de estudiantes sin radiologo", "posible atajo por nivel"]),
    ensure_ascii=False, indent=2), encoding="utf-8")
print("\nGuardado en:", out)
