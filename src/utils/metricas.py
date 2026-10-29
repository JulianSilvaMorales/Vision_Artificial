"""Metricas comunes y guardado de resultados por modelo.

Cada modelo entrega una matriz de puntajes S (una fila por imagen del dataset, una columna por clase):
probabilidades, o la decision_function del SVM (que no son probabilidades; por eso no tiene log loss).
Las filas que un modelo no predijo quedan en NaN y no entran en ningun conjunto evaluado.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             log_loss, precision_recall_fscore_support, roc_auc_score)

from src.utils.config import CLASES, RESULTADOS_DIR

COLUMNAS = ["n", "exactitud", "precision_macro", "recall_macro", "f1_macro", "exactitud_balanceada",
            "roc_auc_macro", "log_loss"]


def metricas(y, S, es_prob=True):
    K = S.shape[1]
    yp = S.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y, yp, labels=range(K), average="macro", zero_division=0)
    auc = float(np.mean([roc_auc_score((y == i).astype(int), S[:, i]) for i in range(K)]))
    ll = float("nan")
    if es_prob:
        P = np.clip(S, 1e-7, 1)
        P = P / P.sum(1, keepdims=True)
        ll = log_loss(y, P, labels=list(range(K)))
    return dict(n=int(len(y)), exactitud=accuracy_score(y, yp), precision_macro=p, recall_macro=r, f1_macro=f,
                exactitud_balanceada=balanced_accuracy_score(y, yp), roc_auc_macro=auc, log_loss=ll)


def tabla_modelo(nombre, S, es_prob, df, conj):
    filas = [dict(modelo=nombre, conjunto=cj, **metricas(df.y.values[m], S[m], es_prob))
             for cj, m in conj.items() if not np.isnan(S[m]).any()]
    return pd.DataFrame(filas)


def guardar_resultados(slug, nombre, S, es_prob, df, conj, extra=None):
    """Escribe en docs/resultados/<slug>/ las metricas, reportes por clase y por nivel y los puntajes por imagen."""
    out = RESULTADOS_DIR / slug
    out.mkdir(parents=True, exist_ok=True)
    tabla = tabla_modelo(nombre, S, es_prob, df, conj)
    tabla.round(5).to_csv(out / "metricas.csv", index=False)

    y = df.y.values
    reportes, por_nivel = [], []
    for cj in ("val_completo", "test_orig", "test_completo"):
        m = conj[cj]
        if np.isnan(S[m]).any():
            continue
        yp = S[m].argmax(1)
        rep = classification_report(y[m], yp, labels=range(len(CLASES)), target_names=CLASES,
                                    output_dict=True, zero_division=0)
        for clase in CLASES:
            reportes.append(dict(conjunto=cj, clase=clase, **{k: rep[clase][k] for k in ("precision", "recall", "f1-score", "support")}))
        cm = confusion_matrix(y[m], yp, labels=range(len(CLASES)))
        pd.DataFrame(cm, index=[f"real_{c}" for c in CLASES], columns=[f"pred_{c}" for c in CLASES]).to_csv(out / f"confusion_{cj}.csv")
        sub = df[m].assign(acierto=(yp == y[m]))
        for nivel, g in sub.groupby("nivel"):
            por_nivel.append(dict(conjunto=cj, nivel=nivel, n=len(g), exactitud=g.acierto.mean(),
                                  **{f"n_{c}": int((g.clase == c).sum()) for c in CLASES}))
    pd.DataFrame(reportes).round(4).to_csv(out / "reporte_por_clase.csv", index=False)
    pd.DataFrame(por_nivel).round(4).to_csv(out / "exactitud_por_nivel.csv", index=False)

    # puntajes de cada imagen: trazabilidad completa y entrada del script de comparacion
    np.savez_compressed(out / "puntajes.npz", S=S.astype(np.float32), archivos=np.array(df.archivo.tolist()),
                        es_prob=es_prob, nombre=nombre)
    if extra is not None:
        (out / "metadata.json").write_text(json.dumps(extra, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return tabla


def cargar_puntajes(slug, df):
    """Lee puntajes.npz de un modelo y los reordena segun df (falla si el dataset no es el mismo)."""
    z = np.load(Path(RESULTADOS_DIR) / slug / "puntajes.npz", allow_pickle=False)
    pos = {a: i for i, a in enumerate(df.archivo)}
    arch = list(z["archivos"])
    assert set(arch) == set(pos), f"{slug}: las imagenes no coinciden con este dataset"
    S = np.full(z["S"].shape, np.nan, dtype=np.float32)
    S[[pos[a] for a in arch]] = z["S"]
    return str(z["nombre"]), S, bool(z["es_prob"])


def imprimir(tabla):
    print(tabla.set_index("conjunto")[COLUMNAS].round(3).to_string())
