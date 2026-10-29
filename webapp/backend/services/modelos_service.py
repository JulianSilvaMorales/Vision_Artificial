"""Informacion de los modelos: pesos disponibles, hash, configuracion y metricas oficiales (docs/resultados)."""
import hashlib
import json
import subprocess
from functools import lru_cache

import pandas as pd

from webapp.backend.core.config import (CASOS_CSV, DISPOSITIVO, IMAGENES_DEMO, MODELOS, MODELOS_DIR, RAIZ_REPO,
                                        RESULTADOS_DIR)

CLAVES_CONFIG = ("base", "mejor_epoca", "epocas_corridas", "criterio_best", "criterio", "elegido", "ls", "lr", "lr0",
                 "pca_componentes", "pca_varianza_explicada", "clasificador", "entrada", "salida")


@lru_cache(maxsize=None)
def _sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


@lru_cache(maxsize=1)
def commit_actual():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ_REPO, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def faltantes():
    return [m for m, d in MODELOS.items() if not (MODELOS_DIR / d["archivo"]).exists()]


def imagenes_demo_ok():
    """False si las imagenes de data/demo son punteros de Git LFS (se clono sin Git LFS)."""
    imgs = sorted(IMAGENES_DEMO.glob("*.png"))
    return bool(imgs) and imgs[0].read_bytes()[:4] == b"\x89PNG"


def estado():
    falta = faltantes()
    n_casos = len(pd.read_csv(CASOS_CSV)) if CASOS_CSV.exists() else 0
    imgs_ok = imagenes_demo_ok()
    if falta:
        est = "faltan_modelos"
    elif not imgs_ok:
        est = "faltan_imagenes_lfs"
    else:
        est = "ok"
    return dict(estado=est, dispositivo=DISPOSITIVO, commit=commit_actual(), imagenes_demo_ok=imgs_ok,
                modelos_disponibles=[m for m in MODELOS if m not in falta], faltan=falta, casos_demo=n_casos)


def info_modelos():
    salida = []
    for modelo, d in MODELOS.items():
        ruta = MODELOS_DIR / d["archivo"]
        meta = json.loads((RESULTADOS_DIR / d["slug"] / "metadata.json").read_text(encoding="utf-8"))
        met = pd.read_csv(RESULTADOS_DIR / d["slug"] / "metricas.csv").set_index("conjunto")
        metricas = {cj: {k: round(float(met.loc[cj, k]), 4) for k in ("n", "exactitud", "f1_macro", "roc_auc_macro")}
                    for cj in ("val_completo", "test_orig", "test_completo")}
        salida.append(dict(modelo=modelo, nombre=meta["modelo"], archivo=d["archivo"], disponible=ruta.exists(),
                           sha256=_sha256(str(ruta)) if ruta.exists() else None, metricas_test=metricas,
                           configuracion={k: v for k, v in meta["config"].items() if k in CLAVES_CONFIG}))
    return salida
