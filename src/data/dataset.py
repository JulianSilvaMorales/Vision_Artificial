"""Indice del dataset de 35k y verificaciones de fuga de datos.

Replica la logica de los cuadernos de Colab: una fila por imagen con su particion, clase,
paciente, nivel y si es original (`__orig`) o copia aumentada (`__augNNN`).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import CLASES, DATA_DIR, MANIFIESTO

RE_ID = re.compile(r"__(\d{4})_(\d{3})_")
RE_NIVEL = re.compile(r"(L[1-5]-(?:L[1-5]|S1))")
PARTICIONES = ("train", "val", "test")


def cargar_indice(data_dir=DATA_DIR):
    """Recorre data_dir/{train,val,test}/<clase>/*.png y devuelve un DataFrame ordenado."""
    data_dir = Path(data_dir)
    filas = []
    for split in PARTICIONES:
        for cdir in sorted((data_dir / split).iterdir()):
            if not cdir.is_dir():
                continue
            for f in sorted(cdir.glob("*.png")):
                m, n = RE_ID.search(f.name), RE_NIVEL.search(f.name)
                filas.append(dict(split=split, clase=cdir.name, archivo=f.name, ruta=str(f),
                                  paciente=m.group(1) if m else None, nivel=n.group(1) if n else None,
                                  es_original=f.name.endswith("__orig.png"),
                                  original=re.sub(r"__(orig|aug\d+)\.png$", "", f.name)))
    df = pd.DataFrame(filas)
    assert sorted(df.clase.unique()) == CLASES, f"Clases inesperadas: {sorted(df.clase.unique())}"
    df["y"] = df.clase.map({c: i for i, c in enumerate(CLASES)})
    return df


def verificar(df, manifiesto=MANIFIESTO):
    """Falla si hay fuga entre particiones o si el indice no coincide con el manifiesto del dataset."""
    assert df.paciente.notna().all(), "Hay archivos sin ID de paciente"
    pp = df.groupby("split").paciente.agg(set)
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not (pp[a] & pp[b]), f"FUGA: pacientes en {a} y {b}"
    orig_split = df[df.es_original].set_index("original").split
    copias = df[~df.es_original]
    assert (copias.original.map(orig_split) == copias.split).all(), "Hay copias fuera de la particion de su original"
    if Path(manifiesto).exists():
        man = pd.read_csv(manifiesto, encoding="utf-8-sig", dtype={"paciente": str})
        man["archivo"] = man.archivo.str.split("/").str[-1]
        a = df.set_index("archivo")[["split", "clase"]]
        b = man.set_index("archivo")[["particion", "clase"]].rename(columns={"particion": "split"})
        assert len(a) == len(b) and a.sort_index().equals(b.sort_index()), "El dataset no coincide con manifest_35k.csv"
    return {k: len(v) for k, v in pp.items()}


def conjuntos(df):
    """Mascaras booleanas de los conjuntos de evaluacion comunes a los 4 modelos."""
    return {
        "train_orig": ((df.split == "train") & df.es_original).values,
        "val_orig": ((df.split == "val") & df.es_original).values,
        "val_completo": (df.split == "val").values,
        "test_orig": ((df.split == "test") & df.es_original).values,
        "test_completo": (df.split == "test").values,
    }


def resumen(df):
    return pd.crosstab([df.split, df.es_original], df.clase, margins=True)
