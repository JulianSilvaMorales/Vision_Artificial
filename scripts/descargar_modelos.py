"""Descarga los pesos de los 4 modelos desde el GitHub Release y verifica su SHA-256.

Solo usa la libreria estandar, asi que se puede correr antes de instalar las dependencias.
Si un archivo ya existe con el hash correcto, no lo vuelve a descargar.

Uso (desde la raiz del repo):
    python scripts/descargar_modelos.py [--destino models]
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/JulianSilvaMorales/Vision_Artificial/releases/download/modelos-v1.0"
ARCHIVOS = {   # mismos hashes que models/README.md
    "yolov8s_cls.pt": "4993c3f972eccea9d1df3841c545b141735cf319639b8213bf5150f562e380e4",
    "resnet50.pt": "7ed202884b3b606a62314888e7720ef660ff8079bf443f458aff0e1e69e4719c",
    "svm.joblib": "57beb318fb4f40edebef9ae00c471d029ddc72b447dce11c9091c7fa23fc6727",
    "random_forest.joblib": "b3b6167d85a44ea860edb4e7dced0837bf9678aa1f7ab708aa845fdcdfb30933",
}


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--destino", default=str(Path(__file__).resolve().parents[1] / "models"))
    destino = Path(ap.parse_args().destino)
    destino.mkdir(parents=True, exist_ok=True)
    fallos = 0
    for nombre, esperado in ARCHIVOS.items():
        ruta = destino / nombre
        if ruta.exists() and sha256(ruta) == esperado:
            print(f"OK (ya estaba)  {nombre}")
            continue
        print(f"Descargando     {nombre} ...", flush=True)
        tmp = ruta.with_suffix(ruta.suffix + ".part")
        urllib.request.urlretrieve(f"{RELEASE}/{nombre}", tmp)
        if sha256(tmp) != esperado:
            tmp.unlink()
            print(f"ERROR           {nombre}: el SHA-256 no coincide; se descarto el archivo")
            fallos += 1
            continue
        tmp.replace(ruta)
        print(f"OK (verificado) {nombre}")
    if fallos:
        sys.exit(1)
    print(f"Los 4 modelos estan en {destino}")


if __name__ == "__main__":
    main()
