"""Carga los scripts de scripts/ por ruta, sin copiarlos: el backend usa el mismo codigo que limpio el dataset.

scripts/ no es un paquete; cada script se importa como modulo suelto. Los bloques `if __name__ == "__main__"` no se ejecutan.
"""
import importlib.util
from functools import lru_cache

from webapp.backend.core.config import SCRIPTS_DIR


@lru_cache(maxsize=None)
def cargar(subcarpeta, nombre):
    ruta = SCRIPTS_DIR / subcarpeta / f"{nombre}.py"
    spec = importlib.util.spec_from_file_location(f"scripts_{subcarpeta}_{nombre}", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo
