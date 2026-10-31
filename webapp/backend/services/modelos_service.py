"""Informacion de los modelos: pesos disponibles y estado del sistema."""
import subprocess
from functools import lru_cache

from webapp.backend.core.config import DETECTOR_PT, DISPOSITIVO, MODELOS, MODELOS_DIR, RAIZ_REPO


@lru_cache(maxsize=1)
def commit_actual():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ_REPO, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def faltantes():
    return [m for m, archivo in MODELOS.items() if not (MODELOS_DIR / archivo).exists()]


def estado():
    falta = faltantes()
    return dict(estado="faltan_modelos" if falta else "ok", dispositivo=DISPOSITIVO, commit=commit_actual(),
                detector_ok=DETECTOR_PT.exists() and DETECTOR_PT.stat().st_size > 1024,
                modelos_disponibles=[m for m in MODELOS if m not in falta], faltan=falta)
