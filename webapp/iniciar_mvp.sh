#!/usr/bin/env bash
# Arranca el MVP en Linux, macOS o WSL: crea el entorno la primera vez, instala dependencias, descarga los modelos
# y levanta el servidor. Uso, desde la raiz del repo: bash webapp/iniciar_mvp.sh
set -e
cd "$(dirname "$0")/.."

VENV=${MVP_VENV:-.venv-mvp}     # otro nombre si el mismo clon ya tiene un .venv-mvp creado desde Windows
PY=$(command -v python3.12 || command -v python3 || true)
if [ -z "$PY" ]; then
  echo "[ERROR] No se encontro Python 3. Instala Python 3.12 (Ubuntu: sudo apt install python3 python3-venv)."
  exit 1
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "[1/4] Creando el entorno virtual $VENV con $PY ..."
  "$PY" -m venv "$VENV"
fi

if [ ! -f "$VENV/.instalado" ]; then
  echo "[2/4] Instalando dependencias (la primera vez tarda varios minutos) ..."
  "$VENV/bin/python" -m pip install --upgrade pip > /dev/null
  "$VENV/bin/python" -m pip install -r webapp/backend/requirements.txt
  touch "$VENV/.instalado"
else
  echo "[2/4] Dependencias ya instaladas."
fi

echo "[3/4] Imagenes de demostracion (Git LFS) y modelos del Release ..."
git lfs pull > /dev/null 2>&1 || echo "      (git lfs no disponible: si las imagenes de data/demo no se ven, instala Git LFS y ejecuta git lfs pull)"
"$VENV/bin/python" scripts/descargar_modelos.py

echo "[4/4] Iniciando el servidor en http://localhost:8000  (Ctrl+C para detenerlo)"
echo "      Primero carga los 4 modelos; cuando diga 'Application startup complete', abre esa direccion."
exec "$VENV/bin/python" -m uvicorn webapp.backend.main:app --port 8000
