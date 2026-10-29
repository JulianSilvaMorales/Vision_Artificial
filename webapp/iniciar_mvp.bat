@echo off
REM Arranca el MVP en Windows: crea el entorno la primera vez, instala dependencias, descarga los modelos,
REM levanta el servidor y abre el navegador. Uso: doble clic, o desde la raiz del repo: webapp\iniciar_mvp.bat
setlocal
cd /d "%~dp0.."

set "VENV=.venv-mvp"
set "PY="
where py >nul 2>nul && py -3.12 --version >nul 2>nul && set "PY=py -3.12"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo [ERROR] No se encontro Python. Instala Python 3.12 desde https://www.python.org/downloads/ y marca "Add python.exe to PATH".
  pause & exit /b 1
)

if not exist "%VENV%\Scripts\python.exe" (
  echo [1/4] Creando el entorno virtual %VENV% con %PY% ...
  %PY% -m venv "%VENV%" || (echo [ERROR] No se pudo crear el entorno. & pause & exit /b 1)
)

if not exist "%VENV%\.instalado" (
  echo [2/4] Instalando dependencias (la primera vez tarda varios minutos^) ...
  "%VENV%\Scripts\python.exe" -m pip install --upgrade pip >nul
  "%VENV%\Scripts\python.exe" -m pip install -r webapp\backend\requirements.txt || (echo [ERROR] Fallo la instalacion. & pause & exit /b 1)
  echo ok> "%VENV%\.instalado"
) else (
  echo [2/4] Dependencias ya instaladas.
)

echo [3/4] Imagenes de demostracion (Git LFS) y modelos del Release ...
where git >nul 2>nul && git lfs pull >nul 2>nul
"%VENV%\Scripts\python.exe" scripts\descargar_modelos.py || (echo [ERROR] No se pudieron descargar los modelos. & pause & exit /b 1)

echo [4/4] Iniciando el servidor en http://127.0.0.1:8000  (Ctrl+C para detenerlo)
echo       Primero carga los 4 modelos; el navegador se abre solo cuando esta listo.
if not defined MVP_SIN_NAVEGADOR set "MVP_ABRIR_NAVEGADOR=http://127.0.0.1:8000"
"%VENV%\Scripts\python.exe" -m uvicorn webapp.backend.main:app --port 8000
pause
