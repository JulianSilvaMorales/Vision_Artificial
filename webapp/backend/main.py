"""Backend del MVP: API de los 4 modelos y pagina de demostracion.

Ejecutar desde la raiz del repo:
    uvicorn webapp.backend.main:app --port 8000
y abrir http://localhost:8000 (documentacion de la API en http://localhost:8000/docs).

Al arrancar carga los 4 modelos (si estan en models/), asi la primera prediccion no espera. Si la variable
MVP_ABRIR_NAVEGADOR tiene una URL (la ponen los scripts webapp/iniciar_mvp.*), abre el navegador cuando el
servidor ya esta listo.

Capas: api/routes (HTTP) -> schemas (validacion) -> services (logica) -> src/inference (modelos).
"""
import os
import threading
import webbrowser
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from webapp.backend.api.routes import casos, prediccion, sistema
from webapp.backend.core.config import FRONTEND_DIR, RESULTADOS_DIR
from webapp.backend.services import modelos_service, prediccion_service


@asynccontextmanager
async def ciclo_de_vida(_app):
    if not modelos_service.faltantes():
        print(f"Modelos cargados en {prediccion_service.precargar()} s", flush=True)
    url = os.environ.get("MVP_ABRIR_NAVEGADOR")
    if url:
        # uvicorn empieza a aceptar conexiones justo despues de este punto
        threading.Timer(1.0, webbrowser.open, args=[url]).start()
    yield


app = FastAPI(title="DeepSpine · MVP de demostracion",
              description="Clasificacion de discos lumbares (RM sagital T2) con 4 modelos. Herramienta academica, "
                          "no diagnostica.", version="0.1.0", lifespan=ciclo_de_vida)
app.include_router(sistema.router)
app.include_router(casos.router)
app.include_router(prediccion.router)
# figuras y tablas oficiales de docs/resultados (solo lectura), para la pestana de metricas
app.mount("/resultados", StaticFiles(directory=RESULTADOS_DIR), name="resultados")
# la pagina del MVP se sirve desde el mismo servidor; se monta al final para no tapar /api y /docs
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
