"""Backend del MVP: API de los 4 modelos y pagina de demostracion.

Ejecutar desde la raiz del repo:
    uvicorn webapp.backend.main:app --port 8000
y abrir http://localhost:8000 (documentacion de la API en http://localhost:8000/docs).

Capas: api/routes (HTTP) -> schemas (validacion) -> services (logica) -> src/inference (modelos).
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from webapp.backend.api.routes import casos, prediccion, sistema
from webapp.backend.core.config import FRONTEND_DIR, RESULTADOS_DIR

app = FastAPI(title="DeepSpine · MVP de demostracion",
              description="Clasificacion de discos lumbares (RM sagital T2) con 4 modelos. Herramienta academica, "
                          "no diagnostica.", version="0.1.0")
app.include_router(sistema.router)
app.include_router(casos.router)
app.include_router(prediccion.router)
# figuras y tablas oficiales de docs/resultados (solo lectura), para la pestana de metricas
app.mount("/resultados", StaticFiles(directory=RESULTADOS_DIR), name="resultados")
# la pagina del MVP se sirve desde el mismo servidor; se monta al final para no tapar /api y /docs
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
