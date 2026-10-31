"""Backend de DeepSpine: API de los 4 modelos, aplicacion para el radiologo y demo del MVP.

Ejecutar desde la raiz del repo:
    uvicorn webapp.backend.main:app --port 8000
y abrir http://localhost:8000 (aplicacion), http://localhost:8000/mvp (demo con casos, simulacion y metricas) o
http://localhost:8000/docs (documentacion de la API).

Al arrancar carga los 4 modelos (si estan en models/), asi la primera prediccion no espera. Si la variable
MVP_ABRIR_NAVEGADOR tiene una URL (la ponen los scripts webapp/iniciar_mvp.*), abre el navegador cuando el
servidor ya esta listo.

Capas: api/routes (HTTP) -> schemas (validacion) -> services (logica) -> src/inference (modelos).
"""
import asyncio
import os
import threading
import webbrowser
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from webapp.backend.api.routes import casos, estudios, prediccion, sistema
from webapp.backend.core.config import CORS_ORIGENES, FRONTEND_APP_DIR, FRONTEND_DIR, RESULTADOS_DIR
from webapp.backend.services import estudios_service, modelos_service, prediccion_service
from webapp.backend.services.estudios_service import EstudioError
from webapp.backend.services.validacion_service import ImagenRechazada


class PaginasSinCache(StaticFiles):
    """Archivos de las paginas con Cache-Control: no-cache: el navegador revalida (ETag) y nunca usa un app.js viejo."""

    async def get_response(self, path, scope):
        r = await super().get_response(path, scope)
        r.headers["Cache-Control"] = "no-cache"
        return r


@asynccontextmanager
async def ciclo_de_vida(_app):
    if not modelos_service.faltantes():
        print(f"Modelos cargados en {prediccion_service.precargar()} s", flush=True)
    estudios_service.limpiar_vencidos()
    try:
        print(f"Detector de discos cargado en {estudios_service.precargar_detector()} s", flush=True)
    except EstudioError as e:       # sin el detector la pestana de estudios no funciona, pero el resto de la app si
        print(f"[AVISO] {e.mensaje}", flush=True)
    url = os.environ.get("MVP_ABRIR_NAVEGADOR")
    if url:
        # uvicorn empieza a aceptar conexiones justo despues de este punto
        threading.Timer(1.0, webbrowser.open, args=[url]).start()
    limpieza = asyncio.create_task(estudios_service.limpieza_periodica())
    yield
    limpieza.cancel()
    estudios_service.limpiar_vencidos(0)       # al detener el servidor no quedan estudios en el disco


app = FastAPI(title="DeepSpine · Apoyo al diagnostico lumbar",
              description="Clasificacion de discos lumbares (RM sagital T2) con 4 modelos. Herramienta academica, "
                          "no diagnostica.", version="0.1.0", lifespan=ciclo_de_vida)
# CORS solo para el desarrollo del frontend en otro puerto (por ejemplo Vite); en produccion se sirve desde este servidor
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGENES, allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(ImagenRechazada)
async def _imagen_rechazada(_req: Request, e: ImagenRechazada):
    return JSONResponse(status_code=422, content={"detail": e.detalle["mensaje"], **e.detalle})


@app.exception_handler(EstudioError)
async def _estudio_error(_req: Request, e: EstudioError):
    return JSONResponse(status_code=e.codigo, content={"detail": e.mensaje})


app.include_router(sistema.router)
app.include_router(casos.router)
app.include_router(prediccion.router)
app.include_router(estudios.router)
# figuras y tablas oficiales de docs/resultados (solo lectura), para la pestana de metricas
app.mount("/resultados", StaticFiles(directory=RESULTADOS_DIR), name="resultados")
@app.get("/mvp", include_in_schema=False)
def _mvp_sin_barra():
    return RedirectResponse("/mvp/")


# las paginas se sirven desde el mismo servidor; se montan al final para no tapar /api y /docs
app.mount("/mvp", PaginasSinCache(directory=FRONTEND_DIR, html=True), name="mvp")
app.mount("/", PaginasSinCache(directory=FRONTEND_APP_DIR, html=True), name="app")
