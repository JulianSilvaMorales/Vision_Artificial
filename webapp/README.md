# webapp · MVP de demostración (temporal)

> **TEMPORAL.** MVP solo para la entrega del Hito 2 (2026-10-30): demuestra la **simulación de resultados** y la **trazabilidad** de los 4 modelos. **No es la webapp final.** La webapp formal (backend ordenado y frontend en React) la desarrolla Juan Manuel Tengonó a partir de esta API. Cuando funcione, **hay que borrar los archivos temporales del MVP** y reescribir este documento (ver [qué borrar y qué conservar](#qué-borrar-y-qué-conservar)).

Se ejecuta en local, en **Windows, Linux o macOS**, sin GPU. No hace falta desplegarlo ni tener el dataset de 35k: usa 45 casos de test incluidos en el repo (`data/demo/`) y los pesos de los 4 modelos del [GitHub Release `modelos-v1.0`](https://github.com/JulianSilvaMorales/Vision_Artificial/releases/tag/modelos-v1.0).

## Guía rápida de arranque

### 1. Requisitos (una sola vez por equipo)

| Programa | Para qué | Cómo comprobarlo |
|---|---|---|
| **Git** | Clonar el repositorio | `git --version` |
| **Git LFS** | Descargar las imágenes de `data/demo/` (sin Git LFS llegan como archivos de texto y la demo avisa del error) | `git lfs version` |
| **Python 3.12 o superior** (probado con 3.12 en Ubuntu y 3.14 en Windows) | Ejecutar el backend y los modelos | `python --version` (Windows) o `python3 --version` |
| Internet, la primera vez | Instalar dependencias (~1 GB con PyTorch) y descargar los modelos (~121 MB) | — |

Instalación:
- **Windows:** Git desde https://git-scm.com (ya trae Git LFS). Python desde https://www.python.org/downloads/ (marca *"Add python.exe to PATH"*).
- **Ubuntu / WSL:** `sudo apt install git git-lfs python3 python3-venv`.
- **macOS:** `brew install git git-lfs python@3.12`.

Después, una sola vez: `git lfs install`.

### 2. Clonar el repositorio

```
git clone https://github.com/JulianSilvaMorales/Vision_Artificial.git
cd Vision_Artificial
```

### 3. Arrancar con un solo comando

| Sistema | Comando (desde la raíz del repo) |
|---|---|
| **Windows** | `webapp\iniciar_mvp.bat`, o doble clic en `webapp/iniciar_mvp.bat` |
| **Linux / macOS / WSL** | `bash webapp/iniciar_mvp.sh` |

El script hace todo en orden:
1. Crea un entorno virtual `.venv-mvp/` en la raíz del repo (solo la primera vez).
2. Instala `webapp/backend/requirements.txt` (solo la primera vez; tarda varios minutos).
3. Ejecuta `git lfs pull` para traer las imágenes de demostración, y `scripts/descargar_modelos.py` para bajar los 4 pesos a `models/` y verificar su SHA-256. Si ya están, no los vuelve a descargar.
4. Levanta el servidor y, en Windows, abre el navegador en **http://localhost:8000**.

Las siguientes veces solo hace los pasos 3 y 4, en unos segundos. Para detener el servidor: `Ctrl + C` en la terminal.

### 4. Paso a paso manual (si prefieres no usar el script)

Windows (PowerShell o CMD):
```
python -m venv .venv-mvp
.venv-mvp\Scripts\activate
pip install -r webapp\backend\requirements.txt
git lfs pull
python scripts\descargar_modelos.py
python -m uvicorn webapp.backend.main:app --port 8000
```

Linux / macOS / WSL:
```
python3 -m venv .venv-mvp
source .venv-mvp/bin/activate
pip install -r webapp/backend/requirements.txt
git lfs pull
python scripts/descargar_modelos.py
python -m uvicorn webapp.backend.main:app --port 8000
```

Abrir **http://localhost:8000**. La documentación interactiva de la API está en **http://localhost:8000/docs**.

### 5. Comprobar que todo está bien
La barra superior de la página debe decir **"4 modelos cargables · CPU · commit … · 45 casos"** en verde. Si está en rojo, dice qué falta. También se puede consultar `http://localhost:8000/api/salud`.

### Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| Barra roja: *"Faltan modelos"* | No se descargaron los pesos | `python scripts/descargar_modelos.py` (necesita internet) |
| Barra roja: *"Las imágenes de data/demo no se descargaron"*, o miniaturas rotas | Se clonó sin Git LFS | `git lfs install` y luego `git lfs pull` |
| `ERROR: SHA-256 no coincide` al descargar | Descarga corrupta o incompleta | Volver a ejecutar el script; borra el archivo malo solo |
| Error al cargar `svm.joblib` o `random_forest.joblib` | Otra versión de scikit-learn | Debe ser la 1.9.1: `pip install scikit-learn==1.9.1` |
| `[Errno 98]` / `10048`: el puerto está en uso | Ya hay algo en el 8000 | Usar otro puerto: `... uvicorn webapp.backend.main:app --port 8001` |
| `ModuleNotFoundError: No module named 'webapp'` o `'src'` | Se ejecutó desde otra carpeta | Ejecutar desde la **raíz** del repo |
| La primera predicción tarda varios segundos | Se están cargando los 4 modelos | Normal; luego son ~0,2 s por imagen |
| `pip` no encuentra `torch` para tu versión de Python | Python demasiado viejo o demasiado nuevo | Usar Python 3.12 a 3.14 |
| Windows: `OSError: [Errno 2] No such file or directory: ...torch\include\...` al instalar | El repo está en una ruta muy larga y Windows limita las rutas a 260 caracteres | Clonar en una ruta corta (por ejemplo `C:\Users\<usuario>\Documents\Vision_Artificial`) o activar las rutas largas de Windows |
| Usas el mismo clon desde Windows y desde WSL | El `.venv-mvp` de un sistema no sirve en el otro | En WSL: `MVP_VENV=.venv-mvp-linux bash webapp/iniciar_mvp.sh` |

Opcional: con GPU NVIDIA y PyTorch con CUDA, `DEEPSPINE_DISPOSITIVO=cuda` usa la GPU. En CPU (por defecto) las predicciones reproducen exactamente la evaluación oficial; en GPU la ResNet usa otra precisión numérica y algún caso límite puede cambiar.

## Qué muestra (pestañas)

| Pestaña | Qué demuestra |
|---|---|
| **1 · Caso a caso** | Eliges uno de los 45 casos de test (paciente, nivel, etiqueta real) y lo pasas por los 4 modelos. Ves la clase y las probabilidades de cada uno, si acierta y si **reproduce** la predicción registrada en la evaluación oficial. Debajo, el **diagrama de trazabilidad**: la imagen original y la de 224 px, las 2.048 características de la ResNet-50, el escalador y el PCA (256 componentes), los puntajes del SVM, los votos de los 300 árboles del Random Forest y el camino independiente de YOLO, cada uno con sus valores reales y su tiempo |
| **2 · Subir imagen** | Lo mismo con un recorte (ROI) propio; sin etiqueta real |
| **3 · Simulación** | Corre los 4 modelos sobre los 45 casos uno por uno y arma **en vivo** la matriz de confusión de cada modelo, con una tabla caso por caso y la columna *reproduce* (4/4 si los 4 modelos dan lo mismo que en `docs/resultados/`) |
| **4 · Modelos y métricas** | Métricas oficiales (test de originales, test completo, validación), hiperparámetro elegido con validación, **SHA-256 del archivo de pesos cargado** y las figuras de `docs/resultados/comparacion/` |

**Importante para la presentación:** los 45 casos están balanceados a propósito (15 por clase, 3 por nivel). Su exactitud (51–60 %) **no** es la del test oficial, que tiene 768 imágenes y 73 % de Protrusión, donde los modelos dan 87,8–88,9 %. Un tercio de los casos es Degeneración, la clase más difícil.

Enlaces directos útiles en clase:
- `http://localhost:8000/#caso=caso_03`: abre el caso 3 y lo ejecuta. Los modelos no coinciden: YOLO falla y los otros 3 aciertan Degeneración.
- `http://localhost:8000/#simular=1`: lanza la simulación al abrir.
- `http://localhost:8000/#tab=modelos`: abre la pestaña de métricas.

## Cómo se garantiza la trazabilidad

| Pregunta | Dónde se responde |
|---|---|
| ¿De dónde sale la imagen? | `data/demo/casos.csv`: paciente, nivel, partición de test, original (no copia aumentada). Se eligió con `scripts/demo/seleccionar_casos_demo.py` (semilla 42) |
| ¿Qué modelo exacto se usó? | `/api/modelos` y la pestaña 4: SHA-256 del archivo, igual al de `models/README.md` y al del Release `modelos-v1.0` |
| ¿Con qué código? | La barra superior muestra el commit de git en ejecución |
| ¿Qué le pasa a la imagen en cada paso? | `POST /api/casos/{id}/predecir` devuelve `pasos` con los valores intermedios y el tiempo de cada etapa |
| ¿El resultado es reproducible? | Cada caso trae `pred_registrada_<modelo>`, tomada de `docs/resultados/<modelo>/puntajes.npz`, y la demo comprueba que coincida (45/45 en CPU) |
| ¿Cómo se entrenó y evaluó cada modelo? | `docs/resultados/`, `docs/modelo_matematico/` y `scripts/entrenamiento/` |

## API

Arquitectura en capas: `api/routes/` (HTTP) → `schemas/` (validación con Pydantic) → `services/` (lógica) → `src/inference/predictor.py` (modelos). Las rutas nunca llaman directamente a los modelos.

| Método y ruta | Qué hace |
|---|---|
| `GET /api/salud` | Estado: modelos disponibles o faltantes, imágenes de demostración, dispositivo, commit, número de casos |
| `GET /api/modelos` | Los 4 modelos: SHA-256, configuración, hiperparámetro elegido y métricas oficiales |
| `GET /api/casos?clase=&nivel=` | Lista de casos de demostración, con filtros |
| `GET /api/casos/{id}/imagen` | PNG del caso |
| `POST /api/casos/{id}/predecir` | Los 4 modelos sobre el caso: predicciones, `pasos` (traza), ensamble, `comparacion` con la etiqueta real y con lo registrado |
| `POST /api/predecir` | Igual, para una imagen subida (`multipart/form-data`, campo `imagen`, máx. 5 MB) |
| `POST /api/casos/simulacion?clase=&nivel=` | Simulación completa en el servidor: exactitud, matriz de confusión y reproducibilidad por modelo |
| `GET /resultados/...` | Archivos de `docs/resultados/` (figuras y tablas oficiales), solo lectura |

## Archivos

```
webapp/
├── README.md                      # esta guia
├── iniciar_mvp.bat                # arranque en Windows (un comando)
├── iniciar_mvp.sh                 # arranque en Linux / macOS / WSL
├── backend/
│   ├── main.py                    # app FastAPI; sirve la API, /resultados y la pagina del MVP
│   ├── requirements.txt           # dependencias minimas para ejecutar el MVP
│   ├── api/routes/                # sistema.py, casos.py, prediccion.py
│   ├── schemas/prediccion.py      # modelos de entrada y salida
│   ├── services/                  # modelos_service, casos_service, prediccion_service
│   ├── core/config.py             # rutas y dispositivo
│   └── utils/imagenes.py
└── frontend/
    └── mvp/                       # index.html, styles.css, app.js (sin dependencias; temporal)
```

## Siguientes pasos para la webapp formal

Responsable: **Juan Manuel Tengonó**. El MVP deja resuelto el consumo de los modelos; lo que falta es la aplicación ordenada y presentable.

**Qué se reutiliza tal cual**
- `src/inference/predictor.py`: la lógica de los 4 modelos (`predecir`, `predecir_todos`, `trazar`). El backend no necesita saber de embeddings, escaladores ni PCA.
- La API de `webapp/backend/`: endpoints, esquemas y servicios. Ya sigue la arquitectura en capas del README.
- `scripts/descargar_modelos.py` y los casos de `data/demo/`.

**Qué hay que construir**
1. **Frontend en React** en `webapp/frontend/src/` (estructura `components/`, `pages/`, `services/` del README):
   - las llamadas a la API van en `services/` (por ejemplo, `api.js` con `fetch` a `/api/...`);
   - una vista por cada pestaña del MVP (caso a caso, subir imagen, simulación, modelos y métricas) como `pages/`;
   - el diagrama de trazabilidad, las tarjetas de predicción y las matrices como `components/`.
2. **Conectar React con el backend:**
   - en desarrollo, React en su propio puerto (por ejemplo, Vite en 5173) llamando a `http://localhost:8000/api`. Hay que agregar `CORSMiddleware` en `main.py` para ese origen;
   - en producción, servir la *build* de React desde FastAPI, cambiando el `app.mount("/", ...)` de `main.py` para que apunte a `webapp/frontend/dist`.
3. **Borrar los archivos temporales del MVP** cuando la versión en React cubra sus 4 pestañas. Ver la tabla de abajo.
4. **Mantener:**
   - el aviso "herramienta académica, no diagnóstica";
   - la aclaración de que los 45 casos son balanceados y no dan la exactitud oficial;
   - la comprobación de reproducibilidad (columna *reproduce*);
   - **CPU por defecto**, para reproducir la evaluación oficial.
5. **Opcional:**
   - pruebas en `tests/` para los endpoints, con `fastapi.testclient`;
   - un historial de predicciones;
   - una opción del ensamble en la interfaz, si el equipo la quiere.

**Reglas que no cambian:** no reentrenar ni cambiar los pesos sin publicar un nuevo Release con sus hashes, y no tocar `data/demo/casos.csv` a mano (se regenera con su script).

### Qué borrar y qué conservar

Cuando la webapp formal funcione, el MVP sobra. Hay que borrar sus archivos temporales para que el repo no quede con archivos innecesarios. Pero **no todo lo de esta carpeta es temporal**: el backend y el `Predictor` son la base de la webapp formal.

| Archivo o carpeta | Qué hacer | Por qué |
|---|---|---|
| `webapp/frontend/mvp/` (index.html, styles.css, app.js) | **Borrar** | Es la página temporal; la reemplaza el frontend en React |
| `webapp/iniciar_mvp.bat`, `webapp/iniciar_mvp.sh` | **Borrar** (o reemplazarlos por scripts de arranque de la webapp formal) | Arrancan el MVP |
| `.venv-mvp*/` en `.gitignore` | **Quitar la línea** si se borran los scripts | Solo la usan esos scripts |
| `app.mount("/", ...)` en `webapp/backend/main.py` | **Cambiar** para servir la *build* de React, o quitarlo | Hoy sirve la página del MVP |
| Este `webapp/README.md` y la sección *Demo (MVP)* del README raíz | **Reescribir** para la webapp formal | Describen el MVP |
| `webapp/backend/` (API, esquemas, servicios) | **Conservar** y mejorar | Es la base del backend formal; no es temporal |
| `src/inference/predictor.py` | **Conservar** | Lógica de los 4 modelos |
| `scripts/descargar_modelos.py` | **Conservar** | Descarga y verifica los pesos del Release |
| `data/demo/` y `scripts/demo/` | **Conservar** si la webapp formal mantiene la simulación y la trazabilidad (recomendado para la sustentación); si no, borrar los dos juntos | Son los casos de test de la demostración |

**No borres nada del MVP antes de que la webapp formal funcione:** hasta entonces es lo único que demuestra los modelos.

Para borrar (desde la raíz del repo, en una rama `feature/tengono-...`):
```
git rm -r webapp/frontend/mvp webapp/iniciar_mvp.bat webapp/iniciar_mvp.sh
git commit -m "chore(frontend): retirar el MVP temporal reemplazado por la webapp formal"
```
