# webapp · MVP de demostración (temporal)

> **Estado:** MVP para la entrega del Hito 2 (2026-10-30). Demuestra la **simulación de resultados** y la **trazabilidad** de los 4 modelos. La webapp formal (backend ordenado y frontend en React) la desarrolla Juan Manuel Tengonó a partir de esta API. Cuando exista, este documento y la carpeta `frontend/mvp/` se reemplazan.

Se ejecuta en local. No hace falta desplegarlo ni tener el dataset de 35k: usa 45 casos de test incluidos en el repo (`data/demo/`) y los pesos del GitHub Release.

## Ejecutar

Requisitos: Python 3.12 (probado en Ubuntu 24.04 / WSL2) y Git LFS para que las imágenes de `data/demo/` se descarguen al clonar. Funciona en CPU; no necesita GPU.

```
git lfs install
git clone https://github.com/JulianSilvaMorales/Vision_Artificial.git
cd Vision_Artificial
python3 -m venv venv && source venv/bin/activate        # Windows: python -m venv venv && venv\Scripts\activate
pip install -r webapp/backend/requirements.txt
python scripts/descargar_modelos.py                      # baja los 4 pesos del Release y verifica su SHA-256
uvicorn webapp.backend.main:app --port 8000
```

Abrir **http://localhost:8000**. La documentación interactiva de la API está en http://localhost:8000/docs.

- La primera predicción tarda unos segundos, porque carga los 4 modelos. Después, ~0,2 s por imagen con los 4 modelos en CPU.
- `DEEPSPINE_DISPOSITIVO=cuda` usa la GPU. En CPU las predicciones reproducen exactamente la evaluación oficial; en GPU la ResNet usa otra precisión numérica y algún caso límite puede cambiar.
- En Windows sin WSL no se ha probado.

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
| `GET /api/salud` | Estado: modelos disponibles o faltantes, dispositivo, commit, número de casos |
| `GET /api/modelos` | Los 4 modelos: SHA-256, configuración, hiperparámetro elegido y métricas oficiales |
| `GET /api/casos?clase=&nivel=` | Lista de casos de demostración, con filtros |
| `GET /api/casos/{id}/imagen` | PNG del caso |
| `POST /api/casos/{id}/predecir` | Los 4 modelos sobre el caso: predicciones, `pasos` (traza), ensamble, `comparacion` con la etiqueta real y con lo registrado |
| `POST /api/predecir` | Igual, para una imagen subida (`multipart/form-data`, campo `imagen`, máx. 5 MB) |
| `POST /api/casos/simulacion?clase=&nivel=` | Simulación completa en el servidor: exactitud, matriz de confusión y reproducibilidad por modelo |

## Archivos

```
webapp/
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

## Para la webapp formal (Tengonó)
- Los endpoints y esquemas de arriba se pueden reutilizar tal cual desde React (`frontend/src/services/`).
- `frontend/mvp/` se puede borrar cuando exista el frontend en React. Entonces hay que quitar en `main.py` el `app.mount("/", ...)` o apuntarlo a la build de React.
- La lógica de los modelos está en `src/inference/predictor.py`; el backend no necesita saber de embeddings, escaladores ni PCA.
- Herramienta académica, no diagnóstica: mantener el aviso visible.
