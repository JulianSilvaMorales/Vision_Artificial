# Vision_Artificial

Proyecto para la materia de Vision Artificial. Incluye entrenamiento de modelos de IA, scripts de procesamiento en Python y una aplicacion web (backend + frontend) para exponer y usar el modelo.

## Descripcion

Clasificacion automatica de **discos intervertebrales lumbares** en imagenes de resonancia magnetica sagital T2 (recortes o ROI de cada disco) en tres clases:

- `Normal`
- `Degeneracion_Discal`
- `Protrusion` (cualquier convexidad del borde posterior del disco, de cualquier grado; las clases Hernia y Abombamiento se fusionaron en esta)

El flujo del proyecto es: recorte del ROI de cada disco, construccion de un dataset balanceado con division por paciente, entrenamiento y comparacion de **cuatro modelos** (YOLOv8-cls, ResNet-50, SVM y Random Forest sobre embeddings) con las mismas metricas, y una aplicacion web que usa los modelos entrenados.

**Dataset construido:** 35.007 imagenes (train 24.501, val 5.253, test 5.253), balanceadas por conteo entre las tres clases mediante aumento de datos, a partir de 5.123 ROI originales de 399 pacientes. La division 70/15/15 es **por paciente**: ningun paciente aparece en dos particiones y cada copia aumentada queda en la particion de su original.

**Limites que se declaran:**
- El tamano efectivo del dataset es el numero de pacientes (399); las copias aumentadas no aportan anatomia nueva.
- Las etiquetas fueron asignadas por estudiantes, sin radiologo; no se midio el acuerdo entre anotadores.
- La fusion de clases se decidio despues de ver los datos; se declara aqui y en la guia de anotacion (`docs/guias/`).
- En el conjunto de originales de test hay solo 42 imagenes de Degeneracion, asi que sus metricas tienen margen de error amplio.
- Herramienta academica, no diagnostica.

## DATASET ORIGINAL 

Lumbar Spine MRI Dataset: https://data.mendeley.com/datasets/k57fr854j2/2

## Integrantes

| Nombre | Rol / Aporte |
| Jhon Jairo Perdomo Perdomo | Líder del Proyecto, Investigación, Dataset, Etiquetado, Documentación |
| Julián Andrés Silva Morales | Entrenamiento y evaluacion (YOLOv8-cls, ResNet-50), Modelos SVM y Random Forest |
| Juan Manuel Tengonó Lasso | Aplicacion web (backend + frontend) |

## Resultados preliminares

Un solo seed (42); se actualizara cuando esten los cuatro modelos.

| Modelo | Estado |
|---|---|
| 1. YOLOv8s-cls | Entrenado. F1 macro 0.755 en test de originales (768 imagenes) y 0.780 en test completo (5.253). |
| 2. ResNet-50 (ajuste fino) | En curso |
| 3. SVM (RBF) sobre embeddings | En curso |
| 4. Random Forest sobre embeddings | En curso |

El detalle, las matrices de confusion y la comparativa se agregaran a `docs/` cuando esten los cuatro modelos.

## Arquitectura del proyecto

Este repositorio combina tres tipos de organizacion, cada una pensada para la parte del proyecto que le corresponde:

### 1. Datos y ML: Cookiecutter Data Science

Las carpetas `data/`, `src/`, `scripts/`, `models/` y `notebooks/` siguen la convencion estandar de proyectos de Machine Learning conocida como Cookiecutter Data Science. Organiza el proyecto por etapa del flujo de trabajo (de donde vienen los datos, que codigo los transforma, como se ejecuta, donde queda el modelo entrenado, donde se explora), en vez de por capas de arquitectura de software tradicional. Es el estandar usado en la industria para este tipo de proyectos.

### 2. Backend: arquitectura en capas (ligera)

`webapp/backend/` sigue una arquitectura en capas simplificada, inspirada en el espiritu de Ports & Adapters (hexagonal) pero sin su complejidad completa:

```
Peticion HTTP -> api/routes/ -> schemas/ (validacion) -> services/ (logica) -> src/inference/ (modelo) -> respuesta
```

Regla clave: las rutas (`api/routes/`) nunca llaman directo al codigo de inferencia (`src/inference/`). Siempre pasan por `services/`, que orquesta la logica. Esto separa "como entra la peticion" de "que hace el proyecto", y permite agregar mas adelante una capa `repositories/` (para base de datos) sin reorganizar nada, si el proyecto llega a necesitarla.

Nota: el backend y el frontend los desarrolla otro integrante del equipo; las tecnologias concretas se documentaran aqui cuando esten definidas. La app debe usar los cuatro modelos entrenados; los modelos SVM y Random Forest requieren primero calcular los embeddings de una ResNet-50 congelada (ImageNet) y aplicar el escalador y el PCA guardados junto al modelo.

### 3. Frontend: organizacion por componentes (React)

`webapp/frontend/` sigue la convencion estandar de proyectos React: separar piezas de UI reutilizables (`components/`) de vistas completas (`pages/`), y centralizar las llamadas a la API en `services/` para no repetir codigo de fetch/axios en cada componente.

## Estructura del proyecto

```
Vision_Artificial/
├── README.md
├── CONTRIBUTING.md
├── .gitignore
├── .gitattributes           # Configuracion de Git LFS
├── requirements.txt
│
├── data/                     # Datos del proyecto
│   ├── raw/                   # Datos originales, sin modificar (solo enlace al dataset de Mendeley)
│   ├── processed/              # Dataset final de 35.007 imagenes (o su enlace), manifiestos y etiquetas
│   └── external/                # Datasets de terceros / descargas externas
│
├── src/                        # Codigo fuente reutilizable (paquete Python)
│   ├── data/                    # Carga y preprocesamiento de datos
│   ├── models/                   # Definicion de arquitecturas del modelo (CNN, etc.)
│   ├── training/                  # Logica de entrenamiento
│   ├── inference/                  # Prediccion usando el modelo ya entrenado
│   └── utils/                       # Funciones auxiliares (config, logging, graficos, metricas)
│
├── scripts/                    # Scripts ejecutables
│   ├── preprocesamiento/        # Ingesta, conversion y limpieza de cortes
│   ├── recorte_roi/              # Recorte del ROI de cada disco
│   ├── etiquetado/                # Semilla, split por paciente y registro de etiquetas
│   ├── guias_visuales/             # Figuras de la guia de anotacion
│   └── build_dataset.py            # Dataset final de 35.007 imagenes
│
├── models/                     # Pesos entrenados (YOLOv8-cls, ResNet-50, SVM, Random Forest)
│
├── notebooks/                  # Notebooks (Colab): preetiquetado, entrenamiento de los 4 modelos y comparacion
│
├── webapp/
│   ├── backend/                 # API que sirve el modelo (arquitectura en capas)
│   │   ├── api/routes/            # Endpoints (ej: POST /predict)
│   │   ├── core/                   # Configuracion, variables de entorno
│   │   ├── services/                # Logica de negocio, orquesta la prediccion
│   │   ├── schemas/                  # Validacion de datos de entrada/salida
│   │   └── utils/                     # Helpers propios del backend
│   │
│   └── frontend/                # Interfaz web (React)
│       ├── public/                # Archivos estaticos (favicon, index.html)
│       └── src/
│           ├── assets/              # Imagenes/iconos/estilos de la app
│           ├── components/           # Piezas de UI reutilizables
│           ├── pages/                 # Vistas completas
│           ├── services/               # Llamadas a la API del backend
│           └── utils/                   # Helpers propios del frontend
│
├── tests/                       # Pruebas unitarias del codigo en src/
├── docs/                        # Guias, informes, modelo matematico, seleccion de modelos, referencias
└── assets/                      # Imagenes/recursos usados en el README
```

Contenido actual y previsto de `scripts/` y `notebooks/`:

| Ruta | Estado | Para que sirve |
|---|---|---|
| `scripts/preprocesamiento/` | En el repo | Filtrado de cortes T2 sagitales (`ingestion.py`), conversion DICOM a PNG, limpieza, filtrado y normalizacion de cortes |
| `scripts/recorte_roi/` | En el repo | Extraccion y recorte del ROI de cada disco, incluido el recorte automatico con el detector YOLO (`autocrop_rois.py`, `aplicar_recortes.py`) |
| `scripts/etiquetado/` | En el repo | Construccion de la semilla de etiquetado, split por paciente, verificacion sin fuga y registro de las etiquetas revisadas |
| `scripts/guias_visuales/` | En el repo | Figuras de la guia de anotacion (`docs/guias/`) |
| `scripts/build_dataset.py` | En el repo | Construye el dataset de 35.007 imagenes: split por paciente 70/15/15, balanceo y aumento dentro de cada particion, verificacion |
| `notebooks/00_entrenamiento_detector_discos.ipynb` | En el repo | Entrenamiento del detector de discos (YOLO) usado para recortar los ROI |
| `notebooks/01_preetiquetado_semilla_v2.ipynb` | En el repo | Modelo semilla (YOLOv8-cls) para preetiquetar los ROI; la revision humana posterior produce `etiquetas_revisadas.csv` |
| `notebooks/02` a `05` | Pendiente | Entrenamiento de los cuatro modelos (YOLOv8s-cls, ResNet-50, SVM y Random Forest sobre embeddings) y su comparacion; se agregan cuando terminen los entrenamientos |

Los scripts de `scripts/` se conservan tal como se usaron durante el trabajo; las rutas de entrada y salida se configuran dentro de cada uno (ver `if __name__ == "__main__"` o sus argumentos).

Nota: esta seccion se debe actualizar a medida que el proyecto avance, para que siempre refleje la organizacion real del repositorio.

## Git LFS

Este proyecto usa Git LFS para versionar archivos pesados (imagenes, CSV, checkpoints de modelos). Los tipos de archivo rastreados por LFS estan definidos en `.gitattributes`. Cualquiera que clone el repo debe tener Git LFS instalado (`git lfs install`) antes de descargar los archivos de `data/` o `models/`.

Pendiente de definir: como se distribuye el dataset de 35.007 imagenes (dentro del repo con LFS, como un solo archivo comprimido, o mediante un enlace externo), segun la cuota de LFS disponible. Mientras tanto, `data/processed/` guarda las etiquetas y manifiestos, y `scripts/build_dataset.py` regenera las imagenes.

## Cambios en archivos de configuracion

- `.gitignore`: se agregaron dos excepciones a la regla `models/*.pt` para poder versionar los pesos del detector de discos y del modelo de preetiquetado:
  ```
  !models/best_disc_detector.pt
  !models/preetiquetado_semilla_v2.pt
  ```
  El resto de los `.pt` y `.h5` de `models/` sigue ignorado.
- `.gitattributes`: sin cambios.

## Instalacion

Requisitos: Python 3.10 o superior y Git LFS.

```
git lfs install
git clone https://github.com/JulianSilvaMorales/Vision_Artificial.git
cd Vision_Artificial
python -m venv venv
venv\Scripts\activate          # en Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

Las dependencias del frontend se documentaran cuando este definido.

## Uso

**Construir el dataset** (requiere los ROI originales recortados y `data/processed/etiquetado/etiquetas_revisadas.csv`):

```
python scripts/build_dataset.py --simular            # vista previa sin escribir imagenes
python scripts/build_dataset.py                       # construye el dataset
python scripts/build_dataset.py --continuar           # reanuda si se interrumpio
```

La ruta base se define con la variable de entorno `DEEPSPINE_BASE` (por defecto, la carpeta actual). El script espera en esa carpeta `t2_disc_roi_patches_v2_ext/` (ROI) y `DeepSpine_Dataset_Etiquetado/etiquetas_revisadas.csv`, y escribe `DeepSpine_Dataset_35k/`.

**Notebooks:** los cuadernos 00 y 01 se ejecutan en Google Colab con GPU. Usan Roboflow: la clave se guarda como secreto de Colab con el nombre `ROBOFLOW_API_KEY`; no se escribe en el cuaderno. Las variables a ajustar (rutas, workspace y proyecto) estan en las primeras celdas.

**Backend y frontend:** (pendiente; se completa cuando la aplicacion este implementada).

## Flujo de trabajo con Git

- Trabajamos con ramas por funcionalidad: `feature/<integrante>-<tarea>`, `fix/<integrante>-<tarea>`.
- Los commits siguen la convencion Conventional Commits (ver `CONTRIBUTING.md`).
- Antes de hacer merge a `main`, se revisa el codigo en Pull Request.
