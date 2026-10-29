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
|---|---|
| Jhon Jairo Perdomo Perdomo | Líder del Proyecto, Investigación, Dataset, Etiquetado, Documentación |
| Julián Andrés Silva Morales | Entrenamiento y evaluacion (YOLOv8-cls, ResNet-50), Modelos SVM y Random Forest |
| Juan Manuel Tengonó Lasso | Aplicacion web (backend + frontend) |

## Resultados

Los cuatro modelos se entrenaron con el mismo dataset, la misma division por paciente y la misma semilla (42), en una RTX 3060 Ti (Ubuntu 24.04 en WSL2, PyTorch 2.15). Las epocas e hiperparametros se eligieron con **validacion** (F1 macro); el **test se evaluo una sola vez**, al final.

Test de originales (768 imagenes en proporcion natural: 42 Degeneracion, 164 Normal, 562 Protrusion):

| Modelo | Exactitud | F1 macro | ROC-AUC | Recall Degeneracion / Normal / Protrusion |
|---|---|---|---|---|
| 1. YOLOv8s-cls (ajuste fino) | 87,8 % | 0,750 | 0,931 | 0,50 / 0,76 / 0,94 |
| **2. ResNet-50 (ajuste fino)**, mejor por validacion | **88,9 %** | **0,769** | 0,919 | 0,50 / 0,81 / 0,94 |
| 3. SVM (RBF) sobre embeddings de la ResNet-50 ajustada | 88,4 % | 0,748 | 0,899 | 0,43 / 0,77 / 0,95 |
| 4. Random Forest sobre los mismos embeddings | 88,4 % | 0,746 | 0,942 | 0,40 / 0,77 / 0,95 |

En el test completo (5.253 imagenes, balanceado con copias aumentadas) la exactitud es de 74–78 % y el F1 macro de 0,73–0,77. Como linea base del curso tambien se entrenaron regresion logistica y KNN sobre los mismos embeddings (87,8 % y 88,8 %), y como comparacion, SVM y Random Forest sobre embeddings de ImageNet sin ajustar (77,6 % y 75,3 %).

![Matrices de confusion de los 4 modelos en test de originales](docs/resultados/comparacion/matrices_confusion_test_orig.png)

Lectura honesta de las cifras:
- **Ningun modelo alcanza el 89 %** de exactitud. La ResNet-50 queda en 88,9 % (683 de 768), y el margen de error sobre 768 imagenes es de ±2,2 puntos, asi que las diferencias entre los 4 modelos no son significativas. Las variantes probadas en validacion (semillas, regularizacion, ensambles) se estancan en 86–87,5 %: el limite parece estar en los datos, no en el modelo.
- **La exactitud sobre originales esta inflada** porque el 73 % es Protrusion. Por eso se reportan tambien F1 macro y recall por clase. **Degeneracion es la clase debil**: se detecta la mitad de los 42 casos.
- **Sobreajuste:** las redes logran su mejor resultado en la epoca 2 y llegan a ~97–99 % en train.
- El SVM y el Random Forest usan las caracteristicas de la ResNet-50 ajustada (modelo 2). Con embeddings de ImageNet sin ajustar rinden mucho menos, asi que dependen del modelo 2; esto se declara en el informe.

Detalle por modelo (metricas por conjunto, por clase y por nivel, matrices, curvas, configuracion y puntajes por imagen): [`docs/resultados/`](docs/resultados/README.md). Registro de los experimentos de ajuste: [`docs/resultados/experimentos_validacion.md`](docs/resultados/experimentos_validacion.md).

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

Nota: el backend y el frontend los desarrolla otro integrante del equipo; las tecnologias concretas se documentaran aqui cuando esten definidas. La app debe usar los cuatro modelos entrenados; los modelos SVM y Random Forest requieren primero calcular los embeddings de la ResNet-50 ajustada (modelo 2) y aplicar el escalador y el PCA guardados junto al modelo. Todo eso lo hace `src/inference/predictor.py`, que la capa `services/` puede llamar directamente.

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
│   ├── entrenamiento/              # Entrenamiento, evaluacion y comparacion de los 4 modelos
│   └── build_dataset.py            # Dataset final de 35.007 imagenes
│
├── models/                     # Pesos del pipeline de datos; los 4 modelos finales se descargan del GitHub Release
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
├── docs/                        # Guias, resultados de los modelos (docs/resultados/), informes y referencias
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
| `scripts/entrenamiento/` | En el repo | Entrenamiento y evaluacion de los cuatro modelos y su comparacion (`01_yolov8s_cls.py` a `05_comparar_modelos.py`); se ejecutan desde la terminal en Ubuntu (WSL2) en lugar de cuadernos |
| `src/` | En el repo | Codigo comun: indice del dataset y verificacion de fuga (`data/`), ResNet-50 (`models/`), evaluacion (`training/`), metricas y graficas (`utils/`) y prediccion para la app (`inference/predictor.py`) |

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

**Entrenar los 4 modelos** (Linux o WSL2 con GPU NVIDIA; el dataset descomprimido en `~/deepspine_data/` con `train/`, `val/` y `test/`, o en la ruta de la variable `DEEPSPINE_DATA`):

```
python scripts/entrenamiento/01_yolov8s_cls.py                          # modelo 1
python scripts/entrenamiento/02_resnet50.py                             # modelo 2
python scripts/entrenamiento/03_embeddings.py --extractor ajustada      # embeddings de la ResNet-50 ajustada
python scripts/entrenamiento/04_svm_random_forest.py --extractor ajustada   # modelos 3 y 4 (+ logistica y KNN)
python scripts/entrenamiento/03_embeddings.py --extractor imagenet      # opcional: comparacion con ImageNet
python scripts/entrenamiento/04_svm_random_forest.py --extractor imagenet
python scripts/entrenamiento/05_comparar_modelos.py                     # tabla y figuras comparativas
```

Cada script deja sus resultados en `docs/resultados/<modelo>/` y los pesos en `models/`; los checkpoints y embeddings quedan fuera del repo, en `~/deepspine_runs/` (`DEEPSPINE_RUNS`). Tiempos medidos en una RTX 3060 Ti: YOLO ~2 min, ResNet-50 ~5 min, embeddings ~35 s, modelos clasicos ~3 min (ImageNet ~10 min). En WSL2 el `pin_memory` de los DataLoader esta desactivado (`src/utils/config.py`) porque provoca errores de memoria.

**Usar los modelos entrenados** (descargar los 4 archivos del GitHub Release en `models/`):

```python
from src.inference.predictor import Predictor
pred = Predictor()
pred.predecir("roi.png", modelo="resnet50")   # "yolov8s_cls", "resnet50", "svm" o "random_forest"
pred.predecir_todos("roi.png")                # los 4 modelos sobre la misma imagen
```

**Notebooks:** los cuadernos 00 y 01 se ejecutan en Google Colab con GPU. Usan Roboflow: la clave se guarda como secreto de Colab con el nombre `ROBOFLOW_API_KEY`; no se escribe en el cuaderno. Las variables a ajustar (rutas, workspace y proyecto) estan en las primeras celdas.

**Backend y frontend:** (pendiente; se completa cuando la aplicacion este implementada).

## Flujo de trabajo con Git

- Trabajamos con ramas por funcionalidad: `feature/<integrante>-<tarea>`, `fix/<integrante>-<tarea>`.
- Los commits siguen la convencion Conventional Commits (ver `CONTRIBUTING.md`).
- Antes de hacer merge a `main`, se revisa el codigo en Pull Request.
