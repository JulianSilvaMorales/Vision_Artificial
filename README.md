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

## DATASET CONSTRUIDO (35.007 IMAGENES)

El dataset final con el que se entrenaron los 4 modelos esta en Google Drive, como un solo archivo `DeepSpine_Dataset_35k.zip` (405 MB): https://drive.google.com/drive/folders/1ZbTE1gb6zv7F2qEBmwRXzHFveLyRrl4I?usp=drive_link

Al descomprimirlo quedan las carpetas `train/`, `val/` y `test/`, cada una con una subcarpeta por clase. Es el contenido que se usa en `data/processed/` (las imagenes no se suben al repo). Para entrenar, se descomprime donde indique `DEEPSPINE_DATA` (por defecto `~/deepspine_data/`); ver [Uso](#uso). Su manifiesto (`data/processed/dataset_35k/manifest_35k.csv`) permite verificar que el zip descargado es el mismo: los scripts de entrenamiento lo comprueban al cargar el dataset.

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

Detalle por modelo (metricas por conjunto, por clase y por nivel, matrices, curvas, configuracion y puntajes por imagen): [`docs/resultados/`](docs/resultados/README.md). Registro de los experimentos de ajuste: [`docs/resultados/experimentos_validacion.md`](docs/resultados/experimentos_validacion.md). Ecuaciones y graficos de los 4 modelos: [`docs/modelo_matematico/`](docs/modelo_matematico/README.md).

## Entrenamiento de los modelos: codigo, funcionamiento y resultados

### Flujo

```
dataset 35k (train / val / test por paciente)
   │
   ├─ 01_yolov8s_cls.py ──────────────► modelo 1: YOLOv8s-cls ajustada
   ├─ 02_resnet50.py ─────────────────► modelo 2: ResNet-50 ajustada
   │                                        │
   │                 03_embeddings.py ◄─────┘  (2048 caracteristicas por imagen, sin la capa final)
   │                        │
   │                 04_svm_random_forest.py ─► modelos 3 y 4: SVM y Random Forest (+ logistica y KNN)
   │
   └─ 05_comparar_modelos.py ─► tabla y figuras comparativas · 06_figuras_modelo_matematico.py ─► figuras del modelo matematico
```

Reglas que cumplen todos los scripts:
- Al cargar el dataset verifican que ningun paciente este en dos particiones, que cada copia aumentada este en la particion de su original y que los archivos coincidan con `manifest_35k.csv`. Si algo falla, se detienen.
- Los parametros se aprenden con train. La epoca y los hiperparametros se eligen con validacion (F1 macro). El test se evalua una sola vez, al final.
- El escalador y el PCA se ajustan solo con train (sin fuga de datos).
- Todo usa la semilla 42.

### Que hace cada script (`scripts/entrenamiento/`)

| Script | Como funciona | Que produce |
|---|---|---|
| `01_yolov8s_cls.py` | Ajuste fino de YOLOv8s-cls (preentrenada en ImageNet) con ultralytics. Configuracion: AdamW (lr 5e-4), 4 capas congeladas, dropout 0,3, coseno, hasta 15 epocas con paciencia 4. Ultralytics guarda la epoca con mejor exactitud en validacion. Despues predice las 35.007 imagenes | `docs/resultados/modelo1_yolov8s_cls/` y `models/yolov8s_cls.pt`. Mejor epoca: 2 de 6 |
| `02_resnet50.py` | Ajuste fino de ResNet-50 (torchvision, ImageNet). Congela `conv1`, `bn1` y `layer1`, y cambia la capa final por Dropout(0,3) + Linear(2048→3). AdamW (lr 1e-4, weight decay 0,01), coseno, suavizado de etiquetas 0,1, aumento en linea suave y precision mixta. Guarda la epoca con mejor F1 macro en validacion, con parada temprana; `--reanudar` continua un entrenamiento cortado | `docs/resultados/modelo2_resnet50/` y `models/resnet50.pt`. Mejor epoca: 2 de 6 |
| `03_embeddings.py` | Pasa las 35.007 imagenes por la ResNet-50 sin su capa final y guarda 2.048 valores por imagen. `--extractor ajustada` (oficial) usa la del modelo 2; `--extractor imagenet` la de ImageNet sin ajustar (comparacion) | `~/deepspine_runs/embeddings_resnet50_*.npz` (~290 MB, fuera del repo). Tarda ~35 s |
| `04_svm_random_forest.py` | StandardScaler y PCA a 256 componentes, ajustados con train. Luego prueba en validacion SVM RBF (C = 0,1 / 0,5 / 2 / 8), Random Forest de 300 arboles (profundidad 10 / 20 / sin limite), regresion logistica (C = 0,01 / 0,1 / 1) y KNN (K = 5 / 15 / 31), y se queda con el mejor de cada uno | `docs/resultados/modelo3_svm/`, `modelo4_random_forest/`, `extra_*/` (o `ablacion_imagenet_*/`), y `models/svm.joblib` y `models/random_forest.joblib`, que guardan dentro el escalador y el PCA. Elegidos: SVM C = 0,1; Random Forest profundidad 20 |
| `05_comparar_modelos.py` | Lee los puntajes de cada modelo, calcula las mismas metricas para todos, agrega un ensamble de los 4 como extra y elige el mejor por F1 macro en validacion | `docs/resultados/comparacion/`: tablas por metrica, recall por clase, matrices de confusion lado a lado, ROC y barras. Mejor por validacion: ResNet-50 |
| `06_figuras_modelo_matematico.py` | Genera los graficos del modelo matematico a partir de los modelos entrenados: softmax, funcion de costo, tasa de aprendizaje, curvas, PCA, SVM, Random Forest y busqueda de hiperparametros | `docs/modelo_matematico/figuras/` y `valores.json` (cada cifra citada en el documento) |
| `07_figura_estructura.py` | Dibuja el diagrama de la estructura del sistema: los dos caminos, la parte hibrida y el ensamble. No necesita el dataset ni los modelos | `docs/modelo_matematico/figuras/09_estructura_sistema.png` |

### Codigo comun (`src/`)

| Archivo | Para que sirve |
|---|---|
| `src/utils/config.py` | Rutas (dataset, corridas, resultados, pesos), semilla, orden de las clases, normalizacion de ImageNet y `PIN_MEMORY = False` (en WSL2 la memoria fijada provoca errores de CUDA) |
| `src/data/dataset.py` | Construye el indice de las 35.007 imagenes (particion, clase, paciente, nivel, original o copia), verifica que no haya fuga y define los conjuntos de evaluacion |
| `src/models/resnet.py` | ResNet-50: construccion, congelado de capas, cargadores de imagenes con y sin aumento, prediccion, carga del modelo 2 y extractores de embeddings |
| `src/training/evaluacion.py` | Cierre comun de cada modelo: metricas, reportes, matrices, curvas y `metadata.json` con la configuracion y las versiones |
| `src/utils/metricas.py`, `src/utils/graficas.py` | Metricas iguales para todos los modelos (exactitud, precision, recall, F1, exactitud balanceada, ROC-AUC, log loss) y graficas |
| `src/inference/predictor.py` | `Predictor` para la aplicacion web: `predecir(imagen, modelo)` y `predecir_todos(imagen)`. Acepta ruta, bytes, PIL o numpy. Devuelve clase, confianza y puntaje por clase, y oculta embeddings, escalador y PCA. Se verifico contra la evaluacion en 20 imagenes de test por modelo |

### Donde quedo lo de los modelos entrenados

| Que | Donde | En el repo |
|---|---|---|
| Metricas, reportes por clase y por nivel, matrices de confusion, curvas, configuracion y puntajes de las 35.007 imagenes de cada modelo | `docs/resultados/<modelo>/` | Si |
| Comparacion de los 4 modelos (tablas y figuras) | `docs/resultados/comparacion/` | Si |
| Registro de los experimentos de ajuste en validacion | `docs/resultados/experimentos_validacion.md` | Si |
| Modelo matematico (ecuaciones, graficos y valores) | `docs/modelo_matematico/` | Si |
| Codigo de entrenamiento, evaluacion e inferencia | `scripts/entrenamiento/` y `src/` | Si |
| Descripcion y hashes SHA-256 de los pesos | `models/README.md` | Si |
| **Pesos de los 4 modelos** (`yolov8s_cls.pt`, `resnet50.pt`, `svm.joblib`, `random_forest.joblib`, ~121 MB en total) | **GitHub Release** [`modelos-v1.0`](https://github.com/JulianSilvaMorales/Vision_Artificial/releases/tag/modelos-v1.0); se descargan en `models/` con `python scripts/descargar_modelos.py` | No: van en la Release para no gastar la cuota de Git LFS |
| Embeddings, checkpoints por epoca y experimentos descartados | `~/deepspine_runs/` de quien entrena | No: se regeneran con los scripts |
| Imagenes del dataset de 35k (`DeepSpine_Dataset_35k.zip`) | [Google Drive](https://drive.google.com/drive/folders/1ZbTE1gb6zv7F2qEBmwRXzHFveLyRrl4I?usp=drive_link) / `scripts/build_dataset.py` | No |

### Proceso seguido
1. **Entorno:** el entrenamiento se hizo en Ubuntu 24.04 (WSL2) con la GPU local, en lugar de Colab, para evitar desconexiones. Se uso PyTorch porque YOLOv8 (ultralytics) solo existe para PyTorch.
2. **Problema resuelto:** en WSL2 los DataLoader fallaban con `CUDA error: out of memory` al reservar memoria fijada. Se desactivo `pin_memory`.
3. **Primera version:** YOLO y ResNet-50 quedaron en 87,8 % de exactitud sobre los originales de test. SVM y Random Forest sobre embeddings de ImageNet sin ajustar quedaron en 80 % y 75 %.
4. **Ajuste con validacion** para intentar llegar al 89 %: varias semillas, regularizacion de la ResNet (suavizado de etiquetas y aumento en linea), ensambles y correccion por la proporcion de clases. Se adopto la ResNet regularizada (mejor F1 en validacion). SVM y Random Forest pasaron a usar las caracteristicas de esa ResNet; su F1 en validacion subio de 0,65 a 0,80 (SVM) y de 0,60 a 0,79 (Random Forest). La correccion por proporcion se descarto porque hundia la deteccion de Degeneracion.
5. **Evaluacion final:** los 4 modelos se evaluaron una sola vez en test, con las metricas de la tabla de [Resultados](#resultados).

## Aplicacion web

Aplicacion local para el radiologo: carga un estudio de resonancia (`.IMA`, `.dcm`, carpeta o `.zip`), recomienda los cortes centrales, detecta los 5 discos L1-S1, los clasifica con el ensamble de los 4 modelos y prepara un informe imprimible con la valoracion del radiologo. Es un backend FastAPI con una interfaz en HTML, CSS y JavaScript sin compilacion. Funciona en Windows, Linux y macOS, sin GPU ni dataset, y sin internet despues de la primera instalacion.

**Arranque en un comando** (requisitos: Git, Git LFS y Python; desde la raiz del repo clonado):

| Sistema | Comando |
|---|---|
| Windows | `webapp\iniciar_app.bat` (o doble clic en el archivo) |
| Linux / macOS / WSL | `bash webapp/iniciar_app.sh` |

El script crea el entorno `.venv-app/`, instala las dependencias (solo la primera vez), trae el detector de discos (Git LFS) y los 4 pesos del Release `modelos-v1.0` con verificacion SHA-256, y levanta el servidor en **http://localhost:8000**.

**Guia completa** (flujo, privacidad, requisitos por sistema, paso a paso manual, problemas frecuentes, rechazo de imagenes y API): [`webapp/README.md`](webapp/README.md). Las metricas de los modelos estan en [Resultados](#resultados) y en `docs/resultados/`.

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

La app usa los cuatro modelos entrenados; los modelos SVM y Random Forest requieren primero calcular los embeddings de la ResNet-50 ajustada (modelo 2) y aplicar el escalador y el PCA guardados junto al modelo. Todo eso lo hace `src/inference/predictor.py`, que la capa `services/` llama directamente.

### 3. Frontend: modulos de JavaScript sin compilacion

`webapp/frontend/clinica/` usa HTML, CSS y modulos ES, sin dependencias ni compilacion, para arrancar con un solo comando y funcionar sin internet. Cada paso de la aplicacion tiene su modulo en `js/`, y las llamadas a la API estan centralizadas en `js/api.js`.

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
│   ├── calibrar_rechazo.py         # Calibra el umbral con el que la app rechaza imagenes atipicas
│   ├── descargar_modelos.py        # Descarga los 4 pesos del GitHub Release y verifica su SHA-256
│   └── build_dataset.py            # Dataset final de 35.007 imagenes
│
├── models/                     # Pesos del pipeline de datos; los 4 modelos finales se descargan del GitHub Release
│
├── notebooks/                  # Notebooks (Colab): preetiquetado, entrenamiento de los 4 modelos y comparacion
│
├── webapp/                     # Aplicacion final (ver webapp/README.md)
│   ├── backend/                 # API FastAPI que sirve los 4 modelos (arquitectura en capas)
│   │   ├── main.py                # App; sirve la API y la aplicacion
│   │   ├── api/routes/            # Endpoints (salud, predecir, estudios)
│   │   ├── core/                   # Configuracion, variables de entorno
│   │   ├── services/                # Logica de negocio, orquesta la prediccion
│   │   ├── schemas/                  # Validacion de datos de entrada/salida
│   │   └── utils/                     # Helpers propios del backend
│   │
│   └── frontend/
│       └── clinica/               # Interfaz para el radiologo (HTML, CSS y JS sin compilacion)
│
├── tests/                       # Pruebas del backend de la aplicacion
├── docs/                        # Guias, resultados (docs/resultados/), modelo matematico (docs/modelo_matematico/) y referencias
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
| `scripts/calibrar_rechazo.py`, `scripts/descargar_modelos.py` | En el repo | Calibracion del rechazo de imagenes atipicas y descarga verificada de los pesos del Release |
| `src/` | En el repo | Codigo comun: indice del dataset y verificacion de fuga (`data/`), ResNet-50 (`models/`), evaluacion (`training/`), metricas y graficas (`utils/`) y prediccion para la app (`inference/predictor.py`) |

Los scripts de `scripts/` se conservan tal como se usaron durante el trabajo; las rutas de entrada y salida se configuran dentro de cada uno (ver `if __name__ == "__main__"` o sus argumentos).

Nota: esta seccion se debe actualizar a medida que el proyecto avance, para que siempre refleje la organizacion real del repositorio.

## Git LFS

Este proyecto usa Git LFS para versionar archivos pesados (imagenes, CSV, checkpoints de modelos). Los tipos de archivo rastreados por LFS estan definidos en `.gitattributes`. Cualquiera que clone el repo debe tener Git LFS instalado (`git lfs install`) antes de descargar los archivos de `data/` o `models/`.

El dataset de 35.007 imagenes no se sube al repo, para no agotar la cuota de LFS: se distribuye como `DeepSpine_Dataset_35k.zip` en [Google Drive](https://drive.google.com/drive/folders/1ZbTE1gb6zv7F2qEBmwRXzHFveLyRrl4I?usp=drive_link). `data/processed/` guarda las etiquetas y manifiestos, y `scripts/build_dataset.py` regenera las imagenes. Los pesos de los 4 modelos tampoco van en LFS: estan en el GitHub Release `modelos-v1.0`.

## Cambios en archivos de configuracion

- `.gitignore`: se agregaron dos excepciones a la regla `models/*.pt` para poder versionar los pesos del detector de discos y del modelo de preetiquetado:
  ```
  !models/best_disc_detector.pt
  !models/preetiquetado_semilla_v2.pt
  ```
  El resto de los `.pt` y `.h5` de `models/` sigue ignorado.
- `.gitignore` (entrenamiento y aplicacion): se ignoran el dataset descomprimido (`data/processed/train/`, `val/`, `test/` y su zip), `notebooks/temporal/`, los `models/*.joblib` y `models/*.npz` (los pesos van en el GitHub Release) y los entornos `.venv-app/` (scripts de arranque) y `.venv-mvp/` (versiones anteriores).
- `.gitattributes`: se fijaron los finales de linea de los scripts de arranque (`*.sh` con LF y `*.bat` con CRLF), para que funcionen al clonar en cualquier sistema. Las reglas de Git LFS no cambiaron.

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

**Entrenar los 4 modelos** (Linux o WSL2 con GPU NVIDIA; el dataset [`DeepSpine_Dataset_35k.zip` de Google Drive](https://drive.google.com/drive/folders/1ZbTE1gb6zv7F2qEBmwRXzHFveLyRrl4I?usp=drive_link) descomprimido en `~/deepspine_data/` con `train/`, `val/` y `test/`, o en la ruta de la variable `DEEPSPINE_DATA`):

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

**Usar los modelos entrenados** (primero `python scripts/descargar_modelos.py`, que baja los 4 pesos del Release a `models/`):

```python
from src.inference.predictor import Predictor
pred = Predictor()
pred.predecir("roi.png", modelo="resnet50")   # "yolov8s_cls", "resnet50", "svm" o "random_forest"
pred.predecir_todos("roi.png")                # los 4 modelos sobre la misma imagen
```

**Notebooks:** los cuadernos 00 y 01 se ejecutan en Google Colab con GPU. Usan Roboflow: la clave se guarda como secreto de Colab con el nombre `ROBOFLOW_API_KEY`; no se escribe en el cuaderno. Las variables a ajustar (rutas, workspace y proyecto) estan en las primeras celdas.

**Backend y frontend:** ver [Aplicacion web](#aplicacion-web) y [`webapp/README.md`](webapp/README.md).

## Flujo de trabajo con Git

- Trabajamos con ramas por funcionalidad: `feature/<integrante>-<tarea>`, `fix/<integrante>-<tarea>`.
- Los commits siguen la convencion Conventional Commits (ver `CONTRIBUTING.md`).
- Antes de hacer merge a `main`, se revisa el codigo en Pull Request.
