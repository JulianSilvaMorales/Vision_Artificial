# webapp · DeepSpine

Aplicación final para el radiólogo. Se ejecuta en local, en **Windows, Linux o macOS**, sin GPU: FastAPI sirve la API de los 4 modelos y la interfaz de `frontend/clinica/` (HTML, CSS y JavaScript, sin compilación ni dependencias). Se eligió esta tecnología en lugar de React porque arranca con un solo comando y funciona sin internet después de la primera instalación.

> Herramienta académica, no diagnóstica. Las etiquetas de entrenamiento las asignaron estudiantes, sin radiólogo.

Las métricas de los modelos no se muestran en la aplicación: están en [`docs/resultados/`](../docs/resultados/README.md).

## Flujo de la aplicación

Cuatro pasos, en tema oscuro (se cambia a claro con el botón de la barra; se recuerda en ese navegador):

1. **Cargar:** arrastra los `.IMA` / `.dcm`, la carpeta exportada del resonador o un `.zip`. Lo que no es un corte (`DICOMDIR`, visores, imágenes) se descarta antes de subir. La subida empieza al elegir o soltar los archivos.
2. **Cortes:** visor con brillo y contraste (arrastrar), zoom, cajas del detector (B) y tira de miniaturas. El detector recomienda los 3 cortes más centrales (★) y quedan elegidos; el radiólogo puede cambiarlos.
3. **Resultados:** conclusión en lenguaje natural y una tarjeta por disco con la clase del ensamble, en cuántos cortes se repite, cuántos votos de los modelos coinciden y una confianza orientativa. En cada disco el radiólogo marca *Concuerdo* o *Discrepo* (con la clase que ve y una nota). El detalle por modelo queda plegado.
4. **Informe:** hallazgos, impresión, recomendación y nombre del radiólogo, con frases rápidas. *Finalizar informe* bloquea el texto y agrega la firma; *Editar* la quita. Se imprime o guarda como PDF (A4) desde el navegador.

El atajo **Analizar una imagen** (barra superior) clasifica un recorte de disco o un corte completo suelto.

La recomendación de cortes y la confianza son reglas fijas de apoyo, sin validación clínica ni calibración.

## Privacidad

- Los DICOM se borran apenas se convierten a PNG. Nombre, fecha de nacimiento e identificadores nunca se devuelven ni se registran.
- Los PNG quedan en una carpeta temporal del equipo (`DEEPSPINE_ESTUDIOS`). Un estudio se borra tras **30 minutos sin usarse**: una limpieza revisa cada minuto mientras el servidor esté activo. También se borran todos al detener el servidor con `Ctrl + C`.
- Si la ventana del servidor se cierra de golpe o el equipo se apaga, la limpieza no alcanza a correr: lo pendiente se borra al arrancar de nuevo si ya venció.
- Las observaciones del radiólogo no se envían al servidor: quedan como borrador en ese navegador y en el PDF que se imprima. No hay que escribir datos que identifiquen al paciente. Si se recarga la página, el estudio, el análisis y el borrador se recuperan mientras el estudio siga en el servidor.

## Guía rápida de arranque

### 1. Requisitos (una sola vez por equipo)

| Programa | Para qué | Cómo comprobarlo |
|---|---|---|
| **Git** | Clonar el repositorio | `git --version` |
| **Git LFS** | Descargar el detector de discos (`models/best_disc_detector.pt`) | `git lfs version` |
| **Python 3.12 a 3.14** | Ejecutar el backend y los modelos | `python --version` (Windows) o `python3 --version` |
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
| **Windows** | `webapp\iniciar_app.bat`, o doble clic en `webapp/iniciar_app.bat` |
| **Linux / macOS / WSL** | `bash webapp/iniciar_app.sh` |

El script hace todo en orden:
1. Crea un entorno virtual `.venv-app/` en la raíz del repo (solo la primera vez).
2. Instala `webapp/backend/requirements.txt` (solo la primera vez o si cambia; tarda varios minutos).
3. Ejecuta `git lfs pull` para traer el detector de discos, y `scripts/descargar_modelos.py` para bajar los 4 pesos a `models/` y verificar su SHA-256. Si ya están, no los vuelve a descargar.
4. Levanta el servidor, que primero carga los modelos (unos segundos), y en Windows abre el navegador en **http://127.0.0.1:8000** cuando ya está listo. Es lo mismo que `http://localhost:8000`.

Las siguientes veces solo hace los pasos 3 y 4, en unos segundos y sin internet. Para detener el servidor: `Ctrl + C` en la terminal.

### 4. Paso a paso manual (si prefieres no usar el script)

Windows (PowerShell o CMD):
```
python -m venv .venv-app
.venv-app\Scripts\activate
pip install -r webapp\backend\requirements.txt
git lfs pull
python scripts\descargar_modelos.py
python -m uvicorn webapp.backend.main:app --port 8000
```

Linux / macOS / WSL:
```
python3 -m venv .venv-app
source .venv-app/bin/activate
pip install -r webapp/backend/requirements.txt
git lfs pull
python scripts/descargar_modelos.py
python -m uvicorn webapp.backend.main:app --port 8000
```

Abrir **http://127.0.0.1:8000** cuando la terminal diga `Application startup complete`. La documentación interactiva de la API está en **http://127.0.0.1:8000/docs**.

### 5. Comprobar que todo está bien

La barra superior debe decir **"Sistema listo · CPU"** con un punto verde. Si está en rojo, dice qué falta. También se puede consultar `http://localhost:8000/api/salud`.

### Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| Barra roja: *"Faltan modelos"* | No se descargaron los pesos | `python scripts/descargar_modelos.py` (necesita internet) |
| Barra roja: *"Falta el detector de discos"* | El detector llegó como puntero de Git LFS | `git lfs install` y `git lfs pull` |
| `ERROR: SHA-256 no coincide` al descargar | Descarga corrupta o incompleta | Volver a ejecutar el script; borra el archivo malo solo |
| Error al cargar `svm.joblib` o `random_forest.joblib` | Otra versión de scikit-learn | Debe ser la 1.9.1: `pip install scikit-learn==1.9.1` |
| `[Errno 98]` / `10048`: el puerto está en uso | Ya hay algo en el 8000 | Usar otro puerto: `... uvicorn webapp.backend.main:app --port 8001` |
| `ModuleNotFoundError: No module named 'webapp'` o `'src'` | Se ejecutó desde otra carpeta | Ejecutar desde la **raíz** del repo |
| El navegador dice *"localhost rechazó la conexión"* | Se abrió antes de que el servidor terminara de cargar los modelos | Esperar a que la terminal diga `Application startup complete` y recargar |
| *"No se pudo convertir ningún corte a PNG"* en Windows | `cv2.imwrite` no escribe en rutas con tildes o símbolos, y la carpeta temporal del usuario las tiene | Definir `DEEPSPINE_ESTUDIOS` con una ruta simple, por ejemplo `set DEEPSPINE_ESTUDIOS=C:\deepspine_tmp` |
| El detector halla pocos discos | Los cortes no son midsagitales, o el equipo es muy distinto al de entrenamiento | Elegir cortes del centro del volumen; ver el aviso de equipo en el paso Cortes |
| Windows: `OSError: [Errno 2] No such file or directory: ...torch\include\...` al instalar | El repo está en una ruta muy larga y Windows limita las rutas a 260 caracteres | Clonar en una ruta corta o activar las rutas largas de Windows |
| Usas el mismo clon desde Windows y desde WSL | El `.venv-app` de un sistema no sirve en el otro | En WSL: `DEEPSPINE_VENV=.venv-app-linux bash webapp/iniciar_app.sh` |

Opcional: con GPU NVIDIA y PyTorch con CUDA, `DEEPSPINE_DISPOSITIVO=cuda` usa la GPU. En CPU (por defecto) las predicciones reproducen exactamente la evaluación oficial; en GPU la ResNet usa otra precisión numérica y algún caso límite puede cambiar.

## Rechazo de imágenes que no son de columna

Los 4 modelos siempre eligen una de las 3 clases, así que antes de predecir se filtran las imágenes (`services/validacion_service.py`):

1. **Formato:** se rechazan las imágenes en color, diminutas o con proporciones raras (`422`, `capa 1`).
2. **Tipo:** casi cuadrada y de hasta 256 px = recorte de disco; más grande o alargada = corte completo, que va al detector y solo se acepta si halla al menos 3 niveles en orden (`capa 2`).
3. **Fuera de distribución:** distancia de Mahalanobis del embedding de la ResNet-50 en el espacio PCA del SVM, contra un umbral calibrado.

**La capa 3 todavía no está activa:** solo actúa si existe `models/umbral_rechazo.json`, y la aplicación avisa en Resultados mientras falte. Se genera una vez con los ROI de validación del dataset (ver `data/processed/README.md`):

```
python scripts/calibrar_rechazo.py --rois <carpeta con ROI de validacion> --negativas <carpeta con unas 30 imagenes que no son discos>
```

Si cambia `svm.joblib`, hay que recalibrar: la API lo marca como `umbral_obsoleto`.

## API

Arquitectura en capas: `api/routes/` (HTTP) → `schemas/` (validación con Pydantic) → `services/` (lógica) → `src/inference/predictor.py` (modelos). Las rutas nunca llaman directamente a los modelos.

| Método y ruta | Qué hace |
|---|---|
| `GET /api/salud` | Estado: modelos disponibles o faltantes, detector, dispositivo y commit |
| `POST /api/predecir` | Recorte de un disco (`multipart/form-data`, campo `imagen`, máx. 5 MB): predicciones, traza y ensamble. `422` con `{motivo, capa, valor, umbral}` si no parece un ROI; `motivo = corte_completo` indica que hay que usar `/api/predecir-corte` |
| `POST /api/predecir-corte` | Corte sagital completo (PNG/JPG): detecta los 5 discos y los clasifica; `422` si el detector no halla al menos 3 discos en orden |
| `POST /api/estudios` | Sube `.IMA` / `.dcm` / `.zip` (campo `archivos`, varios; máx. 500 archivos y 300 MB). Convierte a PNG y devuelve las series, sus cortes y la propuesta de cortes centrales |
| `GET /api/estudios/{id}` · `GET .../series/{serie}/cortes/{n}/imagen` | Ficha del estudio y PNG de un corte |
| `POST /api/estudios/{id}/puntuar` | Ordena los cortes según lo bien que el detector ve los 5 discos y recomienda los 3 mejores |
| `POST /api/estudios/{id}/recortar` | `{serie, cortes[], recortar_zona_lumbar}`: detector y recorte de los 5 ROI de cada corte, sin clasificar |
| `POST /api/estudios/{id}/predecir` | Lo mismo y además los 4 modelos sobre cada ROI |
| `POST /api/estudios/{id}/png.zip` · `DELETE /api/estudios/{id}` | Descarga los PNG elegidos / borra el estudio |

## Archivos

```
webapp/
├── README.md                      # esta guia
├── iniciar_app.bat                # arranque en Windows (un comando)
├── iniciar_app.sh                 # arranque en Linux / macOS / WSL
├── backend/
│   ├── main.py                    # app FastAPI; sirve la API y la aplicacion
│   ├── requirements.txt           # dependencias minimas para ejecutar la aplicacion
│   ├── api/routes/                # sistema.py, prediccion.py, estudios.py
│   ├── schemas/prediccion.py      # modelos de entrada y salida
│   ├── services/                  # modelos_service, prediccion_service, estudios_service, validacion_service
│   ├── core/config.py             # rutas, limites y dispositivo
│   └── utils/                     # imagenes.py, scripts.py
└── frontend/
    └── clinica/                   # index.html, css/ y js/ (modulos ES, sin dependencias ni compilacion)
```

Pruebas, desde la raíz del repo: `python -m pytest tests -q` (requiere `pytest` y `httpx`, que no forman parte de `requirements.txt`).

**Reglas que no cambian:** no reentrenar ni cambiar los pesos sin publicar un nuevo Release con sus hashes, y mantener la CPU por defecto para reproducir la evaluación oficial.
