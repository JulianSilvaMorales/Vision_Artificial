# Vision_Artificial

Proyecto para la materia de Vision Artificial. Incluye entrenamiento de modelos de IA, scripts de procesamiento en Python y una aplicacion web (backend + frontend) para exponer y usar el modelo.

## Descripcion

(Agregar aqui una descripcion corta del problema que resuelve el proyecto, el dataset usado y el objetivo general.)

## Integrantes

| Nombre | Rol / Aporte |
|---|---|
| | |
| | |

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
├── data/                     # Datos del proyecto (versionados con Git LFS)
│   ├── raw/                   # Datos originales, sin modificar
│   ├── processed/              # Dataset limpio final (~32.000 registros)
│   └── external/                # Datasets de terceros / descargas externas
│
├── src/                        # Codigo fuente reutilizable (paquete Python)
│   ├── data/                    # Carga y preprocesamiento de datos
│   ├── models/                   # Definicion de arquitecturas del modelo (CNN, etc.)
│   ├── training/                  # Logica de entrenamiento
│   ├── inference/                  # Prediccion usando el modelo ya entrenado
│   └── utils/                       # Funciones auxiliares (config, logging, graficos)
│
├── scripts/                    # Scripts ejecutables (train.py, evaluate.py, preprocess.py...)
│
├── models/                     # Checkpoints/pesos entrenados (versionados con Git LFS)
│
├── notebooks/                  # Exploracion de datos y pruebas rapidas (Jupyter)
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
├── docs/                        # Documentacion adicional, diagramas, informes
└── assets/                      # Imagenes/recursos usados en el README
```

Nota: esta seccion se debe actualizar a medida que el proyecto avance, para que siempre refleje la organizacion real del repositorio.

## Git LFS

Este proyecto usa Git LFS para versionar archivos pesados (dataset limpio de aproximadamente 32.000 registros, imagenes, checkpoints de modelos). Los tipos de archivo rastreados por LFS estan definidos en `.gitattributes`. Cualquiera que clone el repo debe tener Git LFS instalado antes de descargar los archivos de `data/` o `models/`.

## Instalacion

(Seccion pendiente. Aqui debe quedar documentado, paso a paso, como dejar el proyecto listo para correr en una maquina nueva: creacion de entorno virtual de Python, instalacion de dependencias desde `requirements.txt`, e instalacion de dependencias del frontend. Se completa cuando el entorno del proyecto este definido.)

## Uso

(Seccion pendiente. Aqui debe quedar documentado como ejecutar cada parte del proyecto una vez este implementada: como correr el entrenamiento del modelo, como levantar el backend y como levantar el frontend. Se completa a medida que cada parte quede funcional.)

## Flujo de trabajo con Git

- Trabajamos con ramas por funcionalidad: `feature/nombre-corto`, `fix/nombre-corto`.
- Los commits siguen la convencion Conventional Commits (ver `CONTRIBUTING.md`).
- Antes de hacer merge a `main`, se revisa el codigo en Pull Request.