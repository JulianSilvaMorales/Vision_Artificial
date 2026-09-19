# Guia de contribucion

Este documento explica como trabajar en el repositorio: convencion de commits, manejo de ramas y buenas practicas. Es lo que la profesora pidio para poder rastrear el aporte individual de cada integrante.

## Conventional Commits

Todos los commits deben seguir este formato:

```
<tipo>(<alcance opcional>): <descripcion corta en presente>

<cuerpo opcional explicando el "por que" del cambio>
```

### Tipos mas usados

| Tipo | Cuando usarlo |
|---|---|
| `feat` | Nueva funcionalidad (ej: nuevo endpoint, nueva pantalla, nuevo script) |
| `fix` | Correccion de un error |
| `docs` | Cambios solo en documentacion (README, comentarios, etc.) |
| `style` | Cambios de formato que no afectan la logica (espacios, indentacion) |
| `refactor` | Cambios en el codigo que no agregan funcionalidad ni corrigen bugs |
| `perf` | Cambios que mejoran el rendimiento |
| `test` | Agregar o corregir pruebas |
| `chore` | Tareas de mantenimiento (dependencias, configuracion, estructura de carpetas) |
| `data` | Cambios relacionados con datasets (agregar, limpiar, versionar datos) |

### Alcances sugeridos (segun la estructura del repo)

Usa el nombre de la carpeta principal donde trabajaste como alcance, para que quede claro que parte del proyecto se toco:

`data`, `src`, `scripts`, `models`, `notebooks`, `backend`, `frontend`, `tests`, `docs`

### Ejemplos aplicados a este proyecto

```
feat(models): agregar arquitectura CNN base para clasificacion
fix(src/data): corregir normalizacion incorrecta de imagenes
docs(readme): actualizar estructura de carpetas
chore(deps): agregar opencv-python a requirements.txt
feat(backend): crear endpoint POST /predict
feat(frontend): agregar componente UploadImage
style(frontend): ajustar formato del componente ResultCard
test(src/training): agregar prueba para funcion de carga de dataset
data(raw): agregar nuevas imagenes al dataset de entrenamiento
chore: configurar Git LFS y estructura de carpeta data
```

### Reglas rapidas

- Usa el presente del indicativo: "agregar", no "agregado" ni "agregando".
- Un commit equivale a un cambio logico. Evita commits gigantes que mezclen varias cosas (ej: no mezclar cambios de backend con cambios de frontend en el mismo commit).
- El alcance (lo que va entre parentesis) ayuda a saber en que parte del proyecto se trabajo.
- Commits claros y frecuentes permiten que la profesora vea el aporte individual de cada integrante en el historial de GitHub.

## Ramas

- `main`: version estable del proyecto, siempre debe funcionar.
- `feature/<nombre>`: para nuevas funcionalidades (ej: `feature/endpoint-predict`, `feature/upload-component`).
- `fix/<nombre>`: para correccion de errores.

Flujo sugerido:

```bash
git checkout -b feature/nombre-de-la-tarea
# trabajar y hacer commits siguiendo la convencion
git push origin feature/nombre-de-la-tarea
# abrir un Pull Request hacia main
```

## Pull Requests

- Describe brevemente que hace el PR y por que.
- Menciona que carpetas o partes del proyecto se modificaron.
- Idealmente que otro integrante revise antes de mergear a `main`, para detectar errores y repartir el trabajo de revision.

## Sobre Git LFS

Este repo usa Git LFS para el dataset y los modelos entrenados (ver README). Antes de hacer `git add` de archivos dentro de `data/` o `models/`, asegurate de tener Git LFS instalado (`git lfs install`) para que no se suban como archivos normales pesados.

---

Nota: esta guia puede cambiar a medida que el equipo defina mas convenciones (ej: formato de Pull Request, revisiones obligatorias, etc.).