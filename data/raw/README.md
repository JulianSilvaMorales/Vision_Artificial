# data/raw

Aqui **no se sube** el conjunto original. Se descarga del repositorio publico:

- Lumbar Spine MRI Dataset (Sudirman et al., Mendeley Data): https://data.mendeley.com/datasets/k57fr854j2/2

Estructura local usada por los scripts de `scripts/preprocesamiento/` (rutas relativas a la carpeta de trabajo; no versionadas):

| Carpeta local | Contenido |
|---|---|
| `t2_sagital_unfiltered/` | Conjunto completo descargado (cortes DICOM `.IMA` por paciente) |
| `t2_sagital_filtered/` | Solo las series T2 sagitales, tras `ingestion.py` |

El resto de carpetas intermedias (PNG convertidos, cortes medios, recortes de ROI) se regenera con los scripts y tampoco se versiona.
