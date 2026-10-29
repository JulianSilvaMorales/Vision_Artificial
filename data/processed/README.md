# data/processed

Archivos pequenos (versionados con Git LFS por `.gitattributes`) que documentan el etiquetado y la construccion del dataset de 35.007 imagenes. Las imagenes no estan en el repositorio: se regeneran con `scripts/build_dataset.py` a partir de los ROI y de `etiquetado/etiquetas_revisadas.csv`.

## `etiquetado/`
- `etiquetas_revisadas.csv`: una fila por ROI original elegible. Columnas: `archivo_original`, `nivel`, `paciente`, `revisor` (A o B), `clase_predicha` y `confianza` (modelo de preetiquetado), `clase_final` (decision humana tras revisar) y `corregida` (si la etiqueta humana difiere de la predicha).

## `semilla/`
- `preetiquetas_20261023_0414.csv`: salida del modelo de preetiquetado (clase predicha, confianza, segunda clase, margen y si se marco para revision humana).
- `meta_dicom_por_slice.csv`: identificadores de estudio y serie por corte (sin datos personales: `pid`, `birth` y `sdate` vienen vacios).
- `split_pacientes.csv`: reparto por paciente usado para entrenar el modelo semilla (separador `;`).

## `dataset_35k/`
- `manifest_35k.csv`: una fila por imagen del dataset (35.007). Columnas: `archivo`, `es_original`, `particion`, `clase`, `paciente`, `nivel`, `original`, `revisor` y, para las copias aumentadas, los parametros usados (`semilla`, `angulo`, `zoom`, `tx`, `ty`, `contraste`, `brillo`).
- `split_pacientes.csv`: particion (`train`, `val`, `test`) asignada a cada paciente; ningun paciente aparece en dos particiones.
