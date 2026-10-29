# models

Solo se versionan (con Git LFS) los pesos pequenos que forman parte del pipeline de preparacion de datos. Los pesos de los cuatro modelos finales se agregaran cuando termine su entrenamiento.

| Archivo | Que es | Quien lo usa |
|---|---|---|
| `best_disc_detector.pt` | YOLO de deteccion de discos intervertebrales (cajas) sobre cortes sagitales | `scripts/recorte_roi/autocrop_rois.py` |
| `preetiquetado_semilla_v2.pt` | YOLOv8s-cls de 3 clases (Normal, Degeneracion_Discal, Protrusion) entrenado con la semilla etiquetada a mano, usado para preetiquetar los ROI antes de la revision humana | cuaderno `notebooks/01_preetiquetado_semilla_v2.ipynb` |

Notas:
- El cuaderno `notebooks/00_entrenamiento_detector_discos.ipynb` entrena una variante OBB (cajas orientadas) y la guarda como `best_disc_detector_obb.pt`; el archivo versionado aqui es el detector de cajas que usa `autocrop_rois.py`.
- `.gitignore` ignora `models/*.pt` y `models/*.h5`; solo estos dos archivos tienen excepcion explicita.
