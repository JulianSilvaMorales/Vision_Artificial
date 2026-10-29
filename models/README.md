# models

Solo se versionan (con Git LFS) los pesos pequenos que forman parte del pipeline de preparacion de datos. Los pesos de los cuatro modelos finales se publican en un **GitHub Release** del repositorio; para usarlos, se descargan en esta carpeta con los mismos nombres.

## Pipeline de datos (en el repo)

| Archivo | Que es | Quien lo usa |
|---|---|---|
| `best_disc_detector.pt` | YOLO de deteccion de discos intervertebrales (cajas) sobre cortes sagitales | `scripts/recorte_roi/autocrop_rois.py` |
| `preetiquetado_semilla_v2.pt` | YOLOv8s-cls de 3 clases (Normal, Degeneracion_Discal, Protrusion) entrenado con la semilla etiquetada a mano, usado para preetiquetar los ROI antes de la revision humana | cuaderno `notebooks/01_preetiquetado_semilla_v2.ipynb` |

## Modelos finales (en el GitHub Release)

| Archivo | Modelo | Tamano | SHA-256 |
|---|---|---|---|
| `yolov8s_cls.pt` | 1 · YOLOv8s-cls (ultralytics) | 10,3 MB | `4993c3f972eccea9d1df3841c545b141735cf319639b8213bf5150f562e380e4` |
| `resnet50.pt` | 2 · ResNet-50 ajustada (`state_dict`, clases y dropout) | 94,4 MB | `7ed202884b3b606a62314888e7720ef660ff8079bf443f458aff0e1e69e4719c` |
| `svm.joblib` | 3 · SVM (RBF) + StandardScaler + PCA(256) | 4,0 MB | `57beb318fb4f40edebef9ae00c471d029ddc72b447dce11c9091c7fa23fc6727` |
| `random_forest.joblib` | 4 · Random Forest + StandardScaler + PCA(256) | 12,0 MB | `b3b6167d85a44ea860edb4e7dced0837bf9678aa1f7ab708aa845fdcdfb30933` |

- El SVM y el Random Forest **necesitan `resnet50.pt`**: sus entradas son los embeddings (2048 valores) de la ResNet-50 del modelo 2 sin su capa final.
- Los `.joblib` se crearon con scikit-learn 1.9.1 y los `.pt` con PyTorch 2.15 y ultralytics 8.4. Otra version de scikit-learn puede no cargarlos.
- Se cargan con `src/inference/predictor.py` (`Predictor().predecir(imagen, modelo=...)`), que hace el preprocesamiento.
- Metricas y configuracion de cada modelo: `docs/resultados/`.

Comprobar un archivo descargado (Linux): `sha256sum models/*.pt models/*.joblib`.

## Notas
- El cuaderno `notebooks/00_entrenamiento_detector_discos.ipynb` entrena una variante OBB (cajas orientadas) y la guarda como `best_disc_detector_obb.pt`; el archivo versionado aqui es el detector de cajas que usa `autocrop_rois.py`.
- `.gitignore` ignora `models/*.pt`, `models/*.h5` y `models/*.joblib`; solo los dos pesos del pipeline de datos tienen excepcion explicita.
