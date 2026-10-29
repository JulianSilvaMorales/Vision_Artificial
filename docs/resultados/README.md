# docs/resultados

Resultados del entrenamiento y la evaluación de los 4 modelos (Hito 2). Los genera `scripts/entrenamiento/` y se pueden regenerar con los mismos comandos (semilla 42).

## Cómo se evaluó
- **Mismo dataset para todos:** 35.007 imágenes, división 70/15/15 **por paciente** (279/60/60). Cada script verifica que ningún paciente esté en dos particiones y que el dataset coincida con `data/processed/dataset_35k/manifest_35k.csv`.
- **Selección con validación:** épocas e hiperparámetros se eligieron por F1 macro en validación completa. El test se evaluó una sola vez, con la configuración final. Los experimentos de ajuste están en [experimentos_validacion.md](experimentos_validacion.md).
- **Dos juegos de cifras:**
  - `test_orig`: las 768 imágenes originales de test, en proporción natural (42 Degeneración, 164 Normal, 562 Protrusión).
  - `test_completo`: las 5.253 imágenes de test, balanceadas, con copias aumentadas.
- **Métricas:** exactitud, precisión, recall y F1 (macro), exactitud balanceada, ROC-AUC macro (uno contra el resto), pérdida logarítmica y MSE/RMSE (puntaje de Brier: error al cuadrado entre las probabilidades y la etiqueta en one-hot). El SVM no da probabilidades, así que no tiene pérdida logarítmica ni MSE. El registro de desempeño Train/Test está en `docs/modelo_matematico/` (sección 4.4).

## Resultados en test de originales (768 imágenes)

| Modelo | Exactitud | F1 macro | ROC-AUC | Recall Degeneración | Recall Normal | Recall Protrusión |
|---|---|---|---|---|---|---|
| 1. YOLOv8s-cls | 87,8 % | 0,750 | 0,931 | 0,50 | 0,76 | 0,94 |
| **2. ResNet-50 ajustada** (mejor por validación) | **88,9 %** | **0,769** | 0,919 | 0,50 | 0,81 | 0,94 |
| 3. SVM (RBF) sobre embeddings ajustados | 88,4 % | 0,748 | 0,899 | 0,43 | 0,77 | 0,95 |
| 4. Random Forest sobre embeddings ajustados | 88,4 % | 0,746 | 0,942 | 0,40 | 0,77 | 0,95 |
| extra: ensamble de los 4 (promedio) | 88,9 % | 0,745 | 0,947 | 0,40 | 0,81 | 0,95 |

En el test completo (balanceado, 5.253 imágenes) la exactitud baja a 74–78 %. Ese conjunto pesa igual las tres clases, y Degeneración es la más difícil.

Margen de error de la exactitud sobre 768 imágenes: ±2,2 puntos (IC 95 %). Las diferencias entre los 4 modelos están dentro de ese margen.

## Qué se ve en los resultados
- **Ninguno llega al 89 %.** La ResNet-50 queda en 88,9 % (683 de 768). Todas las variantes probadas en validación se estancan en 86–87,5 %.
- **La exactitud está inflada por la clase mayoritaria:** el 73 % del test de originales es Protrusión, y un modelo que siempre dijera "Protrusión" sacaría 73 %. Por eso se reporta también F1 macro y recall por clase.
- **Degeneración es la clase débil:** solo la mitad de los 42 casos se detecta, y muchos se confunden con Protrusión.
- **Sobreajuste:** las redes alcanzan su mejor época en la 2 y la exactitud de train llega a ~97–99 %.
- **Las características ajustadas importan:** el SVM y el Random Forest sobre embeddings de ImageNet sin ajustar (`ablacion_imagenet_*`) quedan en 78 % y 75 %.
- **Posible atajo por nivel:** la exactitud es mayor en L4-L5 y L5-S1, donde casi todo es Protrusión (ver `exactitud_por_nivel.csv`).

## Contenido de cada carpeta de modelo
| Archivo | Qué es |
|---|---|
| `metricas.csv` | Métricas por conjunto (train_orig, val_orig, val_completo, test_orig, test_completo) |
| `reporte_por_clase.csv` | Precisión, recall, F1 y soporte por clase (val completo, test orig, test completo) |
| `confusion_*.csv`, `matriz_confusion_*.png` | Matrices de confusión (las figuras están normalizadas por fila) |
| `exactitud_por_nivel.csv` | Exactitud por nivel vertebral, con la cantidad de cada clase |
| `historial.csv`, `curvas_entrenamiento.png` | Pérdida y exactitud por época (solo las redes) |
| `metadata.json` | Configuración, hiperparámetros elegidos, búsqueda en validación y versiones de librerías |
| `puntajes.npz` | Puntaje de cada clase para cada una de las 35.007 imágenes (trazabilidad completa) |

Carpetas:
- `modelo1_yolov8s_cls/`, `modelo2_resnet50/`, `modelo3_svm/` y `modelo4_random_forest/`: los 4 modelos oficiales.
- `extra_regresion_logistica/` y `extra_knn/`: líneas base del curso (semanas 9 y 10), sobre los mismos embeddings.
- `ablacion_imagenet_*/`: la misma familia de clasificadores con embeddings de ImageNet sin ajustar.
- `comparacion/`: tablas por métrica, recall por clase, matrices lado a lado, ROC y barras.
