# Experimentos de ajuste (solo validación)

Registro de las pruebas hechas el 2026-10-29 para intentar acercarse al 89 % de exactitud que pide la rúbrica. **Todas las decisiones se tomaron con el conjunto de validación**; el test no se usó para elegir nada y se evaluó una sola vez, con la configuración final.

- *F1 val completo*: F1 macro sobre las 5.253 imágenes de validación (balanceado, con copias). Es el criterio de selección acordado.
- *Val orig.*: las 778 imágenes originales de validación (proporción natural: 72 % Protrusión).
- *Recall D/N/P*: recall de Degeneración, Normal y Protrusión en val orig.

## 1. Primera versión de los modelos

| Modelo | F1 val completo | Exactitud val orig. | F1 val orig. |
|---|---|---|---|
| YOLOv8s-cls (semilla 42) | 0,818 | 0,857 | 0,750 |
| ResNet-50, configuración inicial (sin suavizado ni aumento en línea) | 0,809 | 0,870 | 0,778 |
| SVM sobre embeddings de ResNet-50 ImageNet congelada | 0,636 | 0,781 | 0,598 |
| Random Forest sobre los mismos embeddings | 0,598 | 0,733 | 0,555 |

Las dos redes sobreajustan desde la época 2 (exactitud de train ~99 %). El SVM y el Random Forest sobre características de ImageNet sin ajustar quedan muy por debajo.

## 2. Características ajustadas para SVM y Random Forest

Se reemplazó el extractor de ImageNet congelado por la ResNet-50 ya ajustada (modelo 2, entrenada solo con train):

| Variante | F1 val completo | Exactitud val orig. |
|---|---|---|
| SVM, ImageNet congelada, PCA 256 | 0,636 | 0,781 |
| SVM, ResNet ajustada, PCA 256, C=0,1 | 0,791 | 0,868 |
| SVM, ResNet ajustada, sin PCA, C=0,1 | 0,791 | 0,868 |
| Random Forest, ImageNet congelada | 0,598 | 0,733 |
| Random Forest, ResNet ajustada, PCA 256 | 0,780–0,783 | 0,868–0,869 |
| Random Forest, ResNet ajustada, sin PCA | 0,790–0,791 | 0,866–0,868 |

**Decisión:** los modelos 3 y 4 oficiales usan las características de la ResNet-50 ajustada (PCA 256, que da lo mismo y es más rápido). La versión con ImageNet congelada se conserva como comparación (`ablacion_imagenet_*`). Consecuencia que se declara: los modelos 3 y 4 dependen del modelo 2.

## 3. Variantes de entrenamiento (ResNet-50 y YOLO)

| Variante | F1 val completo | Exactitud val orig. | F1 val orig. | Recall D/N/P |
|---|---|---|---|---|
| ResNet, semilla 42 (inicial) | 0,809 | 0,870 | 0,778 | 0,68 / 0,74 / 0,92 |
| ResNet, semilla 1 | 0,814 | 0,857 | 0,759 | 0,64 / 0,78 / 0,90 |
| ResNet, semilla 2 | 0,796 | 0,868 | 0,764 | 0,61 / 0,68 / 0,95 |
| **ResNet, suavizado 0,1 + aumento en línea** | **0,824** | **0,870** | **0,780** | 0,68 / 0,78 / 0,91 |
| ResNet, lr 5e-5 + suavizado 0,1 | 0,800 | 0,869 | 0,743 | 0,55 / 0,69 / 0,95 |
| YOLO, semilla 42 (oficial) | 0,818 | 0,857 | 0,750 | 0,64 / 0,73 / 0,91 |
| YOLO, semilla 1 | 0,820 | 0,853 | 0,766 | 0,68 / 0,84 / 0,87 |
| YOLO, semilla 2 | 0,799 | 0,853 | 0,735 | 0,50 / 0,79 / 0,90 |

La variación entre semillas (±0,01–0,02 de F1) es del mismo tamaño que las diferencias entre configuraciones. **Decisión:** el modelo 2 oficial usa suavizado de etiquetas 0,1 y aumento en línea (mejor F1 en validación). El YOLO se queda con la semilla 42 (la diferencia con la semilla 1 está dentro del ruido).

## 4. Ensambles y corrección de la proporción de clases

| Variante | F1 val completo | Exactitud val orig. | Recall D/N/P |
|---|---|---|---|
| Ensamble de 5 ResNet | 0,822 | 0,871 | 0,66 / 0,72 / 0,93 |
| Ensamble de 3 YOLO | 0,827 | 0,861 | 0,61 / 0,80 / 0,90 |
| Ensamble de 5 ResNet + 3 YOLO | 0,835 | 0,875 | 0,66 / 0,76 / 0,93 |
| Mismo ensamble × proporción natural (a = 1) | 0,713 | 0,875 | 0,36 / 0,68 / 0,97 |

La corrección por la proporción natural de clases (multiplicar las probabilidades por la frecuencia de cada clase en train) no sube la exactitud y hunde el recall de Degeneración: **se descartó**. El ensamble de 8 redes es lo mejor, pero solo medio punto por encima de la mejor red individual.

## Conclusión

Todas las variantes (dos arquitecturas, tres semillas, regularización, clasificadores clásicos y ensambles) se estancan en **86–87,5 % de exactitud sobre los originales de validación**. Que métodos tan distintos choquen con el mismo techo indica que el límite está en los datos (399 pacientes, 194 originales de Degeneración en train, etiquetas de estudiantes sin radiólogo), no en el modelo.
