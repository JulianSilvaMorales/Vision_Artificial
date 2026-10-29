# data/demo

Casos de demostracion del MVP (`webapp/`): 45 recortes (ROI) **originales de la particion de test**, 15 por clase y 3 por nivel, de 30 pacientes. Ningun modelo los vio durante el entrenamiento ni la seleccion de hiperparametros. Se incluyen en el repo para que la demo funcione sin el dataset de 35k.

- `imagenes/`: los PNG, con el mismo nombre que en el dataset (`T2_TSE_SAG__<paciente>_<corte>_<nivel>__orig.png`).
- `casos.csv`: `id`, `archivo`, `paciente`, `nivel`, `clase_real` (etiqueta de los estudiantes), `particion`, `es_original` y `pred_registrada_<modelo>`, que es la prediccion que cada modelo dio a esa imagen en la evaluacion oficial (`docs/resultados/<modelo>/puntajes.npz`). La demo compara contra ella para mostrar que el resultado se reproduce.

El conjunto esta balanceado a proposito (15 por clase); su exactitud **no** equivale a la del test, que tiene 73 % de Protrusion. Las metricas oficiales estan en `docs/resultados/`.

Se generan con `python scripts/demo/seleccionar_casos_demo.py` (semilla 42). Las imagenes vienen del Lumbar Spine MRI Dataset (Sudirman et al., Mendeley Data): https://data.mendeley.com/datasets/k57fr854j2/2
