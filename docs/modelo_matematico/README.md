# Modelo matemático de los 4 clasificadores y su ensamble

Clasificación de recortes (ROI) de discos intervertebrales lumbares en RM sagital T2 en tres clases: **Degeneración discal**, **Normal** y **Protrusión**.

Este documento explica, con sus ecuaciones, cómo cada uno de los 4 modelos convierte una imagen en una predicción, cómo aprende sus parámetros y cómo se evalúa. También explica la **estructura del sistema completo** (sección 3.5) y el **ensamble** que combina los 4 modelos (sección 3.6). Sigue la secuencia vista en clase (semana 9): **datos → variables → modelo → predicción → error → optimización → evaluación**.

Todas las cifras salen de los modelos entrenados. Las figuras y los valores citados se generan con `scripts/entrenamiento/06_figuras_modelo_matematico.py` y quedan en [`figuras/`](figuras/) y [`valores.json`](valores.json). Las métricas completas están en [`docs/resultados/`](../resultados/README.md).

---

## 1. Planteamiento: X e y

**Datos.** $\mathcal{D} = \lbrace (x_i, y_i)\rbrace _{i=1}^{N}$, con $N = 35.007$ imágenes de 399 pacientes, divididas **por paciente** en train (24.501 imágenes, 279 pacientes), validación (5.253, 60) y test (5.253, 60).

**Variables de entrada (X).** Cada imagen es una matriz de intensidades. El recorte (escala de grises) se redimensiona a $224 \times 224$ píxeles y se repite en los 3 canales que esperan las redes preentrenadas. Los valores se llevan a $[0, 1]$. La ResNet-50 (y, a través de ella, los modelos 3 y 4) normaliza además cada canal $c$ con la media y la desviación de ImageNet, igual que `StandardScaler`; YOLOv8 usa los valores en $[0, 1]$ sin normalizar:

$$x \in [0,1]^{3 \times 224 \times 224}, \qquad \tilde{x}_{c,h,w} = \frac{x_{c,h,w} - \mu_c}{\sigma_c}, \quad \mu = (0{,}485;\ 0{,}456;\ 0{,}406),\ \sigma = (0{,}229;\ 0{,}224;\ 0{,}225)$$

En un modelo tabular, X son columnas como "horas de estudio" o "asistencia". Aquí X son **150.528 valores de píxel**. Por eso los modelos 1 y 2 aprenden primero a extraer características, y los modelos 3 y 4 reciben esas características ya extraídas.

**Variable objetivo (y).** Clasificación **multiclase** con $K = 3$ clases nominales (sin orden):

$$y \in \lbrace 0, 1, 2\rbrace  = \lbrace \text{Degeneración},\ \text{Normal},\ \text{Protrusión}\rbrace , \qquad \mathbf{y} = \text{one-hot}(y) \in \lbrace 0,1\rbrace ^3$$

**Qué se busca.** Una función con parámetros $\theta$ que asigne a cada imagen una probabilidad para cada clase, y luego la clase más probable:

$$f_\theta : \mathbb{R}^{3 \times 224 \times 224} \to \Delta^{2} = \Big\lbrace  p \in [0,1]^3 : \textstyle\sum_k p_k = 1 \Big\rbrace , \qquad \hat{y} = \arg\max_k\ p_k$$

| Modelo | Qué hace con X | Parámetros aprendidos |
|---|---|---|
| 1. YOLOv8s-cls | Red convolucional completa, ajustada de extremo a extremo | 5.084.579 pesos |
| 2. ResNet-50 | Red convolucional residual, ajustada de extremo a extremo | 23.514.179 pesos (225.344 congelados) |
| 3. SVM (RBF) | Clasificador de máximo margen sobre 256 características | 1.684 vectores de soporte y sus coeficientes $\alpha_i$ |
| 4. Random Forest | 300 árboles de decisión sobre las mismas 256 características | Las divisiones (variable, umbral) de cada nodo |

---

## 2. Modelos 1 y 2: redes neuronales convolucionales

### 2.1 Capas
**Convolución.** Un filtro $W$ de tamaño $k \times k$ recorre la imagen y calcula, en cada posición, una suma ponderada de los píxeles vecinos. Detecta bordes, texturas y, en capas profundas, formas como el contorno posterior del disco:

$$(W * x)_{o,i,j} = \sum_{c} \sum_{u=0}^{k-1} \sum_{v=0}^{k-1} W_{o,c,u,v}\ x_{c,\ i+u,\ j+v} + b_o$$

**Normalización por lotes (BatchNorm).** Estandariza cada canal con la media $\mu_B$ y la varianza $\sigma_B^2$ del lote, y luego reescala con dos parámetros aprendidos $\gamma$ y $\beta$:

$$\hat{h} = \gamma\thinspace  \frac{h - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta$$

**Activación** (no linealidad, figura 1): la ResNet-50 usa $\text{ReLU}(z) = \max(0, z)$ y YOLOv8 usa $\text{SiLU}(z) = z \cdot \sigma(z)$.

**Bloque residual (ResNet-50).** La entrada se suma a la salida de tres convoluciones ($1{\times}1$, $3{\times}3$, $1{\times}1$). Así la red solo aprende la *corrección* $\mathcal{F}$, y el gradiente fluye bien por 50 capas. La ResNet-50 apila 16 bloques en 4 etapas de 3, 4, 6 y 3 bloques:

$$h^{(l+1)} = \text{ReLU}\big(\mathcal{F}(h^{(l)};\ W^{(l)}) + h^{(l)}\big)$$

**YOLOv8s-cls.** Columna de convoluciones con paso 2 y bloques C2f (convoluciones con conexiones parciales entre etapas, la misma idea de reutilizar la entrada). Termina en una convolución $1{\times}1$ a 1.280 canales.

**Promedio global y capa final.** El mapa de características final se promedia por canal. Queda un vector $\phi(x)$ (2.048 valores en la ResNet-50, 1.280 en YOLOv8), que pasa por *dropout* ($p = 0{,}3$: durante el entrenamiento se apaga al azar el 30 % de los valores) y por una capa lineal:

$$\phi(x) = \frac{1}{HW}\sum_{i,j} h^{(L)}_{:,i,j}, \qquad z = W\thinspace \phi(x) + b \in \mathbb{R}^3$$

En la ResNet-50 esta capa tiene $2.048 \times 3 + 3 = 6.147$ parámetros.

### 2.2 De puntajes a probabilidades: softmax
La regresión logística de la semana 9 usa la sigmoide para 2 clases. Con 3 clases se usa su generalización, la **softmax**, que convierte los 3 puntajes $z_k$ (*logits*) en probabilidades positivas que suman 1:

$$p_k = P(y = k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^{3} e^{z_j}}$$

Ejemplo (figura 1): los logits $(2{,}0;\ 0{,}5;\ -1{,}0)$ dan las probabilidades $(0{,}786;\ 0{,}175;\ 0{,}039)$, y la predicción es la clase 0. Con $K = 2$ la softmax se reduce exactamente a la sigmoide.

![Activaciones y softmax](figuras/01_activaciones_softmax.png)

### 2.3 Función de costo: entropía cruzada
Como en la regresión logística, el error de una imagen se mide con la **entropía cruzada** (log loss): cuánto se aleja la distribución predicha $p$ de la distribución objetivo $q$. El costo $J(\theta)$ promedia esa pérdida sobre las imágenes:

$$\ell(x_i, y_i) = -\sum_{k=1}^{3} q_{ik} \log p_{ik}, \qquad J(\theta) = \frac{1}{N}\sum_{i=1}^{N} \ell(x_i, y_i) + \text{regularización}$$

- **YOLOv8s-cls:** $q = \mathbf{y}$ (one-hot), así que $\ell = -\log p_{y}$. Si la red da $p_y = 0{,}01$ a la clase correcta, la pérdida es 4,6; si da 0,99, es 0,01. Penaliza fuerte la **seguridad equivocada** (figura 2).
- **ResNet-50, con suavizado de etiquetas** ($\varepsilon = 0{,}1$): en lugar de exigir probabilidad 1 a la clase real, el objetivo reparte un poco de masa entre las demás:

$$q_k = (1-\varepsilon)\thinspace \mathbf{y}_k + \frac{\varepsilon}{K} \quad\Rightarrow\quad q = (0{,}933;\ 0{,}033;\ 0{,}033)$$

  Así la red no puede volverse arbitrariamente segura, lo que reduce el sobreajuste. El costo mínimo posible ya no es 0 sino la entropía de $q$, $H(q) = 0{,}291$. Ese es el piso al que se acerca la pérdida de train (0,305 en la época 6, figura 2).

**¿Por qué no MSE para entrenar?** El MSE de la semana 9 es la función de costo natural cuando $y$ es un número real (regresión). Para probabilidades de clase, la entropía cruzada da gradientes más grandes cuando el error es grande. Además, es la función que se obtiene al maximizar la verosimilitud del modelo. El MSE sí se usa para **evaluar**: se reporta como puntaje de Brier (sección 4.2) en el registro de desempeño (sección 4.4).

![Función de costo](figuras/02_funcion_costo.png)

### 2.4 Optimización: descenso del gradiente con AdamW
**Descenso del gradiente por lotes.** En cada paso se toma un lote de $B = 64$ imágenes y se mueve cada parámetro en la dirección opuesta al gradiente del costo, como la regla $\theta_j := \theta_j - \alpha\thinspace  \partial J / \partial \theta_j$ de la semana 9:

$$g_t = \nabla_\theta \frac{1}{B}\sum_{i \in \text{lote}} \ell(x_i, y_i)$$

El gradiente de cada peso se calcula con la regla de la cadena desde la salida hacia la entrada (retropropagación).

**AdamW.** Mantiene un promedio móvil del gradiente ($m_t$, el "impulso") y de su cuadrado ($v_t$, la escala). Con eso cada parámetro avanza con un paso adaptado a su propia escala. El término $\lambda\thinspace \theta$ (*weight decay*) encoge los pesos hacia 0, y es la regularización L2:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\thinspace  g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\thinspace  g_t^2$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}, \qquad \theta_t = \theta_{t-1} - \alpha_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda\thinspace  \theta_{t-1} \right)$$

Valores: $\beta_1 = 0{,}9$ en la ResNet-50 y $0{,}937$ en YOLOv8 (valor por defecto de ultralytics), $\beta_2 = 0{,}999$, $\epsilon = 10^{-8}$, $\lambda = 0{,}01$. Ultralytics no aplica $\lambda$ a los sesgos ni a las capas de normalización.

**Tasa de aprendizaje en coseno** (figura 3). El tamaño del paso $\alpha_t$ empieza en $\alpha_0$ y baja suavemente hasta $\alpha_{\min}$ en $T$ épocas: pasos grandes al principio, finos al final.

$$\alpha_t = \alpha_{\min} + \tfrac{1}{2}(\alpha_0 - \alpha_{\min})\left(1 + \cos\frac{\pi t}{T}\right)$$

| | YOLOv8s-cls | ResNet-50 |
|---|---|---|
| $\alpha_0$ / $\alpha_{\min}$ | $5\cdot10^{-4}$ / $5\cdot10^{-5}$ (1 época de calentamiento) | $10^{-4}$ / 0 |
| Épocas máximas $T$ / paciencia | 15 / 4 | 10 / 4 |
| Capas congeladas (no se actualizan) | primeros 4 módulos | `conv1`, `bn1`, `layer1` (225.344 pesos) |
| Aumento en línea | recorte aleatorio con el 100 % del área (solo varía la proporción, 3:4 a 4:3) y brillo ±5 % | rotación ±6°, desplazamiento 4 %, zoom 0,95–1,05, brillo y contraste ±10 % |
| Suavizado de etiquetas | no | $\varepsilon = 0{,}1$ |
| Criterio para elegir la época | exactitud top-1 en validación | F1 macro en validación |
| Época elegida / épocas corridas | 2 / 6 | 2 / 6 |

**Ajuste fino (*transfer learning*).** Ninguna red parte de cero: $\theta$ se inicializa con los pesos aprendidos en ImageNet (1,28 millones de fotografías), que ya detectan bordes y texturas. Solo la capa final empieza al azar.

![Tasa de aprendizaje](figuras/03_tasa_aprendizaje.png)

### 2.5 Diagnóstico: sobreajuste y parada temprana
La figura 4 muestra el patrón de las dos redes: la pérdida de train sigue bajando, mientras la de validación sube desde la época 2. El modelo empieza a memorizar las imágenes de train, que además incluyen copias aumentadas de las mismas imágenes. Por eso se guarda la época con mejor desempeño en validación y el entrenamiento se detiene tras 4 épocas sin mejora (**parada temprana**):

$$e^{*} = \arg\max_{e}\ \text{F1}_{\text{macro}}\big(\text{validación};\ \theta^{(e)}\big)$$

![Curvas de aprendizaje](figuras/04_curvas_aprendizaje.png)

---

## 3. Modelos 3 y 4: características profundas + clasificador clásico

Los modelos 3 y 4 no trabajan sobre píxeles. Cada imagen se representa con el vector $\phi(x) \in \mathbb{R}^{2048}$ que produce la ResNet-50 del modelo 2 justo antes de su capa final (sección 2.1). Después se reduce ese vector a 256 dimensiones y se clasifica con un SVM o con un Random Forest.

**Dependencia declarada:** los modelos 3 y 4 dependen del modelo 2. Con un extractor de ImageNet sin ajustar rinden mucho menos (sección 3.4).

### 3.1 Estandarización y PCA
**Estandarización.** Cada una de las 2.048 características se centra y se escala con la media y la desviación **calculadas solo con train**. Calcularlas con todo el dataset filtraría información del test (*data leakage*, semanas 6 y 9):

$$s_j = \frac{\phi_j(x) - \mu_j^{\text{train}}}{\sigma_j^{\text{train}}}$$

**Análisis de componentes principales (PCA).** Se busca la base de direcciones ortogonales en la que los datos de train varían más. Son los vectores propios de la matriz de covarianza, y cada imagen se proyecta sobre los $k = 256$ primeros:

$$\Sigma = \frac{1}{n-1}\thinspace  S^\top S = V \Lambda V^\top, \qquad u = V_k^\top s \in \mathbb{R}^{256}, \qquad \text{varianza explicada} = \frac{\sum_{j=1}^{k}\lambda_j}{\sum_{j=1}^{2048}\lambda_j}$$

Con las características ajustadas, **256 componentes explican el 80,0 % de la varianza** (las 2 primeras ya el 38 %). Con las de ImageNet sin ajustar, solo el 58,7 % (figura 5). Las características ajustadas están más concentradas en pocas direcciones, las que separan las clases.

![Varianza explicada por el PCA](figuras/05_pca_varianza.png)

### 3.2 Modelo 3: máquina de vectores de soporte (SVM) con kernel RBF
**Idea.** Entre todas las fronteras que separan dos clases, el SVM elige la de **máximo margen**: la que queda más lejos de los ejemplos más cercanos de cada lado. Esos ejemplos se llaman vectores de soporte. Para dos clases, $y_i \in \lbrace -1, +1\rbrace $, con margen blando:

$$\min_{w,\thinspace b,\thinspace \xi}\ \ \frac{1}{2}\lVert w \rVert^2 + C \sum_{i=1}^{n} \xi_i \quad \text{sujeto a} \quad y_i\big(w^\top \varphi(u_i) + b\big) \ge 1 - \xi_i,\quad \xi_i \ge 0$$

- $\frac{1}{2}\lVert w\rVert^2$: minimizarlo maximiza el ancho del margen, $2 / \lVert w \rVert$.
- $\xi_i$: holgura que permite que algunas imágenes queden dentro del margen o mal clasificadas.
- $C$: equilibrio entre margen ancho y errores en train. Un $C$ pequeño tolera más errores y da una frontera más suave, menos sobreajustada.

Es equivalente a minimizar la **pérdida bisagra** (*hinge*): $\frac{1}{2}\lVert w\rVert^2 + C\sum_i \max\big(0,\ 1 - y_i f(u_i)\big)$.

**Kernel RBF.** Las clases no se separan con un hiperplano en el espacio original. El SVM resuelve el problema dual, en el que los datos solo aparecen como similitudes $K(u_i, u_j)$; así trabaja implícitamente en un espacio de dimensión infinita (el "truco del kernel"):

$$K(u, v) = \exp\big(-\gamma\thinspace \lVert u - v \rVert^2\big), \qquad \gamma = \frac{1}{d \cdot \mathrm{Var}(U)} = 0{,}00061\ \ (\texttt{gamma="scale"},\ d = 256)$$

$$\max_{\alpha}\ \sum_i \alpha_i - \frac{1}{2}\sum_i\sum_j \alpha_i\alpha_j\thinspace  y_i y_j\thinspace  K(u_i, u_j) \quad \text{s.a.}\ \ 0 \le \alpha_i \le C,\ \ \sum_i \alpha_i y_i = 0$$

$$f(u) = \sum_{i \in \text{SV}} \alpha_i\thinspace  y_i\thinspace  K(u_i, u) + b$$

Solo los vectores de soporte ($\alpha_i > 0$) intervienen en la predicción. Con $C = 0{,}1$, elegido en validación (figura 8), el modelo usa **1.684 vectores de soporte de 24.501 imágenes de train (6,9 %)**: 320 de Degeneración, 660 de Normal y 704 de Protrusión.

La figura 6, a la derecha, muestra cómo decae la similitud con la distancia. Con el $\gamma$ usado, dos imágenes a la distancia mediana (59) tienen similitud ≈ 0,12. Con $\gamma \times 10$ todas las imágenes serían "lejanas" (sobreajuste); con $\gamma / 10$, todas "parecidas" (subajuste).

**Tres clases.** scikit-learn entrena un SVM por cada par de clases (*uno contra uno*: 3 clasificadores) y combina sus votos en un puntaje por clase; la predicción es la clase con mayor puntaje. Estos puntajes **no son probabilidades**; por eso el SVM no tiene log loss y la app los muestra normalizados con softmax.

La figura 6, a la izquierda, es una **ilustración en 2 dimensiones** (las 2 primeras componentes, 1.200 originales de train). En train las clases aparecen muy separadas porque la ResNet ya vio esas imágenes; en validación y test se mezclan más, y por eso la exactitud allí baja a ~88 %.

![SVM con kernel RBF](figuras/06_svm_rbf.png)

### 3.3 Modelo 4: Random Forest
**Árbol de decisión.** Hace preguntas del tipo "¿$u_j \le t$?" y divide las imágenes en dos grupos. En cada nodo elige la variable $j$ y el umbral $t$ que más reducen la **impureza de Gini** (semana 11):

$$G(\text{nodo}) = 1 - \sum_{k=1}^{3} p_k^2, \qquad \Delta G = G(\text{padre}) - \frac{n_L}{n}\thinspace G(L) - \frac{n_R}{n}\thinspace G(R)$$

$G = 0$ si el nodo tiene una sola clase (puro). El máximo es $1 - 1/K$ si las clases están mezcladas en partes iguales: 0,5 con 2 clases, 0,667 con 3. Ejemplo de la clase: 6 compran y 4 no ($G = 1 - 0{,}6^2 - 0{,}4^2 = 0{,}48$). Tras dividir en un grupo (5, 0) con $G = 0$ y otro (1, 4) con $G = 0{,}32$, la impureza ponderada baja a $0{,}5 \cdot 0 + 0{,}5 \cdot 0{,}32 = 0{,}16$.

**Bosque.** Un árbol solo sobreajusta. Se entrenan $T = 300$ árboles distintos entre sí por dos fuentes de azar:
1. **Bootstrap:** cada árbol se entrena con una muestra de train tomada con reemplazo, del mismo tamaño.
2. **Variables al azar:** en cada división solo se consideran $m = \sqrt{256} = 16$ de las 256 variables.

La predicción promedia las probabilidades de los 300 árboles (votación suave). La probabilidad de cada árbol, $p_t(k \mid u)$, es la proporción de la clase $k$ en la hoja donde cae $u$:

$$P(y = k \mid u) = \frac{1}{T}\sum_{t=1}^{T} p_t(k \mid u), \qquad \hat{y} = \arg\max_k P(y = k \mid u)$$

Al promediar árboles diversos baja la varianza (el sobreajuste) sin subir mucho el sesgo. Hiperparámetro elegido en validación (figura 8): profundidad máxima 20. Los 300 árboles llegan a esa profundidad y tienen 779 hojas en promedio.

La **importancia** de cada variable es la disminución total de Gini que aporta, promediada sobre los árboles. Las 10 primeras componentes del PCA concentran el 70 % de la importancia (figura 7).

![Random Forest](figuras/07_random_forest.png)

### 3.4 Extras del curso y comparación con ImageNet
Sobre los mismos 256 componentes se entrenaron, como línea base del curso:
- **Regresión logística multinomial** (semana 9): $p = \text{softmax}(W u + b)$, con regularización L2 de fuerza $1/C$ y $C = 0{,}01$.
- **KNN** (semana 10): predice por mayoría entre los $K = 31$ vecinos más cercanos según la distancia euclídea $d(u, v) = \sqrt{\sum_j (u_j - v_j)^2}$. La escala importa, y por eso se estandariza antes.

Con características de ImageNet sin ajustar, el SVM baja de 88,4 % a 77,6 % y el Random Forest de 88,4 % a 75,3 % de exactitud en el test de originales. Lo que hace funcionar a los modelos 3 y 4 es la representación aprendida por el modelo 2.

![Búsqueda de hiperparámetros](figuras/08_busqueda_hiperparametros.png)

### 3.5 Estructura del sistema: arquitectura híbrida

El sistema tiene **dos caminos**. YOLOv8s-cls va por el suyo, independiente. La ResNet-50 es la **base compartida** de los otros tres modelos: sus características $\phi(x)$ alimentan su propia capa final (modelo 2) y, después de estandarizarlas y reducirlas, al SVM (modelo 3) y al Random Forest (modelo 4). Unir una red profunda que extrae características con un clasificador clásico que decide es lo que se llama **arquitectura híbrida**.

![Estructura del sistema](figuras/09_estructura_sistema.png)

**Las 4 funciones.** Sea $g_Y$ la red YOLOv8s-cls hasta sus logits, $g_R$ la ResNet-50 sin su capa final y $\phi(x) = g_R(x)$. Cada modelo es una composición de funciones:

$$p_1(x) = \text{softmax}\big(g_Y(x)\big)$$

$$p_2(x) = \text{softmax}\big(W\thinspace \phi(x) + b\big)$$

$$u(x) = V_k^\top\thinspace D^{-1}\big(\phi(x) - \mu\big), \qquad D = \text{diag}(\sigma_1, \dots, \sigma_{2048})$$

$$p_3(x) = \text{softmax}\big(\text{SVM}(u(x))\big), \qquad p_4(x) = \frac{1}{T}\sum_{t=1}^{T} p_t\big(u(x)\big)$$

Las tres últimas dependen de la misma $\phi(x)$. Por eso, en la aplicación la ResNet-50 se ejecuta una sola vez por imagen. La softmax de $p_3$ solo lleva los puntajes del SVM a la misma escala de los demás: no son probabilidades calibradas.

**Orden de entrenamiento.** Cada etapa usa solo lo aprendido en las anteriores, y nada se ajusta con el test:

| Paso | Qué se aprende | Con qué datos | Qué queda fijo |
|---|---|---|---|
| 1 | Pesos de YOLOv8s-cls | train; época elegida con validación | — |
| 2 | Pesos de la ResNet-50, incluidos $W$ y $b$ | train; época elegida con validación | — |
| 3 | $\mu$, $D$ y $V_k$ (escalador y PCA) | características de train | La ResNet-50 |
| 4 | SVM y Random Forest | $u$ de train; hiperparámetros elegidos con validación | ResNet-50, escalador y PCA |

**Por qué es híbrida y no un ensamble.** En esta estructura un modelo le pasa sus **características** a otro, y cada uno sigue dando su propia predicción. Un ensamble, en cambio, combina las **respuestas finales** de varios modelos. Esa combinación es la sección siguiente.

### 3.6 Ensamble por promedio

**Idea.** Es la misma de la semana 11: consultar a varios modelos reduce la dependencia de una sola decisión. Cada modelo entrega 3 puntajes para la imagen; se promedian por clase y gana la clase con el mayor promedio (**votación suave**). Con $M = 4$ modelos:

$$P_{\text{ens}}(k \mid x) = \frac{1}{M}\sum_{m=1}^{M} p_m(k \mid x), \qquad \hat{y}_{\text{ens}} = \arg\max_k\ P_{\text{ens}}(k \mid x)$$

El ensamble **no se entrena**: todos los modelos pesan igual ($1/4$) y no tiene parámetros que aprender. Se calcula después, con las salidas de los 4 modelos ya entrenados. Por eso es un **extra** y no un quinto modelo.

**Ejemplo real** (imagen de test: paciente 0319, L1-L2, etiqueta real Degeneración):

| Modelo | Degeneración | Normal | Protrusión | Predicción |
|---|---|---|---|---|
| 1. YOLOv8s-cls | 0,109 | 0,871 | 0,020 | Normal (falla) |
| 2. ResNet-50 | 0,936 | 0,025 | 0,039 | Degeneración |
| 3. SVM (con softmax) | 0,746 | 0,064 | 0,190 | Degeneración |
| 4. Random Forest | 0,866 | 0,050 | 0,083 | Degeneración |
| **Ensamble (promedio)** | **0,664** | 0,253 | 0,083 | **Degeneración** |

Por ejemplo, $P_{\text{ens}}(\text{Degeneración}) = (0{,}109 + 0{,}936 + 0{,}746 + 0{,}866) / 4 = 0{,}664$. Los otros tres modelos corrigen el error de YOLO.

**Por qué puede ayudar, y su límite.** Si cada modelo tiene una varianza de error $\sigma^2$ y la correlación entre modelos es $\rho$, la varianza del promedio de $M$ modelos es:

$$\text{Var}\left(\frac{1}{M}\sum_{m=1}^{M} p_m\right) = \rho\thinspace \sigma^2 + \frac{1-\rho}{M}\thinspace \sigma^2$$

Si los modelos fueran independientes ($\rho = 0$), la varianza se dividiría entre 4. Si fueran idénticos ($\rho = 1$), promediar no serviría de nada. Aquí los modelos 2, 3 y 4 comparten $\phi(x)$, así que se espera una correlación alta y una mejora limitada.

**Resultados** (test de originales; detalle en las secciones 4.3 y 4.4):

| Métrica | Mejor modelo individual | Ensamble |
|---|---|---|
| Exactitud | 0,889 (ResNet-50) | 0,889 |
| F1 macro | 0,769 (ResNet-50) | 0,745 |
| ROC-AUC macro | 0,942 (Random Forest) | **0,947** |
| MSE (Brier) | 0,173 (ResNet-50) | **0,164** |

El ensamble empata en exactitud y da las probabilidades más confiables (mejor ROC-AUC y MSE), pero detecta menos casos de Degeneración: 17 de 42, frente a 21 de la ResNet-50. En validación, el criterio acordado, su F1 macro es 0,822 frente a 0,824 de la ResNet-50. Por eso **la ResNet-50 sigue siendo el modelo principal** y el ensamble se reporta como resultado adicional.

---

## 4. Evaluación

### 4.1 Protocolo
- Los parámetros $\theta$ se aprenden solo con **train**.
- Los hiperparámetros ($C$, profundidad, $K$, época) se eligen con **validación**, por F1 macro.
- El **test** se evalúa una sola vez, al final. Ningún paciente aparece en dos particiones, y cada copia aumentada queda en la partición de su original.
- Se reporta el test de **originales** (768 imágenes en proporción natural: 42 Degeneración, 164 Normal, 562 Protrusión) y el **test completo** (5.253, balanceado con copias).

### 4.2 Matriz de confusión y métricas
En clase se vieron las fórmulas para **2 clases** (positiva y negativa):

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}, \qquad \text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}, \qquad F1 = 2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision} + \text{Recall}}$$

Con **3 clases** se aplican las mismas fórmulas tres veces. Cada clase $k$ se trata como la "positiva" y las otras dos como la "negativa" (uno contra el resto): TP = predijo $k$ y era $k$; FP = predijo $k$ y no era $k$; FN = era $k$ y predijo otra; TN = el resto. Luego se promedian las tres (promedio *macro*). La exactitud no necesita promedio: es el total de aciertos (la diagonal de la matriz) sobre el total de imágenes.

$$\text{Exactitud} = \frac{\sum_k \text{TP}_k}{N}, \qquad \text{Precisión}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FP}_k}, \qquad \text{Recall}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}$$

$$F1_k = 2\thinspace \frac{\text{Precisión}_k \cdot \text{Recall}_k}{\text{Precisión}_k + \text{Recall}_k}, \qquad F1_{\text{macro}} = \frac{1}{K}\sum_k F1_k, \qquad \text{Exactitud balanceada} = \frac{1}{K}\sum_k \text{Recall}_k$$

**Ejemplo con el modelo 2 en el test de originales** (filas = real, columnas = predicha):

| Real \ Predicha | Degeneración | Normal | Protrusión |
|---|---|---|---|
| **Degeneración** (42) | **21** | 6 | 15 |
| **Normal** (164) | 5 | **133** | 26 |
| **Protrusión** (562) | 6 | 27 | **529** |

| Clase | TP | FP | FN | Precisión | Recall | F1 |
|---|---|---|---|---|---|---|
| Degeneración | 21 | 11 | 21 | 0,656 | 0,500 | 0,568 |
| Normal | 133 | 33 | 31 | 0,801 | 0,811 | 0,806 |
| Protrusión | 529 | 41 | 33 | 0,928 | 0,941 | 0,935 |

- **Exactitud** = (21 + 133 + 529) / 768 = 683 / 768 = **0,889**.
- **F1 macro** = (0,568 + 0,806 + 0,935) / 3 = **0,769**.

**Por qué la exactitud sola engaña.** Un modelo que siempre dijera "Protrusión" acertaría 562 / 768 = **73,2 %** sin detectar ni un caso de Degeneración. Es el mismo ejemplo del fraude visto en clase. El F1 macro pesa igual las tres clases y deja ver que Degeneración se detecta solo en la mitad de los casos.

**Intervalo de confianza.** Con 768 imágenes, la exactitud de 88,9 % tiene un intervalo de Wilson al 95 % de **[86,5 %; 91,0 %]**:

$$\frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}, \qquad z = 1{,}96$$

Las diferencias entre los 4 modelos (87,8 % a 88,9 %) están dentro de ese margen.

**ROC-AUC.** Para cada clase, la curva ROC recorre todos los umbrales $\tau$ sobre $p_k$ y grafica la tasa de verdaderos positivos contra la de falsos positivos. El área bajo la curva es la probabilidad de que una imagen de la clase $k$ reciba más puntaje que una que no lo es. Se promedia sobre las 3 clases (uno contra el resto).

**Log loss.** Es la entropía cruzada de la sección 2.3, evaluada con las probabilidades sobre el conjunto de prueba.

**MSE (puntaje de Brier).** Es el MSE de la semana 9 aplicado a la clasificación. El "valor real" de cada imagen es su etiqueta en one-hot (por ejemplo, Protrusión = $(0, 0, 1)$) y la "predicción" son las 3 probabilidades del modelo. Se suman los errores al cuadrado de las 3 clases y se promedia sobre las $N$ imágenes:

$$\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{3}\left(p_{ik} - y_{ik}\right)^2, \qquad \text{RMSE} = \sqrt{\text{MSE}}$$

Vale 0 si el modelo da probabilidad 1 a la clase correcta y como máximo 2. Igual que el MSE de clase, castiga más los errores grandes. Ejemplo con una imagen real de Protrusión:
- Si el modelo da $p = (0{,}1;\ 0{,}2;\ 0{,}7)$: $0{,}1^2 + 0{,}2^2 + 0{,}3^2 = 0{,}14$ (acierta, con algo de duda).
- Si da $p = (0{,}9;\ 0{,}05;\ 0{,}05)$: $0{,}9^2 + 0{,}05^2 + 0{,}95^2 = 1{,}715$ (falla con mucha seguridad).

A diferencia de la exactitud, el MSE también mide si la **confianza** del modelo es la adecuada. No aplica al SVM, porque sus puntajes no son probabilidades.

### 4.3 Resultados (test de originales)

| Modelo | Exactitud | F1 macro | ROC-AUC macro | Recall D / N / P |
|---|---|---|---|---|
| 1. YOLOv8s-cls | 0,878 | 0,750 | 0,931 | 0,50 / 0,76 / 0,94 |
| **2. ResNet-50** (mejor por validación) | **0,889** | **0,769** | 0,919 | 0,50 / 0,81 / 0,94 |
| 3. SVM (RBF) | 0,884 | 0,748 | 0,899 | 0,43 / 0,77 / 0,95 |
| 4. Random Forest | 0,884 | 0,746 | 0,942 | 0,40 / 0,77 / 0,95 |
| extra: ensamble por promedio (sección 3.6) | 0,889 | 0,745 | 0,947 | 0,40 / 0,81 / 0,95 |

![Matrices de confusión de los 4 modelos](../resultados/comparacion/matrices_confusion_test_orig.png)

![Curvas ROC](../resultados/comparacion/roc_test_orig.png)

### 4.4 Registro de desempeño (formato de la clase)

Es la tabla "Registro de desempeño" de la semana 9, con columnas Train y Test.
- **Train:** las 3.577 imágenes originales de entrenamiento.
- **Test:** las 768 originales de test.
- **Test completo:** las 5.253 imágenes de test con copias aumentadas, como referencia adicional.

Precision, recall y F1 son promedios *macro* de las 3 clases. **R² no aplica**: es una métrica de regresión y este problema es de clasificación (como advierte la diapositiva, no todas las métricas aplican a todos los modelos).

| Modelo | Métrica | Train | Test | Test completo |
|---|---|---|---|---|
| **1. YOLOv8s-cls** | MSE / RMSE (Brier) | 0,024 / 0,154 | 0,196 / 0,442 | 0,403 / 0,635 |
| | Accuracy | 0,984 | 0,878 | 0,759 |
| | Precision / Recall | 0,959 / 0,989 | 0,771 / 0,732 | 0,779 / 0,759 |
| | F1 / ROC-AUC | 0,973 / 0,999 | 0,750 / 0,931 | 0,752 / 0,923 |
| **2. ResNet-50** | MSE / RMSE (Brier) | 0,047 / 0,217 | 0,173 / 0,416 | 0,349 / 0,591 |
| | Accuracy | 0,971 | **0,889** | 0,775 |
| | Precision / Recall | 0,948 / 0,981 | 0,795 / 0,751 | 0,799 / 0,775 |
| | F1 / ROC-AUC | 0,964 / 0,998 | 0,769 / 0,919 | 0,768 / 0,915 |
| **3. SVM (RBF)** | MSE / RMSE (Brier) | no aplica | no aplica | no aplica |
| | Accuracy | 0,979 | 0,884 | 0,754 |
| | Precision / Recall | 0,964 / 0,984 | 0,797 / 0,718 | 0,789 / 0,754 |
| | F1 / ROC-AUC | 0,974 / 0,998 | 0,748 / 0,899 | 0,746 / 0,905 |
| **4. Random Forest** | MSE / RMSE (Brier) | 0,010 / 0,098 | 0,174 / 0,416 | 0,370 / 0,608 |
| | Accuracy | 0,998 | 0,884 | 0,743 |
| | Precision / Recall | 0,994 / 0,999 | 0,810 / 0,710 | 0,784 / 0,743 |
| | F1 / ROC-AUC | 0,996 / 1,000 | 0,746 / 0,942 | 0,731 / 0,920 |
| **Extra: ensamble** | MSE / RMSE (Brier) | 0,031 / 0,175 | 0,164 / 0,405 | 0,334 / 0,578 |
| | Accuracy | 0,989 | 0,889 | 0,762 |
| | Precision / Recall | 0,978 / 0,992 | 0,782 / 0,721 | 0,794 / 0,762 |
| | F1 / ROC-AUC | 0,985 / 1,000 | 0,745 / 0,947 | 0,752 / 0,937 |

**Lectura:**
- La distancia entre Train y Test es la que se vio en las curvas de aprendizaje (sección 2.5): **sobreajuste**. En train los modelos casi no se equivocan (MSE de 0,01 a 0,05); en test el MSE sube a ~0,17–0,20.
- Por MSE, el ensamble es el mejor en test (0,164), seguido de la ResNet-50 y el Random Forest (0,173 y 0,174).
- Todas las cifras salen de `docs/resultados/comparacion/tabla_metricas_larga.csv`, que `scripts/entrenamiento/05_comparar_modelos.py` calcula a partir de los puntajes guardados de cada modelo.

---

## 5. Supuestos y limitaciones del modelo
- **Independencia:** se supone que los pacientes son independientes entre sí. Las imágenes de un mismo paciente no lo son, por eso se divide por paciente. Las copias aumentadas no son datos nuevos: el tamaño efectivo es de 399 pacientes, no de 35.007 imágenes.
- **Etiquetas con ruido:** $y$ la asignaron estudiantes, sin radiólogo y sin corte axial. El modelo aprende a reproducir esas etiquetas, no un diagnóstico.
- **Desbalance:** el entrenamiento ve las clases balanceadas (por copias), pero en la proporción natural el 73 % es Protrusión. Probé corregir por esa proporción multiplicando $p_k$ por la frecuencia de cada clase; no mejoró la exactitud y hundió el recall de Degeneración, así que se descartó (`docs/resultados/experimentos_validacion.md`).
- **Sobreajuste:** las redes llegan a ~97–99 % en train y ~89 % en test de originales. La regularización (dropout, weight decay, suavizado, aumento) y la parada temprana lo reducen, pero no lo eliminan.
- **Dependencia de los modelos 3 y 4** respecto del modelo 2: comparten la representación $\phi(x)$, así que no son cuatro opiniones independientes. Eso limita lo que puede ganar el ensamble (sección 3.6).
- **Herramienta académica, no diagnóstica.**
