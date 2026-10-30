"""Figura de la estructura del sistema: los 4 modelos, la arquitectura hibrida y el ensamble por promedio.

Solo dibuja el diagrama; no necesita el dataset ni los modelos. La figura queda en
docs/modelo_matematico/figuras/09_estructura_sistema.png.

Uso (desde la raiz del repo):
    python scripts/entrenamiento/07_figura_estructura.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.utils.config import RAIZ_REPO

SALIDA = RAIZ_REPO / "docs" / "modelo_matematico" / "figuras" / "09_estructura_sistema.png"
ENTRADA, YOLO, RESNET, CLASICO, MODELO, ENSAMBLE = "#e2e8f0", "#fde68a", "#ddd6fe", "#bfdbfe", "#bbf7d0", "#fecaca"


def caja(ax, x, y, ancho, alto, texto, color):
    ax.add_patch(FancyBboxPatch((x, y), ancho, alto, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=color, edgecolor="#334155", linewidth=1))
    ax.text(x + ancho / 2, y + alto / 2, texto, ha="center", va="center", fontsize=8.5)


def flecha(ax, inicio, fin, punteada=False):
    ax.add_patch(FancyArrowPatch(inicio, fin, arrowstyle="-|>", mutation_scale=11, color="#334155", linewidth=1,
                                 linestyle="--" if punteada else "-", shrinkA=0, shrinkB=0))


fig, ax = plt.subplots(figsize=(15, 6.6))
ax.set_xlim(0, 15.4)
ax.set_ylim(0, 7.2)
ax.axis("off")

caja(ax, 0.1, 2.75, 1.7, 1.5, "Imagen del disco\n224 x 224 x 3\n$x$", ENTRADA)

# camino independiente
caja(ax, 2.5, 5.75, 6.9, 0.95, "YOLOv8s-cls (red propia, camino independiente)\nentrenada con las imagenes de train", YOLO)
flecha(ax, (1.8, 3.9), (2.5, 6.2))

# camino hibrido
caja(ax, 2.5, 1.3, 2.3, 3.5, "ResNet-50 ajustada\n(sin capa final)\n\ncaracteristicas\n$\\phi(x)$\n2.048 valores\n\nse calcula una vez", RESNET)
flecha(ax, (1.8, 3.5), (2.5, 3.5))
caja(ax, 5.5, 3.9, 3.9, 0.9, "Capa final de la ResNet\nDropout + Linear(2048 a 3) + softmax", RESNET)
flecha(ax, (4.8, 4.35), (5.5, 4.35))
caja(ax, 5.5, 1.3, 1.5, 2.1, "Escalador\n+ PCA\n$u$\n256 valores\n(ajustados\ncon train)", CLASICO)
flecha(ax, (4.8, 2.35), (5.5, 2.35))
caja(ax, 7.6, 2.5, 1.8, 0.9, "SVM (RBF)\n$C$ = 0,1", CLASICO)
caja(ax, 7.6, 1.3, 1.8, 0.9, "Random Forest\n300 arboles", CLASICO)
flecha(ax, (7.0, 2.6), (7.6, 2.95))
flecha(ax, (7.0, 2.1), (7.6, 1.75))

# salidas
ys = [5.75, 3.9, 2.5, 1.3]
nombres = ["Modelo 1\n$p_1(k \\mid x)$", "Modelo 2\n$p_2(k \\mid x)$", "Modelo 3\n$p_3(k \\mid x)$\n(con softmax)", "Modelo 4\n$p_4(k \\mid x)$"]
for y, n, alto in zip(ys, nombres, (0.95, 0.9, 0.9, 0.9)):
    caja(ax, 10.1, y, 1.7, alto, n, MODELO)
    flecha(ax, (9.4, y + alto / 2), (10.1, y + alto / 2))

# ensamble: se calcula despues, sin entrenamiento
caja(ax, 12.6, 2.45, 2.65, 2.1, "Ensamble (extra)\npromedio de $p_1$ a $p_4$\n\n$P_{ens}(k \\mid x)$\n\nsin entrenamiento", ENSAMBLE)
for y, alto in zip(ys, (0.95, 0.9, 0.9, 0.9)):
    flecha(ax, (11.8, y + alto / 2), (12.6, 3.5), punteada=True)

ax.text(2.5, 0.75, "Parte hibrida: los modelos 2, 3 y 4 comparten las caracteristicas de la ResNet-50 (morado).",
        fontsize=9)
ax.text(2.5, 0.35, "Linea continua: lo que se entrena o se ejecuta en cada modelo. Linea punteada: el promedio, que se calcula despues y no aprende nada.",
        fontsize=9)
fig.tight_layout()
SALIDA.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(SALIDA, dpi=150)
print("figura:", SALIDA)
