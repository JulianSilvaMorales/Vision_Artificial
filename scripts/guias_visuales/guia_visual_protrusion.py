import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch

def base(ax, title):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.add_patch(Rectangle((6.6, 0), 1.6, 10, color="#f2f2f2"))            # canal medular (claro)
    ax.text(7.4, 0.3, "canal", ha="center", fontsize=8, color="gray")
    ax.add_patch(Rectangle((1, 6.1), 5.0, 3.9, color="#9a9a9a"))           # cuerpo superior
    ax.add_patch(Rectangle((1, 0), 5.0, 3.9, color="#9a9a9a"))             # cuerpo inferior
    ax.text(0.9, 9.5, "anterior", fontsize=8, color="gray"); ax.text(6.3, 9.5, "posterior", fontsize=8, color="gray")
    ax.plot([6, 6], [6.1, 3.9], ls="--", color="#d62728", lw=1.6)         # linea de referencia
    ax.plot([6, 6], [10, 6.1], color="#d62728", lw=0.6, alpha=0.4); ax.plot([6, 6], [3.9, 0], color="#d62728", lw=0.6, alpha=0.4)

def disc(ax, poly):
    ax.add_patch(Polygon(poly, closed=True, color="#2b2b2b"))

fig, axs = plt.subplots(1, 4, figsize=(17, 5.6))
# A normal
base(axs[0], "A. Sin protrusión\n(Normal / Degeneración)")
disc(axs[0], [(1, 3.9), (6, 3.9), (6, 6.1), (1, 6.1)])
axs[0].text(5, 2.2, "borde posterior del disco\nen la línea o detrás\n→ protrusión ≤ 0 mm", ha="center", fontsize=9, bbox=dict(fc="white", ec="none", alpha=0.85))
# B abombamiento
base(axs[1], "B. Abombamiento\n(difuso, base ancha)")
t = np.linspace(0, np.pi, 40); x = 6 + 0.7*np.sin(t); y = 3.9 + 2.2*(t/np.pi)
disc(axs[1], [(1, 3.9), (6, 3.9)] + list(zip(x, y)) + [(6, 6.1), (1, 6.1)])
axs[1].annotate("", xy=(6.7, 5.0), xytext=(6, 5.0), arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=2))
axs[1].text(5.0, 2.2, "contorno convexo suave,\nbase ≈ todo el borde posterior\n→ protrusión > 0, ancha", ha="center", fontsize=9, bbox=dict(fc="white", ec="none", alpha=0.85))
# C focal
base(axs[2], "C. Protrusión / hernia\n(focal, en pico)")
disc(axs[2], [(1, 3.9), (6, 3.9), (6, 4.6), (6.4, 4.95), (7.2, 5.1), (6.4, 5.25), (6, 5.6), (6, 6.1), (1, 6.1)])
axs[2].annotate("", xy=(7.2, 5.1), xytext=(6, 5.1), arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=2))
axs[2].text(5.0, 2.2, "bulto localizado (pico),\nbase corta frente al disco\n→ protrusión > 0, focal", ha="center", fontsize=9, bbox=dict(fc="white", ec="none", alpha=0.85))
# D como medir
base(axs[3], "D. Cómo medir en MicroDicom")
disc(axs[3], [(1, 3.9), (6, 3.9)] + list(zip(6 + 1.0*np.sin(t), 3.9 + 2.2*(t/np.pi))) + [(6, 6.1), (1, 6.1)])
axs[3].annotate("", xy=(7.0, 5.0), xytext=(6, 5.0), arrowprops=dict(arrowstyle="<->", color="#2ca02c", lw=2.5))
axs[3].text(6.5, 5.45, "p", color="#2ca02c", fontsize=13, fontweight="bold", ha="center")
axs[3].annotate("✗ no medir desde\nun punto interior", xy=(4.6, 5.0), xytext=(1.2, 7.4), fontsize=8.5, color="#d62728", arrowprops=dict(arrowstyle="->", color="#d62728"))
axs[3].text(5.0, 2.2, "1) línea roja = une las esquinas\nposteriores de los 2 cuerpos\n2) p = perpendicular a esa línea,\n    hasta el punto más posterior del disco", ha="center", fontsize=8.5, bbox=dict(fc="white", ec="none", alpha=0.85))
fig.suptitle("Guía visual (esquema, no es una imagen real): medir la protrusión posterior en sagital T2", fontsize=12)
fig.tight_layout(); fig.savefig("guia_visual_protrusion.png", dpi=110)
