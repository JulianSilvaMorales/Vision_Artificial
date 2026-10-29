import numpy as np, cv2, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = pd.read_csv("seed_v2_borrador/roi_log_anotado.csv", dtype={"ruta": str})
def scale(k, lv="L5-S1"):
    r = d[(d.archivo.str.contains(k)) & (d.nivel == lv)].iloc[0]
    return max((r.x2 - r.x1) * 1.5, (r.y2 - r.y1) * 2.0) * 0.729 / 128
# nombre, U1,U2 (platillo superior), L1,L2 (platillo inferior), estimados a ojo en px del ROI
C = [("0546_008", (15,74), (60,47), (33,91), (70,50)),
     ("0455_007", (10,60), (80,47), (18,90), (80,60))]
fig, axs = plt.subplots(1, 3, figsize=(17, 6.2))
for ax, (k, U1, U2, L1, L2) in zip(axs, C):
    s = scale(k)
    im = cv2.imdecode(np.fromfile(f"seed_v2_borrador/valid_etiquetar/L5-S1_T2_TSE_SAG__{k}_L5-S1.png", np.uint8), 0)
    ax.imshow(im, cmap="gray"); ax.axis("off")
    U1, U2, L1, L2 = map(np.array, (U1, U2, L1, L2)); u = (U2 - U1) / np.linalg.norm(U2 - U1); n = np.array([-u[1], u[0]])
    if n[1] < 0: n = -n                                     # n apunta hacia abajo (hacia el platillo inferior)
    ax.plot(*zip(U1, U2), "--", color="orange", lw=1.6); ax.plot(*zip(L1, L2), "--", color="orange", lw=1.6)
    hs = []
    for t, nm in zip((0.2, 0.5, 0.8), ("ant", "med", "post")):
        P = U1 + (U2 - U1) * t
        # intersección del rayo P + a*n con la recta L1-L2
        M = np.array([n, -(L2 - L1)]).T; a, _ = np.linalg.solve(M, L1 - P); Q = P + a * n
        hs.append(a * s); ax.plot(*zip(P, Q), color="lime", lw=2.4); ax.text(*(Q + n * 4), f"{a*s:.1f}", color="lime", fontsize=11, fontweight="bold")
    ax.set_title(f"{k} (L5-S1)  alturas ant/med/post ≈ {hs[0]:.1f} / {hs[1]:.1f} / {hs[2]:.1f} mm\nmedia ≈ {np.mean(hs):.1f} mm  (estimado, ±1.5 mm)", fontsize=10)
ax = axs[2]; ax.axis("off")
ax.text(0, 1.0, "Cómo medir la altura del disco en L5-S1", fontsize=13, fontweight="bold", va="top")
ax.text(0, 0.9, "1. Dibuja la recta del platillo inferior de L5 (naranja\n   superior) y la del platillo superior de S1 (naranja inferior).\n"
        "2. Mide la distancia PERPENDICULAR al platillo superior\n   (no vertical): L5-S1 es oblicuo y la vertical la sobrestima.\n"
        "3. Tres medidas: tercio anterior, medio y posterior; usa la\n   media (el disco suele ser en cuña, más alto adelante).\n"
        "4. Mide dentro de los bordes oscuros de los platillos,\n   siempre en el mismo criterio.\n"
        "5. En el DICOM con MicroDicom (0.729 mm/px), no en el PNG.\n"
        "6. Compara con la altura de L4-L5 del mismo paciente\n   (cociente L5-S1 / L4-L5): sirve más que el valor absoluto.\n\n"
        "Los puntos de estas dos figuras son una estimación mía a ojo:\n"
        "es una ilustración, no una medida. Un píxel DICOM = 0.73 mm.", fontsize=10.5, va="top")
fig.suptitle("Ejemplo: altura del disco L5-S1 (estimación del lector IA)", fontsize=13)
fig.tight_layout(); fig.savefig("guia_altura_l5s1.png", dpi=95)
