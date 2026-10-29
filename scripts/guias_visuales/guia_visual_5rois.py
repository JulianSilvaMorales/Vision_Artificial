import numpy as np, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
V = "seed_v2_borrador/valid_etiquetar/"
# nombre, mm/px del ROI (de roi_log), P1 (esquina post. cuerpo sup.), P2 (esquina post. cuerpo inf.), T (punta posterior del disco), etiqueta IA actual
C = [("L1-L2_T2_TSE_SAG__0496_006_L1-L2.png", 0.436, (78,48), (78,78), (89,64), "Degeneración (media)"),
     ("L1-L2_T2_TSE_SAG__0455_007_L1-L2.png", 0.504, (83,50), (80,80), (87,64), "Degeneración (alta)"),
     ("L1-L2_T2_TSE_SAG__0107_014_L1-L2.png", 0.444, (83,57), (78,75), (79,68), "Degeneración (baja)"),
     ("L2-L3_T2_TSE_SAG__0174_008_L2-L3.png", 0.495, (76,58), (80,72), (86,64), "Degeneración (media)"),
     ("L2-L3_T2_TSE_SAG__0423_007_L2-L3.png", 0.487, (82,56), (80,75), (86,64), "Degeneración (alta)")]
fig, axs = plt.subplots(2, 3, figsize=(15, 10.5)); axs = axs.ravel()
for ax, (n, s, p1, p2, t, lab) in zip(axs, C):
    im = cv2.imdecode(np.fromfile(V + n, np.uint8), 0)
    ax.imshow(im, cmap="gray", vmin=0, vmax=255); ax.axis("off")
    p1, p2, t = map(np.array, (p1, p2, t)); d = (p2 - p1) / np.linalg.norm(p2 - p1)
    foot = p1 + d * np.dot(t - p1, d); perp = t - foot
    sign = 1 if perp[0] > 0 else -1; mm = sign * np.linalg.norm(perp) * s
    e = p1 - d * 25, p2 + d * 25
    ax.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]], "--", color="red", lw=1.8)
    ax.plot(*p1, "o", color="red", ms=5); ax.plot(*p2, "o", color="red", ms=5)
    ax.annotate("", xy=tuple(t), xytext=tuple(foot), arrowprops=dict(arrowstyle="<->", color="lime", lw=2.2))
    ax.set_title(f"{n.split('__')[1][:8]}  ({n[:5]})\nprotrusión estimada ≈ {mm:+.1f} mm   |   etiqueta IA: {lab}", fontsize=10)
ax = axs[5]; ax.axis("off")
ax.text(0, 0.95, "Cómo leer esta hoja", fontsize=13, fontweight="bold", va="top")
ax.text(0, 0.85, "• Línea roja discontinua: une la esquina posterior\n  del cuerpo vertebral de arriba con la del de abajo.\n"
        "• Flecha verde: distancia perpendicular de esa línea\n  a la punta posterior del disco (p).\n"
        "• Positivo = el disco sale hacia el canal; 0 o negativo =\n  no sobresale.\n"
        "• Los puntos los puse yo a ojo (error ≈ ±3 px ≈ ±1.5 mm):\n  son una ILUSTRACIÓN, no una medida. Midan en el DICOM\n  con MicroDicom (0.729 mm/px).\n"
        "• Valor orientativo de partida (no validado): >~3 mm\n  con base ancha → candidato a abombamiento; pico focal\n  → candidato a hernia; ante la duda → dudoso.", fontsize=10.5, va="top")
fig.suptitle("Guía visual sobre 5 ROIs reales: referencia y protrusión posterior (estimación del lector IA)", fontsize=13)
fig.tight_layout(); fig.savefig("guia_visual_5rois.png", dpi=95)
