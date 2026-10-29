"""Pool de candidatos extra para valid: solo pacientes valid, ROIs elegibles, un corte (mediano) por paciente y nivel,
excluyendo los 70 ya usados. Escribe a carpeta nueva + hojas numeradas para el triaje rápido (Normal/Degeneración/Protrusión)."""
import shutil, cv2, numpy as np, pandas as pd
from pathlib import Path
B = Path(__file__).parent; O = B / "seed_v2_borrador"; ROI = B.parent / "PRUEBAS NUEVAS/t2_disc_roi_patches_v2_ext"
d = pd.read_csv(O / "roi_log_anotado.csv", dtype={"ruta": str, "carpeta": str})
used = set(pd.read_csv(O / "manifest_valid.csv").archivo + "|" + pd.read_csv(O / "manifest_valid.csv").nivel)
e = d[d.ok & (d.split == "valid")].copy(); e["slice"] = e.archivo.str.extract(r"_(\d{3})\.png")[0].astype(int)
rows = []
for (pid, lv), g in e.groupby(["patient_uid", "nivel"]):
    r = g.sort_values("slice").iloc[(len(g) - 1) // 2]
    if r.archivo + "|" + lv not in used: rows.append(r)
c = pd.DataFrame(rows).sort_values(["nivel", "carpeta"]).reset_index(drop=True); c["idx"] = range(71, 71 + len(c))
c["archivo_roi"] = [f"{n}_{a[:-4]}_{n}.png" for a, n in zip(c.archivo, c.nivel)]
out = O / "valid_candidatos"; out.mkdir(exist_ok=True)
for r in c.itertuples(): shutil.copy(ROI / r.nivel / f"{r.archivo[:-4]}_{r.nivel}.png", out / r.archivo_roi)
c[["idx", "archivo_roi", "nivel", "carpeta", "conf", "yc"]].to_csv(O / "valid_candidatos_indice.csv", index=False, encoding="utf-8-sig", sep=";")
sh = B / "lector_ia" / "hojas_candidatos"; sh.mkdir(exist_ok=True)
for k in range(0, len(c), 40):
    t = []
    for r in c.iloc[k:k+40].itertuples():
        i = cv2.cvtColor(cv2.resize(cv2.imdecode(np.fromfile(str(out / r.archivo_roi), np.uint8), 0), (240, 240), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
        cv2.putText(i, str(r.idx), (5, 24), 0, 0.8, (0, 255, 255), 2); cv2.putText(i, r.nivel, (150, 232), 0, 0.5, (0, 255, 0), 1); t.append(i)
    while len(t) % 8: t.append(np.zeros_like(t[0]))
    cv2.imencode(".png", np.vstack([np.hstack(t[j:j+8]) for j in range(0, len(t), 8)]))[1].tofile(str(sh / f"candidatos_{k//40+1}.png"))
print(len(c), "candidatos |", c.groupby("nivel").size().to_dict(), "| pacientes:", c.patient_uid.nunique())
