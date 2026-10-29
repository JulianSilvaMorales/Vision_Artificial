"""Pool de candidatos de TRAIN (un corte mediano por paciente y nivel, solo pacientes train, ROIs elegibles).
Indicador objetivo 'cociente_senal' = media del disco / media de los cuerpos vertebrales vecinos (imagen v2, por serie).
Más bajo = disco más oscuro (candidato a Degeneración). Solo ORDENA para el triaje; no etiqueta."""
import shutil, cv2, numpy as np, pandas as pd
from pathlib import Path
B = Path(__file__).parent; O = B / "seed_v2_borrador"; P = B.parent / "PRUEBAS NUEVAS"
d = pd.read_csv(O / "roi_log_anotado.csv", dtype={"ruta": str, "carpeta": str})
e = d[d.ok & (d.split == "train")].copy(); e["slice"] = e.archivo.str.extract(r"_(\d{3})\.png")[0].astype(int)
rows = [g.sort_values("slice").iloc[(len(g) - 1) // 2] for _, g in e.groupby(["patient_uid", "nivel"])]
c = pd.DataFrame(rows).reset_index(drop=True)
seed = pd.read_csv(O / "manifest_semilla.csv"); sset = set(seed.archivo + "|" + seed.nivel)
c["en_semilla_350"] = [int(a + "|" + n in sset) for a, n in zip(c.archivo, c.nivel)]
cache = {}
def ratio(r):
    im = cache.setdefault(r.ruta, cv2.imdecode(np.fromfile(str(P / "t2_midsagittal_v2_l1s1_cropped_ext" / r.ruta), np.uint8), 0)).astype(float)
    h, w = im.shape; bw, bh = r.x2 - r.x1, r.y2 - r.y1
    xa, xb = int(max(0, r.x1 + .15 * bw)), int(min(w, r.x2 - .15 * bw))
    def m(ya, yb):
        ya, yb = int(max(0, ya)), int(min(h, yb)); return im[ya:yb, xa:xb].mean() if yb - ya > 3 and xb - xa > 3 else np.nan
    disc = m(r.y1 + .2 * bh, r.y2 - .2 * bh); up = m(r.y1 - 1.6 * bh, r.y1 - .4 * bh); lo = m(r.y2 + .4 * bh, r.y2 + 1.6 * bh)
    v = np.nanmean([up, lo]); return disc / v if v and v > 0 else np.nan
c["cociente_senal"] = [ratio(r) for r in c.itertuples()]
c["altura_caja_mm"] = ((c.y2 - c.y1) * 0.729).round(1)
c["archivo_roi"] = [f"{n}_{a[:-4]}_{n}.png" for a, n in zip(c.archivo, c.nivel)]
# orden: primeros 400 = más oscuros; el resto al azar (seed 42)
c["rango_nivel"] = c.groupby("nivel").cociente_senal.rank(method="first")   # rango DENTRO de cada nivel (L5-S1 es más oscuro en general)
c = c.sort_values(["rango_nivel", "nivel"]).reset_index(drop=True)          # intercala niveles: los más oscuros de cada uno primero
top, rest = c.iloc[:400], c.iloc[400:].sample(frac=1, random_state=42)
c = pd.concat([top, rest]).reset_index(drop=True); c["idx"] = range(1001, 1001 + len(c))
c["bloque"] = ["oscuros" if i < 400 else "azar" for i in range(len(c))]
out = O / "train_candidatos"; out.mkdir(exist_ok=True)
for r in c.itertuples(): shutil.copy(P / "t2_disc_roi_patches_v2_ext" / r.nivel / f"{r.archivo[:-4]}_{r.nivel}.png", out / r.archivo_roi)
c[["idx", "archivo_roi", "nivel", "carpeta", "bloque", "cociente_senal", "altura_caja_mm", "en_semilla_350"]].round(3).to_csv(O / "train_candidatos_indice.csv", index=False, encoding="utf-8-sig", sep=";")
t = c[["idx", "archivo_roi", "nivel"]].copy()
for k in ["cubeta_triaje", "clase_final", "protrusion_mm", "n_cortes_con_protrusion", "forma", "comentario"]: t[k] = ""
t.to_csv(O / "train_triaje_plantilla.csv", index=False, encoding="utf-8-sig", sep=";")
sh = B / "lector_ia" / "hojas_train"; sh.mkdir(exist_ok=True)
for k in range(0, len(c), 40):
    ti = []
    for r in c.iloc[k:k+40].itertuples():
        i = cv2.cvtColor(cv2.resize(cv2.imdecode(np.fromfile(str(out / r.archivo_roi), np.uint8), 0), (240, 240), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
        cv2.putText(i, str(r.idx), (5, 24), 0, 0.8, (0, 255, 255), 2); cv2.putText(i, r.nivel, (150, 232), 0, 0.5, (0, 255, 0), 1); ti.append(i)
    while len(ti) % 8: ti.append(np.zeros_like(ti[0]))
    cv2.imencode(".png", np.vstack([np.hstack(ti[j:j+8]) for j in range(0, len(ti), 8)]))[1].tofile(str(sh / f"train_{k//40+1:02d}.png"))
print(len(c), "candidatos | pacientes:", c.carpeta.nunique(), "| por nivel:", c.groupby("nivel").size().to_dict(), "| de la semilla 350:", int(c.en_semilla_350.sum()), "| hojas:", (len(c) + 39) // 40)
print(c.groupby("nivel").cociente_senal.median().round(2).to_dict(), "| NaN:", int(c.cociente_senal.isna().sum()))
