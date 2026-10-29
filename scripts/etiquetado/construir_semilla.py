"""BORRADOR: dataset semilla a partir de los ROIs v2_ext. Solo copia a carpetas nuevas; no sube nada.
Orden: (1) exclusiones -> (2) ID de paciente por hash -> (3) split 80/20 por paciente -> (4) semilla SOLO desde train,
un slice por paciente y nivel, seed=42."""
import csv, hashlib, random, shutil, re
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent / "PRUEBAS NUEVAS"
ROI_DIR = BASE / "t2_disc_roi_patches_v2_ext"
META = Path(__file__).resolve().parent / "meta_dicom_por_slice.csv"   # ruta,pid,study (ver nota en README)
OUT = Path(__file__).resolve().parent / "seed_v2_borrador"
LEVELS = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
SEED, VALID_FRAC, PER_LEVEL, VALID_PER_LEVEL = 42, 0.20, 70, 14
CONF_MIN, EXCL_PACIENTES, EXCL_ARCHIVOS = 0.5, {"0456"}, {"0547_008"}
IOU_DUP = 0.5   # niveles adyacentes con cajas casi idénticas = detección duplicada

def iou(a, b):
    ix = max(0, min(a.x2, b.x2) - max(a.x1, b.x1)); iy = max(0, min(a.y2, b.y2) - max(a.y1, b.y1)); i = ix * iy
    return i / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - i)

d = pd.read_csv(ROI_DIR / "roi_log.csv", dtype={"ruta": str})
d["carpeta"] = d.ruta.str[:4]
razones = {}                      # ruta -> motivo de exclusión de la imagen entera
for ruta, g in d.groupby("ruta"):
    g = g.set_index("nivel")
    if any(e in ruta for e in EXCL_ARCHIVOS) or ruta[:4] in EXCL_PACIENTES:
        razones[ruta] = "lista_exclusion"
    elif not all(l in g.index for l in LEVELS):
        razones[ruta] = "nivel_faltante"
    elif any(iou(g.loc[a], g.loc[b]) > IOU_DUP for a, b in zip(LEVELS[:-1], LEVELS[1:])) \
            or not all(g.loc[b].yc > g.loc[a].yc for a, b in zip(LEVELS[:-1], LEVELS[1:])):
        razones[ruta] = "caja_duplicada_o_yc_no_creciente"
d["excl_img"] = d.ruta.map(razones)
d["excl_conf"] = d.conf < CONF_MIN
d["ok"] = d.excl_img.isna() & ~d.excl_conf

# ID único de paciente: sha256(PatientID|StudyInstanceUID) (PatientID está vacío en este corpus)
m = pd.read_csv(META, dtype=str).fillna("")
m["patient_uid"] = [hashlib.sha256(f"{p}|{s}".encode()).hexdigest()[:12] for p, s in zip(m.pid, m.study)]
d = d.merge(m[["ruta", "patient_uid"]], on="ruta", how="left")
assert d.patient_uid.notna().all()
assert (d.groupby("patient_uid").carpeta.nunique() == 1).all()

# split por paciente (solo pacientes con >=1 ROI elegible)
pacs = sorted(d[d.ok].patient_uid.unique())
random.Random(SEED).shuffle(pacs)
n_val = round(len(pacs) * VALID_FRAC)
valid, train = set(pacs[:n_val]), set(pacs[n_val:])
assert not (valid & train)
d["split"] = d.patient_uid.map(lambda p: "valid" if p in valid else "train" if p in train else "")

# semilla: solo train, un slice por (paciente, nivel): el slice mediano entre los ROIs elegibles
rng = random.Random(SEED)
sel = []
for lvl in LEVELS:
    g = d[(d.nivel == lvl) & d.ok & (d.split == "train")].copy()
    g["slice"] = g.archivo.str.extract(r"_(\d{3})\.png")[0].astype(int)
    elegidos = []
    for pid, gp in g.groupby("patient_uid"):
        gp = gp.sort_values(["slice"])
        elegidos.append(gp.iloc[(len(gp) - 1) // 2])
    elegidos.sort(key=lambda r: r.patient_uid)
    rng.shuffle(elegidos)
    sel += elegidos[:PER_LEVEL]
sel = pd.DataFrame(sel)

OUT.mkdir(exist_ok=True)
(OUT / "seed_train").mkdir(exist_ok=True)
for _, r in sel.iterrows():
    src = ROI_DIR / r.nivel / f"{Path(r.archivo).stem}_{r.nivel}.png"
    shutil.copy(src, OUT / "seed_train" / f"{r.nivel}_{src.name}")
sel[["archivo", "ruta", "carpeta", "patient_uid", "nivel", "slice", "conf", "yc", "pad", "split"]].to_csv(OUT / "manifest_semilla.csv", index=False)
d.drop(columns=[]).to_csv(OUT / "roi_log_anotado.csv", index=False)

# valid a etiquetar a mano (para medir el modelo semilla): mismo criterio, solo pacientes valid, RNG aparte
rngv = random.Random(SEED + 1)
selv = []
for lvl in LEVELS:
    g = d[(d.nivel == lvl) & d.ok & (d.split == "valid")].copy()
    g["slice"] = g.archivo.str.extract(r"_(\d{3})\.png")[0].astype(int)
    el = [gp.sort_values("slice").iloc[(len(gp) - 1) // 2] for _, gp in g.groupby("patient_uid")]
    el.sort(key=lambda r: r.patient_uid); rngv.shuffle(el); selv += el[:VALID_PER_LEVEL]
selv = pd.DataFrame(selv)
(OUT / "valid_etiquetar").mkdir(exist_ok=True)
for _, r in selv.iterrows():
    src = ROI_DIR / r.nivel / f"{Path(r.archivo).stem}_{r.nivel}.png"
    shutil.copy(src, OUT / "valid_etiquetar" / f"{r.nivel}_{src.name}")
selv[["archivo", "ruta", "carpeta", "patient_uid", "nivel", "slice", "conf", "yc", "pad", "split"]].to_csv(OUT / "manifest_valid.csv", index=False)
print("Valid a etiquetar:", len(selv), "| pacientes:", selv.patient_uid.nunique(), "| intersección con semilla train:", len(set(selv.patient_uid) & set(sel.patient_uid)))

# resumen
print("ROIs:", len(d), "| imágenes:", d.ruta.nunique(), "| pacientes:", d.carpeta.nunique())
print("Imágenes excluidas por motivo:", pd.Series(razones).value_counts().to_dict())
print("ROIs con conf<%.1f (en imágenes no excluidas): %d" % (CONF_MIN, (d.excl_conf & d.excl_img.isna()).sum()))
print("ROIs elegibles:", int(d.ok.sum()), "| pacientes elegibles:", len(pacs), "| train/valid:", len(train), len(valid), f"({len(valid)/len(pacs):.1%} valid)")
e = d[d.ok]; print("ROIs elegibles por nivel y split:\n", e.groupby(["nivel", "split"]).size().unstack().loc[LEVELS])
print("Semilla:", len(sel), "parches |", sel.groupby("nivel").size().to_dict(), "| pacientes únicos:", sel.patient_uid.nunique())
print("Intersección train/valid en semilla:", len(set(sel.patient_uid) & valid))
