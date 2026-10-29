import os, pandas as pd, shutil
from pathlib import Path

RAIZ = Path(os.environ.get("DEEPSPINE_BASE", ".")) / "DeepSpine_Dataset_Etiquetado"
NORM = {"normal": "Normal", "degeneracion_discal": "Degeneracion_Discal", "protrusion": "Protrusion", "dudoso": "Dudoso"}

lote = pd.read_csv(RAIZ/"lote_revision.csv", encoding="utf-8-sig")
lote["nombre"] = lote.copia.map(lambda c: Path(c.replace("\\", "/")).name)

filas = []
for rev in "AB":
    for cdir in (RAIZ/rev).iterdir():
        if not cdir.is_dir(): continue
        clase = NORM.get(cdir.name.lower(), cdir.name)
        for f in cdir.rglob("*"):
            if f.is_file() and f.suffix.lower() in {".png", ".jpg"}:
                filas.append(dict(nombre=f.name, clase_final=clase, carpeta=rev))
pos = pd.DataFrame(filas)

dup = pos[pos.nombre.duplicated(keep=False)]
faltan = set(lote.nombre) - set(pos.nombre)
sobran = set(pos.nombre) - set(lote.nombre)
print(f"Imágenes en carpetas: {len(pos)} · en lote: {len(lote)} · duplicadas: {len(dup)} · faltantes: {len(faltan)} · desconocidas: {len(sobran)}")
if len(dup) or faltan or sobran:
    print(dup.head(), list(faltan)[:5], list(sobran)[:5]); raise SystemExit("Hay inconsistencias; no se escribió nada.")

res = lote.merge(pos[["nombre", "clase_final", "carpeta"]], on="nombre", how="left")
res["revisor_mismo"] = res.carpeta == res.revisor           # False = la movieron a la carpeta del otro
res["corregida"] = res.clase_final != res.clase_predicha
res["paciente"] = res.archivo_original.str.extract(r"__(\d{4})_")[0]

for nombre in ["lote_revision.csv", "etiquetas_revisadas.csv"]:
    p = RAIZ/nombre
    b = RAIZ/(p.stem + "_ANTES.csv")
    if p.exists() and not b.exists(): shutil.copy2(p, b)

res.drop(columns=["nombre"]).to_csv(RAIZ/"lote_revision.csv", index=False, encoding="utf-8-sig")
fin = res[["archivo_original", "nivel", "paciente", "revisor", "clase_predicha", "confianza", "clase_final", "corregida"]]
fin.to_csv(RAIZ/"etiquetas_revisadas.csv", index=False, encoding="utf-8-sig")
try:
    fin.to_excel(RAIZ/"etiquetas_revisadas.xlsx", index=False)
except Exception as e:
    print("Sin .xlsx (pip install openpyxl si lo quieres):", e)

print("\nClase final:\n", res.clase_final.value_counts())
print("\nModelo (filas) vs etiqueta final (columnas):\n", pd.crosstab(res.clase_predicha, res.clase_final, margins=True))
print(f"\nCorregidas: {res.corregida.mean():.1%} · movidas a la carpeta del otro revisor: {(~res.revisor_mismo).sum()}")
print("\nPacientes por clase final:\n", res.groupby("clase_final").paciente.nunique())