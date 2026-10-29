"""Verifica que ninguna carpeta de ROIs mezcle pacientes de train y valid.
Uso:  python verificar_split_carpetas.py [carpeta_valid ...] --train [carpeta_train ...]
Sin argumentos revisa las carpetas por defecto. El split oficial sale de seed_v2_borrador\roi_log_anotado.csv."""
import re, sys, glob, os, pandas as pd
from pathlib import Path
O = Path(__file__).parent / "seed_v2_borrador"
d = pd.read_csv(O / "roi_log_anotado.csv", dtype={"carpeta": str})
sp = d.drop_duplicates("carpeta").set_index("carpeta").split.fillna("")
args = sys.argv[1:]; i = args.index("--train") if "--train" in args else len(args)
valid_dirs = args[:i] or [str(O / "valid_etiquetar"), str(O / "valid_candidatos")]
train_dirs = args[i+1:] or [str(O / "seed_train"), str(O / "train_candidatos")]
pat = lambda f: re.search(r"__(\d{4})_", os.path.basename(f)).group(1)
bad = 0
for names, want, other in ((valid_dirs, "valid", "train"), (train_dirs, "train", "valid")):
    for dd in names:
        fs = glob.glob(os.path.join(dd, "**", "*.png"), recursive=True)
        mal = [f for f in fs if sp.get(pat(f), "") != want]
        print(f"{dd}: {len(fs)} archivos, {len(mal)} fuera de '{want}'"); bad += len(mal)
        for f in sorted(mal): print("   ", os.path.basename(f), "-> paciente", pat(f), "es", sp.get(pat(f), "") or "EXCLUIDO")
print("\nRESULTADO:", "OK, sin mezcla" if bad == 0 else f"{bad} archivos con paciente del otro conjunto")
