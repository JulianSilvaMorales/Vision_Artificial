"""
Construye el dataset final: split por paciente 70/15/15, aumentación por partición y manifiesto.

Orden: (1) elegibles -> (2) split por paciente -> (3) aumentar DENTRO de cada partición -> (4) verificar.
Las copias aumentadas nunca cruzan de partición. Las métricas deben calcularse solo sobre originales
(es_original == True en manifest_35k.csv).

Uso:   py construir_dataset_35k.py --simular   (solo split y conteos, sin escribir imágenes)
       py construir_dataset_35k.py             (genera todo)
       --continuar          reanuda sobre una carpeta de salida parcial (omite los archivos ya escritos)
       --limite-seg N       se detiene tras N segundos y deja el resto para la siguiente corrida (con --continuar)
Requiere: pip install pandas numpy opencv-python-headless
"""
import os, sys, time, hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, cv2

BASE   = Path(os.environ.get("DEEPSPINE_BASE", "."))
ROIS   = BASE / "t2_disc_roi_patches_v2_ext"
ETIQ   = BASE / "DeepSpine_Dataset_Etiquetado" / "etiquetas_revisadas.csv"
SALIDA = BASE / "DeepSpine_Dataset_35k"

SEMILLA  = 42
FRAC     = {"train": 0.70, "val": 0.15, "test": 0.15}           # proporción de PACIENTES/ROIs originales
META     = {"train": 8167, "val": 1751, "test": 1751}           # imágenes por clase y partición (total 35,007)
CLASES   = ["Normal", "Degeneracion_Discal", "Protrusion"]
NIVELES  = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
# Aumentación (decisiones del proyecto): sin flips, sin deformación elástica, sin realce
ROT, SHIFT, ZOOM, BRILLO, CONTRASTE = 8.0, 0.08, (0.90, 1.10), 0.10, 0.10
EXCLUIR_PACIENTES = {"0456"}
EXCLUIR_IMAGENES  = {"T2_TSE_SAG__0547_008.png"}


def leer(ruta):      # cv2.imread falla con tildes en Windows
    return cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

def guardar(ruta, img):
    if ruta.exists(): return                 # reanudación: ya escrito
    ok, buf = cv2.imencode(".png", img); assert ok
    tmp = ruta.with_name(ruta.name + ".tmp")
    buf.tofile(str(tmp)); os.replace(tmp, ruta)      # escritura atómica: un corte no deja PNG truncados


def elegibles():
    """Reaplica las reglas de exclusión de la memoria del proyecto sobre roi_log.csv (debe dar 5,123 ROIs / 399 pacientes)."""
    r = pd.read_csv(ROIS / "roi_log.csv")
    r["paciente"] = r.archivo.str.extract(r"__(\d{4})_")[0]
    def iou(a, b):
        w = max(0, min(a.x2, b.x2) - max(a.x1, b.x1)); h = max(0, min(a.y2, b.y2) - max(a.y1, b.y1)); i = w * h
        u = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - i
        return i / u if u else 0
    malas = set()
    for im, g in r.groupby("archivo"):
        g = g.set_index("nivel")
        if any(n not in g.index for n in NIVELES): malas.add(im); continue
        f = [g.loc[n] for n in NIVELES]
        if any(iou(f[i], f[i + 1]) > 0.5 or f[i].yc >= f[i + 1].yc for i in range(4)): malas.add(im)
    r = r[~r.archivo.isin(malas | EXCLUIR_IMAGENES) & ~r.paciente.isin(EXCLUIR_PACIENTES) & (r.conf >= 0.5)]
    r["archivo_original"] = r.apply(lambda x: x.archivo[:-4] + "_" + x.nivel + ".png", axis=1)
    return set(r.archivo_original)


def repartir(df, intentos=20000):
    """Cada paciente va entero a una partición (70/15/15 de los pacientes). Se prueban muchos repartos aleatorios
    con semilla fija y se queda el que mejor mantiene la proporción de cada clase (peso x3) y de cada nivel en las tres particiones."""
    rng = np.random.default_rng(SEMILLA)
    pac = df.paciente.values
    X = pd.concat([pd.get_dummies(df.clase_final).reindex(columns=CLASES, fill_value=0),
                   pd.get_dummies(df.nivel).reindex(columns=NIVELES, fill_value=0)], axis=1).astype(float)
    X = X.groupby(pac).sum()
    ids, M = X.index.to_numpy(), X.to_numpy()
    n = len(ids); n_tr, n_va = round(FRAC["train"] * n), round(FRAC["val"] * n)
    tot = M.sum(0); w = np.array([3.0] * len(CLASES) + [1.0] * len(NIVELES))
    fr = [FRAC["train"], FRAC["val"], FRAC["test"]]
    mejor, mejor_err = None, np.inf
    for _ in range(intentos):
        perm = rng.permutation(n)
        trozos = (perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:])
        err = sum((w * ((M[idx].sum(0) / tot - f) / f) ** 2).sum() for idx, f in zip(trozos, fr))
        if err < mejor_err: mejor, mejor_err = trozos, err
    nombres = ["train", "val", "test"]
    asign = {}
    for nom, idx in zip(nombres, mejor):
        for i in idx: asign[str(ids[i])] = nom
    return asign


def aumentar(img, rng):
    h, w = img.shape[:2]
    ang, z = rng.uniform(-ROT, ROT), rng.uniform(*ZOOM)
    tx, ty = rng.uniform(-SHIFT, SHIFT) * w, rng.uniform(-SHIFT, SHIFT) * h
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, z); M[:, 2] += (tx, ty)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    a, b = 1 + rng.uniform(-CONTRASTE, CONTRASTE), rng.uniform(-BRILLO, BRILLO) * 255
    out = np.clip(out.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    return out, dict(angulo=round(ang, 2), zoom=round(z, 3), tx=round(tx, 1), ty=round(ty, 1), contraste=round(a, 3), brillo=round(b, 1))


def main(simular, continuar=False, limite=None):
    df = pd.read_csv(ETIQ, encoding="utf-8-sig")
    df = df[df.clase_final.isin(CLASES)]                      # fuera "Dudoso" si hubiera
    ok = elegibles()
    print(f"Etiquetados: {len(df)} · elegibles: {len(ok)} · usados: {df.archivo_original.isin(ok).sum()}")
    df = df[df.archivo_original.isin(ok)].copy()
    df["paciente"] = df.paciente.astype(str).str.zfill(4)
    print(f"Pacientes: {df.paciente.nunique()}")

    asign = repartir(df)
    df["particion"] = df.paciente.map(asign)
    print("\nPacientes por partición:", pd.Series(asign).value_counts().to_dict())
    print("\nORIGINALES por partición y clase:\n", pd.crosstab(df.particion, df.clase_final).reindex(columns=CLASES))
    print("\nORIGINALES por partición y nivel:\n", pd.crosstab(df.particion, df.nivel).reindex(columns=NIVELES))
    print("\nPacientes con >=1 Degeneración por partición:\n", df[df.clase_final == "Degeneracion_Discal"].groupby("particion").paciente.nunique())

    rng = np.random.default_rng(SEMILLA)
    plan = {}
    for part in FRAC:
        for c in CLASES:
            n = int(((df.particion == part) & (df.clase_final == c)).sum())
            assert 0 < n <= META[part], f"{part}/{c}: {n} originales vs meta {META[part]}"
            plan[(part, c)] = (n, META[part], META[part] / n)
    print("\nMultiplicador de aumentación (imágenes finales / originales):")
    print(pd.Series({f"{p}/{c}": f"x{m:.1f}" for (p, c), (n, t, m) in plan.items()}).to_string())
    if simular:
        print("\n(simulación: no se escribió nada)"); return

    assert continuar or not SALIDA.exists(), f"{SALIDA} ya existe; renómbrala o muévela antes (o usa --continuar)"
    fin = time.time() + limite if limite else None

    # Plan determinista: (partición, clase, carpeta, fila, nº de copias) por cada original
    items = []
    for part in FRAC:
        for c in CLASES:
            sub = df[(df.particion == part) & (df.clase_final == c)].reset_index(drop=True)
            n, meta = len(sub), META[part]
            base, resto = divmod(meta - n, n)
            copias = np.full(n, base); copias[rng.choice(n, resto, replace=False)] += 1
            dest = SALIDA / part / c; dest.mkdir(parents=True, exist_ok=True)
            existentes = set(os.listdir(dest))
            for r, k in zip(sub.itertuples(), copias):
                items.append((part, c, dest, existentes, r, int(k)))
            print(f"  {part}/{c}: {n} originales -> {meta}")

    def nombres(r, k):
        stem = r.archivo_original[:-4]
        return [f"{stem}__orig.png"] + [f"{stem}__aug{i:03d}.png" for i in range(k)]

    def procesar(item, pool=None, futuros=None):
        """Lee el original, genera las copias (determinista) y, si hay pool, escribe solo lo que falta."""
        part, c, dest, existentes, r, k = item
        img = leer(ROIS / r.nivel / r.archivo_original)
        assert img is not None and img.dtype == np.uint8, f"No se pudo leer/8-bit: {r.archivo_original}"
        stem = r.archivo_original[:-4]
        base = dict(particion=part, clase=c, paciente=r.paciente, nivel=r.nivel, original=r.archivo_original, revisor=r.revisor)
        filas_i = [dict(archivo=f"{part}/{c}/{stem}__orig.png", es_original=True, **base)]
        if pool and f"{stem}__orig.png" not in existentes: futuros.append(pool.submit(guardar, dest / f"{stem}__orig.png", img))
        for i in range(k):
            sem = int(hashlib.sha256(f"{SEMILLA}|{r.archivo_original}|{i}".encode()).hexdigest()[:8], 16)
            out, par = aumentar(img, np.random.default_rng(sem))
            nom = f"{stem}__aug{i:03d}.png"
            if pool and nom not in existentes: futuros.append(pool.submit(guardar, dest / nom, out))
            filas_i.append(dict(archivo=f"{part}/{c}/{nom}", es_original=False, semilla=sem, **base, **par))
        return filas_i

    pendientes = [it for it in items if any(nm not in it[3] for nm in nombres(it[4], it[5]))]
    print(f"\nItems con archivos faltantes: {len(pendientes)} de {len(items)}")
    if pendientes:
        pool = ThreadPoolExecutor(max_workers=16); futuros = []
        for it in pendientes:
            if fin and time.time() > fin: break
            procesar(it, pool, futuros)
        for f in futuros: f.result()
        pool.shutdown()
        escritos = sum(1 for _ in SALIDA.rglob("*.png"))
        if fin and time.time() > fin:
            print(f"\nPARCIAL: {escritos} archivos escritos de {sum(it[5] + 1 for it in items)}. Vuelve a correr con --continuar."); return

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:
        filas = [fila for lote in ex.map(procesar, items) for fila in lote]
    print(f"Filas del manifiesto: {len(filas)} ({time.time() - t0:.0f} s)", flush=True)
    presentes = {f"{d.parent.parent.name}/{d.parent.name}/{d.name}" for d in SALIDA.glob("*/*/*.png")}
    faltan = [x["archivo"] for x in filas if x["archivo"] not in presentes]
    assert not faltan, f"Faltan {len(faltan)} archivos, p. ej. {faltan[:3]}"
    # Integridad (reanudable): todo PNG debe terminar con el chunk IEND; lo ya verificado se recuerda en .verificados.txt
    def entero(f):
        try:
            with open(f, "rb") as fh:
                fh.seek(-12, 2); return fh.read(12) == b"\x00\x00\x00\x00IEND\xaeB`\x82"
        except OSError:                      # archivo vacío o más corto que el cierre PNG
            return False
    reg = SALIDA / ".verificados.txt"
    hechos = set(reg.read_text().split("\n")) if reg.exists() else set()
    por_ver = [x["archivo"] for x in filas if x["archivo"] not in hechos]
    malos = []
    with ThreadPoolExecutor(max_workers=64) as ex, open(reg, "a") as out:
        for i in range(0, len(por_ver), 2000):
            if fin and time.time() > fin: break
            lote = por_ver[i:i + 2000]
            for arch, ok in zip(lote, ex.map(lambda a: entero(SALIDA / a), lote)):
                if ok: out.write(arch + "\n")
                else: malos.append(arch)
            out.flush()
    if malos:
        for arch in malos: os.replace(SALIDA / arch, SALIDA / (arch + ".corrupto"))
        print(f"{len(malos)} PNG truncados movidos a *.corrupto; vuelve a correr con --continuar para regenerarlos."); return
    hechos = set(reg.read_text().split("\n"))
    quedan = sum(1 for x in filas if x["archivo"] not in hechos)
    if quedan:
        print(f"INTEGRIDAD PARCIAL: quedan {quedan} archivos por verificar. Vuelve a correr con --continuar."); return
    print(f"Integridad: {len(filas)} PNG completos ({time.time() - t0:.0f} s)", flush=True)
    man = pd.DataFrame(filas)
    man.to_csv(SALIDA / "manifest_35k.csv", index=False, encoding="utf-8-sig")
    pd.Series(asign, name="particion").rename_axis("paciente").reset_index().to_csv(SALIDA / "split_pacientes.csv", index=False)
    verificar(man)


def verificar(man):
    print("\n=== VERIFICACIÓN ===")
    pp = man.groupby("particion").paciente.agg(set)
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not (pp[a] & pp[b]), f"FUGA: pacientes en {a} y {b}: {sorted(pp[a] & pp[b])[:5]}"
    o = man[man.es_original].set_index("original").particion
    aug = man[~man.es_original]
    assert (aug.original.map(o) == aug.particion).all(), "Hay copias en una partición distinta a la de su original"
    assert man.archivo.is_unique
    print("Pacientes disjuntos entre particiones: OK · copias dentro de la partición de su original: OK")
    print("\nTotal:", len(man), "(rúbrica: >35,000)")
    print(pd.crosstab(man.particion, man.clase, margins=True))
    print("\nOriginales vs total por partición:", {p: (int(man[(man.particion == p) & man.es_original].shape[0]), int((man.particion == p).sum())) for p in FRAC})
    print("Proporción de imágenes:", (man.particion.value_counts(normalize=True).round(4)).to_dict())


if __name__ == "__main__":
    lim = int(sys.argv[sys.argv.index("--limite-seg") + 1]) if "--limite-seg" in sys.argv else None
    main("--simular" in sys.argv, "--continuar" in sys.argv, lim)
