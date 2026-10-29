"""Modelo 2 · ResNet-50 (ImageNet) con ajuste fino sobre el dataset de 35k.

Congela conv1, bn1 y layer1 y reemplaza la capa final por Dropout + Linear(3). AdamW con tasa baja y coseno.
La mejor epoca se elige por F1 macro en val completo, con parada temprana. Cada epoca deja un estado para
reanudar (--reanudar). El test se evalua una sola vez al final.

Configuracion por defecto (elegida con validacion entre 5 variantes, ver docs/resultados/README.md): suavizado de
etiquetas 0.1 y aumento en linea suave (rotacion 6 grados, desplazamiento 4 %, zoom 0.95-1.05, brillo y contraste
10 %), ademas de las copias aumentadas del dataset. Para la configuracion inicial: --ls 0 --sin-aumento --epocas 8 --paciencia 3.

Uso (en Ubuntu, desde la raiz del repo):
    python scripts/entrenamiento/02_resnet50.py [--epocas 10] [--lr 1e-4] [--reanudar]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, log_loss

from src.data.dataset import cargar_indice, conjuntos, resumen, verificar
from src.models.resnet import cargador, construir_resnet, inferir, modo_entrenamiento
from src.training.evaluacion import cerrar_modelo
from src.utils.config import CLASES, MODELOS_DIR, RUNS_DIR, SEMILLA, fijar_semilla

SLUG, NOMBRE = "modelo2_resnet50", "2 · ResNet-50 ajustada"

ap = argparse.ArgumentParser()
ap.add_argument("--epocas", type=int, default=10)
ap.add_argument("--paciencia", type=int, default=4)
ap.add_argument("--ls", type=float, default=0.1, help="suavizado de etiquetas")
ap.add_argument("--sin-aumento", action="store_true", help="sin aumento en linea")
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--dropout", type=float, default=0.3)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--reanudar", action="store_true")
args = ap.parse_args()

fijar_semilla(SEMILLA)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
assert DEV == "cuda", "No hay GPU visible en WSL"
df = cargar_indice()
print(resumen(df))
print("Pacientes por particion:", verificar(df), "· sin fuga · coincide con manifest_35k.csv")
conj = conjuntos(df)
TR, VA = (df.split == "train").values, conj["val_completo"]

run_dir = RUNS_DIR / SLUG
run_dir.mkdir(parents=True, exist_ok=True)
ESTADO, MEJOR = run_dir / "estado.pt", run_dir / "best.pt"

modelo = construir_resnet(args.dropout).to(DEV).to(memory_format=torch.channels_last)
params = [p for p in modelo.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epocas)
escala = torch.amp.GradScaler(DEV)
crit = nn.CrossEntropyLoss(label_smoothing=args.ls)
hist, mejor_f1, mala, ep0 = [], -1.0, 0, 0

if args.reanudar and ESTADO.exists():
    e = torch.load(ESTADO, map_location=DEV)
    modelo.load_state_dict(e["modelo"]); opt.load_state_dict(e["opt"]); sched.load_state_dict(e["sched"])
    hist, mejor_f1, mala, ep0 = e["hist"], e["mejor_f1"], e["mala"], e["epoca"] + 1
    print(f"Reanudando desde la epoca {ep0 + 1} (mejor F1 val = {mejor_f1:.3f})")

dl_tr = cargador(df.ruta.values[TR], df.y.values[TR], shuffle=True, batch=args.batch, workers=args.workers,
                 aumento=not args.sin_aumento)
dl_va = cargador(df.ruta.values[VA], batch=128, workers=args.workers)
yv = df.y.values[VA]
for ep in range(ep0, args.epocas):
    t0 = time.time()
    modo_entrenamiento(modelo)
    perdida, aciertos, n = 0.0, 0, 0
    for x, y in dl_tr:
        x = x.to(DEV, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(DEV, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEV):
            salida = modelo(x)
            l = crit(salida, y)
        escala.scale(l).backward()
        escala.step(opt)
        escala.update()
        perdida += l.item() * len(y); aciertos += (salida.argmax(1) == y).sum().item(); n += len(y)
    sched.step()
    Pv = inferir(modelo, dl_va, DEV)
    f1v = f1_score(yv, Pv.argmax(1), average="macro")
    hist.append(dict(epoca=ep + 1, train_loss=perdida / n, train_acc=aciertos / n,
                     val_loss=log_loss(yv, Pv, labels=list(range(len(CLASES)))), val_acc=float((Pv.argmax(1) == yv).mean()),
                     val_f1_macro=f1v, lr=opt.param_groups[0]["lr"], seg=round(time.time() - t0)))
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in hist[-1].items()}, flush=True)
    if f1v > mejor_f1:
        mejor_f1, mala = f1v, 0
        torch.save(modelo.state_dict(), MEJOR)
    else:
        mala += 1
    torch.save(dict(modelo=modelo.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), hist=hist,
                    mejor_f1=mejor_f1, mala=mala, epoca=ep), ESTADO)
    if mala >= args.paciencia:
        print("Parada temprana: sin mejora en F1 macro de validacion.")
        break

hist = pd.DataFrame(hist)
modelo.load_state_dict(torch.load(MEJOR, map_location=DEV))
S = inferir(modelo, cargador(df.ruta.values, batch=128, workers=args.workers), DEV)
mejor_ep = int(hist.loc[hist.val_f1_macro.idxmax(), "epoca"])
config = dict(base="torchvision resnet50 IMAGENET1K_V2", congeladas="conv1, bn1, layer1", **vars(args),
              optimizador="AdamW", scheduler="CosineAnnealingLR", amp=True, mejor_epoca=mejor_ep,
              criterio_best="F1 macro en val completo")
cerrar_modelo(SLUG, NOMBRE, S, True, df, conj, config, hist)
MODELOS_DIR.mkdir(exist_ok=True)
torch.save(dict(state_dict=modelo.state_dict(), clases=CLASES, dropout=args.dropout), MODELOS_DIR / "resnet50.pt")
print("Pesos guardados en", MODELOS_DIR / "resnet50.pt")
