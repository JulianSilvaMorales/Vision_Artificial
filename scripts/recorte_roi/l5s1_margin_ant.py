"""Re-recorta L5-S1 con margin_ant=0.20 usando las cajas ya registradas (sin re-ejecutar YOLO). Escribe solo a carpeta nueva."""
import sys, cv2, numpy as np, pandas as pd
from pathlib import Path
B = Path(__file__).resolve().parent.parent / "PRUEBAS NUEVAS"
sys.path.insert(0, str(B))
from autocrop_rois import YOLODiscAutoCropper
MARG = 0.20
c = YOLODiscAutoCropper.__new__(YOLODiscAutoCropper)
c.margin_ant, c.margin_post, c.margin_vert = MARG, 0.40, 0.50
d = pd.read_csv(B / "t2_disc_roi_patches_v2_ext/roi_log.csv", dtype={"ruta": str})
d = d[d.nivel == "L5-S1"]
out = Path(__file__).parent / "roi_L5-S1_margin_ant020"; out.mkdir(exist_ok=True)
rows = []
for r in d.itertuples():
    img = cv2.imdecode(np.fromfile(str(B / "t2_midsagittal_v2_l1s1_cropped_ext" / r.ruta), np.uint8), 0)
    p, pads = c._expand_square(img, r.x1, r.y1, r.x2, r.y2)
    p = cv2.resize(p, (128, 128), interpolation=cv2.INTER_AREA if p.shape[0] > 128 else cv2.INTER_CUBIC)
    cv2.imencode(".png", p)[1].tofile(str(out / f"{Path(r.archivo).stem}_L5-S1.png"))
    rows.append((r.archivo, max(pads.values()), pads["left"], pads["right"], pads["top"], pads["bottom"]))
pd.DataFrame(rows, columns=["archivo", "pad", "pad_left", "pad_right", "pad_top", "pad_bottom"]).to_csv(out / "pads.csv", index=False)
x = pd.DataFrame(rows, columns=["a", "pd_", "l", "r", "t", "b"])
print(len(x), "ROIs | pad>0:", (x.pd_ > 0).sum(), "| pad_left>0:", (x.l > 0).sum(), "max", x.pd_.max())
