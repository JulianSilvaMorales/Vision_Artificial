"""Hoja de contacto: cajas YOLO (todas las detecciones, conf>=0.10) sobre el recorte antiguo, para imágenes con yc no creciente."""
import cv2, numpy as np
from pathlib import Path
from ultralytics import YOLO
B = Path(__file__).resolve().parent.parent / "PRUEBAS NUEVAS"
imgs = ["0069/T2_TSE_SAG__0069_008.png","0195/T2_TSE_SAG__0195_008.png","0469/T2_TSE_SAG__0469_006.png",
        "0497/T2_TSE_SAG__0497_008.png","0510/T2_TSE_SAG__0510_007.png","0510/T2_TSE_SAG__0510_008.png","0518/T2_TSE_SAG__0518_009.png"]
m = YOLO(str(B / "best_disc_detector.pt"))
col = {"L1-L2":(255,0,0),"L2-L3":(0,200,0),"L3-L4":(0,200,255),"L4-L5":(0,0,255),"L5-S1":(255,0,255)}
tiles = []
for r in imgs:
    im = cv2.imdecode(np.fromfile(str(B/"t2_midsagittal_l1s1_cropped"/r), np.uint8), cv2.IMREAD_COLOR)
    res = m(im, conf=0.10, iou=0.4, verbose=False)[0]
    for b in res.boxes:
        l = m.names[int(b.cls[0])]; x1,y1,x2,y2 = map(int,b.xyxy[0].tolist())
        cv2.rectangle(im,(x1,y1),(x2,y2),col[l],1)
        cv2.putText(im,f"{l} {float(b.conf[0]):.2f}",(x1+2,y1+11),cv2.FONT_HERSHEY_SIMPLEX,0.35,col[l],1)
    cv2.putText(im,r.split("__")[1],(4,im.shape[0]-6),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,255),1)
    im = cv2.resize(im,(384,int(im.shape[0]*384/im.shape[1])))
    tiles.append(im)
H = max(t.shape[0] for t in tiles)
tiles = [cv2.copyMakeBorder(t,0,H-t.shape[0],0,0,cv2.BORDER_CONSTANT) for t in tiles]
while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
g = np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)])
out = Path(__file__).parent / "hoja_yc_no_creciente.png"
cv2.imencode(".png", g)[1].tofile(str(out)); print(out, g.shape)
