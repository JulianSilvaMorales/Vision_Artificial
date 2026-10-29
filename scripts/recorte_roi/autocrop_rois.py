import csv
import cv2
import random
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class YOLODiscAutoCropper:
    def __init__(self, model_weights_path, patch_size=(128, 128), conf_thresh=0.35,
                 margin_post=0.40, margin_ant=0.10, margin_vert=0.50):
        self.patch_size = patch_size
        self.conf_thresh = conf_thresh
        # fracciones del ancho/alto de la caja
        self.margin_post = margin_post   # hacia atrás (canal) -> a la DERECHA en tu imagen
        self.margin_ant = margin_ant     # hacia adelante -> a la IZQUIERDA
        self.margin_vert = margin_vert   # arriba y abajo (platillos y parte de cuerpos)
        self.model = YOLO(model_weights_path)
        self.levels = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]

    def _expand_square(self, img, x1, y1, x2, y2):
        h, w = img.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        # caja expandida (posterior = derecha)
        nx1 = x1 - self.margin_ant * bw
        nx2 = x2 + self.margin_post * bw
        ny1 = y1 - self.margin_vert * bh
        ny2 = y2 + self.margin_vert * bh
        # volverla cuadrada tomando el lado mayor, centrada
        cx, cy = (nx1 + nx2) / 2, (ny1 + ny2) / 2
        side = max(nx2 - nx1, ny2 - ny1)
        sx1, sy1 = int(round(cx - side / 2)), int(round(cy - side / 2))
        sx2, sy2 = int(round(cx + side / 2)), int(round(cy + side / 2))
        # recorte con relleno si se sale de la imagen (sin estirar)
        pads = dict(top=max(0, -sy1), bottom=max(0, sy2 - h),
                    left=max(0, -sx1), right=max(0, sx2 - w))
        pad = max(pads.values())
        if pad > 0:
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                     cv2.BORDER_CONSTANT, value=int(np.median(img)))
            sx1 += pad; sx2 += pad; sy1 += pad; sy2 += pad
        return img[sy1:sy2, sx1:sx2], pads

    @staticmethod
    def _read(path, flags=cv2.IMREAD_GRAYSCALE):
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)

    def crop_dataset(self, input_folder, output_folder, detect_folder=None):
        """input_folder: recortes v2 (de aquí se recorta).
        detect_folder: recortes antiguos (con ellos se detecta, igual que en el entrenamiento de YOLO)."""
        input_path, output_path = Path(input_folder), Path(output_folder)
        detect_path = Path(detect_folder) if detect_folder else input_path
        for level in self.levels:
            (output_path / level).mkdir(parents=True, exist_ok=True)
        png_files = sorted(input_path.rglob("*.png"))

        # if max_images:
        #     png_files = png_files[:max_images]

        # if sample_patients:
        #     por_pac = {}
        #     for f in png_files:
        #         por_pac.setdefault(f.parent.name, []).append(f)   # carpeta = paciente
        #     random.seed(7)
        #     pacs = random.sample(sorted(por_pac), min(sample_patients, len(por_pac)))
        #     png_files = [random.choice(por_pac[p]) for p in pacs]

        # offset vertical aplicado al extender el recorte v2 hacia arriba (aplicar_recortes.py)
        offsets = {}
        coord_csv = input_path / "coordenadas_recortes.csv"
        with open(coord_csv, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                offsets[r["archivo"]] = int(r["offset_arriba"])

        fallos, total = [], 0
        log_rows = []

        for fp in png_files:
            try:
                rel = fp.relative_to(input_path)
                off_arriba = offsets[str(rel)]
                img = self._read(fp)                                   # v2
                det = self._read(detect_path / rel, cv2.IMREAD_COLOR)  # antigua
                if img is None or det is None or img.shape[1] != det.shape[1] or img.shape[0] < det.shape[0] + off_arriba:
                    fallos.append((str(rel), "no se pudo leer o geometría incompatible")); continue
                res = self.model(det, conf=self.conf_thresh, iou=0.4, verbose=False)[0]

                best = {}
                for box in res.boxes:
                    lvl = self.model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    if lvl not in best or conf > best[lvl][0]:
                        best[lvl] = (conf, *map(int, box.xyxy[0].tolist()))

                for lvl, (conf, x1, y1, x2, y2) in best.items():
                    # la caja viene del recorte antiguo: se lleva al marco del v2 extendido
                    y1 += off_arriba; y2 += off_arriba
                    patch, pads = self._expand_square(img, x1, y1, x2, y2)
                    if patch.size == 0:
                        continue
                    interp = cv2.INTER_AREA if patch.shape[0] > self.patch_size[0] else cv2.INTER_CUBIC
                    patch_r = cv2.resize(patch, self.patch_size, interpolation=interp)
                    ok, buf = cv2.imencode(".png", patch_r)
                    buf.tofile(str(output_path / lvl / f"{fp.stem}_{lvl}.png"))
                    total += 1
                    # ¿La caja YOLO toca el borde izquierdo del recorte antiguo?
                    toca_izq = int(x1 <= 2)
                    yc = (y1 + y2) / 2
                    log_rows.append([fp.name, str(rel), lvl, f"{conf:.3f}", x1, y1, x2, y2, f"{yc:.1f}",
                                     off_arriba, max(pads.values()), pads["top"], pads["bottom"],
                                     pads["left"], pads["right"], toca_izq])
            except Exception as e:
                fallos.append((str(fp.name), repr(e)))

        print(f"Parches generados: {total} | Fallos: {len(fallos)}")
        if fallos:
            (output_path / "fallos.txt").write_text("\n".join("\t".join(x) for x in fallos), encoding="utf-8")
        # Log por parche con flag de borde izquierdo tocado
        log_file = output_path / "roi_log.csv"
        with open(log_file, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["archivo", "ruta", "nivel", "conf", "x1", "y1", "x2", "y2", "yc", "offset_arriba",
                    "pad", "pad_top", "pad_bottom", "pad_left", "pad_right", "toca_izq"])
            w.writerows(log_rows)
        toca_n = sum(int(r[-1]) for r in log_rows)
        print(f"Log guardado: {log_file.name} ({len(log_rows)} parches, toca_izq={toca_n})")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    MODELO_YOLO_PT = BASE_DIR / "best_disc_detector.pt"
    if not MODELO_YOLO_PT.exists():
        MODELO_YOLO_PT = Path("./best_disc_detector.pt")

    cropper = YOLODiscAutoCropper(
        model_weights_path=str(MODELO_YOLO_PT),
        patch_size=(128, 128),
        conf_thresh=0.35,
    )

    cropper.crop_dataset(
        input_folder=BASE_DIR / "t2_midsagittal_v2_l1s1_cropped_ext",
        detect_folder=BASE_DIR / "t2_midsagittal_l1s1_cropped",
        output_folder=BASE_DIR / "t2_disc_roi_patches_v2_ext",
    )