import csv
from collections import defaultdict
from pathlib import Path
import numpy as np, pydicom, cv2

META = ["Manufacturer", "ManufacturerModelName", "MagneticFieldStrength",
        "RepetitionTime", "EchoTime", "SliceThickness", "PixelSpacing", "StationName"]

def load_pixels(ds):
    px = ds.pixel_array.astype(np.float32)
    return px * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))

def convert(input_dir, output_dir, p_low=1.0, p_high=99.5):
    inp, out = Path(input_dir), Path(output_dir)
    files = [f for f in inp.rglob("*") if f.suffix.lower() in (".ima", ".dcm")]
    series = defaultdict(list)                      # (carpeta, SeriesInstanceUID) -> archivos
    for f in files:
        ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        series[(f.parent, str(getattr(ds, "SeriesInstanceUID", "")))].append(f)

    rows, fallos = [], []
    for (folder, uid), fl in series.items():
        try:
            data = {f: pydicom.dcmread(str(f)) for f in fl}
            px = {f: load_pixels(d) for f, d in data.items()}
            pool = np.concatenate([p.ravel() for p in px.values()])
            pool = pool[pool > 0.02 * np.percentile(pool, 99.5)]   # solo tejido
            lo, hi = np.percentile(pool, (p_low, p_high))           # UNA escala por serie
            for f in fl:
                img = np.round((np.clip(px[f], lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)
                dest = out / f.relative_to(inp).with_suffix(".png")
                dest.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dest), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                rows.append([str(f.relative_to(inp)), uid, lo, hi] +
                            [str(getattr(data[f], k, "")) for k in META])
        except Exception as e:
            fallos.append((str(folder), repr(e)))

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metadatos_series.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["archivo", "serie_uid", "lo", "hi"] + META); w.writerows(rows)
    if fallos:
        (out / "fallos.txt").write_text("\n".join("\t".join(x) for x in fallos), encoding="utf-8")
    print(f"{len(rows)} PNG generados, {len(fallos)} series con error")

if __name__ == "__main__":
    convert("./t2_midsagittal_curated", "./t2_midsagittal_png_v2")