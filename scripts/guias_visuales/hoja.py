import sys
from pathlib import Path
import cv2
import numpy as np

nivel = sys.argv[1] if len(sys.argv) > 1 else "L5-S1"
base = Path("./roi_prueba") / nivel
files = sorted(base.glob("*.png"))[:30]
tiles = []
for f in files:
    img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(cv2.resize(img, (200, 200)), cv2.COLOR_GRAY2BGR)
    cv2.putText(img, f.stem.split("__")[1][:8], (4, 194), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    tiles.append(img)
while len(tiles) % 6:
    tiles.append(np.zeros_like(tiles[0]))
grid = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
cv2.imencode(".png", grid)[1].tofile(f"hoja_{nivel}.png")
print("Guardada:", f"hoja_{nivel}.png", f"({len(files)} ROIs)")