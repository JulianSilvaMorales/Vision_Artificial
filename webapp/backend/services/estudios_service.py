"""Estudios de resonancia subidos por el radiologo: DICOM/IMA -> PNG -> cinco ROI -> cuatro modelos.

Flujo (cada paso reutiliza el codigo de scripts/, el mismo que limpio el dataset):
  subir .IMA/.dcm/.zip -> convertidor_dcm_a_png.convert (PNG por corte, normalizado por serie)
  -> elegir serie y cortes midsagitales (propuesta: 30 %-70 % central, como limpiar_slices.py)
  -> LumbarRegionCropper (L1-S1) -> best_disc_detector.pt (YOLODiscAutoCropper) -> 5 ROI de 128x128
  -> Predictor.trazar() sobre cada ROI.

Privacidad: los DICOM se borran apenas se convierten; solo quedan los PNG, que no llevan datos del paciente, y el
estudio completo se borra solo a los ESTUDIO_TTL_MIN minutos. Nunca se devuelven ni se registran nombre, fecha de
nacimiento ni identificadores.
"""
import io
import json
import shutil
import time
import uuid
import zipfile
from collections import Counter
from functools import lru_cache

import cv2
import numpy as np
import pydicom

from webapp.backend.core.config import (DETECTOR_PT, ESTUDIO_MAX_ARCHIVOS, ESTUDIO_MAX_BYTES, ESTUDIO_TTL_MIN,
                                        ESTUDIOS_DIR)
from webapp.backend.services import prediccion_service
from webapp.backend.services.validacion_service import ImagenRechazada
from webapp.backend.utils.imagenes import a_base64_png
from webapp.backend.utils.scripts import cargar

NIVELES = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
CONF_DETECTOR, IOU_DETECTOR = 0.35, 0.4
MIN_NIVELES_COLUMNA = 3          # un corte se acepta como "columna lumbar" si el detector halla al menos 3 niveles en orden


class EstudioError(Exception):
    """Error de uso (archivo invalido, estudio vencido...) que la API traduce a un codigo HTTP."""

    def __init__(self, mensaje, codigo=400):
        super().__init__(mensaje)
        self.mensaje, self.codigo = mensaje, codigo


# ---------- almacenamiento temporal ----------
def _dir(id_estudio):
    if not id_estudio.isalnum():
        raise EstudioError("Estudio no encontrado", 404)
    d = ESTUDIOS_DIR / id_estudio
    if not (d / "estudio.json").exists():
        raise EstudioError("Estudio no encontrado o vencido", 404)
    return d


def limpiar_vencidos():
    if not ESTUDIOS_DIR.exists():
        return
    limite = time.time() - ESTUDIO_TTL_MIN * 60
    for d in ESTUDIOS_DIR.iterdir():
        if d.is_dir() and d.stat().st_mtime < limite:
            shutil.rmtree(d, ignore_errors=True)


def borrar(id_estudio):
    shutil.rmtree(_dir(id_estudio), ignore_errors=True)


def _info(id_estudio):
    d = _dir(id_estudio)
    d.touch()          # cada uso renueva el plazo de borrado
    return json.loads((d / "estudio.json").read_text(encoding="utf-8"))


# ---------- subida y conversion ----------
def _expandir(archivos):
    """Devuelve [(nombre, bytes)] con los .zip abiertos. Controla el tamano total (zip bombs)."""
    salida, total = [], 0
    for nombre, datos in archivos:
        if nombre.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(datos)) as z:
                    for m in z.infolist():
                        if m.is_dir():
                            continue
                        total += m.file_size
                        if total > ESTUDIO_MAX_BYTES:
                            raise EstudioError("El estudio supera 300 MB", 413)
                        salida.append((m.filename.rsplit("/", 1)[-1], z.read(m)))
            except zipfile.BadZipFile:
                raise EstudioError(f"{nombre} no es un .zip valido")
        else:
            total += len(datos)
            if total > ESTUDIO_MAX_BYTES:
                raise EstudioError("El estudio supera 300 MB", 413)
            salida.append((nombre, datos))
    if len(salida) > ESTUDIO_MAX_ARCHIVOS:
        raise EstudioError(f"Maximo {ESTUDIO_MAX_ARCHIVOS} archivos por estudio", 413)
    return salida


def _plano(ds):
    """'sagital', 'coronal' o 'axial' segun ImageOrientationPatient; None si el archivo no lo trae."""
    try:
        o = np.array(ds.ImageOrientationPatient, dtype=float)
        n = np.cross(o[:3], o[3:])
    except (AttributeError, ValueError, TypeError):
        return None, None
    return ["sagital", "coronal", "axial"][int(np.abs(n).argmax())], n


def _posicion(ds, normal):
    try:
        return float(np.dot(np.array(ds.ImagePositionPatient, dtype=float), normal))
    except (AttributeError, ValueError, TypeError):
        return None


def _propuestos(n):
    """Cortes centrales propuestos (30 %-70 %), igual que scripts/preprocesamiento/limpiar_slices.py."""
    ini, fin = int(n * 0.30), int(n * 0.70)
    if ini >= fin:
        ini, fin = max(0, n // 2 - 1), min(n, n // 2 + 2)
    return list(range(ini, fin))


def crear(archivos):
    """Recibe [(nombre, bytes)], convierte a PNG con convertidor_dcm_a_png.convert y devuelve la ficha del estudio."""
    limpiar_vencidos()
    archivos = _expandir(archivos)
    series, omitidos = {}, 0
    for nombre, datos in archivos:
        try:
            ds = pydicom.dcmread(io.BytesIO(datos), stop_before_pixels=True)
            if not (hasattr(ds, "Rows") and hasattr(ds, "Columns")):
                raise ValueError("sin imagen")
        except Exception:
            omitidos += 1
            continue
        plano, normal = _plano(ds)
        pos = _posicion(ds, normal) if normal is not None else None
        inst = int(getattr(ds, "InstanceNumber", 0) or 0)
        uid = str(getattr(ds, "SeriesInstanceUID", "") or "sin_serie")
        s = series.setdefault(uid, dict(archivos=[], ds=ds, plano=plano))
        s["archivos"].append((pos if pos is not None else float(inst), inst, datos))
    if not series:
        raise EstudioError("Ningun archivo es un corte DICOM/IMA valido")

    id_estudio = uuid.uuid4().hex
    base = ESTUDIOS_DIR / id_estudio
    entrada, salida = base / "dicom", base / "png"
    try:
        fichas = []
        for k, (uid, s) in enumerate(sorted(series.items(), key=lambda kv: -len(kv[1]["archivos"]))):
            s["archivos"].sort(key=lambda t: (t[0], t[1]))
            carpeta = entrada / f"s{k}"
            carpeta.mkdir(parents=True)
            for i, (_, _, datos) in enumerate(s["archivos"]):
                (carpeta / f"{i:04d}.dcm").write_bytes(datos)
            ds = s["ds"]
            desc = str(getattr(ds, "SeriesDescription", "") or "")
            n = len(s["archivos"])
            fichas.append(dict(
                id=f"s{k}", descripcion=desc, n_cortes=n, plano=s["plano"] or "desconocido",
                t2_sagital=("T2" in desc.upper() and s["plano"] == "sagital"),
                equipo=dict(fabricante=str(getattr(ds, "Manufacturer", "") or ""),
                            modelo=str(getattr(ds, "ManufacturerModelName", "") or ""),
                            campo_tesla=str(getattr(ds, "MagneticFieldStrength", "") or "")),
                filas=int(ds.Rows), columnas=int(ds.Columns), propuestos=_propuestos(n)))
        cargar("preprocesamiento", "convertidor_dcm_a_png").convert(entrada, salida)
    except Exception:
        shutil.rmtree(base, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(entrada, ignore_errors=True)       # los DICOM llevan datos del paciente: no se conservan

    for f in fichas:
        f["cortes_ok"] = sum((salida / f["id"] / f"{i:04d}.png").exists() for i in range(f["n_cortes"]))
    if not any(f["cortes_ok"] for f in fichas):
        shutil.rmtree(base, ignore_errors=True)
        raise EstudioError("No se pudo convertir ningun corte a PNG. Si la ruta temporal tiene tildes o simbolos "
                           "(Windows), define DEEPSPINE_ESTUDIOS con una carpeta sin ellos, por ejemplo C:\\deepspine_tmp.", 500)
    # serie preseleccionada: la T2 sagital con mas cortes, si no la primera sagital, si no la primera
    elegida = next((f for f in fichas if f["t2_sagital"]), None) or next((f for f in fichas if f["plano"] == "sagital"), fichas[0])
    info = dict(id=id_estudio, series=fichas, serie_sugerida=elegida["id"], archivos_omitidos=omitidos)
    (base / "estudio.json").write_text(json.dumps(info), encoding="utf-8")
    return info


def ficha(id_estudio):
    return _info(id_estudio)


def ruta_corte(id_estudio, serie, n):
    info = _info(id_estudio)
    f = next((s for s in info["series"] if s["id"] == serie), None)
    if f is None or not (0 <= n < f["n_cortes"]):
        raise EstudioError("Corte no encontrado", 404)
    ruta = ESTUDIOS_DIR / id_estudio / "png" / serie / f"{n:04d}.png"
    if not ruta.exists():
        raise EstudioError("Ese corte no se pudo convertir", 404)
    return ruta


def _leer_gris(ruta):
    return cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


# ---------- deteccion y recorte de los 5 ROI ----------
@lru_cache(maxsize=1)
def _cropper():
    pesos = DETECTOR_PT
    if not pesos.exists() or pesos.stat().st_size < 1024:      # un puntero de Git LFS pesa ~130 bytes
        raise EstudioError("Falta models/best_disc_detector.pt (llego como puntero de Git LFS). "
                           "Ejecuta: git lfs install y git lfs pull", 503)
    return cargar("recorte_roi", "autocrop_rois").YOLODiscAutoCropper(str(pesos), patch_size=(128, 128),
                                                                    conf_thresh=CONF_DETECTOR)


def precargar_detector():
    t0 = time.perf_counter()
    _cropper()
    return round(time.perf_counter() - t0, 1)


def detectar_rois(gris, recortar_zona_lumbar=True):
    """Corte completo (gris) -> {nivel: dict(confianza, caja, roi(np 128x128))}, con las cajas en coordenadas del corte."""
    cropper = _cropper()
    y0 = 0
    region = gris
    if recortar_zona_lumbar:
        lumbar = cargar("recorte_roi", "recorte").LumbarRegionCropper()
        region = lumbar.crop_image(gris)
        y0 = int(gris.shape[0] * lumbar.top_crop_ratio)
    res = cropper.model(cv2.cvtColor(region, cv2.COLOR_GRAY2BGR), conf=CONF_DETECTOR, iou=IOU_DETECTOR, verbose=False)[0]
    mejores = {}
    for box in res.boxes:
        nivel = cropper.model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        if nivel in NIVELES and (nivel not in mejores or conf > mejores[nivel][0]):
            mejores[nivel] = (conf, *map(int, box.xyxy[0].tolist()))
    salida = {}
    for nivel, (conf, x1, y1, x2, y2) in mejores.items():
        parche, _ = cropper._expand_square(region, x1, y1, x2, y2)
        if parche.size == 0:
            continue
        interp = cv2.INTER_AREA if parche.shape[0] > 128 else cv2.INTER_CUBIC
        salida[nivel] = dict(confianza=round(conf, 3), caja=[x1, y1 + y0, x2, y2 + y0],
                             roi=cv2.resize(parche, (128, 128), interpolation=interp))
    return salida


def _orden_ok(rois):
    """Los niveles detectados deben ir de arriba hacia abajo en el orden L1-L2 ... L5-S1 (centro vertical creciente)."""
    yc = [(r["caja"][1] + r["caja"][3]) / 2 for n, r in sorted(rois.items(), key=lambda kv: NIVELES.index(kv[0]))]
    return all(a < b for a, b in zip(yc, yc[1:]))


def _png(arreglo):
    from PIL import Image
    return a_base64_png(Image.fromarray(arreglo))


def _describir_corte(gris, rois, **extra):
    ok = _orden_ok(rois)
    return dict(alto=int(gris.shape[0]), ancho=int(gris.shape[1]), n_detectados=len(rois), orden_ok=ok,
                columna_valida=len(rois) >= MIN_NIVELES_COLUMNA and ok,
                niveles={n: (dict(detectado=True, confianza=rois[n]["confianza"], caja=rois[n]["caja"],
                                  roi=_png(rois[n]["roi"])) if n in rois else dict(detectado=False)) for n in NIVELES},
                **extra)


def recortar(id_estudio, serie, cortes, recortar_zona_lumbar=True):
    """Detector + recorte de los 5 ROI para cada corte elegido, sin clasificar."""
    out = []
    for n in cortes:
        gris = _leer_gris(ruta_corte(id_estudio, serie, n))
        out.append(_describir_corte(gris, detectar_rois(gris, recortar_zona_lumbar), corte=n))
    return dict(estudio=id_estudio, serie=serie, cortes=out)


# ---------- clasificacion ----------
def _clasificar_roi(roi):
    """4 modelos sobre un ROI, con rechazo fuera de distribucion. Devuelve la traza sin la imagen de 224 px."""
    try:
        tr = prediccion_service.trazar_roi(roi)
    except ImagenRechazada as e:
        return dict(rechazado=e.detalle)
    tr.pop("imagen_224", None)
    return tr


def _resumen(cortes):
    """Por nivel: votos del ensamble entre los cortes elegidos y clase mayoritaria de cada modelo."""
    res = {}
    for nivel in NIVELES:
        trazas = [c["niveles"][nivel]["traza"] for c in cortes
                  if c["niveles"][nivel].get("traza") and "predicciones" in c["niveles"][nivel]["traza"]]
        if not trazas:
            res[nivel] = dict(n=0, clase=None, votos_ensamble={}, por_modelo={})
            continue
        votos = Counter(t["ensamble"]["clase"] for t in trazas)
        por_modelo = {}
        for p in trazas[0]["predicciones"]:
            c = Counter(t_p["clase"] for t in trazas for t_p in t["predicciones"] if t_p["modelo"] == p["modelo"])
            por_modelo[p["modelo"]] = c.most_common(1)[0][0]
        res[nivel] = dict(n=len(trazas), clase=votos.most_common(1)[0][0], votos_ensamble=dict(votos), por_modelo=por_modelo)
    return res


def predecir_cortes(id_estudio, serie, cortes, recortar_zona_lumbar=True):
    out = []
    for n in cortes:
        gris = _leer_gris(ruta_corte(id_estudio, serie, n))
        rois = detectar_rois(gris, recortar_zona_lumbar)
        c = _describir_corte(gris, rois, corte=n)
        if c["columna_valida"]:
            for nivel in rois:
                c["niveles"][nivel]["traza"] = _clasificar_roi(rois[nivel]["roi"])
        out.append(c)
    return dict(estudio=id_estudio, serie=serie, cortes=out, resumen=_resumen(out))


def analizar_corte_subido(img_pil):
    """Imagen subida suelta que es un corte completo (no un ROI): se detectan los discos y se clasifican."""
    gris = np.asarray(img_pil.convert("L"))
    rois = detectar_rois(gris)
    c = _describir_corte(gris, rois, corte=0)
    if not c["columna_valida"]:
        raise ImagenRechazada("sin_columna", f"El detector hallo {len(rois)} de 5 discos lumbares"
                              + ("" if _orden_ok(rois) else " y no estan en orden") + "; no parece un corte sagital lumbar.",
                              2, len(rois), MIN_NIVELES_COLUMNA)
    for nivel in rois:
        c["niveles"][nivel]["traza"] = _clasificar_roi(rois[nivel]["roi"])
    return dict(estudio=None, serie=None, cortes=[c], resumen=_resumen([c]))


def descargar_zip(id_estudio, serie, cortes):
    """PNG de los cortes elegidos, en un .zip."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n in cortes:
            z.write(ruta_corte(id_estudio, serie, n), f"corte_{n:03d}.png")
    return buf.getvalue()
