"""Pruebas del backend de la webapp: rechazo de imagenes, estudios de resonancia y votos del Random Forest.

Las pruebas que necesitan los pesos de models/ o las imagenes de data/demo (Git LFS) se saltan si faltan.
Ejecutar desde la raiz del repo:  python -m pytest tests -q
"""
import datetime
import io
import zipfile

import numpy as np
import pydicom
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from webapp.backend.main import app
from webapp.backend.services import estudios_service, modelos_service, validacion_service

client = TestClient(app)          # sin `with`: no corre el arranque que carga los modelos
hay_modelos = not modelos_service.faltantes()
sin_modelos = pytest.mark.skipif(not hay_modelos, reason="faltan los pesos en models/")


def png(arr, modo="L"):
    b = io.BytesIO()
    Image.fromarray(arr, modo).save(b, format="PNG")
    return b.getvalue()


def subir_imagen(contenido, ruta="/api/predecir", nombre="x.png"):
    return client.post(ruta, files={"imagen": (nombre, contenido, "image/png")})


# ---------- rechazo de imagenes (B) ----------
def test_foto_en_color_se_rechaza():
    rng = np.random.default_rng(0)
    foto = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)           # ruido en color: no es una resonancia
    r = subir_imagen(png(foto, "RGB"))
    assert r.status_code == 422 and r.json()["motivo"] == "color" and r.json()["capa"] == 1


def test_imagen_diminuta_y_proporcion_rara():
    assert subir_imagen(png(np.zeros((16, 16), np.uint8))).json()["motivo"] == "muy_pequena"
    assert subir_imagen(png(np.zeros((40, 400), np.uint8))).json()["motivo"] == "proporcion"


def test_archivo_que_no_es_imagen_es_400():
    r = client.post("/api/predecir", files={"imagen": ("x.png", b"esto no es una imagen", "image/png")})
    assert r.status_code == 400


def test_png_truncado_es_400_y_no_500():
    datos = png(np.zeros((128, 128), np.uint8))
    assert subir_imagen(datos[: len(datos) // 2]).status_code == 400


def test_imagen_de_mas_de_5mb_es_413():
    assert subir_imagen(b"0" * (5 * 1024 * 1024 + 10)).status_code == 413


def test_tipo_de_imagen():
    assert validacion_service.tipo_imagen(Image.new("L", (128, 128))) == "roi"
    assert validacion_service.tipo_imagen(Image.new("L", (320, 320))) == "corte"
    assert validacion_service.tipo_imagen(Image.new("L", (128, 256))) == "corte"


@sin_modelos
def test_corte_completo_se_deriva_al_detector():
    r = subir_imagen(png(np.full((320, 320), 90, np.uint8)))
    assert r.status_code == 422 and r.json()["motivo"] == "corte_completo"


def test_umbral_fuera_de_distribucion(monkeypatch):
    class Pca:
        components_ = np.eye(3)

    monkeypatch.setattr(validacion_service, "umbral", lambda: None)
    assert validacion_service.validar_fuera_de_distribucion(1e9, Pca()) == "sin_umbral"
    monkeypatch.setattr(validacion_service, "umbral", lambda: dict(d2_umbral=100.0, pca_huella=validacion_service.huella_pca(Pca())))
    assert validacion_service.validar_fuera_de_distribucion(99.0, Pca()) == "aplicada"
    with pytest.raises(validacion_service.ImagenRechazada) as e:
        validacion_service.validar_fuera_de_distribucion(101.0, Pca())
    assert e.value.detalle["motivo"] == "fuera_de_distribucion" and e.value.detalle["capa"] == 3
    monkeypatch.setattr(validacion_service, "umbral", lambda: dict(d2_umbral=100.0, pca_huella="otra"))
    assert validacion_service.validar_fuera_de_distribucion(1e9, Pca()) == "umbral_obsoleto"


# ---------- estudios de resonancia (A) ----------
def dicom(serie_uid, instancia, x, descripcion="t2_tse_sag", plano="sagital", filas=64, columnas=48):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientName, ds.PatientID, ds.PatientBirthDate = "Perez^Juan", "12345", "19800101"     # no deben salir de la API
    ds.SeriesInstanceUID, ds.SeriesDescription, ds.InstanceNumber = serie_uid, descripcion, instancia
    ds.Manufacturer, ds.ManufacturerModelName, ds.MagneticFieldStrength = "SIEMENS", "Aera", 1.5
    ds.ImageOrientationPatient = [0, 1, 0, 0, 0, -1] if plano == "sagital" else [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [x, 0, 0] if plano == "sagital" else [0, 0, x]
    ds.Rows, ds.Columns = filas, columnas
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit, ds.PixelRepresentation = 15, 0
    rng = np.random.default_rng(instancia)
    ds.PixelData = (rng.integers(0, 800, (filas, columnas)) + 100).astype(np.uint16).tobytes()
    b = io.BytesIO()
    ds.save_as(b)
    return b.getvalue()


@pytest.fixture
def estudio(tmp_path, monkeypatch):
    monkeypatch.setattr(estudios_service, "ESTUDIOS_DIR", tmp_path)
    uid, axial = generate_uid(), generate_uid()
    # se suben desordenados para comprobar que se ordenan por posicion y no por nombre
    archivos = [(f"IM{i}.IMA", dicom(uid, i, x=float(10 - i))) for i in range(10)]
    archivos.append(("loc.dcm", dicom(axial, 1, 0.0, descripcion="localizer", plano="axial")))
    archivos.append(("notas.txt", b"no es dicom"))
    r = client.post("/api/estudios", files=[("archivos", (n, d, "application/octet-stream")) for n, d in archivos])
    assert r.status_code == 200, r.text
    return r.json()


def test_subida_convierte_ordena_y_propone_cortes(estudio, tmp_path):
    assert len(estudio["series"]) == 2 and estudio["archivos_omitidos"] == 1
    s = next(x for x in estudio["series"] if x["t2_sagital"])
    assert estudio["serie_sugerida"] == s["id"] and s["n_cortes"] == 10 and s["cortes_ok"] == 10
    assert s["propuestos"] == [3, 4, 5, 6]                                       # 30 %-70 % de 10 cortes
    assert s["equipo"]["fabricante"] == "SIEMENS" and s["plano"] == "sagital"
    assert not (tmp_path / estudio["id"] / "dicom").exists()                      # los DICOM no se conservan
    assert "Perez" not in str(estudio) and "12345" not in str(estudio)             # nada de datos del paciente
    img = client.get(f"/api/estudios/{estudio['id']}/series/{s['id']}/cortes/0/imagen")
    assert img.status_code == 200 and img.content[:4] == b"\x89PNG"
    # orden por posicion: el corte 0 es el de x = 1 (instancia 9), el de mayor instancia
    assert client.get(f"/api/estudios/{estudio['id']}/series/{s['id']}/cortes/99/imagen").status_code == 404


def test_zip_y_borrado(tmp_path, monkeypatch):
    monkeypatch.setattr(estudios_service, "ESTUDIOS_DIR", tmp_path)
    uid = generate_uid()
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for i in range(4):
            z.writestr(f"carpeta/IM{i}", dicom(uid, i, float(i)))          # DICOM sin extension, como en los CD
    r = client.post("/api/estudios", files=[("archivos", ("estudio.zip", b.getvalue(), "application/zip"))])
    assert r.status_code == 200 and r.json()["series"][0]["n_cortes"] == 4
    id_estudio = r.json()["id"]
    assert client.delete(f"/api/estudios/{id_estudio}").status_code == 204
    assert client.get(f"/api/estudios/{id_estudio}").status_code == 404


def test_archivos_invalidos_y_zip_malo(tmp_path, monkeypatch):
    monkeypatch.setattr(estudios_service, "ESTUDIOS_DIR", tmp_path)
    assert client.post("/api/estudios", files=[("archivos", ("a.dcm", b"basura", "x"))]).status_code == 400
    assert client.post("/api/estudios", files=[("archivos", ("a.zip", b"no zip", "x"))]).status_code == 400
    assert client.get("/api/estudios/../../etc").status_code in (404, 405)


def test_vencimiento_borra_estudios(estudio, tmp_path, monkeypatch):
    import os, time
    carpeta = tmp_path / estudio["id"]
    viejo = time.time() - 3600
    os.utime(carpeta, (viejo, viejo))
    estudios_service.limpiar_vencidos()
    assert not carpeta.exists()


def roi_falso(conf=0.9):
    return dict(confianza=conf, caja=[0, 0, 10, 10], roi=np.zeros((128, 128), np.uint8))


def test_recorte_y_prediccion_con_detector_simulado(estudio, monkeypatch):
    serie = estudio["serie_sugerida"]

    def detector(_gris, _recortar=True):
        # 5 discos de arriba hacia abajo, como los hallaria el detector
        return {n: dict(roi_falso(), caja=[10, 10 + 40 * i, 30, 30 + 40 * i]) for i, n in enumerate(estudios_service.NIVELES)}

    monkeypatch.setattr(estudios_service, "detectar_rois", detector)
    r = client.post(f"/api/estudios/{estudio['id']}/recortar", json=dict(serie=serie, cortes=[3, 4]))
    assert r.status_code == 200
    c = r.json()["cortes"][0]
    assert c["columna_valida"] and c["n_detectados"] == 5 and c["niveles"]["L3-L4"]["roi"].startswith("data:image/png")

    # sin modelos reales: se simula la clasificacion para probar el resumen por nivel
    def falsa(_roi):
        p = [dict(modelo=m, nombre=m, clase="Normal", confianza=.9, puntajes={}, es_probabilidad=True)
             for m in ("yolov8s_cls", "resnet50", "svm", "random_forest")]
        return dict(predicciones=p, ensamble=dict(clase="Normal", puntajes={}), coinciden=True, pasos={}, ms_total=1)

    monkeypatch.setattr(estudios_service, "_clasificar_roi", falsa)
    monkeypatch.setattr(estudios_service.modelos_service if hasattr(estudios_service, "modelos_service") else modelos_service,
                        "faltantes", lambda: [])
    r = client.post(f"/api/estudios/{estudio['id']}/predecir", json=dict(serie=serie, cortes=[3, 4, 5]))
    assert r.status_code == 200
    res = r.json()["resumen"]
    assert res["L4-L5"]["n"] == 3 and res["L4-L5"]["clase"] == "Normal"
    assert len(r.json()["cortes"]) == 3


def test_corte_sin_columna_no_se_clasifica(estudio, monkeypatch):
    monkeypatch.setattr(estudios_service, "detectar_rois", lambda g, r=True: {"L5-S1": roi_falso()})
    monkeypatch.setattr(modelos_service, "faltantes", lambda: [])
    llamado = []
    monkeypatch.setattr(estudios_service, "_clasificar_roi", lambda roi: llamado.append(1))
    r = client.post(f"/api/estudios/{estudio['id']}/predecir", json=dict(serie=estudio["serie_sugerida"], cortes=[4]))
    assert r.status_code == 200 and not r.json()["cortes"][0]["columna_valida"] and not llamado


def test_niveles_fuera_de_orden_no_son_columna():
    rois = {"L1-L2": dict(roi_falso(), caja=[0, 100, 10, 120]), "L2-L3": dict(roi_falso(), caja=[0, 10, 10, 30]),
            "L3-L4": dict(roi_falso(), caja=[0, 50, 10, 70])}
    assert not estudios_service._orden_ok(rois)


def test_seleccion_invalida_es_422(estudio):
    r = client.post(f"/api/estudios/{estudio['id']}/recortar", json=dict(serie=estudio["serie_sugerida"], cortes=[]))
    assert r.status_code == 422


def test_descarga_png(estudio):
    r = client.post(f"/api/estudios/{estudio['id']}/png.zip", json=dict(serie=estudio["serie_sugerida"], cortes=[3, 4]))
    assert r.status_code == 200 and zipfile.ZipFile(io.BytesIO(r.content)).namelist() == ["corte_003.png", "corte_004.png"]


# ---------- modelos ----------
@sin_modelos
def test_votos_rapidos_del_random_forest_igualan_a_predict_proba():
    from webapp.backend.services import prediccion_service
    pred = prediccion_service.predictor()
    b = pred._cargar("random_forest")
    emb = np.random.default_rng(1).normal(size=2048).astype(np.float32)
    z = b["pca"].transform(b["escalador"].transform(emb[None]))
    lentos = np.bincount([int(a.predict_proba(z)[0].argmax()) for a in b["modelo"].estimators_], minlength=3)
    z32 = np.ascontiguousarray(z, dtype=np.float32)
    rapidos = np.bincount([int(a.tree_.predict(z32)[0].argmax()) for a in b["modelo"].estimators_], minlength=3)
    assert (lentos == rapidos).all()


@sin_modelos
def test_imagen_en_gris_y_en_color_dan_lo_mismo_tras_pasar_a_gris():
    from src.inference.predictor import abrir_imagen
    g = np.random.default_rng(2).integers(0, 255, (128, 128), dtype=np.uint8)
    a = np.asarray(abrir_imagen(Image.fromarray(g)))
    assert a.shape == (128, 128, 3) and (a[..., 0] == g).all() and (a[..., 1] == g).all()


@pytest.mark.skipif(not (hay_modelos and modelos_service.imagenes_demo_ok()), reason="faltan modelos o imagenes demo (Git LFS)")
def test_los_45_casos_demo_se_reproducen_y_pasan_el_rechazo():
    for caso in client.get("/api/casos").json():
        r = client.post(f"/api/casos/{caso['id']}/predecir")
        assert r.status_code == 200 and all(c["coincide_con_registrada"] for c in r.json()["comparacion"])
        with open(f"data/demo/imagenes/{caso['archivo']}", "rb") as f:
            assert subir_imagen(f.read()).status_code == 200            # un ROI real no debe rechazarse


@sin_modelos
def test_roi_en_gris_pasa_la_validacion_y_se_rechaza_con_umbral_bajo(monkeypatch):
    ruido = np.random.default_rng(3).integers(0, 255, (128, 128), dtype=np.uint8)
    monkeypatch.setattr(validacion_service, "umbral", lambda: None)
    r = subir_imagen(png(ruido))
    assert r.status_code == 200 and r.json()["pasos"]["validacion"]["capa3"] == "sin_umbral"
    assert r.json()["pasos"]["validacion"]["d2"] > 0
    # con un umbral de 1 (imposible de cumplir) la misma imagen se rechaza antes de clasificar
    monkeypatch.setattr(validacion_service, "umbral", lambda: dict(d2_umbral=1.0))
    r = subir_imagen(png(ruido))
    assert r.status_code == 422 and r.json()["motivo"] == "fuera_de_distribucion" and r.json()["capa"] == 3


def test_detectar_rois_usa_el_codigo_real_de_los_scripts(monkeypatch):
    """Con un YOLO simulado: comprueba la integracion con LumbarRegionCropper y YOLODiscAutoCropper de scripts/."""
    from webapp.backend.utils.scripts import cargar
    autocrop = cargar("recorte_roi", "autocrop_rois")
    cropper = autocrop.YOLODiscAutoCropper.__new__(autocrop.YOLODiscAutoCropper)      # sin cargar pesos
    cropper.patch_size, cropper.conf_thresh = (128, 128), 0.35
    cropper.margin_post, cropper.margin_ant, cropper.margin_vert = 0.40, 0.10, 0.50

    class Caja:
        def __init__(self, cls, conf, xyxy):
            class T(list):                 # como los tensores de ultralytics: caja.cls[0], caja.xyxy[0].tolist()
                def tolist(self):
                    return list(self)

            self.cls, self.conf, self.xyxy = [cls], [conf], [T(xyxy)]

        def __getitem__(self, i):
            return self

    class FakeYolo:
        names = {i: n for i, n in enumerate(estudios_service.NIVELES)}

        def __call__(self, img, **kw):
            assert img.ndim == 3 and img.shape[2] == 3                           # el detector recibe BGR
            cajas = [Caja(i, 0.9 - 0.05 * i, [100, 30 + 60 * i, 140, 60 + 60 * i]) for i in range(5)]
            cajas.append(Caja(0, 0.4, [10, 10, 30, 30]))                          # duplicado de menor confianza
            return [type("R", (), {"boxes": cajas})()]

    cropper.model = FakeYolo()
    monkeypatch.setattr(estudios_service, "_cropper", lambda: cropper)
    gris = np.random.default_rng(4).integers(0, 255, (400, 320), dtype=np.uint8)
    rois = estudios_service.detectar_rois(gris)
    assert list(rois) == estudios_service.NIVELES and all(r["roi"].shape == (128, 128) for r in rois.values())
    assert rois["L1-L2"]["confianza"] == 0.9                                        # se queda con la de mayor confianza
    assert rois["L1-L2"]["caja"][1] == 30 + int(400 * 0.15)                         # caja en coordenadas del corte completo
    assert estudios_service._orden_ok(rois)
    sin_recorte = estudios_service.detectar_rois(gris, recortar_zona_lumbar=False)
    assert sin_recorte["L1-L2"]["caja"][1] == 30


def test_puntuar_recomienda_los_cortes_mas_centrales(estudio, monkeypatch):
    # cortes 3..6 propuestos: el 3 ve 3 discos, el 4 los 5 pero fuera de orden, el 5 y el 6 los 5 en orden (el 6 con mas confianza)
    def caja(i):
        return [10, 10 + 40 * i, 30, 30 + 40 * i]

    def rois(n, conf, orden=True):
        niveles = estudios_service.NIVELES[:n]
        orden_y = range(len(niveles)) if orden else reversed(range(len(niveles)))
        return {nv: dict(roi_falso(conf), caja=caja(i)) for nv, i in zip(niveles, orden_y)}

    respuestas = iter([rois(3, 0.9), rois(5, 0.9, orden=False), rois(5, 0.7), rois(5, 0.8)])
    monkeypatch.setattr(estudios_service, "detectar_rois", lambda g, r=True: next(respuestas))
    r = client.post(f"/api/estudios/{estudio['id']}/puntuar", json=dict(serie=estudio["serie_sugerida"], cortes=[3, 4, 5, 6]))
    assert r.status_code == 200
    d = r.json()
    p = {c["corte"]: c for c in d["cortes"]}
    # el 4 no es columna valida (discos fuera de orden); el 3 si (3 discos en orden): se recomiendan 6, 5 y 3
    assert d["recomendados"] == [3, 5, 6]
    assert p[6]["posicion"] == 1 and p[5]["posicion"] == 2
    assert not p[4]["recomendado"] and p[4]["motivo"] == "los discos detectados no estan en orden"
    assert p[3]["motivo"] == "solo 3 de 5 discos detectados"
    assert p[6]["puntaje"] == 0.8


def test_puntuar_sin_columna_no_recomienda_nada(estudio, monkeypatch):
    monkeypatch.setattr(estudios_service, "detectar_rois", lambda g, r=True: {})
    r = client.post(f"/api/estudios/{estudio['id']}/puntuar", json=dict(serie=estudio["serie_sugerida"], cortes=[3, 4]))
    assert r.status_code == 200 and r.json()["recomendados"] == []
    assert r.json()["cortes"][0]["motivo"] == "solo 0 de 5 discos detectados"


# ---------- paginas: aplicacion en la raiz y demo en /mvp ----------
def test_raiz_sirve_la_app_y_mvp_la_demo():
    r = client.get("/")
    assert r.status_code == 200 and "Apoyo al diagnóstico lumbar" in r.text and 'src="js/main.js"' in r.text
    r = client.get("/mvp", follow_redirects=False)
    assert r.status_code in (301, 302, 307, 308) and r.headers["location"] == "/mvp/"
    r = client.get("/mvp/")
    assert r.status_code == 200 and "MVP de demostración" in r.text


def test_paginas_sin_cache():
    for ruta in ("/", "/js/main.js", "/css/base.css", "/mvp/app.js"):
        r = client.get(ruta)
        assert r.status_code == 200 and r.headers["cache-control"] == "no-cache", ruta
