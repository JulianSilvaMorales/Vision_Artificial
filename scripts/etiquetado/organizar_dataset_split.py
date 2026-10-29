#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: organizar_dataset_split.py
Descripción:
    Organiza un dataset de ROIs de resonancia lumbar en conjuntos 'train' y 'valid'
    mediante una partición 80/20 a nivel de PACIENTE (evita data leakage).
    Cumple con todas las restricciones del enunciado:
    - Regex estricto y reporte de errores.
    - Split a nivel de paciente.
    - Tolerancia de ±2% en número de imágenes (18% - 22%).
    - Presencia de los 5 niveles discales en validación con representación balanceada.
    - Soporte condicional para estratificación por clases si existe 'labels.csv'.
    - Copia no destructiva de archivos en estructura plana.
    - Generación de manifest.csv y reporte de verificación.
"""

import os
import sys
import re
import csv
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict, Counter

# Asegurar codificación UTF-8 en salida estándar para consolas Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


REGEX_PATTERN = re.compile(
    r'^(L[1-5]-(?:L[1-5]|S1))_T2_TSE_SAG__(\d{4})_(\d{3})_(L[1-5]-(?:L[1-5]|S1))\.png$'
)
DISC_LEVELS = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Organizar dataset lumbar por paciente en train/valid (80/20)."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="seed_dataset_enhanced",
        help="Ruta de la carpeta de origen con las imágenes .png.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset_split",
        help="Ruta de la carpeta de salida para train/valid y manifest.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla inicial para el generador pseudoaleatorio.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Tolerancia del porcentaje de validación (por defecto 0.02 = ±2%%).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Ruta opcional a labels.csv con columnas filename,clase.",
    )
    return parser.parse_args()


def load_labels_if_exists(labels_path: Path):
    if not labels_path or not labels_path.is_file():
        return None
    labels = {}
    with open(labels_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "filename" in row and "clase" in row:
                labels[row["filename"].strip()] = row["clase"].strip()
    return labels


def find_labels_file(source_dir: Path, custom_path_str: str = None):
    if custom_path_str:
        p = Path(custom_path_str)
        if p.is_file():
            return p
    # Search common locations
    candidates = [
        source_dir / "labels.csv",
        source_dir.parent / "labels.csv",
        Path("./labels.csv"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def main():
    args = parse_arguments()
    source_dir = Path(args.source).resolve()
    output_dir = Path(args.output).resolve()
    initial_seed = args.seed
    tolerance = args.tolerance

    print("=" * 70)
    print("ORGANIZACIÓN Y PARTICIÓN DE DATASET LUMBAR (TRAIN / VALID 80/20)")
    print("=" * 70)
    print(f"Carpeta origen: {source_dir}")
    print(f"Carpeta salida: {output_dir}")
    print(f"Semilla base:   {initial_seed}")
    print(f"Tolerancia:     ±{tolerance * 100:.1f}%")

    if not source_dir.is_dir():
        print(f"ERROR: La carpeta de origen '{source_dir}' no existe.")
        return 1

    # Crear carpeta de salida
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = output_dir / "train"
    valid_dir = output_dir / "valid"
    train_dir.mkdir(exist_ok=True)
    valid_dir.mkdir(exist_ok=True)

    # Comprobar si existe labels.csv
    labels_file = find_labels_file(source_dir, args.labels)
    labels_map = None
    if labels_file:
        print(f"Archivo de etiquetas encontrado: {labels_file}")
        labels_map = load_labels_if_exists(labels_file)
        print(f"Total etiquetas cargadas: {len(labels_map)}")
    else:
        print("Archivo de etiquetas 'labels.csv' no detectado. Se omite estratificación por clase.")

    # 1. Recorrer y parsear carpeta de origen con regex
    valid_records = []
    error_records = []

    for entry in sorted(source_dir.iterdir()):
        if not entry.is_file():
            continue
        filename = entry.name
        match = REGEX_PATTERN.match(filename)
        if not match:
            error_records.append((filename, "Formato de nombre no coincide con regex"))
            continue

        g1_level = match.group(1)
        g2_patient = match.group(2)
        g3_slice = match.group(3)
        g4_level = match.group(4)

        if g1_level != g4_level:
            error_records.append(
                (filename, f"Inconsistencia en niveles discales: {g1_level} != {g4_level}")
            )
            continue

        record = {
            "filename": filename,
            "paciente": g2_patient,
            "nivel": g1_level,
            "slice": g3_slice,
            "path": entry,
        }
        if labels_map and filename in labels_map:
            record["clase"] = labels_map[filename]
        valid_records.append(record)

    # Guardar reporte de errores
    error_file = output_dir / "errores.txt"
    with open(error_file, "w", encoding="utf-8") as f:
        if error_records:
            f.write(f"REPORTE DE ARCHIVOS CON ERROR ({len(error_records)} archivos):\n")
            f.write("-" * 60 + "\n")
            for fn, err in error_records:
                f.write(f"{fn}\t-> {err}\n")
        else:
            f.write("0 errores encontrados. Todos los archivos cumplen estrictamente la regex y validaciones.\n")

    print(f"\n[PASO 1] Análisis de archivos:")
    print(f"  - Archivos analizados:  {len(valid_records) + len(error_records)}")
    print(f"  - Archivos válidos:     {len(valid_records)}")
    print(f"  - Archivos con errores: {len(error_records)} (registrados en {error_file.name})")

    if not valid_records:
        print("ERROR: No hay imágenes válidas para procesar.")
        return 1

    # 2. Agrupar por paciente
    patient_to_images = defaultdict(list)
    for rec in valid_records:
        patient_to_images[rec["paciente"]].append(rec)

    total_valid_imgs = len(valid_records)
    total_patients = len(patient_to_images)
    target_valid_imgs = 0.20 * total_valid_imgs
    min_valid_imgs = (0.20 - tolerance) * total_valid_imgs
    max_valid_imgs = (0.20 + tolerance) * total_valid_imgs

    print(f"\n[PASO 2] Agrupación por paciente:")
    print(f"  - Pacientes únicos:        {total_patients}")
    print(f"  - Total imágenes válidas:  {total_valid_imgs}")
    print(f"  - Meta validación (20%):   {target_valid_imgs:.1f} imágenes")
    print(f"  - Rango permitido (±2%):   [{min_valid_imgs:.1f}, {max_valid_imgs:.1f}] imágenes")

    # 3. Partición por paciente con búsqueda de semilla reproducible
    sorted_patients = sorted(list(patient_to_images.keys()))
    current_seed = initial_seed
    best_split = None
    max_attempts = 1000

    print(f"\n[PASO 3] Búsqueda de semilla para el split...")

    for attempt in range(max_attempts):
        rng = random.Random(current_seed)
        shuffled_patients = list(sorted_patients)
        rng.shuffle(shuffled_patients)

        valid_patients = []
        train_patients = []
        valid_img_count = 0

        # Asignación secuencial a valid buscando minimizar diferencia con 20%
        for p in shuffled_patients:
            c = len(patient_to_images[p])
            if valid_img_count + c <= target_valid_imgs:
                valid_patients.append(p)
                valid_img_count += c
            else:
                # Decidir si incluir este paciente acerca más al objetivo que detenerse
                if abs((valid_img_count + c) - target_valid_imgs) < abs(valid_img_count - target_valid_imgs):
                    valid_patients.append(p)
                    valid_img_count += c
                break

        valid_patient_set = set(valid_patients)
        train_patients = [p for p in shuffled_patients if p not in valid_patient_set]

        # Verificar rango de tolerancia
        if not (min_valid_imgs <= valid_img_count <= max_valid_imgs):
            current_seed += 1
            continue

        # Verificar representación de niveles discales en valid
        valid_levels_count = Counter()
        for p in valid_patients:
            for rec in patient_to_images[p]:
                valid_levels_count[rec["nivel"]] += 1

        # Todos los 5 niveles deben estar presentes y con conteo razonable (>= 5 imágenes por nivel)
        has_all_levels = all(valid_levels_count[lvl] >= 5 for lvl in DISC_LEVELS)
        if not has_all_levels:
            current_seed += 1
            continue

        # Si hay labels, verificar distribución de clases
        if labels_map:
            valid_classes_count = Counter()
            total_classes_count = Counter()
            for rec in valid_records:
                total_classes_count[rec.get("clase", "desconocida")] += 1
            for p in valid_patients:
                for rec in patient_to_images[p]:
                    valid_classes_count[rec.get("clase", "desconocida")] += 1

            # Cada clase debe tener ~20% (tolerancia ±5%)
            class_balance_ok = True
            for cls_name, total_cls in total_classes_count.items():
                cls_ratio = valid_classes_count[cls_name] / total_cls if total_cls > 0 else 0
                if not (0.15 <= cls_ratio <= 0.25):
                    class_balance_ok = False
                    break
            if not class_balance_ok:
                current_seed += 1
                continue

        # Encontró split óptimo
        best_split = {
            "seed": current_seed,
            "valid_patients": valid_patients,
            "train_patients": train_patients,
            "valid_img_count": valid_img_count,
            "valid_levels_count": valid_levels_count,
        }
        break

    if not best_split:
        print(f"ERROR: No se encontró split válido tras {max_attempts} semillas.")
        return 1

    selected_seed = best_split["seed"]
    valid_patients = best_split["valid_patients"]
    train_patients = best_split["train_patients"]
    valid_set = set(valid_patients)

    print(f"  -> ¡Semilla seleccionada con éxito: seed={selected_seed}!")
    print(f"  -> Pacientes asignados: {len(train_patients)} train, {len(valid_patients)} valid")

    # 4. Copiar archivos a carpetas train/ y valid/
    print(f"\n[PASO 4] Copiando archivos (estructura plana, nombres originales)...")
    train_records = []
    valid_split_records = []

    for p in train_patients:
        for rec in patient_to_images[p]:
            src_file = rec["path"]
            dest_file = train_dir / rec["filename"]
            shutil.copy2(src_file, dest_file)
            rec_copy = dict(rec)
            rec_copy["split"] = "train"
            train_records.append(rec_copy)

    for p in valid_patients:
        for rec in patient_to_images[p]:
            src_file = rec["path"]
            dest_file = valid_dir / rec["filename"]
            shutil.copy2(src_file, dest_file)
            rec_copy = dict(rec)
            rec_copy["split"] = "valid"
            valid_split_records.append(rec_copy)

    all_split_records = train_records + valid_split_records

    # 5. Generar manifest.csv
    print(f"\n[PASO 5] Generando manifest.csv...")
    manifest_file = output_dir / "manifest.csv"
    with open(manifest_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "paciente", "nivel", "slice", "split"]
        if labels_map:
            fieldnames.append("clase")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in sorted(all_split_records, key=lambda x: (x["split"], x["paciente"], x["filename"])):
            row = {
                "filename": rec["filename"],
                "paciente": rec["paciente"],
                "nivel": rec["nivel"],
                "slice": rec["slice"],
                "split": rec["split"],
            }
            if labels_map:
                row["clase"] = rec.get("clase", "")
            writer.writerow(row)
    print(f"  -> Guardado en: {manifest_file.resolve()}")

    # Generar README.md explicativo en output_dir
    readme_file = output_dir / "README.md"
    with open(readme_file, "w", encoding="utf-8") as f:
        f.write("# Dataset Lumbar ROI Split (Train / Valid)\n\n")
        f.write(f"- **Origen:** `{source_dir}`\n")
        f.write(f"- **Semilla utilizada:** `{selected_seed}`\n")
        f.write(f"- **Estrategia de partición:** Por paciente (0 fuga de datos)\n")
        f.write(f"- **Proporción objetivo:** 80% train / 20% valid (tolerancia ±2%)\n")
        f.write(f"- **Pacientes train:** {len(train_patients)}\n")
        f.write(f"- **Pacientes valid:** {len(valid_patients)}\n")
        f.write(f"- **Imágenes train:** {len(train_records)} ({len(train_records)/total_valid_imgs*100:.2f}%)\n")
        f.write(f"- **Imágenes valid:** {len(valid_split_records)} ({len(valid_split_records)/total_valid_imgs*100:.2f}%)\n\n")
        f.write("## Distribución por nivel discal\n\n")
        f.write("| Nivel | Train | Valid | Total | % Valid |\n")
        f.write("|---|---|---|---|---|\n")
        train_lvls = Counter(r["nivel"] for r in train_records)
        valid_lvls = Counter(r["nivel"] for r in valid_split_records)
        for lvl in DISC_LEVELS:
            t = train_lvls[lvl]
            v = valid_lvls[lvl]
            tot = t + v
            v_pct = (v / tot * 100) if tot > 0 else 0
            f.write(f"| {lvl} | {t} | {v} | {tot} | {v_pct:.1f}% |\n")

    # 6. VERIFICACIÓN OBLIGATORIA
    print(f"\n[PASO 6] VERIFICACIÓN OBLIGATORIA:")
    print("=" * 70)

    # 6.1 Intersección de pacientes
    intersection = set(train_patients).intersection(valid_set)
    print(f"1. Intersección de pacientes train/valid: {len(intersection)}")
    if len(intersection) != 0:
        print(f"¡ALERTA CRÍTICA! Se detectaron pacientes compartidos: {intersection}")
        print("Abortando proceso...")
        return 1
    print("   -> OK: Intersección = 0. No hay fuga de datos (data leakage).")

    # 6.2 Total de imágenes
    actual_train_imgs = len(list(train_dir.glob("*.png")))
    actual_valid_imgs = len(list(valid_dir.glob("*.png")))
    total_copied_imgs = actual_train_imgs + actual_valid_imgs
    print(f"2. Conteo de imágenes:")
    print(f"   - Origen válidas: {total_valid_imgs}")
    print(f"   - Train (disco):  {actual_train_imgs}")
    print(f"   - Valid (disco):  {actual_valid_imgs}")
    print(f"   - Suma copiada:   {total_copied_imgs}")
    assert total_copied_imgs == total_valid_imgs, "¡Error! Suma de imágenes no coincide con origen."
    print("   -> OK: Total de imágenes train + valid coincide exactamente con origen.")

    # 6.3 Pacientes y porcentajes
    pct_train_imgs = (actual_train_imgs / total_valid_imgs) * 100
    pct_valid_imgs = (actual_valid_imgs / total_valid_imgs) * 100
    pct_train_pts = (len(train_patients) / total_patients) * 100
    pct_valid_pts = (len(valid_patients) / total_patients) * 100
    print(f"3. Resumen por conjunto:")
    print(f"   - TRAIN: {len(train_patients)} pacientes ({pct_train_pts:.1f}%), {actual_train_imgs} imágenes ({pct_train_imgs:.2f}%)")
    print(f"   - VALID: {len(valid_patients)} pacientes ({pct_valid_pts:.1f}%), {actual_valid_imgs} imágenes ({pct_valid_imgs:.2f}%)")

    # 6.4 Conteo de imágenes por nivel discal
    train_levels_counter = Counter(r["nivel"] for r in train_records)
    valid_levels_counter = Counter(r["nivel"] for r in valid_split_records)
    print(f"4. Conteo de imágenes por nivel discal:")
    print(f"   {'Nivel':<10} | {'Train':<8} | {'Valid':<8} | {'Total':<8} | {'% en Valid':<10}")
    print(f"   {'-'*10}-|-{'-'*8}-|-{'-'*8}-|-{'-'*8}-|-{'-'*10}")
    for lvl in DISC_LEVELS:
        t_c = train_levels_counter[lvl]
        v_c = valid_levels_counter[lvl]
        tot_c = t_c + v_c
        v_pct = (v_c / tot_c * 100) if tot_c > 0 else 0
        print(f"   {lvl:<10} | {t_c:<8} | {v_c:<8} | {tot_c:<8} | {v_pct:.1f}%")

    # 6.5 Conteo por clase (si existe labels.csv)
    if labels_map:
        train_classes_counter = Counter(r.get("clase") for r in train_records)
        valid_classes_counter = Counter(r.get("clase") for r in valid_split_records)
        all_classes = sorted(list(set(list(train_classes_counter.keys()) + list(valid_classes_counter.keys()))))
        print(f"\n5. Conteo por clase:")
        print(f"   {'Clase':<20} | {'Train':<8} | {'Valid':<8} | {'Total':<8} | {'% en Valid':<10}")
        print(f"   {'-'*20}-|-{'-'*8}-|-{'-'*8}-|-{'-'*8}-|-{'-'*10}")
        for cls_name in all_classes:
            t_c = train_classes_counter[cls_name]
            v_c = valid_classes_counter[cls_name]
            tot_c = t_c + v_c
            v_pct = (v_c / tot_c * 100) if tot_c > 0 else 0
            print(f"   {cls_name:<20} | {t_c:<8} | {v_c:<8} | {tot_c:<8} | {v_pct:.1f}%")
    else:
        print("\n5. Conteo por clase: No aplica (no existe archivo labels.csv).")

    print("=" * 70)
    print("PROCESO COMPLETADO EXITOSAMENTE SIN ERRORES")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
