import os
import shutil
from pathlib import Path

class MidsagittalSliceFilter:
    """
    Clase para filtrar los slices periféricos de una resonancia magnética.
    Conserva únicamente el 40% central del volumen sagital (Midsagittal region),
    donde el canal medular y la región L1-S1 son claramente visibles.
    """
    def __init__(self, start_ratio: float = 0.30, end_ratio: float = 0.70):
        # start_ratio = 0.30 descarta el 30% inicial
        # end_ratio = 0.70 descarta el 30% final
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio

    def filter_patient_slices(self, input_folder: str, output_folder: str):
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        if not input_path.exists():
            print(f"Error: La ruta de origen '{input_path.resolve()}' no existe.")
            return

        output_path.mkdir(parents=True, exist_ok=True)
        total_procesados = 0
        total_descartados = 0

        # Iterar sobre las carpetas de cada paciente (0001, 0002, etc.)
        for patient_folder in input_path.iterdir():
            if patient_folder.is_dir():
                # Obtener todos los archivos .ima y ordenarlos numéricamente por su nombre/secuencia
                ima_files = sorted(list(patient_folder.glob("*.ima")), key=lambda x: x.name)
                num_slices = len(ima_files)

                if num_slices == 0:
                    continue

                # Calcular los índices del rango central
                idx_start = int(num_slices * self.start_ratio)
                idx_end = int(num_slices * self.end_ratio)

                # Si hay muy pocos slices, asegurar que al menos se conserve el slice del centro
                if idx_start >= idx_end:
                    idx_start = max(0, (num_slices // 2) - 1)
                    idx_end = min(num_slices, (num_slices // 2) + 2)

                # Definir la carpeta de salida para el paciente
                target_patient_dir = output_path / patient_folder.name
                target_patient_dir.mkdir(parents=True, exist_ok=True)

                # Copiar únicamente los cortes centrales seleccionados
                for idx, file_path in enumerate(ima_files):
                    if idx_start <= idx < idx_end:
                        shutil.copy(file_path, target_patient_dir / file_path.name)
                        total_procesados += 1
                    else:
                        total_descartados += 1

        print("\n--- DEPURACIÓN DE SLICES MIDSAGITALES COMPLETADA ---")
        print(f"Slices centrales conservados (útiles L1-S1): {total_procesados}")
        print(f"Slices periféricos descartados (ruido óseo/muscular): {total_descartados}")
        print(f"Ubicación del dataset curado: {output_path.resolve()}")

# --- EJECUCIÓN DEL FILTRO DE SLICES ---
if __name__ == "__main__":
    # Carpeta donde dejaste las T2 Sagitales en el paso anterior
    CARPETA_T2_SAGITAL = "./t2_sagital_filtered"
    
    # Nombre de la nueva carpeta para el dataset depurado y curado
    CARPETA_DATASET_CURADO = "./t2_midsagittal_curated"

    filter_tool = MidsagittalSliceFilter(start_ratio=0.30, end_ratio=0.70)
    filter_tool.filter_patient_slices(
        input_folder=CARPETA_T2_SAGITAL,
        output_folder=CARPETA_DATASET_CURADO
    )