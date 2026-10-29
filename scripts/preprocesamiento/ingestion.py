import os
import shutil
from pathlib import Path

class LumbarDatasetCleaner:
    """
    Clase encargada de filtrar y copiar únicamente las imágenes T2 Sagitales
    explorando directamente las carpetas de los pacientes.
    """
    def __init__(self, raw_dataset_path: str, filtered_dataset_path: str):
        self.raw_path = Path(raw_dataset_path)
        self.filtered_path = Path(filtered_dataset_path)

    def clean_and_structure_dataset(self):
        """Verifica la existencia del directorio e inicia el filtrado directo."""
        if not self.raw_path.exists():
            print(f"Error: La ruta de origen '{self.raw_path.resolve()}' no existe.")
            return

        self.filtered_path.mkdir(parents=True, exist_ok=True)
        count = 0

        # Iterar directamente sobre las carpetas de los pacientes (0001, 0002, etc.)
        for patient_folder in self.raw_path.iterdir():
            if patient_folder.is_dir():
                # Buscar todos los archivos .ima dentro de la carpeta del paciente (incluso en subcarpetas)
                ima_files = list(patient_folder.rglob("*.ima"))
                
                # Filtrar solo aquellos que corresponden a T2 Sagital por su nombre de archivo
                for file_path in ima_files:
                    file_name_upper = file_path.name.upper()
                    
                    # Criterio: Debe ser T2 Sagital y NO contener localizadores ni artefactos
                    if "T2" in file_name_upper and "SAG" in file_name_upper:
                        if "LOCALIZER" not in file_name_upper and "POSDISP" not in file_name_upper:
                            target_dir = self.filtered_path / patient_folder.name
                            target_dir.mkdir(parents=True, exist_ok=True)
                            
                            shutil.copy(file_path, target_dir / file_path.name)
                            count += 1

        print(f"\nProceso finalizado con éxito.")
        print(f"Archivos T2 Sagital conservados: {count}")
        print(f"Ubicación de guardado: {self.filtered_path.resolve()}")

# --- CONFIGURACIÓN DE RUTAS ---
if __name__ == "__main__":
    RUTA_ORIGEN = "./t2_sagital_unfiltered"
    RUTA_DESTINO = "./t2_sagital_filtered"

    cleaner = LumbarDatasetCleaner(
        raw_dataset_path=RUTA_ORIGEN, 
        filtered_dataset_path=RUTA_DESTINO
    )
    cleaner.clean_and_structure_dataset()