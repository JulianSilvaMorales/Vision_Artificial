import random
import shutil
from pathlib import Path

def create_seed_dataset(input_roi_folder: str, seed_output_folder: str, samples_per_level: int = 70):
    """
    Selecciona aleatoriamente parches de cada nivel intervertebral
    para conformar el lote semilla de anotación manual (~350 parches).
    """
    input_path = Path(input_roi_folder)
    seed_path = Path(seed_output_folder)
    seed_path.mkdir(parents=True, exist_ok=True)
    
    total_copied = 0
    levels = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
    
    for level in levels:
        level_dir = input_path / level
        if not level_dir.exists():
            continue
            
        all_patches = list(level_dir.glob("*.png"))
        selected_patches = random.sample(all_patches, min(samples_per_level, len(all_patches)))
        
        for patch_path in selected_patches:
            dest = seed_path / f"{level}_{patch_path.name}"
            shutil.copy(patch_path, dest)
            total_copied += 1
            
    print(f"Dataset semilla creado con {total_copied} parches en: {seed_path.resolve()}")

if __name__ == "__main__":
    create_seed_dataset(
        input_roi_folder="./t2_disc_roi_patches",
        seed_output_folder="./seed_dataset_unlabeled",
        samples_per_level=70
    )