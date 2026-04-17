import shutil
from pathlib import Path

# --- Config ---
UCF_ROOT = Path(r"F:\IIMSTC\VScode\AI Surveillance\data\ucf_crime_subset")
OUT_ROOT  = Path(r"F:\IIMSTC\VScode\AI Surveillance\data\accident")

# Categories to treat as "accident"
ACCIDENT_CLASSES = ['RoadAccidents', 'Fighting', 'Assault', 'Explosion']
NORMAL_CLASSES   = ['NormalVideos']

def copy_images(src_folder, dst_folder):
    dst_folder.mkdir(parents=True, exist_ok=True)
    copied = 0
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        for img in src_folder.glob(ext):
            shutil.copy(img, dst_folder / img.name)
            copied += 1
    return copied

for split in ['Train', 'Test']:
    out_split = 'train' if split == 'Train' else 'val'

    # Copy accident images
    total_accident = 0
    for cls in ACCIDENT_CLASSES:
        src = UCF_ROOT / split / cls
        if src.exists():
            n = copy_images(src, OUT_ROOT / out_split / 'accident')
            print(f"  {split}/{cls} -> {n} images")
            total_accident += n

    # Copy normal images
    total_normal = 0
    for cls in NORMAL_CLASSES:
        src = UCF_ROOT / split / cls
        if src.exists():
            n = copy_images(src, OUT_ROOT / out_split / 'normal')
            print(f"  {split}/{cls} -> {n} images")
            total_normal += n

    print(f"\n{out_split} -> accident: {total_accident}, normal: {total_normal}\n")

print("Done. Dataset ready at:", OUT_ROOT)