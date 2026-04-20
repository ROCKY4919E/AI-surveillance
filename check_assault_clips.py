# check_folder.py
from pathlib import Path

base = Path('data/ucf_crime_subset/train/Assault')
print(f"Exists: {base.exists()}")
print(f"Contents:")
for item in list(base.iterdir())[:20]:
    print(f"  {item.name}  (is_dir={item.is_dir()})")

# Check actual file extensions
all_files = list(base.rglob('*.*'))
exts = set(f.suffix for f in all_files)
print(f"\nFile extensions found: {exts}")
print(f"Total files: {len(all_files)}")
if all_files:
    print(f"Sample filenames:")
    for f in all_files[:5]:
        print(f"  {f.name}")