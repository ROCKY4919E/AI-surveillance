# prepare_assault.py
import os, shutil, random
from pathlib import Path
from collections import defaultdict

DATASET_ROOT = 'data/ucf_crime_subset'
OUTPUT_DIR   = 'data/assault'
CRIME_FOLDER = 'Assault'
CRIME_LABEL  = 'assault'
VAL_RATIO    = 0.15
RANDOM_SEED  = 42
EXTENSIONS   = {'.png', '.jpg', '.jpeg'}

random.seed(RANDOM_SEED)

def get_clip_id(filename):
    """
    Assault001_x264_1000.png  ->  Assault001_x264
    Strips trailing frame number after last underscore.
    """
    stem  = Path(filename).stem          # 'Assault001_x264_1000'
    parts = stem.rsplit('_', 1)          # ['Assault001_x264', '1000']
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else stem

def collect_by_clip(folder):
    clips = defaultdict(list)
    for f in Path(folder).rglob('*'):
        if f.suffix.lower() in EXTENSIONS:
            clips[get_clip_id(f.name)].append(f)
    return clips

def copy_frames(frame_list, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for f in frame_list:
        shutil.copy2(f, Path(out_dir) / f.name)
    return len(frame_list)

def split_clips(clips_dict, val_ratio):
    clip_ids = list(clips_dict.keys())
    random.shuffle(clip_ids)
    split        = int(len(clip_ids) * (1 - val_ratio))
    train_frames = [f for cid in clip_ids[:split] for f in clips_dict[cid]]
    val_frames   = [f for cid in clip_ids[split:] for f in clips_dict[cid]]
    return train_frames, val_frames

def select_normal_by_frame_count(normal_clips_all, target_frames):
    clip_ids = list(normal_clips_all.keys())
    random.shuffle(clip_ids)
    selected, total = {}, 0
    for cid in clip_ids:
        if total >= target_frames:
            break
        selected[cid] = normal_clips_all[cid]
        total += len(normal_clips_all[cid])
    return selected

def prepare():
    if Path(OUTPUT_DIR).exists():
        shutil.rmtree(OUTPUT_DIR)
        print(f"Removed old {OUTPUT_DIR}/")

    for split_name in ['train', 'test']:
        split_src = Path(DATASET_ROOT) / split_name
        print(f"\nProcessing {split_name}/")

        assault_clips  = collect_by_clip(split_src / 'Assault')
        assault_total  = sum(len(v) for v in assault_clips.values())
        print(f"  Assault clips : {len(assault_clips)}  ({assault_total} frames)")

        normal_clips_all = collect_by_clip(split_src / 'NormalVideos')
        normal_clips     = select_normal_by_frame_count(normal_clips_all, assault_total)
        normal_total     = sum(len(v) for v in normal_clips.values())
        print(f"  Normal clips  : {len(normal_clips)}  ({normal_total} frames)")

        if split_name == 'train':
            for clips, label in [(assault_clips, 'assault'), (normal_clips, 'normal')]:
                tr_frames, val_frames = split_clips(clips, VAL_RATIO)
                n_tr  = copy_frames(tr_frames,  f'{OUTPUT_DIR}/train/{label}')
                n_val = copy_frames(val_frames, f'{OUTPUT_DIR}/val/{label}')
                print(f"  [{label}] train={n_tr}  val={n_val}")
        else:
            for clips, label in [(assault_clips, 'assault'), (normal_clips, 'normal')]:
                all_frames = [f for v in clips.values() for f in v]
                n_te = copy_frames(all_frames, f'{OUTPUT_DIR}/test/{label}')
                print(f"  [{label}] test={n_te}")

    print(f"\nDone. Dataset ready at: {OUTPUT_DIR}/")

if __name__ == '__main__':
    prepare()