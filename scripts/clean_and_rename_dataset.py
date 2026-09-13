"""Clean up dataset directories:
1. Move positive sibling videos, frames, images, and labels from clear_negative to positive.
2. Remove all desert and forest clear_negative directories.
3. Rename all hard_negative directories, subdirectories, and files to negative.
4. Update manifest.json.
5. Re-run split generation for Option B.
"""

import os
import shutil
import json
from pathlib import Path


def main():
    print("=" * 60)
    print("STEP 1: Move positive siblings out of clear_negative")
    print("=" * 60)

    # 1. Raw videos
    clear_neg_raw = Path("data/raw/desert/clear_negative/rgb")
    pos_raw = Path("data/raw/desert/positive/rgb")
    if clear_neg_raw.exists():
        for f in sorted(clear_neg_raw.glob("desert_RGB_positive_*.mp4")):
            dst = pos_raw / f.name
            print(f"  Moving raw video: {f.name} -> {dst}")
            shutil.move(str(f), str(dst))

    # 2. Processed frames
    clear_neg_frames = Path("data/processed/frames/desert/clear_negative/rgb")
    pos_frames = Path("data/processed/frames/desert/positive/rgb")
    if clear_neg_frames.exists():
        for d in sorted(clear_neg_frames.glob("desert_RGB_positive_*")):
            dst = pos_frames / d.name
            print(f"  Moving frames dir: {d.name} -> {dst}")
            if dst.exists():
                shutil.rmtree(str(dst))
            shutil.move(str(d), str(dst))

    # 3. Processed images
    clear_neg_images = Path("data/processed/images/desert/clear_negative/rgb")
    pos_images = Path("data/processed/images/desert/positive/rgb")
    if clear_neg_images.exists():
        pos_images.mkdir(parents=True, exist_ok=True)
        for f in sorted(clear_neg_images.glob("desert_RGB_positive_*.jpg")):
            dst = pos_images / f.name
            shutil.move(str(f), str(dst))
        print(f"  Moved positive images from {clear_neg_images} -> {pos_images}")

    # 4. Processed labels
    clear_neg_labels = Path("data/processed/labels/desert/clear_negative/rgb")
    pos_labels = Path("data/processed/labels/desert/positive/rgb")
    if clear_neg_labels.exists():
        for d in sorted(clear_neg_labels.glob("desert_RGB_positive_*")):
            if d.is_dir():
                dst = pos_labels / d.name
                print(f"  Moving label dir: {d.name} -> {dst}")
                if dst.exists():
                    shutil.rmtree(str(dst))
                shutil.move(str(d), str(dst))
        for f in sorted(clear_neg_labels.glob("desert_RGB_positive_*.txt")):
            dst = pos_labels / f.name
            shutil.move(str(f), str(dst))
        print(f"  Moved positive flat label files to {pos_labels}")

    print("\n" + "=" * 60)
    print("STEP 2: Remove clear_negative directories")
    print("=" * 60)
    dirs_to_remove = [
        Path("videos/desert/clear_negative"),
        Path("data/raw/desert/clear_negative"),
        Path("data/raw/forest/clear_negative"),
        Path("data/processed/label_studio_videos/clear_negative"),
        Path("data/processed/frames/desert/clear_negative"),
        Path("data/processed/images/desert/clear_negative"),
        Path("data/processed/labels/desert/clear_negative"),
    ]
    for d in dirs_to_remove:
        if d.exists():
            print(f"  Removing directory: {d}")
            shutil.rmtree(str(d))
        else:
            print(f"  Already absent: {d}")

    print("\n" + "=" * 60)
    print("STEP 3: Rename hard_negative -> negative across files & folders")
    print("=" * 60)

    # A. videos/desert/hard_negative -> videos/desert/negative
    v_hard = Path("videos/desert/hard_negative")
    v_neg = Path("videos/desert/negative")
    if v_hard.exists():
        if v_neg.exists():
            shutil.rmtree(str(v_neg))
        v_hard.rename(v_neg)
        print(f"  Renamed folder: {v_hard} -> {v_neg}")
    if v_neg.exists():
        for f in sorted(v_neg.rglob("*hard_negative*")):
            if f.is_file():
                new_name = f.name.replace("hard_negative", "negative")
                f.rename(f.parent / new_name)
                print(f"    Renamed file: {f.name} -> {new_name}")

    # B. data/raw/desert/hard_negative -> data/raw/desert/negative
    r_des_hard = Path("data/raw/desert/hard_negative")
    r_des_neg = Path("data/raw/desert/negative")
    if r_des_hard.exists():
        if r_des_neg.exists():
            shutil.rmtree(str(r_des_neg))
        r_des_hard.rename(r_des_neg)
        print(f"  Renamed folder: {r_des_hard} -> {r_des_neg}")
    if r_des_neg.exists():
        for f in sorted(r_des_neg.rglob("*hard_negative*")):
            if f.is_file():
                new_name = f.name.replace("hard_negative", "negative")
                f.rename(f.parent / new_name)
                print(f"    Renamed file: {f.name} -> {new_name}")

    # C. data/raw/forest/hard_negative -> data/raw/forest/negative
    r_for_hard = Path("data/raw/forest/hard_negative")
    r_for_neg = Path("data/raw/forest/negative")
    if r_for_hard.exists():
        if r_for_neg.exists():
            shutil.rmtree(str(r_for_neg))
        r_for_hard.rename(r_for_neg)
        print(f"  Renamed folder: {r_for_hard} -> {r_for_neg}")
    if r_for_neg.exists():
        for f in sorted(r_for_neg.rglob("*hard_negative*")):
            if f.is_file():
                new_name = f.name.replace("hard_negative", "negative")
                f.rename(f.parent / new_name)
                print(f"    Renamed file: {f.name} -> {new_name}")

    # D. data/processed/label_studio_videos/hard_negative -> negative
    ls_hard = Path("data/processed/label_studio_videos/hard_negative")
    ls_neg = Path("data/processed/label_studio_videos/negative")
    if ls_hard.exists():
        if ls_neg.exists():
            shutil.rmtree(str(ls_neg))
        ls_hard.rename(ls_neg)
        print(f"  Renamed folder: {ls_hard} -> {ls_neg}")
    if ls_neg.exists():
        for f in sorted(ls_neg.rglob("*hard_negative*")):
            if f.is_file():
                new_name = f.name.replace("hard_negative", "negative")
                f.rename(f.parent / new_name)

    # E. data/processed/frames/desert/hard_negative -> negative
    fr_hard = Path("data/processed/frames/desert/hard_negative")
    fr_neg = Path("data/processed/frames/desert/negative")
    if fr_hard.exists():
        if fr_neg.exists():
            shutil.rmtree(str(fr_neg))
        fr_hard.rename(fr_neg)
        print(f"  Renamed folder: {fr_hard} -> {fr_neg}")
    if fr_neg.exists():
        # Rename subdirectories first
        for d in sorted(fr_neg.rglob("*hard_negative*")):
            if d.is_dir():
                new_dname = d.name.replace("hard_negative", "negative")
                d.rename(d.parent / new_dname)
        # Rename files
        for f in sorted(fr_neg.rglob("*hard_negative*")):
            if f.is_file():
                new_fname = f.name.replace("hard_negative", "negative")
                f.rename(f.parent / new_fname)

    # F. data/processed/images/desert/hard_negative -> negative
    im_hard = Path("data/processed/images/desert/hard_negative")
    im_neg = Path("data/processed/images/desert/negative")
    if im_hard.exists():
        if im_neg.exists():
            shutil.rmtree(str(im_neg))
        im_hard.rename(im_neg)
        print(f"  Renamed folder: {im_hard} -> {im_neg}")
    if im_neg.exists():
        for f in sorted(im_neg.rglob("*hard_negative*")):
            if f.is_file():
                new_name = f.name.replace("hard_negative", "negative")
                f.rename(f.parent / new_name)

    # G. data/processed/labels/desert/hard_negative -> negative
    lb_hard = Path("data/processed/labels/desert/hard_negative")
    lb_neg = Path("data/processed/labels/desert/negative")
    if lb_hard.exists():
        if lb_neg.exists():
            shutil.rmtree(str(lb_neg))
        lb_hard.rename(lb_neg)
        print(f"  Renamed folder: {lb_hard} -> {lb_neg}")
    if lb_neg.exists():
        # Rename subdirectories
        for d in sorted(lb_neg.rglob("*hard_negative*")):
            if d.is_dir():
                new_dname = d.name.replace("hard_negative", "negative")
                d.rename(d.parent / new_dname)
        # Rename files
        for f in sorted(lb_neg.rglob("*hard_negative*")):
            if f.is_file():
                new_fname = f.name.replace("hard_negative", "negative")
                f.rename(f.parent / new_fname)

    print("\n" + "=" * 60)
    print("STEP 4: Update data/manifest.json")
    print("=" * 60)
    manifest_path = Path("data/manifest.json")
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as fp:
            manifest = json.load(fp)

        for biome in ["desert", "forest", "altitude"]:
            if biome in manifest:
                b_data = manifest[biome]
                if "clear_negative" in b_data:
                    del b_data["clear_negative"]
                if "hard_negative" in b_data:
                    neg_val = b_data.pop("hard_negative")
                    neg_val["scenario"] = "negative"
                    b_data["negative"] = neg_val

        with open(manifest_path, "w", encoding="utf-8") as fp:
            json.dump(manifest, fp, indent=2)
        print("  Updated data/manifest.json to only contain 'positive' and 'negative'.")

    print("\n" + "=" * 60)
    print("CLEANUP AND RENAMING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
