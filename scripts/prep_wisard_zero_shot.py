"""Extract real-world WiSARD test episode and provide zero-shot evaluation harness."""

import os
import zipfile
from pathlib import Path

WISARD_ZIP = Path("D:/Downloads/WiSARDv1.zip")
OUTPUT_DIR = Path("data/wisard_test")


def extract_wisard_snippet(max_frames: int = 240):
    """Extract a 240-frame snippet from the Mount Baker snow sequence."""
    if not WISARD_ZIP.exists():
        print(f"WiSARD zip not found at {WISARD_ZIP}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vis_dir = OUTPUT_DIR / "vis" / "images"
    vis_labels_dir = OUTPUT_DIR / "vis" / "labels"
    ir_dir = OUTPUT_DIR / "ir" / "images"
    vis_dir.mkdir(parents=True, exist_ok=True)
    vis_labels_dir.mkdir(parents=True, exist_ok=True)
    ir_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening {WISARD_ZIP} to extract {max_frames} frames of Baker snow sequence...")
    with zipfile.ZipFile(WISARD_ZIP, "r") as z:
        vis_imgs = sorted([n for n in z.namelist() if n.startswith("220109_Baker_Enterprise_VIS_1/") and n.endswith(".jpg")])
        ir_imgs = sorted([n for n in z.namelist() if n.startswith("220109_Baker_Enterprise_IR_1/") and n.endswith(".jpg")])

        n_frames = min(max_frames, len(vis_imgs), len(ir_imgs))
        print(f"Found {len(vis_imgs)} VIS and {len(ir_imgs)} IR images. Extracting {n_frames} pairs...")

        for i in range(n_frames):
            v_name = vis_imgs[i]
            ir_name = ir_imgs[i]
            base_idx = f"baker_snow_{i:04d}"

            # Extract VIS
            v_bytes = z.read(v_name)
            (vis_dir / f"{base_idx}.jpg").write_bytes(v_bytes)

            # Extract IR
            ir_bytes = z.read(ir_name)
            (ir_dir / f"{base_idx}.jpg").write_bytes(ir_bytes)

            # Extract label
            txt_name = v_name[:-4] + ".txt"
            if txt_name in z.namelist():
                lbl_bytes = z.read(txt_name)
                (vis_labels_dir / f"{base_idx}.txt").write_bytes(lbl_bytes)
            else:
                (vis_labels_dir / f"{base_idx}.txt").write_bytes(b"")

    print(f"Extraction complete! Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    extract_wisard_snippet(max_frames=240)
