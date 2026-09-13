"""Generate compact 3-biome multimodal comparison figure for MMSAR manuscript.

Creates a 3-row, 2-column composite image showing Visible RGB and Pseudo-Thermal
channels across Arid Desert, Temperate Forest, and Snow/Alpine biomes,
explicitly highlighting the high-albedo snow brightness inversion artifact.
"""

from pathlib import Path
import cv2
import numpy as np


def draw_yolo_box(img: np.ndarray, bbox: list[float], color=(0, 255, 0), thickness=2, label="Person"):
    """Draw YOLO normalized bbox [xc, yc, w, h] on image."""
    h, w = img.shape[:2]
    xc, yc, bw, bh = bbox
    x1 = int((xc - bw / 2.0) * w)
    y1 = int((yc - bh / 2.0) * h)
    x2 = int((xc + bw / 2.0) * w)
    y2 = int((yc + bh / 2.0) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(
            img, label, (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
        )


def main():
    # 1. Desert
    d_rgb_p = Path("docs/manuscript/figures/desert_rgb_annotated.png")
    d_th_p = Path("docs/manuscript/figures/desert_th_annotated.png")
    d_rgb = cv2.imread(str(d_rgb_p))
    d_th = cv2.imread(str(d_th_p))

    # 2. Forest: forest_RGB_positive_1_left / frame 64
    f_rgb_p = Path("data/processed/frames/forest/positive/rgb/forest_RGB_positive_1_left/frame_000064.png")
    f_th_p = Path("data/processed/frames/forest/positive/thermal/forest_Thermal_positive_1_left/frame_000064.png")
    f_lbl_p = Path("data/processed/labels/forest/positive/rgb/forest_RGB_positive_1_left_frame_000064.txt")
    f_rgb = cv2.imread(str(f_rgb_p))
    f_th = cv2.imread(str(f_th_p))
    if f_lbl_p.exists() and f_lbl_p.stat().st_size > 0:
        box = [float(x) for x in f_lbl_p.read_text().strip().split()[1:]]
        draw_yolo_box(f_rgb, box, color=(0, 255, 0), thickness=2, label="Person")
        draw_yolo_box(f_th, box, color=(0, 255, 255), thickness=2, label="Person")

    # 3. Snow: snow_RGB_positive_10_left / frame 99
    s_rgb_p = Path("data/processed/frames/snow/positive/rgb/snow_RGB_positive_10_left/frame_000099.png")
    s_th_p = Path("data/processed/frames/snow/positive/thermal/snow_Thermal_positive_10_left/frame_000099.png")
    s_lbl_p = Path("data/processed/labels/snow/positive/rgb/snow_RGB_positive_10_left_frame_000099.txt")
    s_rgb = cv2.imread(str(s_rgb_p))
    s_th = cv2.imread(str(s_th_p))
    if s_lbl_p.exists() and s_lbl_p.stat().st_size > 0:
        box = [float(x) for x in s_lbl_p.read_text().strip().split()[1:]]
        draw_yolo_box(s_rgb, box, color=(0, 255, 0), thickness=2, label="Person")
        draw_yolo_box(s_th, box, color=(0, 255, 255), thickness=2, label="Person (Cool)")

    # Resize all to uniform 360x360 for compact high-density rendering
    dim = (360, 360)
    d_rgb_r = cv2.resize(d_rgb, dim)
    d_th_r = cv2.resize(d_th, dim)
    f_rgb_r = cv2.resize(f_rgb, dim)
    f_th_r = cv2.resize(f_th, dim)
    s_rgb_r = cv2.resize(s_rgb, dim)
    s_th_r = cv2.resize(s_th, dim)

    def add_banner(img, title, subtitle=None):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (dim[0], 28 if not subtitle else 42), (25, 25, 25), -1)
        cv2.putText(out, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        if subtitle:
            cv2.putText(out, subtitle, (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 220, 255), 1, cv2.LINE_AA)
        return out

    d_rgb_r = add_banner(d_rgb_r, "Desert: Visible RGB", "Midday illumination, rock clutter")
    d_th_r = add_banner(d_th_r, "Desert: Pseudo-TIR", "Thermal crossover on heated substrate")

    f_rgb_r = add_banner(f_rgb_r, "Forest: Visible RGB", "Canopy shade & foliage dropout")
    f_th_r = add_banner(f_th_r, "Forest: Pseudo-TIR", "Vegetation penetration")

    s_rgb_r = add_banner(s_rgb_r, "Snow: Visible RGB", "High-albedo snow glare")
    s_th_r = add_banner(s_th_r, "Snow: Pseudo-TIR", "Artifact: Snow appears hot, target cool")

    row1 = np.hstack([d_rgb_r, d_th_r])
    row2 = np.hstack([f_rgb_r, f_th_r])
    row3 = np.hstack([s_rgb_r, s_th_r])

    border = np.zeros((4, row1.shape[1], 3), dtype=np.uint8) + 200
    composite = np.vstack([row1, border, row2, border, row3])

    out_path = Path("docs/manuscript/figures/figure_2_biome_samples.png")
    cv2.imwrite(str(out_path), composite, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"Saved composite figure to: {out_path} ({composite.shape})")


if __name__ == "__main__":
    main()
