"""Physically Decoupled Thermal Infrared Synthesis for Snow Biomes.

Decouples the pseudo-thermal signal from simple RGB brightness:
1. Background Snow: high RGB albedo mapped to cold sub-zero thermal radiance (dark navy/purple).
2. Exposed Rocks/Clutter: mapped to intermediate thermal band (cyan/green/dull tones).
3. Human Target: segmented body shape with realistic thermal gradient (hot core, tapering clothing, optical PSF bloom).
4. Microbolometer Sensor Noise: subtle realistic high-frequency sensor grain.
"""

from pathlib import Path
import subprocess
import cv2
import numpy as np
import imageio_ffmpeg


def generate_snow_thermal_frame(
    frame_bgr: np.ndarray,
    lut: np.ndarray,
    person_box: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    h_img, w_img = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    # 1. Snow terrain classification
    snow_mask = (v > 145) & (s < 55)

    # 2. Thermal radiance map (0-255 index into LUT)
    thermal_val = np.zeros((h_img, w_img), dtype=np.float32)

    # Cold snow: [16, 36] (deep navy / dark purple in Melo LUT)
    v_norm = np.clip((v.astype(np.float32) - 145.0) / 110.0, 0.0, 1.0)
    thermal_val[snow_mask] = 16.0 + v_norm[snow_mask] * 18.0

    # Exposed rocks / bare earth: [45, 82] (cyan / green in Melo LUT)
    non_snow = ~snow_mask
    rock_norm = np.clip((145.0 - v.astype(np.float32)) / 145.0, 0.0, 1.0)
    thermal_val[non_snow] = 45.0 + rock_norm[non_snow] * 35.0

    current_person_mask = None

    # 3. Target thermal insertion if person_box is provided and within frame
    if person_box is not None:
        bx, by, bw, bh = person_box
        x1 = max(0, bx)
        y1 = max(0, by)
        x2 = min(w_img, bx + bw)
        y2 = min(h_img, by + bh)

        cw = x2 - x1
        ch = y2 - y1

        if cw > 15 and ch > 15:
            crop = frame_bgr[y1:y2, x1:x2]
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # Robust Otsu segmentation for silhouette
            _, thresh = cv2.threshold(gray_crop, 0, 1.0, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            seg = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel).astype(np.float32)

            # Thermal heat gradient inside body
            yy, xx = np.ogrid[:ch, :cw]
            head_grad = np.clip(1.0 - (yy / max(1.0, ch * 0.35)), 0.0, 1.0)
            torso_grad = np.clip(1.0 - np.abs(yy - ch * 0.45) / max(1.0, ch * 0.45), 0.0, 1.0)
            edge_taper = np.clip((ch - yy) / max(1.0, ch * 0.12), 0.0, 1.0)

            # Heat profile: Head ~ 225, Torso ~ 195, Lower ~ 170
            temp_profile = (165.0 + 35.0 * torso_grad + 30.0 * head_grad) * edge_taper
            heat_map = temp_profile * seg

            # Optical Point Spread Function (PSF) bloom
            heat_bloom = cv2.GaussianBlur(heat_map, (7, 7), 1.8)
            bloom_mask = heat_bloom > 30.0

            sub_t = thermal_val[y1:y2, x1:x2]
            sub_t[bloom_mask] = np.maximum(sub_t[bloom_mask], heat_bloom[bloom_mask])
            thermal_val[y1:y2, x1:x2] = sub_t
            current_person_mask = seg

    # 4. Microbolometer sensor noise
    noise = np.random.normal(0, 1.6, (h_img, w_img)).astype(np.float32)
    thermal_val = np.clip(thermal_val + noise, 0, 255).astype(np.uint8)

    # 5. Colormap via Melo LUT
    thermal_bgr = lut[thermal_val]
    return thermal_bgr, current_person_mask


def process_snow_video(
    in_path: Path,
    out_path: Path,
    lut_path: Path,
    initial_bbox: tuple[int, int, int, int] | None = None,
):
    lut = np.load(str(lut_path))
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {in_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "bgr24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(out_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    tracker = None
    if initial_bbox is not None:
        ok, f0 = cap.read()
        if not ok:
            cap.release()
            return
        tracker = cv2.TrackerMIL_create()
        tracker.init(f0, initial_bbox)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_box = None
            if tracker is not None:
                if frame_idx == 0:
                    current_box = initial_bbox
                else:
                    success, updated_box = tracker.update(frame)
                    if success:
                        bx, by, bw, bh = map(int, updated_box)
                        if bx < w - 20 and by < h - 20 and bx + bw > 20 and by + bh > 20:
                            current_box = (bx, by, bw, bh)
                        else:
                            current_box = None
                    else:
                        current_box = None

            thermal_frame, _ = generate_snow_thermal_frame(frame, lut, person_box=current_box)
            proc.stdin.write(thermal_frame.tobytes())
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames...")
    finally:
        cap.release()
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    print(f"Video saved successfully to {out_path} ({frame_idx} frames).")


if __name__ == "__main__":
    import sys
    in_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\Downloads\Snow_RGB_positive_1.mp4")
    out_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(r"D:\Downloads\Snow_Thermal_decoupled_test_1.mp4")
    lut_f = Path("scripts/melobytes_medium_low_lut.npy")
    bbox = (945, 180, 75, 115) if "positive_1" in in_file.name else None
    process_snow_video(in_file, out_file, lut_f, initial_bbox=bbox)
