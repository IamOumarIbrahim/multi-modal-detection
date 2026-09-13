from pathlib import Path
import datetime

p_rgb = Path("C:/Dev/repos/Public repos/DMS-Eval/runs/detect/runs/detect/yolo11n_rgb_60e/results.csv")
p_th = Path("C:/Dev/repos/Public repos/DMS-Eval/runs/detect/runs/detect/yolo11n_thermal_60e/results.csv")

if p_th.exists():
    lines = p_th.read_text().strip().splitlines()
    if len(lines) > 1:
        header = [h.strip() for h in lines[0].split(",")]
        last = [v.strip() for v in lines[-1].split(",")]
        row = dict(zip(header, last))
        cur_epoch = int(row.get("epoch", 0))
        time_sec = float(row.get("time", 0))
        sec_per_epoch = time_sec / max(1, cur_epoch)
        rem_sec = (60 - cur_epoch) * sec_per_epoch
        eta = datetime.datetime.now() + datetime.timedelta(seconds=rem_sec)
        print(f"[Phase 2/2] YOLO11n-Thermal: Epoch {cur_epoch}/60")
        print(f"Metrics: mAP50 = {float(row.get('metrics/mAP50(B)', 0)):.4f}, Prec = {float(row.get('metrics/precision(B)', 0)):.4f}, Rec = {float(row.get('metrics/recall(B)', 0)):.4f}")
        print(f"Speed: {sec_per_epoch:.1f}s/epoch | Remaining: {rem_sec/3600:.2f} hours")
        print(f"Overall Completion ETA (Dubai): {eta.strftime('%I:%M %p')}")
    else:
        print("[Phase 2/2] YOLO11n-Thermal: In Epoch 1/60")
elif p_rgb.exists():
    lines = p_rgb.read_text().strip().splitlines()
    if len(lines) > 1:
        header = [h.strip() for h in lines[0].split(",")]
        last = [v.strip() for v in lines[-1].split(",")]
        row = dict(zip(header, last))
        cur_epoch = int(row.get("epoch", 0))
        time_sec = float(row.get("time", 0))
        sec_per_epoch = time_sec / max(1, cur_epoch)
        rem_rgb_sec = (60 - cur_epoch) * sec_per_epoch
        rem_th_sec = 60 * sec_per_epoch
        total_rem_sec = rem_rgb_sec + rem_th_sec
        eta = datetime.datetime.now() + datetime.timedelta(seconds=total_rem_sec)
        print(f"[Phase 1/2] YOLO11n-RGB: Epoch {cur_epoch}/60")
        print(f"Metrics: mAP50 = {float(row.get('metrics/mAP50(B)', 0)):.4f}, Prec = {float(row.get('metrics/precision(B)', 0)):.4f}, Rec = {float(row.get('metrics/recall(B)', 0)):.4f}")
        print(f"Speed: {sec_per_epoch:.1f}s/epoch | Remaining: {total_rem_sec/3600:.2f} hours")
        print(f"Overall Completion ETA (Dubai): {eta.strftime('%I:%M %p')}")
else:
    print("Training outputs not found yet.")
