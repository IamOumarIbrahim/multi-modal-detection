#   MMSAR - Label Studio video tracks to per-frame YOLO labels
#        Oumar Ibrahim U22200741

# label studio gives keyframes in percent (0..100) with 1-indexed frame numbers
# between two keyframes the box is linearly interpolated


def keyframe_box(kf):
    return {
        "x": float(kf["x"]),
        "y": float(kf["y"]),
        "width": float(kf["width"]),
        "height": float(kf["height"]),
    }


def frame_number(kf):
    return kf.get("frame", 1)


# returns {frame: box or None} for frames 1..total_frames
def interpolate_video_sequence(sequence, total_frames):
    boxes = {}
    if len(sequence) == 0:
        for f in range(1, total_frames + 1):
            boxes[f] = None
        return boxes

    kfs = sorted(sequence, key=frame_number)
    first = kfs[0]
    last = kfs[-1]

    for f in range(1, total_frames + 1):
        # before the first keyframe nothing is labelled
        if f < first["frame"]:
            boxes[f] = None
            continue

        # after the last keyframe hold the last box (if the track is still on)
        if f >= last["frame"]:
            if last.get("enabled", True):
                boxes[f] = keyframe_box(last)
            else:
                boxes[f] = None
            continue

        exact = None
        for k in kfs:
            if k.get("frame") == f:
                exact = k
                break
        if exact is not None:
            if exact.get("enabled", True):
                boxes[f] = keyframe_box(exact)
            else:
                boxes[f] = None
            continue

        # find the two keyframes around f
        prev = first
        nxt = last
        for i in range(len(kfs) - 1):
            if kfs[i]["frame"] < f < kfs[i + 1]["frame"]:
                prev = kfs[i]
                nxt = kfs[i + 1]
                break

        # a disabled keyframe means the person left the frame
        if not prev.get("enabled", True):
            boxes[f] = None
            continue

        f0 = prev["frame"]
        f1 = nxt["frame"]
        if f0 == f1:
            boxes[f] = keyframe_box(prev)
        else:
            t = (f - f0) / float(f1 - f0)
            box = {}
            for key in ["x", "y", "width", "height"]:
                a = float(prev[key])
                b = float(nxt[key])
                box[key] = a + t * (b - a)
            boxes[f] = box

    return boxes


# percent top-left box -> "0 xc yc w h" normalised, clamped inside the image
def box_to_yolo_line(box, class_id=0):
    x = box["x"] / 100.0
    y = box["y"] / 100.0
    w = box["width"] / 100.0
    h = box["height"] / 100.0

    xc = max(0.0001, min(0.9999, x + w / 2.0))
    yc = max(0.0001, min(0.9999, y + h / 2.0))
    w = max(0.0001, min(1.0, w))
    h = max(0.0001, min(1.0, h))
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"


# first video track in a task, or [] if the clip has no person
def task_sequence(task):
    anns = task.get("annotations", [])
    if len(anns) == 0 or "result" not in anns[0]:
        return []
    for res in anns[0]["result"]:
        if res.get("type") == "videorectangle" and "value" in res:
            return res["value"].get("sequence", [])
    return []
