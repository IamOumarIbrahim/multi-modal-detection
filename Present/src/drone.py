"""A quadcopter, rendered here (numpy ray-marching, no external images) for the Mixed Media cover.

Writes assets/drone_render.png: shaded grey levels (RGB) + alpha (body/arms/motors/gear opaque, propeller discs
semi-transparent). The cover prints it as a halftone cut-out sticker.

Run:  python -I drone.py
"""
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent.parent / "assets" / "drone_render.png"


def norm(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def sd_round_box(p, b, r):
    q = np.abs(p) - b
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0) - r


def sd_capsule(p, a, b, r):
    pa, ba = p - a, b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0, 1)[..., None]
    return np.linalg.norm(pa - ba * h, axis=-1) - r


def sd_cyl(p, c, r, h):          # vertical cylinder centred at c, radius r, half height h
    q = p - c
    d = np.stack([np.linalg.norm(q[..., [0, 2]], axis=-1) - r, np.abs(q[..., 1]) - h], -1)
    return np.minimum(d.max(-1), 0) + np.linalg.norm(np.maximum(d, 0), axis=-1)


ARM = 0.95
MOTORS = [np.array([sx * ARM * 0.72, 0.02, sz * ARM * 0.70]) for sx in (-1, 1) for sz in (-1, 1)]
PROP_Y, PROP_R = 0.16, 0.36


def scene(p):
    d = sd_round_box(p, np.array([0.26, 0.07, 0.40]), 0.07)                              # body
    d = np.minimum(d, sd_round_box(p - np.array([0, 0.11, -0.05]), np.array([0.16, 0.03, 0.22]), 0.04))  # top shell
    for m in MOTORS:
        d = np.minimum(d, sd_capsule(p, np.array([0, 0.0, 0]), m * np.array([1, 0, 1]), 0.045))       # arms
        d = np.minimum(d, sd_cyl(p, m + np.array([0, 0.05, 0]), 0.085, 0.06))                       # motors
        d = np.minimum(d, sd_cyl(p, m + np.array([0, 0.13, 0]), 0.025, 0.03))                       # shafts
    for sx in (-1, 1):                                                                              # landing gear
        d = np.minimum(d, sd_capsule(p, np.array([sx * 0.30, -0.34, -0.42]), np.array([sx * 0.30, -0.34, 0.42]), 0.025))
        for sz in (-0.25, 0.25):
            d = np.minimum(d, sd_capsule(p, np.array([sx * 0.16, -0.05, sz]), np.array([sx * 0.30, -0.34, sz * 1.15]), 0.02))
    d = np.minimum(d, np.linalg.norm(p - np.array([0, -0.13, 0.40]), axis=-1) - 0.085)              # camera gimbal
    d = np.minimum(d, sd_cyl(p, np.array([0, -0.06, 0.40]), 0.03, 0.05))
    return d


def normal(p):
    e = 1e-3
    ex = np.array([e, 0, 0]); ey = np.array([0, e, 0]); ez = np.array([0, 0, e])
    n = np.stack([scene(p + ex) - scene(p - ex), scene(p + ey) - scene(p - ey), scene(p + ez) - scene(p - ez)], -1)
    return norm(n)


def render(w=1600, h=1200):
    eye = np.array([2.35, 1.75, 2.75])
    tgt = np.array([0.0, -0.02, 0.0])
    fw = norm(tgt - eye)
    rt = norm(np.cross(fw, np.array([0, 1.0, 0])))
    up = np.cross(rt, fw)
    f = 1 / np.tan(np.radians(36) / 2)
    ys, xs = np.mgrid[0:h, 0:w]
    u = (xs - w / 2) / (h / 2)
    v = -(ys - h / 2) / (h / 2)
    rd = norm(u[..., None] * rt + v[..., None] * up + f * fw).reshape(-1, 3)
    ro = np.broadcast_to(eye, rd.shape)
    t = np.zeros(len(rd))
    hit = np.zeros(len(rd), bool)
    act = np.arange(len(rd))
    for _ in range(110):
        p = ro[act] + rd[act] * t[act, None]
        d = scene(p)
        t[act] += d * 0.9
        done = d < 1e-3
        hit[act[done]] = True
        act = act[~done & (t[act] < 8)]
        if not len(act):
            break
    shade = np.zeros(len(rd))
    idx = np.nonzero(hit)[0]
    p = ro[idx] + rd[idx] * t[idx, None]
    n = normal(p)
    L = norm(np.array([-0.45, 1.0, 0.35]))
    lam = np.clip(n @ L, 0, 1)
    ao = np.ones(len(idx))
    for k in range(1, 5):                                     # cheap ambient occlusion along the normal
        dd = 0.03 * k
        ao -= np.clip(dd - scene(p + n * dd), 0, 1) * (1.6 / k)
    ao = np.clip(ao, 0, 1)
    hv = norm(L - rd[idx])
    spec = np.clip((n * hv).sum(-1), 0, 1) ** 40
    rim = (1 - np.clip(-(n * rd[idx]).sum(-1), 0, 1)) ** 3
    shade[idx] = np.clip((0.18 + 0.72 * lam) * ao * 0.62 + spec * 0.55 + rim * 0.12, 0, 1)
    # propeller discs: blurred spinning blades = translucent discs with a darker rim and hub streaks
    alpha = hit.astype(float)
    for m in MOTORS:
        c = m + np.array([0, PROP_Y, 0])
        tt = (c[1] - eye[1]) / rd[:, 1]
        q = eye + rd * tt[:, None]
        r = np.linalg.norm((q - c)[:, [0, 2]], axis=-1)
        front = (tt > 0) & (r < PROP_R) & (~hit | (tt < t))
        ang = np.arctan2((q - c)[:, 2], (q - c)[:, 0])
        streak = 0.5 + 0.5 * np.cos(ang * 2 + r * 3)
        a = np.where(front, 0.16 + 0.22 * (r / PROP_R) ** 6 + 0.10 * streak * (r / PROP_R), 0)
        g = 0.42
        shade = shade * (1 - a) + g * a
        alpha = np.maximum(alpha, a * 1.6)
    img = (np.clip(shade, 0, 1) * 255).reshape(h, w).astype(np.uint8)
    al = (np.clip(alpha, 0, 1) * 255).reshape(h, w).astype(np.uint8)
    im = Image.merge("RGBA", [Image.fromarray(img)] * 3 + [Image.fromarray(al)])
    return im.crop(im.getchannel("A").getbbox())


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    render().save(OUT)
    print(OUT)
