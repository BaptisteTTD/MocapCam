"""Détection 2D corps + mains + visage (133 points COCO-WholeBody) avec RTMW via rtmlib.
Les modèles (~200 Mo) sont téléchargés automatiquement au premier lancement."""
import os
import time

import numpy as np

from .video import video_info, iter_frames

MAX_PERSONS = 4


def run_video(video, out_npz, mode="performance", device="cpu", backend="onnxruntime",
              force=False, log=print):
    if not force and os.path.exists(out_npz) and os.path.getmtime(out_npz) >= os.path.getmtime(video):
        return load(out_npz)
    from rtmlib import Wholebody
    model = Wholebody(mode=mode, backend=backend, device=device, to_openpose=False)
    info = video_info(video)
    T = info["frames"]
    kp = np.full((T, MAX_PERSONS, 133, 2), np.nan, np.float32)
    sc = np.zeros((T, MAX_PERSONS, 133), np.float32)
    t0 = time.time()
    last = -1
    for i, frame in iter_frames(video):
        if i >= T:
            break
        k, s = model(frame)
        if k is not None and len(k):
            k = np.asarray(k)[:MAX_PERSONS]
            s = np.asarray(s)[:MAX_PERSONS]
            kp[i, : len(k)] = k
            sc[i, : len(s)] = s
        last = i
        if i % 100 == 0:
            el = time.time() - t0
            log(f"    {os.path.basename(video)} : image {i}/{T}  ({(i + 1) / max(el, 1e-6):.1f} img/s)")
    kp, sc = kp[: last + 1], sc[: last + 1]
    np.savez_compressed(out_npz, kp=kp, sc=sc, width=info["width"], height=info["height"])
    return dict(kp=kp, sc=sc, width=info["width"], height=info["height"])


def load(path):
    d = np.load(path)
    return dict(kp=d["kp"], sc=d["sc"], width=int(d["width"]), height=int(d["height"]))
