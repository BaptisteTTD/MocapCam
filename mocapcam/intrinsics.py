"""Calibration optique de chaque téléphone (une seule fois) avec un damier
affiché sur un écran. La taille réelle des cases n'a pas d'importance ici."""
import json
import os

import cv2
import numpy as np

from .video import iter_frames, video_info

DEFAULT_PATTERN = (9, 6)  # coins intérieurs (damier de 10 x 7 cases)


def make_checkerboard(path, pattern=DEFAULT_PATTERN, size=(1920, 1080)):
    cols, rows = pattern[0] + 1, pattern[1] + 1
    w, h = size
    sq = int(min(w * 0.9 / cols, h * 0.9 / rows))
    img = np.full((h, w), 255, np.uint8)
    x0, y0 = (w - cols * sq) // 2, (h - rows * sq) // 2
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                img[y0 + r * sq: y0 + (r + 1) * sq, x0 + c * sq: x0 + (c + 1) * sq] = 0
    cv2.imwrite(path, img)
    return path


def _detect(gray, pattern):
    ok, corners = cv2.findChessboardCornersSB(
        gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY)
    if not ok:
        return None
    return corners.reshape(-1, 2).astype(np.float32)


def calibrate_from_video(video, pattern=DEFAULT_PATTERN, max_views=60, step=None,
                         log=print):
    info = video_info(video)
    step = step or max(1, info["frames"] // (max_views * 4))
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    views, size = [], None
    for i, frame in iter_frames(video, step=step):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        c = _detect(gray, pattern)
        if c is not None:
            views.append(c)
    log(f"  damier trouvé sur {len(views)} images")
    if len(views) < 8:
        raise RuntimeError("Pas assez d'images avec le damier (minimum 8). Filmez l'écran "
                           "sous des angles variés, en remplissant aussi les bords de l'image.")
    # garder des vues variées : tri par position/tailles du damier
    if len(views) > max_views:
        sel = np.linspace(0, len(views) - 1, max_views).astype(int)
        views = [views[i] for i in sel]
    obj = [objp] * len(views)
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, views, size, None, None)
    log(f"  erreur RMS : {rms:.3f} px (bon si < 0.5, acceptable si < 1)")
    return dict(width=size[0], height=size[1], K=K.tolist(),
                dist=dist.ravel().tolist(), rms=float(rms), views=len(views))


def save(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    d["K"] = np.array(d["K"], float)
    d["dist"] = np.array(d["dist"], float)
    return d


def scaled_to(intr, width, height):
    """Adapte les paramètres si la vidéo a une autre résolution (même cadrage)."""
    if intr["width"] == width and intr["height"] == height:
        return intr
    sx, sy = width / intr["width"], height / intr["height"]
    if abs(sx - sy) > 0.01:
        raise RuntimeError(
            f"Format d'image différent de la calibration ({intr['width']}x{intr['height']} "
            f"vs {width}x{height}) : filmez dans la même orientation et le même format.")
    out = dict(intr)
    K = intr["K"].copy()
    K[0] *= sx
    K[1] *= sy
    out.update(K=K, width=width, height=height)
    return out
