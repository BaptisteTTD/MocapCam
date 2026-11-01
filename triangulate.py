"""Reconstruction 3D : choix de la personne, correction des inversions gauche/droite,
triangulation pondérée, rejet des points aberrants, comblement des trous, lissage."""
import itertools

import cv2
import numpy as np
from scipy.signal import butter, filtfilt

from .skeleton import CORE_IDX, SWAP_LR, HAND_IDX, FACE_IDX, BODY_IDX, INDEX


def undistort_normalized(pts, K, dist):
    """Pixels -> coordonnées normalisées (sans distorsion). pts (...,2), NaN conservés."""
    shp = pts.shape
    flat = pts.reshape(-1, 2).astype(np.float64)
    ok = np.isfinite(flat).all(1)
    out = np.full_like(flat, np.nan)
    if ok.any():
        u = cv2.undistortPoints(flat[ok].reshape(-1, 1, 2), K, dist).reshape(-1, 2)
        out[ok] = u
    return out.reshape(shp)


def triangulate(xn, w, Rt):
    """xn (V,N,2) coords normalisées, w (V,N) poids, Rt (V,3,4).
    Retourne X (N,3) et erreur de reprojection normalisée (V,N)."""
    V, N, _ = xn.shape
    x = np.nan_to_num(xn)
    A = np.zeros((N, 2 * V, 4))
    for v in range(V):
        P = Rt[v]
        A[:, 2 * v] = w[v, :, None] * (x[v, :, 0:1] * P[2] - P[0])
        A[:, 2 * v + 1] = w[v, :, None] * (x[v, :, 1:2] * P[2] - P[1])
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[:, -1]
    X = Xh[:, :3] / Xh[:, 3:4]
    err = np.full((V, N), np.nan)
    Xh1 = np.concatenate([X, np.ones((N, 1))], 1)
    for v in range(V):
        pc = Xh1 @ Rt[v].T
        with np.errstate(invalid="ignore", divide="ignore"):
            pr = pc[:, :2] / pc[:, 2:3]
            e = np.linalg.norm(pr - xn[v], axis=1)
            e[pc[:, 2] <= 0] = np.inf
        err[v] = e
    nviews = (w > 0).sum(0)
    X[nviews < 2] = np.nan
    return X, err


def _thresholds(cfg):
    thr = np.full(133, cfg.get("score_body", 0.3), float)
    thr[HAND_IDX] = cfg.get("score_hands", 0.3)
    thr[FACE_IDX] = cfg.get("score_face", 0.5)
    return thr


def reconstruct(cams, poses, delays, fps, volume, cfg=None, log=print, frames=None,
                return_choices=False):
    """cams : liste de dict(K, dist, R, t). poses : liste de dict(kp (T,P,133,2), sc (T,P,133)).
    delays[v] : décalage (s) de la vue v par rapport à la vue 0 (référence).
    Retourne X (T,133,3) en mètres dans le repère du sol, et des statistiques."""
    cfg = cfg or {}
    V = len(cams)
    Rt = np.stack([np.hstack([c["R"], np.asarray(c["t"], float).reshape(3, 1)]) for c in cams])
    fx = np.array([c["K"][0, 0] for c in cams])
    thr = _thresholds(cfg)
    max_err_px = cfg.get("max_reproj_px", 25.0)
    lo, hi = volume
    T = poses[0]["kp"].shape[0]
    X_all = np.full((T, 133, 3), np.nan)
    chosen_err = np.full(T, np.nan)
    prev_pelvis = None
    swaps = 0
    lh, rh = INDEX["l_hip"], INDEX["r_hip"]

    # pré-calcul des coordonnées normalisées
    norm = [undistort_normalized(p["kp"], c["K"], c["dist"]) for p, c in zip(poses, cams)]

    choices = {}
    for i in (range(T) if frames is None else frames):
        per_view = []
        for v in range(V):
            j = int(round(i + delays[v] * fps))
            if j < 0 or j >= poses[v]["kp"].shape[0]:
                per_view.append([])
                continue
            sc = poses[v]["sc"][j]
            people = [p for p in range(sc.shape[0]) if np.nanmax(sc[p]) > 0]
            per_view.append([(norm[v][j, p], sc[p], j, p) for p in people])
        if sum(1 for pv in per_view if pv) < 2:
            continue
        best = None
        for combo in itertools.product(*[range(len(pv)) if pv else [None] for pv in per_view]):
            for flips in itertools.product([False, True], repeat=V - 1):
                xs = np.full((V, 133, 2), np.nan)
                ws = np.zeros((V, 133))
                for v in range(V):
                    if combo[v] is None:
                        continue
                    x, s = per_view[v][combo[v]][:2]
                    if v > 0 and flips[v - 1]:
                        x, s = x[SWAP_LR], s[SWAP_LR]
                    xs[v], ws[v] = x, np.where(s >= thr, s, 0.0)
                core = CORE_IDX
                X, err = triangulate(xs[:, core], ws[:, core], Rt)
                e_px = err * fx[:, None]
                valid = (ws[:, core] > 0) & np.isfinite(e_px)
                if valid.sum() < 6:
                    continue
                med = float(np.median(e_px[valid]))
                pelvis = np.nanmean(X[[CORE_IDX.index(lh), CORE_IDX.index(rh)]], 0)
                pen = 0.0
                if not np.all(np.isfinite(pelvis)):
                    pen += 50
                elif np.any(pelvis < lo) or np.any(pelvis > hi):
                    pen += 100
                elif prev_pelvis is not None:
                    pen += 20 * max(0.0, np.linalg.norm(pelvis - prev_pelvis) - 0.3)
                score = med + pen + 0.5 * sum(flips)  # à égalité, pas d'inversion
                if best is None or score < best[0]:
                    best = (score, xs, ws, flips, pelvis, med,
                            [None if c is None else per_view[v][c][2:] for v, c in enumerate(combo)])
        if best is None or best[0] > max_err_px * 2:
            continue
        _, xs, ws, flips, pelvis, med, picked = best
        choices[i] = (picked, flips)
        swaps += int(any(flips))
        X, err = triangulate(xs, ws, Rt)
        e_px = np.nanmax(np.where(ws > 0, err * fx[:, None], 0), axis=0)
        X[e_px > max_err_px] = np.nan
        X_all[i] = X
        chosen_err[i] = med
        if np.all(np.isfinite(pelvis)):
            prev_pelvis = pelvis
        if frames is None and i % 300 == 0:
            log(f"    triangulation {i}/{T}")
    stats = dict(frames=T, frames_ok=int(np.isfinite(chosen_err).sum()),
                 median_reproj_px=float(np.nanmedian(chosen_err)) if np.isfinite(chosen_err).any() else None,
                 lr_swaps_fixed=swaps)
    if return_choices:
        return X_all, stats, choices
    return X_all, stats


# ----------------------------------------------------------------------------- nettoyage

def remove_spikes(X, fps, max_jump=0.15, win=7):
    """Supprime les points qui s'écartent brutalement de leur médiane locale."""
    from numpy.lib.stride_tricks import sliding_window_view
    T = X.shape[0]
    if T < win:
        return X
    pad = win // 2
    Xp = np.concatenate([np.repeat(X[:1], pad, 0), X, np.repeat(X[-1:], pad, 0)], 0)
    Xp = np.where(np.isfinite(Xp), Xp, np.nan)
    w = sliding_window_view(Xp, win, axis=0)  # (T,133,3,win)
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        med = np.nanmedian(w, axis=-1)
    d = np.linalg.norm(X - med, axis=-1)
    out = X.copy()
    out[d > max_jump] = np.nan
    return out


def fill_gaps(X, max_gap):
    """Interpolation linéaire des trous courts (<= max_gap images)."""
    T, J, _ = X.shape
    out = X.copy()
    t = np.arange(T)
    for j in range(J):
        ok = np.isfinite(X[:, j, 0])
        if ok.sum() < 2 or ok.all():
            continue
        idx = np.where(ok)[0]
        interp = np.stack([np.interp(t, idx, X[idx, j, k]) for k in range(3)], 1)
        # longueur de chaque trou
        miss = ~ok
        start = None
        for f in range(T + 1):
            m = f < T and miss[f]
            if m and start is None:
                start = f
            elif not m and start is not None:
                if start > 0 and f < T and (f - start) <= max_gap:
                    out[start:f, j] = interp[start:f]
                start = None
    return out


def smooth(X, fps, cutoff_body=6.0, cutoff_hands=8.0, order=2):
    """Filtre de Butterworth aller-retour (sans décalage) : possible car en différé."""
    out = X.copy()
    cut = np.full(X.shape[1], cutoff_body)
    cut[HAND_IDX] = cutoff_hands
    for c in np.unique(cut):
        wn = min(c / (fps / 2.0), 0.99)
        b, a = butter(order, wn)
        padlen = 3 * max(len(a), len(b))
        for j in np.where(cut == c)[0]:
            ok = np.isfinite(X[:, j, 0])
            # segments continus
            edges = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(int), [0]])))
            for s, e in zip(edges[::2], edges[1::2]):
                if e - s > padlen:
                    out[s:e, j] = filtfilt(b, a, X[s:e, j], axis=0)
    return out


def postprocess(X, fps, cfg=None):
    cfg = cfg or {}
    X = remove_spikes(X, fps, max_jump=cfg.get("max_jump_m", 0.15))
    X = fill_gaps(X, max_gap=int(cfg.get("max_gap_s", 0.3) * fps))
    X = smooth(X, fps, cfg.get("cutoff_body_hz", 6.0), cfg.get("cutoff_hands_hz", 8.0))
    return X
