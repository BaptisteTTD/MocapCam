"""Reconstruction 3D : choix de la personne, correction des inversions gauche/droite,
triangulation pondérée, rejet des points aberrants, comblement des trous, lissage."""
import itertools

import cv2
import numpy as np
from scipy.signal import butter, filtfilt

from .skeleton import CORE_IDX, SWAP_LR, HAND_IDX, FACE_IDX, BODY_IDX, INDEX, LHAND_IDX, RHAND_IDX


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


# Segments dont la longueur ne change pas (ou peu) au cours d'une prise. Avec deux caméras à
# la même hauteur, gauche et droite sont sur la même ligne épipolaire : associer la hanche
# gauche d'une vue à la hanche droite de l'autre donne aussi un point 3D « cohérent », avec une
# erreur de reprojection parfois plus faible, mais un corps déformé (bassin 50 % trop large,
# tourné de 90°). La constance de ces longueurs départage les deux hypothèses.
SEGMENTS = [("l_hip", "r_hip"), ("l_shoulder", "r_shoulder"),
            ("l_shoulder", "r_hip"), ("r_shoulder", "l_hip"),
            ("l_hip", "l_knee"), ("r_hip", "r_knee"), ("l_knee", "l_ankle"), ("r_knee", "r_ankle"),
            ("l_shoulder", "l_elbow"), ("r_shoulder", "r_elbow"), ("l_elbow", "l_wrist"), ("r_elbow", "r_wrist")]
SEG_GROUP = np.array([0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])   # gauche et droite : même longueur
_SEG_CORE = [(CORE_IDX.index(INDEX[a]), CORE_IDX.index(INDEX[b])) for a, b in SEGMENTS]


def _segment_lengths(Xcore):
    return np.array([np.linalg.norm(Xcore[a] - Xcore[b]) for a, b in _SEG_CORE])


def _structure_dev(lengths, ref):
    """Écart relatif moyen aux longueurs de référence (0 = corps parfaitement constant)."""
    r = ref[SEG_GROUP]
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.minimum(np.abs(lengths / r - 1.0), 1.0)
    d = d[np.isfinite(d)]
    return float(d.mean()) if d.size >= 3 else 0.5


def reconstruct(cams, poses, delays, fps, volume, cfg=None, log=print, frames=None,
                return_choices=False, return_errors=False):
    """cams : liste de dict(K, dist, R, t). poses : liste de dict(kp (T,P,133,2), sc (T,P,133)).
    delays[v] : décalage (s) de la vue v par rapport à la vue 0 (référence).
    Retourne X (T,133,3) en mètres dans le repère du sol, et des statistiques.

    Pour chaque image, toutes les hypothèses (quelle personne dans chaque vue, gauche/droite
    inversées ou non) sont triangulées. On retient celle qui combine faible erreur de
    reprojection, corps de proportions constantes, et bassin proche de l'image précédente."""
    cfg = cfg or {}
    V = len(cams)
    Rt = np.stack([np.hstack([c["R"], np.asarray(c["t"], float).reshape(3, 1)]) for c in cams])
    fx = np.array([c["K"][0, 0] for c in cams])
    thr = _thresholds(cfg)
    max_err_px = cfg.get("max_reproj_px", 25.0)
    w_struct = cfg.get("structure_weight", 40.0)
    lo, hi = volume
    T = poses[0]["kp"].shape[0]
    X_all = np.full((T, 133, 3), np.nan)
    chosen_err = np.full(T, np.nan)
    lh, rh = INDEX["l_hip"], INDEX["r_hip"]
    ilh, irh = CORE_IDX.index(lh), CORE_IDX.index(rh)

    # pré-calcul des coordonnées normalisées
    norm = [undistort_normalized(p["kp"], c["K"], c["dist"]) for p, c in zip(poses, cams)]

    def build(per_view, combo, flips):
        xs = np.full((V, 133, 2), np.nan)
        ws = np.zeros((V, 133))
        for v in range(V):
            if combo[v] is None:
                continue
            x, s = per_view[v][combo[v]][:2]
            if v > 0 and flips[v - 1]:
                x, s = x[SWAP_LR], s[SWAP_LR]
            xs[v], ws[v] = x, np.where(s >= thr, s, 0.0)
        return xs, ws

    # 1) toutes les hypothèses de chaque image
    cands = {}
    views_of = {}
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
        lst = []
        for combo in itertools.product(*[range(len(pv)) if pv else [None] for pv in per_view]):
            for flips in itertools.product([False, True], repeat=V - 1):
                xs, ws = build(per_view, combo, flips)
                X, err = triangulate(xs[:, CORE_IDX], ws[:, CORE_IDX], Rt)
                e_px = err * fx[:, None]
                valid = (ws[:, CORE_IDX] > 0) & np.isfinite(e_px)
                if valid.sum() < 6:
                    continue
                med = float(np.median(e_px[valid]))
                pelvis = np.nanmean(X[[ilh, irh]], 0)
                pen = 0.0
                if not np.all(np.isfinite(pelvis)):
                    pen += 50
                elif np.any(pelvis < lo) or np.any(pelvis > hi):
                    pen += 100
                lst.append(dict(med=med, pen=pen + 0.5 * sum(flips), pelvis=pelvis,
                                lengths=_segment_lengths(X), combo=combo, flips=flips))
        if lst:
            cands[i] = lst
            views_of[i] = per_view

    # 2) longueurs de référence du corps (médianes), puis choix image par image
    def choose(ref):
        out, prev_pelvis = {}, None
        for i in sorted(cands):
            best = None
            for c in cands[i]:
                score = c["med"] + c["pen"]
                if prev_pelvis is not None and np.all(np.isfinite(c["pelvis"])):
                    score += 20 * max(0.0, np.linalg.norm(c["pelvis"] - prev_pelvis) - 0.3)
                if ref is not None:
                    score += w_struct * _structure_dev(c["lengths"], ref)
                if best is None or score < best[0]:
                    best = (score, c)
            if best is None or best[1]["med"] > max_err_px * 2:
                continue
            out[i] = best[1]
            if np.all(np.isfinite(best[1]["pelvis"])):
                prev_pelvis = best[1]["pelvis"]
        return out

    def reference(sel):
        L = np.array([c["lengths"] for c in sel.values()])
        if len(L) < 10:
            return None
        ref = np.array([np.nanmedian(L[:, SEG_GROUP == g]) if np.isfinite(L[:, SEG_GROUP == g]).any()
                        else np.nan for g in range(SEG_GROUP.max() + 1)])
        return ref if np.isfinite(ref).sum() >= 3 else None

    sel = choose(None)
    ref = reference(sel) if w_struct > 0 else None
    if ref is not None:
        sel = choose(ref)
        ref = reference(sel)      # une seconde passe, avec des références plus propres
        sel = choose(ref) if ref is not None else sel

    choices = {}
    swaps = 0
    E_all = np.full((T, len(CORE_IDX)), np.nan)   # erreur de reprojection (px) des points du corps
    for i, c in sel.items():
        per_view = views_of[i]
        xs, ws = build(per_view, c["combo"], c["flips"])
        choices[i] = ([None if k is None else per_view[v][k][2:] for v, k in enumerate(c["combo"])],
                      c["flips"])
        swaps += int(any(c["flips"]))
        X, err = triangulate(xs, ws, Rt)
        e_px = np.nanmax(np.where(ws > 0, err * fx[:, None], 0), axis=0)
        E_all[i] = np.where(np.isfinite(X[CORE_IDX, 0]), e_px[CORE_IDX], np.nan)
        X[e_px > max_err_px] = np.nan
        X_all[i] = X
        chosen_err[i] = c["med"]
        if frames is None and i % 300 == 0:
            log(f"    triangulation {i}/{T}")
    stats = dict(frames=T, frames_ok=int(np.isfinite(chosen_err).sum()),
                 median_reproj_px=float(np.nanmedian(chosen_err)) if np.isfinite(chosen_err).any() else None,
                 lr_swaps_fixed=swaps,
                 body_lengths_m=None if ref is None else [round(float(v), 3) for v in ref])
    if return_errors:
        return X_all, stats, E_all
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


def clean_hands(X, rel=0.35, wrist_tol=0.6):
    """Une main est un objet rigide : longueur de paume (poignet -> base du majeur) et largeur
    des jointures (index -> auriculaire) constantes. Quand elles s'écartent de plus de 35 % de
    leur valeur habituelle (main vue par la tranche, floue, ou mal reconnue), ou que le poignet
    de la main ne colle pas à celui du corps, la main de cette image est écartée : elle sera
    interpolée entre les images voisines au lieu de faire tourner l'avant-bras d'un coup."""
    out = X.copy()
    n_bad = 0
    for side, hidx in (("l", LHAND_IDX), ("r", RHAND_IDX)):
        h = lambda k: X[:, hidx[k]]
        palm = np.linalg.norm(h(0) - h(9), axis=1)
        kn = np.linalg.norm(h(5) - h(17), axis=1)
        wr = np.linalg.norm(h(0) - X[:, INDEX[f"{side}_wrist"]], axis=1)
        if np.isfinite(palm).sum() < 10:
            continue
        rp, rk = np.nanmedian(palm), np.nanmedian(kn)
        with np.errstate(invalid="ignore"):
            bad = ((np.abs(palm / rp - 1) > rel) | (np.abs(kn / rk - 1) > rel)
                   | (wr > wrist_tol * rp))
        bad &= np.isfinite(palm)
        out[np.ix_(bad, hidx)] = np.nan
        n_bad += int(bad.sum())
    return out, n_bad


def postprocess(X, fps, cfg=None):
    cfg = cfg or {}
    X, _ = clean_hands(X)
    X = remove_spikes(X, fps, max_jump=cfg.get("max_jump_m", 0.15))
    X = fill_gaps(X, max_gap=int(cfg.get("max_gap_s", 0.3) * fps))
    X = smooth(X, fps, cfg.get("cutoff_body_hz", 6.0), cfg.get("cutoff_hands_hz", 8.0))
    return X


def trim_take(X, fps, min_segment_s=1.0, min_gap_s=0.5):
    """Coupe le début et la fin de prise sans acteur exploitable : quelques images isolées
    (moins de min_segment_s), séparées du reste par un trou, avant le jeu ou après (quand
    l'acteur va arrêter les téléphones). Retourne (X coupé, première image gardée)."""
    ok = np.isfinite(X[:, CORE_IDX, 0]).mean(1) >= 0.5
    T = len(ok)
    segs, s0 = [], None
    for t in range(T + 1):
        if t < T and ok[t]:
            s0 = t if s0 is None else s0
        elif s0 is not None:
            segs.append((s0, t))
            s0 = None
    if not segs:
        return X, 0
    # on regroupe les morceaux séparés par de petits trous
    merged = [list(segs[0])]
    for a, b in segs[1:]:
        if a - merged[-1][1] < min_gap_s * fps:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    big = [m for m in merged if m[1] - m[0] >= min_segment_s * fps] or [max(merged, key=lambda m: m[1] - m[0])]
    first, last = big[0][0], big[-1][1]
    return X[first:last], first


# ----------------------------------------------------------------------------- cadences différentes

def _centroid(kp, sc, thr=0.3):
    ok = (sc >= thr) & np.isfinite(kp[:, 0])
    if ok.sum() < 3:
        return None
    return kp[ok].mean(0)


def resample_pose(pose, fps_src, fps_dst, delay, T_dst, match_px=150.0):
    """Ramène les points 2D d'une caméra sur la base de temps commune (fps_dst), en tenant
    compte de son décalage (delay, s) : un événement au temps t de la référence apparaît au
    temps t + delay dans cette caméra. Interpolation linéaire entre les deux images voisines
    (avec appariement des personnes), donc synchro plus fine qu'une image et cadences mixtes
    (ex. 30 et 60 i/s) gérées sans dupliquer d'images."""
    kp, sc = pose["kp"], pose["sc"]
    Ts, P = kp.shape[:2]
    out_kp = np.full((T_dst,) + kp.shape[1:], np.nan, np.float32)
    out_sc = np.zeros((T_dst,) + sc.shape[1:], np.float32)
    s = (np.arange(T_dst) / fps_dst + delay) * fps_src
    a = np.floor(s).astype(int)
    w = s - a
    for i in np.where((a >= 0) & (a < Ts))[0]:
        ai, wi = a[i], float(w[i])
        bi = min(ai + 1, Ts - 1)
        if bi == ai or wi < 0.02 or wi > 0.98:
            k = bi if wi > 0.98 else ai
            out_kp[i], out_sc[i] = kp[k], sc[k]
            continue
        used = set()
        for p in range(P):
            if not np.nanmax(sc[ai, p]) > 0:
                continue
            ca = _centroid(kp[ai, p], sc[ai, p])
            best, bd = None, match_px
            if ca is not None:
                for q in range(P):
                    if q in used or not np.nanmax(sc[bi, q]) > 0:
                        continue
                    cb = _centroid(kp[bi, q], sc[bi, q])
                    if cb is not None and np.linalg.norm(cb - ca) < bd:
                        best, bd = q, np.linalg.norm(cb - ca)
            if best is None:          # pas de correspondance : image la plus proche
                out_kp[i, p], out_sc[i, p] = kp[ai, p], sc[ai, p]
                continue
            used.add(best)
            ka, kb = kp[ai, p], kp[bi, best]
            sa, sb = sc[ai, p], sc[bi, best]
            both = (sa > 0) & (sb > 0) & np.isfinite(ka[:, 0]) & np.isfinite(kb[:, 0])
            near_k, near_s = (ka, sa) if wi < 0.5 else (kb, sb)
            out_kp[i, p] = np.where(both[:, None], (1 - wi) * ka + wi * kb, near_k)
            out_sc[i, p] = np.where(both, np.minimum(sa, sb), near_s * (np.abs(wi - 0.5) > 0.25))
    return dict(kp=out_kp, sc=out_sc, width=pose.get("width"), height=pose.get("height"))
