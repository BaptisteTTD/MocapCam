"""Calibration automatique de la position des caméras grâce aux boules colorées
posées au sol à des positions mesurées (une fois) au mètre ruban.

À chaque prise : image médiane (sans la personne) -> détection des boules par couleur
-> choix de la meilleure combinaison (cohérence géométrique) -> PnP -> pose caméra.
Le sol (Z = 0), l'origine et les axes sont donc identiques d'une session à l'autre.
"""
import itertools

import cv2
import numpy as np

# Teintes OpenCV (0-180). Ajustables dans studio.json ("hsv": [h, tolérance, s_min, v_min]).
COLOR_PRESETS = {
    "rouge": (0, 10, 110, 70),
    "orange": (14, 6, 130, 90),
    "jaune": (28, 8, 110, 90),
    "vert": (65, 18, 90, 50),
    "cyan": (92, 8, 110, 70),
    "bleu": (112, 12, 110, 50),
    "violet": (138, 12, 80, 50),
    "magenta": (160, 10, 100, 70),
}
DRAW_BGR = {"rouge": (0, 0, 255), "orange": (0, 140, 255), "jaune": (0, 230, 255),
            "vert": (0, 200, 0), "cyan": (255, 255, 0), "bleu": (255, 80, 0),
            "violet": (200, 0, 140), "magenta": (255, 0, 255)}


def hsv_params(marker):
    if "hsv" in marker:
        return tuple(marker["hsv"])
    return COLOR_PRESETS[marker["color"]]


def color_mask(hsv, params):
    h0, tol, smin, vmin = params
    h = hsv[..., 0].astype(np.int16)
    dh = np.abs(h - h0)
    dh = np.minimum(dh, 180 - dh)
    m = (dh <= tol) & (hsv[..., 1] >= smin) & (hsv[..., 2] >= vmin)
    m = m.astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m


def blob_candidates(mask, min_area=12, max_frac=0.01, top=4):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    H, W = mask.shape
    cands = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_frac * H * W:
            continue
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        fill = area / float(w * h)            # disque ≈ 0.785
        aspect = min(w, h) / float(max(w, h))  # vu de biais : ellipse
        if fill < 0.5 or aspect < 0.3:
            continue
        score = area * fill * (0.5 + aspect)
        cands.append((score, (float(cent[i][0]), float(cent[i][1])), int(area)))
    cands.sort(key=lambda c: -c[0])
    return cands[:top]


def detect_candidates(image_bgr, markers, min_area=12):
    hsv = cv2.cvtColor(cv2.GaussianBlur(image_bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    cache = {}
    out = []
    for m in markers:
        key = tuple(hsv_params(m))
        if key not in cache:
            cache[key] = blob_candidates(color_mask(hsv, key), min_area=min_area)
        out.append(cache[key])
    return out


LENS_F_SCALE = {"wide": 0.75, "ultra": 0.36}   # focale / largeur d'image (×1 ≈ 26 mm, ×0,5 ≈ 13 mm)


def default_K(width, height, lens="wide"):
    """Focale de départ selon l'objectif du téléphone (×1 ≈ 26 mm équivalent,
    ×0,5 ≈ 13 mm), centre optique au milieu de l'image."""
    f = LENS_F_SCALE.get(lens or "wide", 0.75) * max(width, height)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])


def _solve(obj, img, K, dist):
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
    R, _ = cv2.Rodrigues(rvec)
    # caméra devant les points
    zc = (R @ obj.T + tvec).T[:, 2]
    if np.any(zc <= 0):
        return None
    return R, tvec.ravel(), err


def _solve_with_focal(obj, img, K0, R, t, prior_sigma=0.35):
    """Affine pose + focale (centre optique fixé au milieu, sans distorsion).
    Un a priori faible retient la focale près de la valeur de départ quand
    les boules ne suffisent pas à la déterminer."""
    from scipy.optimize import least_squares
    f0, cx, cy = K0[0, 0], K0[0, 2], K0[1, 2]
    rvec0 = cv2.Rodrigues(R)[0].ravel()

    def res(p):
        f = f0 * np.exp(p[6])
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        pr, _ = cv2.projectPoints(obj, p[:3], p[3:6], K, None)
        return np.concatenate([(pr.reshape(-1, 2) - img).ravel(), [p[6] / prior_sigma]])

    sol = least_squares(res, np.concatenate([rvec0, t, [0.0]]), method="lm")
    p = sol.x
    f = f0 * np.exp(p[6])
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    R2 = cv2.Rodrigues(p[:3])[0]
    pr, _ = cv2.projectPoints(obj, p[:3], p[3:6], K, None)
    err = np.linalg.norm(pr.reshape(-1, 2) - img, axis=1)
    if np.any((R2 @ obj.T + p[3:6, None])[2] <= 0):
        return None
    return K, R2, p[3:6].copy(), err


def solve_camera_pose(image_bgr, markers, K=None, dist=None, min_markers=4, max_err_px=8.0,
                      estimate_focal=False, lens="wide"):
    """Retourne un dict : R, t (monde -> caméra), K, centre, erreurs, détections.
    Sans K (pas de calibration optique), la focale est estimée avec les boules."""
    H, W = image_bgr.shape[:2]
    if K is None:
        K, dist, estimate_focal = default_K(W, H, lens), np.zeros(5), True
    dist = np.zeros(5) if dist is None else dist
    cands = detect_candidates(image_bgr, markers)
    world = np.array([m["position"] for m in markers], np.float64)
    options = [[None] + [c[1] for c in cl] for cl in cands]
    # focale inconnue : tolérance plus large au premier tri
    thr = max_err_px * (6 if estimate_focal else 2)
    found = []
    for combo in itertools.product(*options):
        used = [i for i, p in enumerate(combo) if p is not None]
        if len(used) < min_markers:
            continue
        pts = [combo[i] for i in used]
        if len(set(pts)) < len(pts):  # une même tache pour deux boules
            continue
        sol = _solve(world[used], np.array(pts, np.float64), K, dist)
        if sol is None:
            continue
        R, t, err = sol
        if err.max() > thr:
            continue
        rms = float(np.sqrt(np.mean(err ** 2)))
        found.append((rms + 1.5 * (len(markers) - len(used)), combo, used, R, t, err))
    found.sort(key=lambda f: f[0])
    best = None
    for _, combo, used, R, t, err in found[: (10 if estimate_focal else 1)]:
        obj = world[used]
        img = np.array([combo[i] for i in used], np.float64)
        Kc = K
        if estimate_focal:
            sol = _solve_with_focal(obj, img, K, R, t)
            if sol is None:
                continue
            Kc, R, t, err = sol
            if err.max() > max_err_px * 2:
                continue
        rms = float(np.sqrt(np.mean(err ** 2)))
        # pénalité par boule manquante : on préfère utiliser toutes les boules
        score = rms + 1.5 * (len(markers) - len(used))
        if best is None or score < best["score"]:
            best = dict(score=score, R=R, t=t, K=Kc, dist=dist, rms=rms, used=used,
                        focal_estimated=estimate_focal,
                        points={markers[i]["name"]: combo[i] for i in used})
    if best is None:
        raise RuntimeError(
            "Boules introuvables ou incohérentes : vérifiez qu'au moins "
            f"{min_markers} boules sont visibles, et les couleurs (commande 'marqueurs').")
    best["center"] = (-best["R"].T @ best["t"]).tolist()
    best["missing"] = [m["name"] for i, m in enumerate(markers) if i not in best["used"]]
    best["marker_obs"] = [(markers[i]["position"], combo_pt) for i, combo_pt in
                          ((i, best["points"][markers[i]["name"]]) for i in best["used"])]
    return best


def volume_box(markers, margin=0.5, height=2.3):
    p = np.array([m["position"] for m in markers], float)
    lo = [p[:, 0].min() - margin, p[:, 1].min() - margin, 0.0]
    hi = [p[:, 0].max() + margin, p[:, 1].max() + margin, height]
    return np.array(lo), np.array(hi)


def draw_debug(image_bgr, markers, K, dist, pose, volume=None):
    img = image_bgr.copy()
    rvec, _ = cv2.Rodrigues(pose["R"])
    t = np.asarray(pose["t"], float)
    world = np.array([m["position"] for m in markers], np.float64)
    proj, _ = cv2.projectPoints(world, rvec, t, K, dist)
    for m, p in zip(markers, proj.reshape(-1, 2)):
        col = DRAW_BGR.get(m.get("color"), (255, 255, 255))
        cv2.circle(img, tuple(int(v) for v in p), 14, col, 2)
        cv2.putText(img, m["name"], (int(p[0]) + 16, int(p[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        if m["name"] in pose["points"]:
            q = pose["points"][m["name"]]
            cv2.drawMarker(img, (int(q[0]), int(q[1])), (255, 255, 255),
                           cv2.MARKER_CROSS, 12, 2)
    # axes : X rouge, Y vert, Z bleu (1 m)
    ax = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64)
    pa, _ = cv2.projectPoints(ax, rvec, t, K, dist)
    pa = pa.reshape(-1, 2).astype(int)
    for k, col in zip(range(1, 4), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        cv2.arrowedLine(img, tuple(pa[0]), tuple(pa[k]), col, 3)
    if volume is not None:
        lo, hi = volume
        corners = np.array(list(itertools.product(*zip(lo, hi))), np.float64)
        pc, _ = cv2.projectPoints(corners, rvec, t, K, dist)
        pc = pc.reshape(-1, 2).astype(int)
        for i, j in itertools.combinations(range(8), 2):
            if bin(i ^ j).count("1") == 1:
                cv2.line(img, tuple(pc[i]), tuple(pc[j]), (200, 200, 200), 1)
    cv2.putText(img, f"erreur {pose['rms']:.2f} px", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img
