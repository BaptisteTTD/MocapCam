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


MIN_CAMERA_HEIGHT = 0.15   # m : une caméra ne peut pas être sous le sol


def _camera_height(R, t):
    return float((-R.T @ np.asarray(t, float).reshape(3))[2])


def _solutions(obj, img, K, dist):
    """Toutes les poses plausibles. Les boules étant toutes au sol (points coplanaires),
    il existe deux solutions « miroir » : on garde celles où la caméra est au-dessus du sol."""
    out = []
    coplanar = np.ptp(obj[:, 2]) < 0.01
    cands = []
    if coplanar and len(obj) >= 4:
        try:
            n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
            cands += list(zip(rvecs, tvecs))
        except cv2.error:
            pass
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if ok:
        cands.append((rvec, tvec))
    for rvec, tvec in cands:
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec.copy(), tvec.copy())
        R, _ = cv2.Rodrigues(rvec)
        if np.any((R @ obj.T + tvec).T[:, 2] <= 0):      # boules devant la caméra
            continue
        if _camera_height(R, tvec) < MIN_CAMERA_HEIGHT:
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
        out.append((R, tvec.ravel(), err))
    return out


def _solve(obj, img, K, dist):
    sols = _solutions(obj, img, K, dist)
    if not sols:
        return None
    return min(sols, key=lambda s: float(np.mean(s[2] ** 2)))


def _solve_with_focal(obj, img, K0, R, t, prior_sigma=0.15, prior_px=3.0):
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
        # a priori : un écart d'un sigma sur la focale « coûte » prior_px pixels
        return np.concatenate([(pr.reshape(-1, 2) - img).ravel(), [prior_px * p[6] / prior_sigma]])

    sol = least_squares(res, np.concatenate([rvec0, t, [0.0]]), method="lm")
    p = sol.x
    f = f0 * np.exp(p[6])
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    R2 = cv2.Rodrigues(p[:3])[0]
    pr, _ = cv2.projectPoints(obj, p[:3], p[3:6], K, None)
    err = np.linalg.norm(pr.reshape(-1, 2) - img, axis=1)
    if np.any((R2 @ obj.T + p[3:6, None])[2] <= 0) or _camera_height(R2, p[3:6]) < MIN_CAMERA_HEIGHT:
        return None
    return K, R2, p[3:6].copy(), err


def spread_ok(obj, rel=0.04):
    """Vrai s'il existe 4 boules dont aucun trio n'est aligné : la focale peut alors être
    estimée par les boules. Sinon (par exemple 3 boules alignées + 1), seule la focale
    nominale de l'objectif est fiable."""
    p = np.asarray(obj, float)[:, :2]
    if len(p) < 4:
        return False
    span2 = max(float(np.sum((a - b) ** 2)) for a, b in itertools.combinations(p, 2))
    if span2 <= 0:
        return False
    def area(i, j, k):
        a, b = p[j] - p[i], p[k] - p[i]
        return abs(a[0] * b[1] - a[1] * b[0]) / 2

    for q in itertools.combinations(range(len(p)), 4):
        if all(area(i, j, k) > rel * span2 for i, j, k in itertools.combinations(q, 3)):
            return True
    return False


def mirrored(markers):
    """Disposition symétrique (X -> -X) : mêmes distances entre boules, mais
    B/C de l'autre côté de la ligne A-D."""
    return [dict(m, position=[-m["position"][0]] + list(m["position"][1:])) for m in markers]


def swapped(markers, i, j):
    out = [dict(m) for m in markers]
    out[i]["position"], out[j]["position"] = markers[j]["position"], markers[i]["position"]
    return out


def find_candidates(image_bgr, markers, frames=None):
    """Taches candidates pour chaque boule (dépend seulement des couleurs)."""
    if frames:
        return detect_candidates_multi(list(frames) + [image_bgr], markers)
    return detect_candidates(image_bgr, markers)


def detect_candidates_multi(frames, markers, min_presence=0.25, top=4):
    """Cherche les boules sur plusieurs images de la vidéo et ne garde que les taches
    immobiles et présentes souvent : une boule cachée un moment par l'acteur est
    retrouvée, une tache qui bouge (vêtement) est écartée ou reléguée."""
    H, W = frames[0].shape[:2]
    rad = 0.008 * max(W, H)
    per = [detect_candidates(f, markers) for f in frames]
    out = []
    for mi in range(len(markers)):
        pts = [c[1] for fr in per for c in fr[mi]]
        clusters = []
        for p in pts:
            for cl in clusters:
                if np.hypot(p[0] - cl["c"][0], p[1] - cl["c"][1]) < rad:
                    cl["p"].append(p)
                    cl["c"] = np.median(np.array(cl["p"]), 0)
                    break
            else:
                clusters.append(dict(p=[p], c=np.array(p, float)))
        scored = []
        for cl in clusters:
            arr = np.array(cl["p"])
            presence = min(1.0, len(arr) / len(frames))
            if presence < min_presence:
                continue
            spread = float(np.median(np.linalg.norm(arr - cl["c"], axis=1)))
            scored.append((presence / (1.0 + spread), (float(cl["c"][0]), float(cl["c"][1])), len(arr)))
        scored.sort(key=lambda x: -x[0])
        out.append(scored[:top])
    return out


def fit_score(err, n_missing, focal_free=True):
    """Qualité d'un ajustement (plus bas = mieux). L'écart est corrigé du nombre de
    paramètres : avec 4 boules et la focale libre, presque n'importe quelle disposition
    « colle », ce qui ne prouve rien ; avec 5 boules, un petit écart est significatif.
    Chaque boule non utilisée coûte 2 px."""
    n_obs = 2 * len(err)
    dof = max(n_obs - (7 if focal_free else 6), 0.5)
    sigma = float(np.sqrt(np.sum(np.asarray(err) ** 2) / dof))
    return sigma + 2.0 * n_missing


def solve_camera_pose(image_bgr, markers, K=None, dist=None, min_markers=4, max_err_px=8.0,
                      estimate_focal=False, lens="wide", frames=None, cands=None):
    """Retourne un dict : R, t (monde -> caméra), K, centre, erreurs, détections.
    Sans K (pas de calibration optique), la focale est estimée avec les boules, autour
    de la focale nominale de l'objectif ; elle reste à cette valeur quand les boules
    vues sont presque alignées (3 alignées + 1)."""
    H, W = image_bgr.shape[:2]
    if K is None:
        K, dist, estimate_focal = default_K(W, H, lens), np.zeros(5), True
    dist = np.zeros(5) if dist is None else dist
    if cands is None:
        cands = find_candidates(image_bgr, markers, frames)
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
        for R, t, err in _solutions(world[used], np.array(pts, np.float64), K, dist):
            if err.max() > thr:
                continue
            found.append((fit_score(err, len(markers) - len(used), False), combo, used, R, t, err))
    found.sort(key=lambda f: f[0])
    best = None
    for _, combo, used, R, t, err in found[: (12 if estimate_focal else 1)]:
        obj = world[used]
        img = np.array([combo[i] for i in used], np.float64)
        Kc = K
        spread = spread_ok(obj)
        well = spread and len(used) >= 5
        if estimate_focal:
            # focale libre autour de la valeur nominale de l'objectif ; retenue plus fort quand
            # les boules la déterminent mal (moins de 5 boules, ou 3 alignées + 1). Elle est
            # de toute façon affinée ensuite avec le corps de l'acteur.
            sol = _solve_with_focal(obj, img, K, R, t, prior_sigma=0.20 if well else 0.10)
            if sol is None:
                continue
            Kc, R, t, err = sol
        if err.max() > max_err_px * (3 if estimate_focal else 2):
            continue
        rms = float(np.sqrt(np.mean(err ** 2)))
        score = fit_score(err, len(markers) - len(used), estimate_focal)
        if best is None or score < best["score"]:
            best = dict(score=score, R=R, t=t, K=Kc, dist=dist, rms=rms, used=used,
                        focal_estimated=bool(estimate_focal and well), aligned=not spread,
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
