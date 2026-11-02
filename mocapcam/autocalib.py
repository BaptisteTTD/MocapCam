"""Calibration automatique sans damier.

Les boules au sol donnent une première estimation de la focale et de la position de
chaque caméra. Ensuite, le corps de l'acteur, vu par les deux caméras sur des centaines
d'images, fournit des milliers de points communs. Un ajustement global (bundle adjustment)
affine en même temps, pour chaque caméra : position, orientation, focale et déformation
de l'objectif. Les boules restent dans le calcul et fixent le sol, l'origine et l'échelle.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from .skeleton import BODY_IDX, SWAP_LR
from . import triangulate as tri

NCP = 9  # paramètres par caméra : rotation (3), translation (3), log focale, k1, k2
# a priori sur la déformation selon l'objectif (l'ultra grand-angle déforme davantage)
DIST_SIGMA = {"wide": (0.08, 0.03), "ultra": (0.30, 0.15)}


def _project(cp, X, cx, cy, f0):
    """cp (9,) paramètres caméra, X (N,3) -> pixels (N,2). Modèle OpenCV (k1, k2)."""
    R = Rotation.from_rotvec(cp[:3]).as_matrix()
    Xc = X @ R.T + cp[3:6]
    z = Xc[:, 2:3]
    x = Xc[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)
    r2 = np.sum(x * x, axis=1, keepdims=True)
    xd = x * (1 + cp[7] * r2 + cp[8] * r2 * r2)
    f = f0 * np.exp(cp[6])
    return np.column_stack([f * xd[:, 0] + cx, f * xd[:, 1] + cy])


def collect(cams, poses, delays, fps, volume, cfg, max_frames=240, min_score=0.5):
    """Sélectionne des images, la bonne personne et l'éventuelle inversion G/D,
    et renvoie les observations pixel des articulations du corps."""
    T = poses[0]["kp"].shape[0]
    frames = np.linspace(0, T - 1, min(max_frames, T)).astype(int)
    X, _, choices = tri.reconstruct(cams, poses, delays, fps, volume, cfg,
                                    log=lambda *a: None, frames=frames, return_choices=True)
    obs = []  # (point_id, cam, u, v, poids)
    pts = []
    for i, (picked, flips) in choices.items():
        per_cam = []
        for v, pk in enumerate(picked):
            if pk is None:
                per_cam.append(None)
                continue
            j, p = pk
            kp, sc = poses[v]["kp"][j, p], poses[v]["sc"][j, p]
            if v > 0 and flips[v - 1]:
                kp, sc = kp[SWAP_LR], sc[SWAP_LR]
            per_cam.append((kp, sc))
        for jt in BODY_IDX:
            if not np.isfinite(X[i, jt]).all():
                continue
            seen = [(v, pc[0][jt], pc[1][jt]) for v, pc in enumerate(per_cam)
                    if pc is not None and pc[1][jt] >= min_score and np.isfinite(pc[0][jt]).all()]
            if len(seen) < 2:
                continue
            pid = len(pts)
            pts.append(X[i, jt])
            for v, uv, s in seen:
                obs.append((pid, v, uv[0], uv[1], s))
    return np.array(pts), np.array(obs)


def refine(cams, poses, delays, fps, volume, cfg=None, fixed_intrinsics=None, log=print):
    """Affine les caméras. fixed_intrinsics[v] = True : focale/distorsion de la caméra v
    viennent d'une calibration au damier et ne sont pas modifiées."""
    cfg = cfg or {}
    V = len(cams)
    fixed_intrinsics = fixed_intrinsics or [False] * V
    pts, obs = collect(cams, poses, delays, fps, volume, cfg)
    if len(pts) < 50:
        log("  pas assez de points communs pour affiner (acteur trop peu visible) : "
            "estimation par les boules conservée")
        return cams, None
    f0 = np.array([c["K"][0, 0] for c in cams])
    cxy = np.array([[c["K"][0, 2], c["K"][1, 2]] for c in cams])
    cam0 = np.zeros((V, NCP))
    for v, c in enumerate(cams):
        cam0[v, :3] = Rotation.from_matrix(c["R"]).as_rotvec()
        cam0[v, 3:6] = np.asarray(c["t"], float)
        dv = np.asarray(c["dist"], float).ravel()
        cam0[v, 7] = float(dv[0]) if dv.size > 0 else 0.0
        cam0[v, 8] = float(dv[1]) if dv.size > 1 else 0.0
    # observations des boules (positions connues)
    mobs = []
    for v, c in enumerate(cams):
        for world, uv in c.get("marker_obs", []):
            mobs.append((v, *world, *uv))
    mobs = np.array(mobs, float)

    npnt = len(pts)
    pid = obs[:, 0].astype(int)
    cid = obs[:, 1].astype(int)
    uv = obs[:, 2:4]
    w = np.sqrt(obs[:, 4])
    w_mark = 4.0
    sig_f = 0.5
    sig = [DIST_SIGMA.get(c.get("lens") or "wide", DIST_SIGMA["wide"]) for c in cams]

    def unpack(p):
        return p[: V * NCP].reshape(V, NCP), p[V * NCP:].reshape(npnt, 3)

    def residuals(p):
        cp, X = unpack(p)
        r = np.empty((len(obs), 2))
        for v in range(V):
            m = cid == v
            r[m] = (_project(cp[v], X[pid[m]], *cxy[v], f0[v]) - uv[m]) * w[m, None]
        out = [r.ravel()]
        if len(mobs):
            rm = np.empty((len(mobs), 2))
            for v in range(V):
                m = mobs[:, 0] == v
                if m.any():
                    rm[m] = (_project(cp[v], mobs[m, 1:4], *cxy[v], f0[v]) - mobs[m, 4:6]) * w_mark
            out.append(rm.ravel())
        pri = []
        for v in range(V):
            if fixed_intrinsics[v]:
                pri += [(cp[v, 6]) * 1e3, (cp[v, 7] - cam0[v, 7]) * 1e3, (cp[v, 8] - cam0[v, 8]) * 1e3]
            else:
                pri += [cp[v, 6] / sig_f, cp[v, 7] / sig[v][0], cp[v, 8] / sig[v][1]]
        out.append(np.array(pri))
        return np.concatenate(out)

    # structure creuse du jacobien
    n_res = 2 * len(obs) + 2 * len(mobs) + 3 * V
    A = lil_matrix((n_res, V * NCP + 3 * npnt), dtype=int)
    rows = np.arange(len(obs))
    for k in range(NCP):
        A[2 * rows, cid * NCP + k] = 1
        A[2 * rows + 1, cid * NCP + k] = 1
    for k in range(3):
        A[2 * rows, V * NCP + 3 * pid + k] = 1
        A[2 * rows + 1, V * NCP + 3 * pid + k] = 1
    base = 2 * len(obs)
    for i, mo in enumerate(mobs):
        v = int(mo[0])
        A[base + 2 * i, v * NCP: (v + 1) * NCP] = 1
        A[base + 2 * i + 1, v * NCP: (v + 1) * NCP] = 1
    base += 2 * len(mobs)
    for v in range(V):
        A[base + 3 * v, v * NCP + 6] = 1
        A[base + 3 * v + 1, v * NCP + 7] = 1
        A[base + 3 * v + 2, v * NCP + 8] = 1

    p0 = np.concatenate([cam0.ravel(), pts.ravel()])
    r0 = residuals(p0)
    sol = least_squares(residuals, p0, jac_sparsity=A, x_scale="jac", loss="soft_l1",
                        f_scale=3.0, method="trf", max_nfev=60)
    cp, _ = unpack(sol.x)
    r1 = sol.fun[: 2 * len(obs)].reshape(-1, 2) / w[:, None]
    e0 = np.linalg.norm(r0[: 2 * len(obs)].reshape(-1, 2) / w[:, None], axis=1)
    e1 = np.linalg.norm(r1, axis=1)
    out = []
    info = []
    for v, c in enumerate(cams):
        f = f0[v] * np.exp(cp[v, 6])
        K = np.array([[f, 0, cxy[v, 0]], [0, f, cxy[v, 1]], [0, 0, 1.0]])
        R = Rotation.from_rotvec(cp[v, :3]).as_matrix()
        t = cp[v, 3:6].copy()
        d = dict(c, K=K, dist=np.array([cp[v, 7], cp[v, 8], 0, 0, 0.0]), R=R, t=t,
                 center=(-R.T @ t).tolist())
        out.append(d)
        info.append(dict(name=c.get("name", str(v)), focal_px=float(f), k1=float(cp[v, 7]), k2=float(cp[v, 8]),
                         moved_cm=float(np.linalg.norm(np.array(d["center"]) - np.array(c["center"])) * 100)
                         if "center" in c else None))
    stats = dict(points=int(npnt), observations=int(len(obs)),
                 reproj_before_px=float(np.median(e0)), reproj_after_px=float(np.median(e1)),
                 cameras=info)
    return out, stats
