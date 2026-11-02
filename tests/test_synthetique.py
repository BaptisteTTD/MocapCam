"""Test de bout en bout sur données de synthèse (nécessite le module bpy : pip install bpy).

  python tests/test_synthetique.py

Vérifie : synchro audio, calibration optique, détection des boules + PnP,
triangulation (bruit, inversions G/D, mains cachées, personne parasite),
lissage, export, puis application dans Blender sur 3 squelettes différents
(généré, A-pose, FBX Mixamo typique : échelle 0.01 + rotation 90°).
"""
import json
import math
import os
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "blender_addon"))

from mocapcam import intrinsics, markers as mk, sync, triangulate, pipeline  # noqa: E402
from mocapcam.skeleton import JOINT_NAMES, INDEX, HAND_IDX, BODY_IDX, SWAP_LR  # noqa: E402

RNG = np.random.default_rng(0)
FPS = 60
W, H = 1920, 1080
K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
DIST = np.zeros(5)
OUT = tempfile.mkdtemp(prefix="mocapcam_test_")
RESULTS = {}


def ok(name, cond, detail):
    RESULTS[name] = (bool(cond), detail)
    print(f"[{'OK' if cond else 'ÉCHEC'}] {name} : {detail}")


def look_at(C, target):
    C, target = np.asarray(C, float), np.asarray(target, float)
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ C


CAMS_TRUE = [look_at([-1.0, -1.8, 1.4], [1.25, 1.25, 0.9]),
             look_at([3.4, -1.4, 1.2], [1.25, 1.25, 0.9])]
# objectifs réels, inconnus du logiciel quand il n'y a pas de damier
K_TRUE = [np.array([[1440.0, 0, 960], [0, 1440.0, 540], [0, 0, 1]]),
          np.array([[1580.0, 0, 960], [0, 1580.0, 540], [0, 0, 1]])]
DIST_TRUE = [np.array([0.0, 0, 0, 0, 0]), np.array([-0.06, 0, 0, 0, 0])]


# --------------------------------------------------------------------------- 1. synchro

def test_sync():
    sr = 16000
    n = sr * 12
    base = RNG.normal(0, 0.01, n).astype(np.float32)
    clap = np.zeros(n, np.float32)
    t0 = int(3.2 * sr)
    clap[t0:t0 + 400] = RNG.normal(0, 0.8, 400) * np.exp(-np.arange(400) / 80)
    a = base + clap
    true_d = 0.437
    shift = int(true_d * sr)
    b = np.concatenate([RNG.normal(0, 0.01, shift).astype(np.float32), a])[:n] * 0.6
    d, conf = sync.audio_delay(a, b, sr)
    ok("synchro audio", abs(d - true_d) < 0.003, f"trouvé {d * 1000:.1f} ms (vrai {true_d * 1000:.0f} ms), confiance {conf:.1f}")


# --------------------------------------------------------------------------- 2. intrinsèques

def test_intrinsics():
    board = cv2.imread(intrinsics.make_checkerboard(os.path.join(OUT, "damier.png")), 0)
    bh, bw = board.shape
    vid = os.path.join(OUT, "damier.mp4")
    vw = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    sw = 0.6  # écran de 60 cm
    corners_board = np.array([[0, 0], [bw, 0], [bw, bh], [0, bh]], np.float32)
    obj = np.array([[0, 0, 0], [sw, 0, 0], [sw, sw * bh / bw, 0], [0, sw * bh / bw, 0]])
    obj -= obj.mean(0)
    for i in range(60):
        rv = RNG.normal(0, 0.35, 3)
        tv = np.array([RNG.uniform(-0.25, 0.25), RNG.uniform(-0.15, 0.15), RNG.uniform(0.75, 1.2)])
        pts, _ = cv2.projectPoints(obj, rv, tv, K, DIST)
        Hm = cv2.getPerspectiveTransform(corners_board, pts.reshape(-1, 2).astype(np.float32))
        img = cv2.warpPerspective(board, Hm, (W, H), borderValue=110)
        vw.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    vw.release()
    data = intrinsics.calibrate_from_video(vid, log=lambda *a: None)
    fx = data["K"][0][0]
    ok("calibration optique (damier)", abs(fx - 1500) < 15 and data["rms"] < 1.0,
       f"focale {fx:.1f} px (vraie 1500), RMS {data['rms']:.2f} px, {data['views']} vues")


# --------------------------------------------------------------------------- 3. boules au sol

def color_bgr(name):
    h = mk.COLOR_PRESETS[name][0]
    hsv = np.uint8([[[h, 210, 210]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def render_floor(R, t, markers, Kc=K, Dc=DIST):
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (70, 95, 120)
    img = (img + RNG.normal(0, 8, img.shape) + (yy[..., None] / H) * 30).clip(0, 255).astype(np.uint8)
    # parasites : grand meuble rouge, petite tache jaune hors zone
    cv2.rectangle(img, (1500, 100), (1800, 400), color_bgr("rouge"), -1)
    cv2.circle(img, (200, 150), 9, color_bgr("jaune"), -1)
    rvec, _ = cv2.Rodrigues(R)
    for m in markers:
        p, _ = cv2.projectPoints(np.array([m["position"]], float), rvec, t, Kc, Dc)
        depth = (R @ m["position"] + t)[2]
        r = max(3, int(round(Kc[0, 0] * 0.02 / depth)))
        cv2.circle(img, tuple(int(round(v)) for v in p.ravel()), r, color_bgr(m["color"]), -1, cv2.LINE_AA)
    return img


def test_markers():
    cfg = dict(pipeline.DEFAULT_STUDIO)
    mlist = [dict(m, position=[m["position"][0], m["position"][1], 0.01]) for m in cfg["markers"]]
    msg = []
    worst = 0
    for ci, (R, t) in enumerate(CAMS_TRUE):
        img = render_floor(R, t, mlist, K_TRUE[ci], DIST_TRUE[ci])
        pose = mk.solve_camera_pose(img, mlist)          # sans calibration optique
        vol = mk.volume_box(mlist)
        cv2.imwrite(os.path.join(OUT, f"calibration_cam{ci}.jpg"),
                    mk.draw_debug(img, mlist, pose["K"], pose["dist"], pose, vol))
        dC = np.linalg.norm(np.array(pose["center"]) + R.T @ t)
        df = pose["K"][0, 0] / K_TRUE[ci][0, 0] - 1
        worst = max(worst, dC)
        msg.append(f"cam{ci}: écart {dC * 100:.1f} cm, focale {pose['K'][0, 0]:.0f} px "
                   f"(vraie {K_TRUE[ci][0, 0]:.0f}, {df * 100:+.1f} %)")
    ok("boules seules (sans damier) : première estimation", worst < 0.4, "; ".join(msg))


def _rigid2d(src, dst):
    """Transformation rigide du plan (sans miroir) qui amène src sur dst."""
    a, b = src - src.mean(0), dst - dst.mean(0)
    U, _, Vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1, d]) @ U.T
    return Rm, dst.mean(0) - Rm @ src.mean(0), d > 0


CAMS_LAYOUT = [look_at([-1.6, -2.4, 1.6], [1.25, 1.25, 0.2]),
               look_at([4.3, -1.9, 1.5], [1.25, 1.25, 0.2])]


def test_layout_errors():
    """Boules mal placées comme chez un vrai utilisateur : posées à ±4 cm près, deux boules
    échangées ou tout en miroir, et une caméra qui ne voit que 4 boules (dont 3 alignées).
    Le logiciel doit retrouver la bonne disposition et la bonne position des caméras."""
    cfg = dict(pipeline.DEFAULT_STUDIO)
    mlist = [dict(m, position=[m["position"][0], m["position"][1], 0.01]) for m in cfg["markers"]]
    rng = np.random.default_rng(3)
    idx = {m["name"]: i for i, m in enumerate(mlist)}
    mirror_in_place = [dict(m, position=[2.5 - m["position"][0]] + list(m["position"][1:])) for m in mlist]
    cases = {
        "telle que décrite": mlist,
        "miroir": mirror_in_place,
        "A<->D": mk.swapped(mlist, idx["A"], idx["D"]),
        "B<->C": mk.swapped(mlist, idx["B"], idx["C"]),
    }
    msg, good = [], True
    for label, phys in cases.items():
        phys = [dict(m, position=[m["position"][0] + rng.uniform(-0.04, 0.04),
                                  m["position"][1] + rng.uniform(-0.04, 0.04), 0.01]) for m in phys]
        views = []
        for ci, (R, t) in enumerate(CAMS_LAYOUT):
            # caméra 1 : la boule posée au coin (2,5 ; 0) est cachée par un meuble
            shown = [m for m in phys if not (ci == 1 and np.hypot(m["position"][0] - 2.5, m["position"][1]) < 0.2)]
            img = render_floor(R, t, shown, K_TRUE[ci], DIST_TRUE[ci])
            views.append(dict(name=f"cam{ci}", bg=img, K=None, dist=None, lens="wide",
                              cands=mk.find_candidates(img, mlist)))
        ml, poses, lab = pipeline.choose_layout(views, mlist, log=lambda m: None)
        # repère choisi par le logiciel <-> repère réel : doit être une transformation rigide
        src = np.array([m["position"][:2] for m in phys])
        dst = np.array([{x["name"]: x for x in ml}[m["name"]]["position"][:2] for m in phys])
        Rm, tm, proper = _rigid2d(src, dst)
        fit = np.abs(src @ Rm.T + tm - dst).max()
        errs = []
        for p, (R, t) in zip(poses, CAMS_LAYOUT):
            if p is None:
                errs.append(np.inf)
                continue
            C = -R.T @ t
            C2 = np.array([*(Rm @ C[:2] + tm), C[2]])
            errs.append(np.linalg.norm(np.array(p["center"]) - C2))
        # première estimation seulement (focale nominale, boules imprécises) : le corps
        # de l'acteur l'affine ensuite
        found = proper and fit < 0.15 and max(errs) < 0.8
        good &= found
        msg.append(f"{label} -> {lab} (caméras à {', '.join(f'{e * 100:.0f}' for e in errs)} cm)")
    ok("boules mal placées (échange, miroir, ±4 cm, 3 alignées + 1)", good, " ; ".join(msg))


# --------------------------------------------------------------------------- 4. mouvement de référence (Blender)

def reset_blender():
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy


def make_ground_truth(T=300, fps=60):
    bpy = reset_blender()
    import mocapcam_blender as mb
    from mathutils import Euler, Quaternion
    empty = {n: np.full((10, 3), np.nan) for n in JOINT_NAMES}
    arm = mb.create_mixamo_armature(bpy.context, empty, name="GT")
    arm.location = (1.25, 1.4, 0.0)
    P = mb.PREFIX
    anim = {
        "Hips": lambda s: Euler((0.1 * math.sin(s), 0.35 * math.sin(0.5 * s), 0)),
        "Spine1": lambda s: Euler((0.15 * math.sin(s), 0, 0.1 * math.cos(s))),
        "Neck": lambda s: Euler((0, 0.3 * math.sin(0.7 * s), 0)),
        "Head": lambda s: Euler((0.2 * math.sin(1.3 * s), 0.3 * math.sin(0.9 * s), 0)),
        "LeftArm": lambda s: Euler((0.5 * math.sin(s), 0.3, -0.9 + 0.6 * math.sin(0.8 * s))),
        "LeftForeArm": lambda s: Euler((0, 0.6 * math.sin(s), 0.9 + 0.7 * math.sin(1.2 * s))),
        "LeftHand": lambda s: Euler((0.4 * math.sin(2 * s), 0, 0.3 * math.cos(s))),
        "RightArm": lambda s: Euler((-0.4 * math.sin(s), -0.2, 1.0 + 0.5 * math.sin(0.7 * s))),
        "RightForeArm": lambda s: Euler((0, -0.5 * math.sin(s), -1.0 - 0.6 * math.sin(s))),
        "RightHand": lambda s: Euler((0.5 * math.sin(1.5 * s), 0.3 * math.sin(s), 0)),
        "LeftUpLeg": lambda s: Euler((0.5 * math.sin(s), 0, 0.1)),
        "LeftLeg": lambda s: Euler((0.6 * (1 - math.cos(s)), 0, 0)),
        "RightUpLeg": lambda s: Euler((-0.5 * math.sin(s), 0, -0.1)),
        "RightLeg": lambda s: Euler((0.6 * (1 + math.cos(s)), 0, 0)),
        "LeftFoot": lambda s: Euler((0.3 * math.sin(s), 0, 0)),
    }
    for side, sg in (("Left", 1), ("Right", -1)):
        for f in ("Index", "Middle", "Ring", "Pinky"):
            for k in (1, 2, 3):
                anim[f"{side}Hand{f}{k}"] = (lambda s, k=k, sg=sg, f=f:
                                            Euler((0, 0, -sg * 0.7 * (1 + math.sin(1.5 * s + k + len(f))) / 2 * (1.2 if k == 1 else 1.0))))
        for k in (1, 2, 3):
            anim[f"{side}HandThumb{k}"] = lambda s, sg=sg: Euler((0.4 * math.sin(s), 0, -sg * 0.3 * (1 + math.sin(s))))
    X = np.full((T, 133, 3), np.nan)
    for f in range(T):
        s = 2 * math.pi * (f * 60.0 / fps) / 150.0
        for bn, fn in anim.items():
            pb = arm.pose.bones[P + bn]
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = fn(s).to_quaternion()
        arm.pose.bones[P + "Hips"].location = (0.15 * math.sin(0.5 * s), 0.0, 0.05 * math.sin(2 * s))
        bpy.context.view_layer.update()
        J = mb.virtual_joints(arm, P, posed=True)
        for n, v in J.items():
            if n in INDEX:
                X[f, INDEX[n]] = v
    return X


# --------------------------------------------------------------------------- 5. caméras virtuelles + triangulation

def project(X, R, t, Kc=K, Dc=DIST):
    shp = X.shape
    pr, _ = cv2.projectPoints(X.reshape(-1, 3).astype(np.float64), cv2.Rodrigues(R)[0], t, Kc, Dc)
    return pr.reshape(*shp[:-1], 2)


def synth_2d(X_gt):
    T = X_gt.shape[0]
    poses = []
    truth_valid = np.isfinite(X_gt[..., 0])
    for ci, (R, t) in enumerate(CAMS_TRUE):
        kp = np.full((T, 2, 133, 2), np.nan, np.float32)
        sc = np.zeros((T, 2, 133), np.float32)
        uv = project(np.nan_to_num(X_gt), R, t, K_TRUE[ci], DIST_TRUE[ci])
        noise = RNG.normal(0, 1.5, uv.shape)
        noise[:, HAND_IDX] = RNG.normal(0, 2.0, (T, len(HAND_IDX), 2))
        uv = uv + noise
        s = np.where(truth_valid, RNG.uniform(0.6, 0.95, (T, 133)), 0.0)
        # main droite cachée pour la caméra 0 entre 100 et 130 (occlusion)
        if ci == 0:
            s[100:130, 112:133] = 0.05
        # erreurs du détecteur : quelques points aberrants
        bad = RNG.random((T, 133)) < 0.01
        uv[bad] += RNG.normal(0, 80, (bad.sum(), 2))
        # inversion gauche/droite sur la caméra 1, une image sur 7
        if ci == 1:
            for f in range(0, T, 7):
                uv[f] = uv[f][SWAP_LR]
                s[f] = s[f][SWAP_LR]
        # personne parasite (spectateur) en première position une image sur 3
        for f in range(T):
            slot = 0
            if f % 3 == 0:
                kp[f, 0] = uv[f] + np.array([700.0, -50.0]) * (1 if ci == 0 else -1)
                sc[f, 0] = s[f] * 0.9
                slot = 1
            kp[f, slot] = uv[f]
            sc[f, slot] = s[f]
        poses.append(dict(kp=kp, sc=sc, width=W, height=H))
    return poses


def test_reconstruction(X_gt):
    cfg = pipeline.DEFAULT_STUDIO
    poses = synth_2d(X_gt)
    mlist = [dict(m, position=[m["position"][0], m["position"][1], 0.01]) for m in cfg["markers"]]
    vol = mk.volume_box(mlist, 0.5, 2.3)
    # comme le logiciel réel, sans damier : boules -> première estimation
    cams = []
    for ci, (R, t) in enumerate(CAMS_TRUE):
        img = render_floor(R, t, mlist, K_TRUE[ci], DIST_TRUE[ci])
        pz = mk.solve_camera_pose(img, mlist)
        cams.append(dict(name=f"cam{ci}", K=pz["K"], dist=np.zeros(5), R=pz["R"], t=pz["t"],
                         center=pz["center"], marker_obs=pz["marker_obs"], width=W, height=H))
    # décalage de synchro : la caméra 1 a démarré 2 images plus tôt
    shift = 2
    poses[1]["kp"] = np.concatenate([np.full_like(poses[1]["kp"][:shift], np.nan), poses[1]["kp"]])
    poses[1]["sc"] = np.concatenate([np.zeros_like(poses[1]["sc"][:shift]), poses[1]["sc"]])
    delays = [0.0, shift / FPS]
    from mocapcam import autocalib
    X0, _ = triangulate.reconstruct(cams, poses, delays, FPS, vol, cfg["reconstruction"], log=lambda *a: None)
    cams, ac = autocalib.refine(cams, poses, delays, FPS, vol, cfg["reconstruction"], log=print)
    msg = []
    worst_f, worst_c = 0, 0
    for ci, (R, t) in enumerate(CAMS_TRUE):
        df = cams[ci]["K"][0, 0] / K_TRUE[ci][0, 0] - 1
        dC = np.linalg.norm(np.array(cams[ci]["center"]) + R.T @ t)
        worst_f, worst_c = max(worst_f, abs(df)), max(worst_c, dC)
        msg.append(f"cam{ci}: focale {cams[ci]['K'][0, 0]:.0f} px (vraie {K_TRUE[ci][0, 0]:.0f}, {df * 100:+.2f} %), "
                   f"k1 {cams[ci]['dist'][0]:+.3f} (vrai {DIST_TRUE[ci][0]:+.3f}), position à {dC * 100:.1f} cm")
    ok("auto-calibration boules + corps (sans damier)", worst_f < 0.01 and worst_c < 0.03,
       "; ".join(msg) + f" ; reprojection {ac['reproj_before_px']:.1f} -> {ac['reproj_after_px']:.1f} px")
    X, stats = triangulate.reconstruct(cams, poses, delays, FPS, vol, cfg["reconstruction"], log=lambda *a: None)
    Xs = triangulate.postprocess(X, FPS, cfg["filtering"])
    valid = np.isfinite(X_gt[..., 0])
    def err(idx, A):
        e = np.linalg.norm(A[:, idx] - X_gt[:, idx], axis=-1)
        m = valid[:, idx] & np.isfinite(e)
        return np.median(e[m]) * 1000, np.percentile(e[m], 95) * 1000, m.sum() / valid[:, idx].sum()
    body_idx = [i for i in BODY_IDX]
    eb_raw = err(body_idx, X)
    eb0 = err(body_idx, triangulate.postprocess(X0, FPS, cfg["filtering"]))
    print(f"    (pour comparaison, avec les boules seules : corps {eb0[0]:.1f} mm)")
    eb = err(body_idx, Xs)
    eh = err(HAND_IDX, Xs)
    ok("triangulation corps", eb[0] < 10 and eb[2] > 0.97,
       f"erreur médiane {eb[0]:.1f} mm (brut {eb_raw[0]:.1f}), 95 % < {eb[1]:.1f} mm, couverture {eb[2] * 100:.0f} %")
    ok("triangulation mains", eh[0] < 10 and eh[2] > 0.9,
       f"erreur médiane {eh[0]:.1f} mm, 95 % < {eh[1]:.1f} mm, couverture {eh[2] * 100:.0f} %")
    ok("inversions G/D + personne parasite", stats["lr_swaps_fixed"] >= (X_gt.shape[0] // 7) - 2,
       f"{stats['lr_swaps_fixed']} inversions corrigées, {stats['frames_ok']}/{stats['frames']} images")
    cams_full = [dict(c, name=n) for n, c in zip(["iphone", "poco"], cams)]
    cfg2 = json.loads(json.dumps(cfg))
    for m in cfg2["markers"]:
        m["position_3d"] = [m["position"][0], m["position"][1], 0.01]
    path = pipeline.export_motion(os.path.join(OUT, "test.mocap.json"), Xs, FPS, cams_full, cfg2, vol, stats)
    return path


# --------------------------------------------------------------------------- 6b. synchro par le mouvement

def test_sync_motion(X_gt):
    """Le son d'un téléphone est décalé de son image (90 ms ici, comme mesuré sur un vrai
    Poco) : le bip donne donc un décalage faux. La synchro par le mouvement doit le corriger."""
    from mocapcam import syncbody
    T = X_gt.shape[0]
    true_d = 0.090
    t = np.arange(T)
    poses = []
    for ci, (R, tc) in enumerate(CAMS_TRUE):
        # image j de la caméra 1 = instant j/FPS - true_d de la référence
        src = t - (true_d * FPS if ci == 1 else 0.0)
        Xs = np.full_like(X_gt, np.nan)
        for j in range(X_gt.shape[1]):
            valid = np.isfinite(X_gt[:, j, 0])
            if valid.sum() > 2:
                for k in range(3):
                    Xs[:, j, k] = np.interp(src, t[valid], X_gt[valid, j, k], left=np.nan, right=np.nan)
        uv = project(np.nan_to_num(Xs), R, tc, K_TRUE[ci], DIST_TRUE[ci]) + RNG.normal(0, 1.5, (T, 133, 2))
        sc = np.where(np.isfinite(Xs[..., 0]), 0.8, 0.0)
        uv[~np.isfinite(Xs[..., 0])] = np.nan
        poses.append(dict(kp=uv[:, None].astype(np.float32), sc=sc[:, None].astype(np.float32), width=W, height=H))
    cams = [dict(K=K_TRUE[i], dist=DIST_TRUE[i], R=R, t=tc) for i, (R, tc) in enumerate(CAMS_TRUE)]
    mlist = [dict(m, position=[m["position"][0], m["position"][1], 0.01]) for m in pipeline.DEFAULT_STUDIO["markers"]]
    vol = mk.volume_box(mlist, 0.5, 2.3)
    logs = []
    d = syncbody.refine(cams, poses, ["cam0", "cam1"], [FPS, FPS], FPS, [0.0, 0.0], T, vol,
                        pipeline.DEFAULT_STUDIO["reconstruction"], log=logs.append)
    err = abs(d[1] - true_d) * 1000
    ok("synchro par le mouvement (son décalé de l'image)", err < 4,
       f"décalage trouvé {d[1] * 1000:.1f} ms (vrai {true_d * 1000:.0f} ms, bip : 0 ms) ; " + (logs[-1].strip() if logs else ""))


# --------------------------------------------------------------------------- 7. export FBX

def test_fbx(path):
    """Capture -> FBX (squelette Mixamo animé) -> réimport : mêmes positions, mêmes noms d'os."""
    import bpy
    import mocapcam_blender as mb
    dst = os.path.join(OUT, "test_mixamo.fbx")
    rep = mb.capture_to_fbx(path, dst)
    arm = bpy.data.objects["Armature"]
    sc = bpy.context.scene
    frames = [sc.frame_start, (sc.frame_start + sc.frame_end) // 2, sc.frame_end]

    def heads(obj, frs):
        out = []
        for f in frs:
            bpy.context.scene.frame_set(f)
            bpy.context.view_layer.update()
            out.append({b.name: np.array(obj.matrix_world @ b.head) for b in obj.pose.bones})
        return out
    ref = heads(arm, frames)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=dst)
    a2 = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]
    act = a2.animation_data.action
    off = int(round(act.frame_range[0])) - frames[0]   # Blender place le temps 0 à l'image 1
    got = heads(a2, [f + off for f in frames])
    err = max(np.linalg.norm(g[n] - r[n]) for g, r in zip(got, ref) for n in r)
    names = {b.name for b in a2.data.bones}
    n_frames = int(round(act.frame_range[1] - act.frame_range[0])) + 1
    # pose de repos du fichier = T-pose (bras à l'horizontale, jambes verticales)
    M = a2.matrix_world.to_3x3()
    tpose = {"LeftArm": (1, 0, 0), "LeftForeArm": (1, 0, 0), "RightArm": (-1, 0, 0), "RightForeArm": (-1, 0, 0),
             "LeftUpLeg": (0, 0, -1), "RightLeg": (0, 0, -1), "Spine": (0, 0, 1)}
    t_err = max(math.degrees(math.acos(min(1.0, float(np.dot(
        np.array((M @ (a2.data.bones["mixamorig:" + n].tail_local - a2.data.bones["mixamorig:" + n].head_local)).normalized()),
        np.array(v, float)))))) for n, v in tpose.items())
    ok("export FBX squelette Mixamo", err < 1e-3 and len(names) == 65 and t_err < 2
       and all(n.startswith("mixamorig:") for n in names) and n_frames == rep["frames"],
       f"{len(names)} os 'mixamorig:', pose de repos T-pose (écart {t_err:.1f}°), {n_frames} images à "
       f"{rep['fps']:g} i/s, animation « {act.name} », écart après réimport {err * 1000:.2f} mm, "
       f"{os.path.getsize(dst) / 1e6:.1f} Mo")


# --------------------------------------------------------------------------- 5b. boules imprécises

def test_imprecise_balls(X_gt):
    """Conditions réelles : boules posées à ±4 cm près, focale inconnue (objectif nominal
    ±10 %), une caméra qui ne voit que 3 boules alignées + 1. Le corps de l'acteur doit
    corriger la première estimation."""
    from mocapcam import autocalib
    cfg = pipeline.DEFAULT_STUDIO
    rng = np.random.default_rng(7)
    mlist = [dict(m, position=[m["position"][0], m["position"][1], 0.01]) for m in cfg["markers"]]
    phys = [dict(m, position=[m["position"][0] + rng.uniform(-0.04, 0.04),
                              m["position"][1] + rng.uniform(-0.04, 0.04), 0.01]) for m in mlist]
    views = []
    for ci, (R, t) in enumerate(CAMS_TRUE):
        shown = [m for m in phys if not (ci == 1 and m["name"] == "B")]
        img = render_floor(R, t, shown, K_TRUE[ci], DIST_TRUE[ci])
        views.append(dict(name=f"cam{ci}", bg=img, K=None, dist=None, lens="wide",
                          cands=mk.find_candidates(img, mlist)))
    ml, pz, lab = pipeline.choose_layout(views, mlist, log=lambda m: None)
    vol = mk.volume_box(mlist, 0.5, 2.3)
    cams = [dict(name=f"cam{ci}", K=p["K"], dist=np.zeros(5), R=p["R"], t=p["t"], center=p["center"],
                 marker_obs=p["marker_obs"], width=W, height=H, lens="wide",
                 f_nominal=mk.default_K(W, H, "wide")[0, 0]) for ci, p in enumerate(pz)]
    first = [np.linalg.norm(np.array(c["center"]) + R.T @ t) for c, (R, t) in zip(cams, CAMS_TRUE)]
    poses = synth_2d(X_gt)
    delays = [0.0, 0.0]
    cams, ac = autocalib.refine(cams, poses, delays, FPS, vol, cfg["reconstruction"], log=lambda *a: None)
    X, stats = triangulate.reconstruct(cams, poses, delays, FPS, vol, cfg["reconstruction"], log=lambda *a: None)
    Xs = triangulate.postprocess(X, FPS, cfg["filtering"])
    msg, worst_c, worst_f = [], 0, 0
    for ci, (R, t) in enumerate(CAMS_TRUE):
        dC = np.linalg.norm(np.array(cams[ci]["center"]) + R.T @ t)
        df = cams[ci]["K"][0, 0] / K_TRUE[ci][0, 0] - 1
        worst_c, worst_f = max(worst_c, dC), max(worst_f, abs(df))
        msg.append(f"cam{ci}: {first[ci] * 100:.0f} -> {dC * 100:.1f} cm, focale {df * 100:+.1f} %")
    # le repère (sol, origine) ne peut pas être plus juste que la pose des boules : on mesure
    # à part le décalage global et la qualité du mouvement une fois ce décalage retiré
    A = Xs[:, BODY_IDX].reshape(-1, 3)
    B = X_gt[:, BODY_IDX].reshape(-1, 3)
    m = np.isfinite(A).all(1) & np.isfinite(B).all(1)
    a0, b0 = A[m].mean(0), B[m].mean(0)
    U, _, Vt = np.linalg.svd((A[m] - a0).T @ (B[m] - b0))
    Rr = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    e_raw = np.median(np.linalg.norm(A[m] - B[m], axis=1)) * 1000
    e_al = np.median(np.linalg.norm((A[m] - a0) @ Rr.T + b0 - B[m], axis=1)) * 1000
    tilt = np.degrees(np.arccos(np.clip(Rr[2, 2], -1, 1)))
    h, h_gt = pipeline.actor_height(Xs), pipeline.actor_height(X_gt)
    ok("boules imprécises + 3 alignées : affinage par le corps",
       worst_f < 0.03 and e_al < 10 and e_raw < 60 and tilt < 2 and abs(h / h_gt - 1) < 0.03,
       "; ".join(msg) + f" ; corps {e_al:.1f} mm (repère décalé de {e_raw:.0f} mm, sol incliné de "
       f"{tilt:.1f}°) ; taille estimée {h:.2f} m (vraie {h_gt:.2f} m)")


# --------------------------------------------------------------------------- 5c. ultra grand-angle

def test_ultrawide(X_gt):
    """Poco à l'objectif ×0,5 : focale ~2 fois plus courte et forte déformation."""
    from mocapcam import autocalib
    saved = (K_TRUE[1].copy(), DIST_TRUE[1].copy())
    K_TRUE[1] = np.array([[700.0, 0, 960], [0, 700.0, 540], [0, 0, 1]])
    DIST_TRUE[1] = np.array([-0.12, 0.035, 0, 0, 0])
    try:
        mlist = [dict(m, position=[m["position"][0], m["position"][1], 0.016]) for m in pipeline.DEFAULT_STUDIO["markers"]]
        vol = mk.volume_box(mlist)
        poses = synth_2d(X_gt)
        cams = []
        for ci, (R, t) in enumerate(CAMS_TRUE):
            lens = "ultra" if ci == 1 else "wide"
            img = render_floor(R, t, mlist, K_TRUE[ci], DIST_TRUE[ci])
            pz = mk.solve_camera_pose(img, mlist, lens=lens)
            cams.append(dict(name=f"cam{ci}", K=pz["K"], dist=np.zeros(5), R=pz["R"], t=pz["t"], lens=lens,
                             center=pz["center"], marker_obs=pz["marker_obs"], width=W, height=H))
        f_init = cams[1]["K"][0, 0]
        cams, ac = autocalib.refine(cams, poses, [0.0, 0.0], FPS, vol, pipeline.DEFAULT_STUDIO["reconstruction"], log=lambda *a: None)
        X, _ = triangulate.reconstruct(cams, poses, [0.0, 0.0], FPS, vol, pipeline.DEFAULT_STUDIO["reconstruction"], log=lambda *a: None)
        X = triangulate.postprocess(X, FPS, pipeline.DEFAULT_STUDIO["filtering"])
        valid = np.isfinite(X_gt[..., 0])
        e = np.linalg.norm(X[:, BODY_IDX] - X_gt[:, BODY_IDX], axis=-1)
        m = valid[:, BODY_IDX] & np.isfinite(e)
        c1 = cams[1]
        ok("ultra grand-angle (×0,5) sur une caméra", np.median(e[m]) * 1000 < 8,
           f"focale {f_init:.0f} -> {c1['K'][0, 0]:.0f} px (vraie 700), k1 {c1['dist'][0]:+.3f} (vrai -0.120), "
           f"k2 {c1['dist'][1]:+.3f} (vrai +0.035) ; corps : erreur médiane {np.median(e[m]) * 1000:.1f} mm")
    finally:
        K_TRUE[1], DIST_TRUE[1] = saved


# --------------------------------------------------------------------------- 5b. cadences différentes

def test_mixed_fps():
    """iPhone 60 i/s, Poco 30 i/s, décalé d'une demi-image : interpolation vs images dupliquées."""
    X120 = make_ground_truth(T=480, fps=120)       # vérité à 120 i/s (4 s)
    X60 = X120[::2]
    cams = [dict(K=K_TRUE[i], dist=DIST_TRUE[i], R=R, t=t) for i, (R, t) in enumerate(CAMS_TRUE)]
    def obs(Xs, ci):
        uv = project(np.nan_to_num(Xs), *CAMS_TRUE[ci], K_TRUE[ci], DIST_TRUE[ci]) + RNG.normal(0, 1.0, Xs.shape[:2] + (2,))
        sc = np.where(np.isfinite(Xs[..., 0]), 0.9, 0.0).astype(np.float32)
        return dict(kp=uv[:, None].astype(np.float32), sc=sc[:, None])
    p0 = obs(X60, 0)                              # 60 i/s, référence
    p1 = obs(X120[1::4], 1)                       # 30 i/s, démarre 1/120 s plus tard
    d1 = -1 / 120
    mlist = [dict(m, position=[m["position"][0], m["position"][1], 0.01]) for m in pipeline.DEFAULT_STUDIO["markers"]]
    vol = mk.volume_box(mlist)
    T = X60.shape[0]
    valid = np.isfinite(X60[..., 0])
    def err(X):
        e = np.linalg.norm(X - X60, axis=-1)
        m = valid & np.isfinite(e)
        return np.median(e[m]) * 1000, np.percentile(e[m], 95) * 1000
    # ancienne méthode : la vidéo à 30 i/s est dupliquée à 60 i/s, décalage arrondi à l'image
    dup = dict(kp=np.repeat(p1["kp"], 2, 0)[:T], sc=np.repeat(p1["sc"], 2, 0)[:T])
    Xa, _ = triangulate.reconstruct(cams, [p0, dup], [0.0, d1], 60, vol, log=lambda *a: None)
    # nouvelle méthode : interpolation temporelle
    r0 = triangulate.resample_pose(p0, 60, 60, 0.0, T)
    r1 = triangulate.resample_pose(p1, 30, 60, d1, T)
    Xb, _ = triangulate.reconstruct(cams, [r0, r1], [0.0, 0.0], 60, vol, log=lambda *a: None)
    ea, eb = err(Xa), err(Xb)
    ok("cadences mixtes 60 + 30 i/s", eb[0] < ea[0] and eb[0] < 8,
       f"images dupliquées : médiane {ea[0]:.1f} mm (95 % < {ea[1]:.0f} mm) ; "
       f"interpolation : médiane {eb[0]:.1f} mm (95 % < {eb[1]:.0f} mm)")


# --------------------------------------------------------------------------- 6. Blender : application

def test_blender(path, X_gt):
    import mathutils
    bpy = reset_blender()
    import importlib
    import mocapcam_blender as mb
    importlib.reload(mb)
    mb.register()
    d, J = mb.load_motion(path)
    T = d["frame_count"]
    # (a) via l'opérateur d'import (squelette généré + caméras + boules)
    bpy.context.scene.mocapcam_filepath = path
    res = bpy.ops.mocapcam.import_capture(filepath=path, create_points=True)
    gen = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]
    # (b) squelette en A-pose
    apose = mb.create_mixamo_armature(bpy.context, J, name="Apose", pose="A")
    apose.location = (4, 0, 0)
    mb.apply_motion(apose, J, FPS)
    # (c) façon FBX Mixamo : objet tourné de 90° sur X, échelle 0.01, autres proportions
    M = mathutils.Matrix.Translation((-3, 0, 0)) @ mathutils.Matrix.Rotation(math.radians(90), 4, "X") @ mathutils.Matrix.Scale(0.01, 4)
    Jbig = {k: v * 1.15 for k, v in J.items()}
    fbx = mb.create_mixamo_armature(bpy.context, Jbig, name="FBXlike", obj_matrix=M)
    for b in fbx.data.bones:      # préfixe différent comme certains exports
        pass
    mb.apply_motion(fbx, J, FPS)

    check = ["l_wrist", "r_wrist", "l_ankle", "r_ankle", "l_elbow", "r_knee", "nose",
             "lh_8", "lh_4", "rh_12", "rh_20", "l_big_toe"]
    dir_pairs = [("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"), ("r_elbow", "r_wrist"),
                 ("l_hip", "l_knee"), ("r_knee", "r_ankle"), ("lh_5", "lh_6"), ("lh_6", "lh_7"),
                 ("rh_9", "rh_10"), ("rh_1", "rh_2"), ("lh_0", "lh_9")]
    def angle(u, v):
        c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        return math.degrees(math.acos(np.clip(c, -1, 1)))

    body_pairs = [p for p in dir_pairs if not p[0].startswith(("lh_", "rh_"))]
    finger_pairs = [p for p in dir_pairs if p[0].startswith(("lh_", "rh_"))]
    for arm, label in ((gen, "squelette généré"), (apose, "squelette A-pose"), (fbx, "squelette type FBX Mixamo")):
        pos_err = []
        fid = {"corps": [], "doigts": []}      # Blender vs points capturés (fidélité de l'add-on)
        tot = {"corps": [], "doigts": []}      # Blender vs mouvement réel (bilan complet)
        for f in range(0, T, 5):
            bpy.context.scene.frame_set(1 + f)
            Jp = mb.virtual_joints(arm, mb.find_prefix(arm), posed=True)
            for key, pairs in (("corps", body_pairs), ("doigts", finger_pairs)):
                for a, b in pairs:
                    p = Jp[b] - Jp[a]
                    m = J[b][f] - J[a][f]
                    if np.isfinite(m).all():
                        fid[key].append(angle(m, p))
                    tot[key].append(angle(X_gt[f, INDEX[b]] - X_gt[f, INDEX[a]], p))
            if arm is gen:
                rel_g = X_gt[f] - (X_gt[f, INDEX["l_hip"]] + X_gt[f, INDEX["r_hip"]]) / 2
                hip_p = (Jp["l_hip"] + Jp["r_hip"]) / 2
                for n in check:
                    pos_err.append(np.linalg.norm((Jp[n] - hip_p) - rel_g[INDEX[n]]))
        fb, ff = np.array(fid["corps"]), np.array(fid["doigts"])
        tb, tf = np.array(tot["corps"]), np.array(tot["doigts"])
        msg = (f"fidélité add-on : corps {np.median(fb):.2f}° (95 % < {np.percentile(fb, 95):.1f}°), "
               f"doigts {np.median(ff):.2f}° | bilan vs réel : corps {np.median(tb):.1f}°, "
               f"doigts {np.median(tf):.1f}° (95 % < {np.percentile(tf, 95):.0f}°)")
        if pos_err:
            pe = np.array(pos_err) * 1000
            msg += f" | extrémités {np.median(pe):.0f} mm"
        ok(f"Blender {label}", np.median(fb) < 1 and np.percentile(fb, 95) < 3
           and np.median(ff) < 1.5 and np.median(tb) < 2, msg)
    out_blend = os.path.join(OUT, "resultat_test.blend")
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    return out_blend


if __name__ == "__main__":
    print(f"Dossier de test : {OUT}")
    test_sync()
    test_intrinsics()
    test_markers()
    test_layout_errors()
    test_mixed_fps()
    X_gt = make_ground_truth()
    path = test_reconstruction(X_gt)
    test_ultrawide(X_gt)
    test_sync_motion(X_gt)
    test_imprecise_balls(X_gt)
    test_blender(path, X_gt)
    test_fbx(path)
    n_ok = sum(1 for v in RESULTS.values() if v[0])
    print(f"\n{n_ok}/{len(RESULTS)} tests réussis")
    with open(os.path.join(OUT, "resultats.json"), "w", encoding="utf-8") as f:
        json.dump({k: {"ok": v[0], "detail": v[1]} for k, v in RESULTS.items()}, f, indent=2, ensure_ascii=False)
    sys.exit(0 if n_ok == len(RESULTS) else 1)
