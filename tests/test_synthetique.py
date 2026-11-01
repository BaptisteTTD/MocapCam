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
    test_mixed_fps()
    X_gt = make_ground_truth()
    path = test_reconstruction(X_gt)
    test_blender(path, X_gt)
    n_ok = sum(1 for v in RESULTS.values() if v[0])
    print(f"\n{n_ok}/{len(RESULTS)} tests réussis")
    with open(os.path.join(OUT, "resultats.json"), "w", encoding="utf-8") as f:
        json.dump({k: {"ok": v[0], "detail": v[1]} for k, v in RESULTS.items()}, f, indent=2, ensure_ascii=False)
    sys.exit(0 if n_ok == len(RESULTS) else 1)
