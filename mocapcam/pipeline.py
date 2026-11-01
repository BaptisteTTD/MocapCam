"""Enchaînement complet d'une prise."""
import json
import os
import time

import cv2
import numpy as np

from . import autocalib, intrinsics, markers as mk, pose2d, sync, triangulate, video
from .skeleton import JOINT_NAMES

DEFAULT_STUDIO = {
    "cameras": ["iphone", "poco"],
    "fps": 60,
    "max_height": 1080,
    "marker_height": 0.01,
    "markers": [
        {"name": "A", "color": "rouge", "position": [0.0, 0.0, 0.0]},
        {"name": "B", "color": "vert", "position": [2.5, 0.0, 0.0]},
        {"name": "C", "color": "bleu", "position": [2.5, 2.5, 0.0]},
        {"name": "D", "color": "jaune", "position": [0.0, 2.5, 0.0]},
        {"name": "E", "color": "rouge", "position": [0.0, 1.25, 0.0]},
    ],
    "volume": {"margin": 0.5, "height": 2.3},
    "pose": {"mode": "performance", "device": "cpu", "backend": "onnxruntime"},
    "reconstruction": {"score_body": 0.3, "score_hands": 0.3, "score_face": 0.5,
                       "max_reproj_px": 25.0},
    "filtering": {"max_jump_m": 0.15, "max_gap_s": 0.3,
                  "cutoff_body_hz": 6.0, "cutoff_hands_hz": 8.0},
}


def load_studio(studio_dir):
    path = os.path.join(studio_dir, "studio.json")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    for k, v in DEFAULT_STUDIO.items():
        cfg.setdefault(k, v)
    h = cfg["marker_height"]
    for m in cfg["markers"]:
        p = list(m["position"]) + [0.0] * (3 - len(m["position"]))
        m["position_3d"] = [p[0], p[1], p[2] + h]
    return cfg


def init_studio(studio_dir):
    os.makedirs(os.path.join(studio_dir, "intrinseques"), exist_ok=True)
    path = os.path.join(studio_dir, "studio.json")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_STUDIO, f, indent=2, ensure_ascii=False)
    return path


def _markers_for_solver(cfg):
    return [dict(m, position=m["position_3d"]) for m in cfg["markers"]]


def calibrate_extrinsics(cfg, studio_dir, norm_videos, work, log=print):
    cams = []
    mlist = _markers_for_solver(cfg)
    vol = mk.volume_box(mlist, cfg["volume"]["margin"], cfg["volume"]["height"])
    for name, path in norm_videos.items():
        ipath = os.path.join(studio_dir, "intrinseques", f"{name}.json")
        info = video.video_info(path)
        bg = video.median_frame(path)
        cv2.imwrite(os.path.join(work, f"fond_{name}.jpg"), bg)
        if os.path.exists(ipath):   # calibration au damier (optionnelle)
            intr = intrinsics.scaled_to(intrinsics.load(ipath), info["width"], info["height"])
            pose = mk.solve_camera_pose(bg, mlist, intr["K"], intr["dist"])
            fixed = True
        else:                       # sans damier : focale estimée avec les boules
            pose = mk.solve_camera_pose(bg, mlist)
            fixed = False
        dbg = mk.draw_debug(bg, mlist, pose["K"], pose["dist"], pose, vol)
        cv2.imwrite(os.path.join(work, f"calibration_{name}.jpg"), dbg)
        miss = f" (non vues : {', '.join(pose['missing'])})" if pose["missing"] else ""
        src = "damier" if fixed else "estimée par les boules"
        log(f"  {name} : erreur {pose['rms']:.2f} px, focale {pose['K'][0, 0]:.0f} px ({src}), "
            f"caméra à ({pose['center'][0]:.2f}, {pose['center'][1]:.2f}, {pose['center'][2]:.2f}) m{miss}")
        if pose["rms"] > 4:
            log(f"  ATTENTION {name} : erreur élevée, vérifiez calibration_{name}.jpg")
        cams.append(dict(name=name, K=pose["K"], dist=np.asarray(pose["dist"], float), R=pose["R"],
                         t=np.asarray(pose["t"]), center=pose["center"], rms=pose["rms"],
                         width=info["width"], height=info["height"],
                         marker_obs=pose["marker_obs"], intrinsics_fixed=fixed))
    return cams, vol


def load_take_meta(take_dir):
    """Métadonnées écrites par MocapCam Studio (take.json), ou None pour une prise manuelle."""
    path = os.path.join(take_dir, "take.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_delays(norm_videos, manual=None, take_meta=None, log=print):
    names = list(norm_videos)
    manual = manual or {}
    audio = {n: video.read_audio(norm_videos[n]) for n in names}

    # 1) prise lancée par MocapCam Studio : signal sonore (chirp) joué par le PC
    if take_meta and take_meta.get("chirp_at_pc") and not manual:
        cams = take_meta["cameras"]
        ok, t_chirp, expected = True, {}, {}
        for n in names:
            st = cams.get(n, {}).get("started_at_pc")
            exp = (take_meta["chirp_at_pc"] - st) / 1000.0 if st is not None else None
            expected[n] = exp
            win = (exp - 1.0, exp + 1.5) if exp is not None else None
            t, conf = sync.find_chirp(audio[n], search=win)
            if t is None or conf < 6:
                t, conf = sync.find_chirp(audio[n])   # recherche sur toute la piste
            log(f"  {n} : signal sonore à {t if t is not None else float('nan'):.4f} s (confiance {conf:.0f})")
            if t is None or conf < 6:
                ok = False
            t_chirp[n] = t
        if ok:
            delays = [t_chirp[n] - t_chirp[names[0]] for n in names]
            for n, d in zip(names[1:], delays[1:]):
                log(f"  {n} : décalage {d * 1000:+.1f} ms par rapport à {names[0]} (signal sonore)")
            return delays
        if all(expected[n] is not None for n in names):
            log("  ATTENTION : signal sonore peu audible, synchro par les horloges des téléphones "
                "(précision ~1 image). Montez le volume du PC.")
            return [expected[n] - expected[names[0]] for n in names]

    # 2) prise manuelle : clap au début
    delays = [0.0]
    for n in names[1:]:
        if n in manual:
            d, conf = float(manual[n]), None
        else:
            d, conf = sync.audio_delay(audio[names[0]], audio[n])
        delays.append(d)
        c = "manuel" if conf is None else f"confiance {conf:.1f}"
        log(f"  {n} : décalage {d * 1000:+.1f} ms par rapport à {names[0]} ({c})")
        if conf is not None and conf < 1.5:
            log("  ATTENTION : synchro audio peu fiable. Faites un clap net au début, "
                "ou indiquez le décalage avec --decalage.")
    return delays


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def export_motion(path, X, fps, cams, cfg, vol, stats):
    pos = [[None if not np.isfinite(p).all() else [round(float(v), 4) for v in p]
            for p in frame] for frame in X]
    data = {
        "format": "mocapcam", "version": 1, "fps": fps, "units": "m", "up": "Z",
        "joint_names": JOINT_NAMES, "frame_count": int(X.shape[0]),
        "positions": pos,
        "cameras": [dict(name=c["name"], K=np.asarray(c["K"]).tolist(),
                         dist=np.asarray(c["dist"]).ravel().tolist(), width=c["width"],
                         height=c["height"], R=np.asarray(c["R"]).tolist(),
                         t=np.asarray(c["t"]).tolist(), center=list(c["center"]))
                    for c in cams],
        "markers": [dict(name=m["name"], color=m["color"], position=m["position_3d"])
                    for m in cfg["markers"]],
        "volume": {"min": vol[0].tolist(), "max": vol[1].tolist()},
        "stats": stats,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), default=_jsonable)
    return path


def process_take(studio_dir, take_dir, force=False, manual_delays=None, only_calib=False,
                 log=print):
    t0 = time.time()
    cfg = load_studio(studio_dir)
    work = os.path.join(take_dir, "mocapcam")
    os.makedirs(work, exist_ok=True)
    meta = load_take_meta(take_dir)
    if meta:   # prise MocapCam Studio : les caméras sont celles de la prise
        names = [n for n, c in meta["cameras"].items() if c.get("file")]
    else:
        names = cfg["cameras"]
    raw = video.find_camera_videos(take_dir, names)
    fps = cfg["fps"]

    log("1/6 Normalisation des vidéos (cadence fixe)…")
    norm = {}
    for n, p in raw.items():
        norm[n] = video.normalize(p, os.path.join(work, f"norm_{n}.mp4"), fps,
                                  cfg["max_height"], force=force)

    log("2/6 Synchronisation par le son…")
    delays = compute_delays(norm, manual_delays, meta, log)

    log("3/6 Calibration automatique par les boules au sol…")
    cams, vol = calibrate_extrinsics(cfg, studio_dir, norm, work, log)
    if only_calib:
        log(f"Images de contrôle dans {work}")
        return None

    log("4/6 Détection 2D corps + mains (long la première fois)…")
    poses = []
    for n, p in norm.items():
        poses.append(pose2d.run_video(p, os.path.join(work, f"pose2d_{n}.npz"),
                                      force=force, log=log, **cfg["pose"]))

    log("5/6 Affinage des caméras avec le corps de l'acteur, puis triangulation 3D…")
    cams, ac = autocalib.refine(cams, poses, delays, fps, vol, cfg["reconstruction"],
                                fixed_intrinsics=[c["intrinsics_fixed"] for c in cams], log=log)
    if ac:
        log(f"  {ac['observations']} observations ; erreur {ac['reproj_before_px']:.1f} px "
            f"-> {ac['reproj_after_px']:.1f} px")
        for c in ac["cameras"]:
            log(f"  {c['name']} : focale {c['focal_px']:.0f} px, déformation k1 {c['k1']:+.3f}")
    X, stats = triangulate.reconstruct(cams, poses, delays, fps, vol,
                                       cfg["reconstruction"], log)
    log(f"  images reconstruites : {stats['frames_ok']}/{stats['frames']}, "
        f"erreur médiane {stats['median_reproj_px'] or float('nan'):.1f} px, "
        f"inversions gauche/droite corrigées : {stats['lr_swaps_fixed']}")

    log("6/6 Nettoyage, lissage et export…")
    X = triangulate.postprocess(X, fps, cfg["filtering"])
    ok = np.isfinite(X[..., 0])
    stats["coverage_body"] = float(ok[:, :23].mean())
    stats["coverage_hands"] = float(ok[:, 91:].mean())
    stats["delays_s"] = dict(zip(norm.keys(), delays))
    stats["autocalibration"] = ac
    name = os.path.basename(os.path.normpath(take_dir))
    out = export_motion(os.path.join(take_dir, f"{name}.mocap.json"), X, fps, cams, cfg, vol, stats)
    with open(os.path.join(work, "rapport.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=_jsonable)
    log(f"Terminé en {time.time() - t0:.0f} s -> {out}")
    log(f"  couverture corps {stats['coverage_body'] * 100:.0f} %, mains {stats['coverage_hands'] * 100:.0f} %")
    return out
