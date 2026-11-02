"""Enchaînement complet d'une prise."""
import itertools
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


def _lens(meta, name):
    return ((meta or {}).get("cameras", {}).get(name, {}) or {}).get("lens") or "wide"


def _evaluate_layout(views, ml):
    """Pose de chaque caméra pour une disposition donnée ; score total (plus bas = mieux)."""
    total, poses = 0.0, []
    for v in views:
        try:
            p = mk.solve_camera_pose(v["bg"], ml, v["K"], v["dist"], lens=v["lens"], cands=v["cands"])
        except RuntimeError:
            p = None
        poses.append(p)
        total += p["score"] if p else 100.0
    return total, poses


def _layout_ok(poses, max_rms=8.0):
    # focale nominale (pas encore affinée) : quelques pixels d'écart restent normaux
    return all(p is not None and p["rms"] <= max_rms for p in poses)


FEMININE = {"vert": "verte", "bleu": "bleue", "violet": "violette"}


def _boule(m):
    return f"la boule {FEMININE.get(m['color'], m['color'])} ({m['name']})"


def choose_layout(views, mlist, log=print, swaps=True):
    """Disposition des boules qui explique le mieux toutes les caméras.

    * telle que décrite dans studio.json, ou son symétrique (B et C de l'autre côté de la
      ligne A-D : mêmes distances, seul le sens change) : pris en compte sans rien dire de plus ;
    * deux boules de couleurs différentes échangées : retenu seulement si c'est nettement
      mieux, avec un message pour l'utilisateur.
    Retourne (marqueurs, poses, description) ; poses[i] vaut None si la caméra i échoue."""
    base = []
    for mir in (False, True):
        ml = mk.mirrored(mlist) if mir else mlist
        sc, ps = _evaluate_layout(views, ml)
        base.append((sc, mir, ps, ml))
    base.sort(key=lambda r: (round(r[0], 2), r[1]))
    found = []
    for i, j in itertools.combinations(range(len(mlist)) if swaps else [], 2):
        if mlist[i]["color"] == mlist[j]["color"]:
            continue
        for mir in (False, True):
            ml = mk.swapped(mlist, i, j)
            ml = mk.mirrored(ml) if mir else ml
            sc, ps = _evaluate_layout(views, ml)
            found.append((sc, mir, ps, ml, (i, j)))
    found.sort(key=lambda r: (round(r[0], 2), r[1]))   # à égalité : l'explication sans miroir
    sc, mir, poses, ml = base[0]
    if found and _layout_ok(found[0][2]) and (not _layout_ok(poses) or found[0][0] < 0.7 * sc):
        sc, mir, poses, ml, (i, j) = found[0]
        a, b = mlist[i], mlist[j]
        log(f"  ATTENTION : {_boule(a)} et {_boule(b)} semblent échangées par rapport à studio.json. "
            "Le traitement en tient compte ; vous pouvez les laisser ainsi.")
        return ml, poses, f"{a['name']}<->{b['name']}" + (" miroir" if mir else "")
    if mir and _layout_ok(poses):
        log("  disposition des boules en miroir par rapport à studio.json (B et C de l'autre côté "
            "de la ligne A-D) : prise en compte automatiquement")
    return ml, poses, "miroir" if mir else "telle que décrite"


def calibrate_extrinsics(cfg, studio_dir, norm_videos, work, log=print, meta=None):
    mlist = _markers_for_solver(cfg)
    views = []
    for name, path in norm_videos.items():
        ipath = os.path.join(studio_dir, "intrinseques", f"{name}.json")
        info = video.video_info(path)
        bg = video.median_frame(path)
        cv2.imwrite(os.path.join(work, f"fond_{name}.jpg"), bg)
        fixed = os.path.exists(ipath)   # calibration optique au damier (optionnelle)
        if fixed:
            intr = intrinsics.scaled_to(intrinsics.load(ipath), info["width"], info["height"])
            K, dist = intr["K"], intr["dist"]
        else:                           # sans damier : focale de l'objectif, affinée ensuite
            K, dist = None, None
        frames = video.sample_frames(path, 30)
        views.append(dict(name=name, info=info, bg=bg, K=K, dist=dist, fixed=fixed, lens=_lens(meta, name),
                          cands=mk.find_candidates(bg, mlist, frames)))
    mlist, poses, layout = choose_layout(views, mlist, log)
    failed = [v["name"] for v, p in zip(views, poses) if p is None]
    if failed:
        raise RuntimeError(
            f"Boules introuvables ou incohérentes pour : {', '.join(failed)}. Chaque caméra doit voir au "
            "moins 4 boules, et leur disposition doit correspondre à studio.json (mêmes distances). "
            f"Regardez fond_*.jpg dans {work}.")
    # positions réellement utilisées (miroir / inversion) : reprises pour l'export
    by_name = {m["name"]: m for m in mlist}
    for m in cfg["markers"]:
        m["position_3d"] = list(by_name[m["name"]]["position"])
    vol = mk.volume_box(mlist, cfg["volume"]["margin"], cfg["volume"]["height"])
    cams = []
    for v, pose in zip(views, poses):
        name, info = v["name"], v["info"]
        dbg = mk.draw_debug(v["bg"], mlist, pose["K"], pose["dist"], pose, vol)
        cv2.imwrite(os.path.join(work, f"calibration_{name}.jpg"), dbg)
        miss = f" (non utilisées : {', '.join(pose['missing'])})" if pose["missing"] else ""
        if v["fixed"]:
            src = "damier"
        elif pose.get("aligned"):
            src = "approximative : boules vues presque alignées"
        else:
            src = "estimée par les boules"
        log(f"  {name} : erreur {pose['rms']:.2f} px, focale {pose['K'][0, 0]:.0f} px ({src}), "
            f"caméra à ({pose['center'][0]:.2f}, {pose['center'][1]:.2f}, {pose['center'][2]:.2f}) m{miss}")
        if pose["rms"] > 4:
            log(f"  ATTENTION {name} : erreur élevée, vérifiez calibration_{name}.jpg")
        f_nom = (v["K"] if v["K"] is not None else mk.default_K(info["width"], info["height"], v["lens"]))[0, 0]
        cams.append(dict(name=name, K=pose["K"], dist=np.asarray(pose["dist"], float), R=pose["R"],
                         t=np.asarray(pose["t"]), center=pose["center"], rms=pose["rms"],
                         width=info["width"], height=info["height"],
                         marker_obs=pose["marker_obs"], intrinsics_fixed=v["fixed"],
                         lens=v["lens"], f_nominal=float(f_nom)))
    return cams, vol


def actor_height(X):
    """Taille approximative de l'acteur (m) d'après la longueur médiane des jambes
    (hanche -> cheville ≈ 0,49 × la taille). Sert à repérer une erreur d'échelle :
    distances entre boules de studio.json différentes de la réalité."""
    J = {n: i for i, n in enumerate(JOINT_NAMES)}
    legs = []
    for side in ("l", "r"):
        a = np.linalg.norm(X[:, J[f"{side}_hip"]] - X[:, J[f"{side}_knee"]], axis=1)
        b = np.linalg.norm(X[:, J[f"{side}_knee"]] - X[:, J[f"{side}_ankle"]], axis=1)
        legs.append(a + b)
    leg = np.concatenate(legs)
    leg = leg[np.isfinite(leg)]
    if len(leg) < 10:
        return None
    return float(np.median(leg) / 0.49)


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

    # 1) prise lancée par MocapCam Studio : bips joués par le PC (départ, et arrêt)
    chirps = (take_meta or {}).get("chirps_at_pc") or (
        [take_meta["chirp_at_pc"]] if take_meta and take_meta.get("chirp_at_pc") else [])
    if chirps and not manual:
        offs = [(c - chirps[0]) / 1000.0 for c in chirps]
        found = {}
        for n in names:
            peaks = sync.find_chirps(audio[n], max_n=len(chirps) + 2)
            found[n] = sync.match_chirps(peaks, offs)
            desc = ", ".join(f"{t:.3f} s" for _, t in sorted(found[n].items())) or "aucun"
            log(f"  {n} : bip(s) trouvé(s) à {desc}")
        common = set.intersection(*(set(f) for f in found.values())) if found else set()
        if common:
            ks = sorted(common)
            delays = []
            for n in names:
                d = [found[n][k] - found[names[0]][k] for k in ks]
                delays.append(float(np.mean(d)))
                if n != names[0]:
                    drift = (d[-1] - d[0]) * 1000 if len(d) > 1 else None
                    extra = f", dérive {drift:+.1f} ms entre les 2 bips" if drift is not None else ""
                    log(f"  {n} : décalage {delays[-1] * 1000:+.1f} ms par rapport à {names[0]}"
                        f" ({len(ks)} bip(s){extra})")
                    if drift is not None and abs(drift) > 20:
                        log("  ATTENTION : forte dérive entre les deux bips, vérifiez les cadences.")
            return delays
        cams = take_meta.get("cameras", {})
        st = {n: cams.get(n, {}).get("started_at_pc") for n in names}
        if all(v is not None for v in st.values()):
            log("  ATTENTION : bip peu audible, synchro par les horloges des téléphones "
                "(précision ~1 image). Montez le volume du PC.")
            return [-(st[n] - st[names[0]]) / 1000.0 for n in names]

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
    norm, cam_fps = {}, {}
    for n, p in raw.items():
        # chaque vidéo garde sa cadence d'origine (30 ou 60 i/s…) : pas d'images dupliquées
        cam_fps[n] = min(video.native_fps(p), 120)
        norm[n] = video.normalize(p, os.path.join(work, f"norm_{n}_{cam_fps[n]}.mp4"), cam_fps[n],
                                  cfg["max_height"], force=force)
    fps = max(cam_fps.values())          # base de temps commune = la plus rapide
    log("  cadences : " + ", ".join(f"{n} {f} i/s" for n, f in cam_fps.items())
        + (f" -> interpolées sur une base commune de {fps} i/s" if len(set(cam_fps.values())) > 1 else ""))

    log("2/6 Synchronisation par le son…")
    delays = compute_delays(norm, manual_delays, meta, log)
    delays_s = list(delays)

    log("3/6 Calibration automatique par les boules au sol…")
    cams, vol = calibrate_extrinsics(cfg, studio_dir, norm, work, log, meta=meta)
    if only_calib:
        log(f"Images de contrôle dans {work}")
        return None

    log("4/6 Détection 2D corps + mains (long la première fois)…")
    poses = []
    for n, p in norm.items():
        poses.append(pose2d.run_video(p, os.path.join(work, f"pose2d_{n}_{cam_fps[n]}.npz"),
                                      force=force, log=log, **cfg["pose"]))

    # toutes les caméras ramenées sur la même base de temps (décalage sous-image inclus)
    names = list(norm)
    T = int(poses[0]["kp"].shape[0] / cam_fps[names[0]] * fps)
    poses = [triangulate.resample_pose(p, cam_fps[n], fps, d, T) for p, n, d in zip(poses, names, delays)]
    delays = [0.0] * len(poses)

    log("5/6 Affinage des caméras avec le corps de l'acteur, puis triangulation 3D…")
    cams, ac = autocalib.refine(cams, poses, delays, fps, vol, cfg["reconstruction"],
                                fixed_intrinsics=[c["intrinsics_fixed"] for c in cams], log=log)
    if ac:
        log(f"  {ac['observations']} observations ; erreur {ac['reproj_before_px']:.1f} px "
            f"-> {ac['reproj_after_px']:.1f} px")
        for c in ac["cameras"]:
            log(f"  {c['name']} : focale {c['focal_px']:.0f} px, déformation k1 {c['k1']:+.3f} k2 {c['k2']:+.3f}")
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
    stats["delays_s"] = dict(zip(norm.keys(), delays_s))
    stats["camera_fps"] = cam_fps
    stats["autocalibration"] = ac
    h = actor_height(X)
    stats["actor_height_m"] = h
    if h is not None:
        log(f"  taille estimée de l'acteur : {h:.2f} m")
        if not 1.35 <= h <= 2.15:
            log(f"  ATTENTION : taille peu plausible. Les distances entre boules de studio.json ne "
                f"correspondent sans doute pas à la réalité (facteur ≈ {1.75 / h:.2f} si vous mesurez "
                "environ 1,75 m). Mesurez-les au mètre ruban et corrigez studio.json.")
    name = os.path.basename(os.path.normpath(take_dir))
    out = export_motion(os.path.join(take_dir, f"{name}.mocap.json"), X, fps, cams, cfg, vol, stats)
    with open(os.path.join(work, "rapport.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=_jsonable)
    log(f"Terminé en {time.time() - t0:.0f} s -> {out}")
    log(f"  couverture corps {stats['coverage_body'] * 100:.0f} %, mains {stats['coverage_hands'] * 100:.0f} %")
    return out
