"""Synchro fine par le mouvement de l'acteur.

Le bip donne l'instant où chaque téléphone a *entendu* le son. Mais dans un téléphone, le son
et l'image ne sont pas horodatés de la même façon : selon le modèle, l'image peut être en
avance ou en retard de plusieurs dizaines de millisecondes (100 ms mesurées sur un Poco X3).
À 60 i/s, c'est 6 images : un bras qui bouge vite n'est alors plus au même endroit dans les
deux vues, et la triangulation le fait trembler ou le perd.

On cherche donc, autour du décalage donné par le son, celui pour lequel les points du corps
vus par les caméras se recoupent le mieux en 3D (erreur de reprojection minimale).
"""
import numpy as np

from . import triangulate


def _cost(cams, raw, cam_fps, fps, delays, T, volume, rcfg, frames, cap=30.0):
    poses = [triangulate.resample_pose(r, f, fps, d, T) for r, f, d in zip(raw, cam_fps, delays)]
    _, _, E = triangulate.reconstruct(cams, poses, [0.0] * len(poses), fps, volume, rcfg,
                                      log=lambda *a: None, frames=frames, return_errors=True)
    Ef = E[frames]
    return float(np.mean(np.where(np.isfinite(Ef), np.minimum(Ef, cap), cap)))


def _motion_frames(raw0, fps0, fps, T, n=400):
    """Images de la base commune où l'acteur bouge le plus (les plus utiles pour la synchro),
    plus quelques images régulières."""
    kp = raw0["kp"][:, 0]                         # personne principale, (T0, 133, 2)
    sp = np.nanmean(np.linalg.norm(np.diff(kp[:, :23], axis=0), axis=-1), axis=1)
    sp = np.concatenate([[0.0], np.nan_to_num(sp)])
    t0 = np.clip((np.arange(T) / fps * fps0).astype(int), 0, len(sp) - 1)
    speed = sp[t0]
    k = min(n, T)
    moving = np.argsort(-speed)[: k // 2]
    regular = np.linspace(0, T - 1, k - len(moving)).astype(int)
    return np.unique(np.concatenate([moving, regular]))


def refine(cams, raw, names, cam_fps, fps, delays, T, volume, rcfg, log=print, search_s=0.35):
    """Retourne les décalages (s) affinés. raw[k] : points 2D bruts de la caméra k (cadence
    cam_fps[k]) ; delays[k] : décalage de départ (bip). La caméra 0 sert de référence."""
    delays = list(delays)
    frames = _motion_frames(raw[0], cam_fps[0], fps, T)
    for k in range(1, len(cams)):
        def cost(d):
            dl = list(delays)
            dl[k] = d
            return _cost(cams, raw, cam_fps, fps, dl, T, volume, rcfg, frames)
        d0 = delays[k]
        c0 = cost(d0)
        grid = d0 + np.arange(-search_s, search_s + 1e-9, 0.02)
        cs = np.array([cost(d) for d in grid])
        best = grid[int(np.argmin(cs))]
        fine = best + np.arange(-0.02, 0.0201, 0.004)
        cf = np.array([cost(d) for d in fine])
        i = int(np.argmin(cf))
        d1 = fine[i]
        if 0 < i < len(fine) - 1:              # sommet de la parabole : précision < 1 ms
            a, b, c = np.polyfit(fine[i - 1:i + 2], cf[i - 1:i + 2], 2)
            if a > 0:
                d1 = float(np.clip(-b / (2 * a), fine[i - 1], fine[i + 1]))
        c1 = cost(d1)
        if c1 < c0 - 0.2 and abs(d1 - d0) < search_s - 0.01:
            log(f"  synchro fine par le mouvement : {names[k]} {(d1 - d0) * 1000:+.0f} ms par rapport au bip "
                f"(erreur moyenne {c0:.1f} -> {c1:.1f} px)")
            delays[k] = float(d1)
        elif abs(d1 - d0) >= search_s - 0.01:
            log(f"  synchro fine par le mouvement : {names[k]} sans minimum net, décalage du bip conservé")
        else:
            log(f"  synchro fine par le mouvement : {names[k]} déjà bonne ({(d1 - d0) * 1000:+.0f} ms)")
    return delays
