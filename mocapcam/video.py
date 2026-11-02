"""Outils vidéo : ffmpeg (fourni par imageio-ffmpeg), normalisation, audio, lecture."""
import os
import shutil
import subprocess

import numpy as np

VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".MP4", ".MOV")


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        raise RuntimeError("ffmpeg introuvable : pip install imageio-ffmpeg") from e


def normalize(src, dst, fps=60, max_height=1080, force=False):
    """Ré-encode en cadence fixe (les téléphones filment en cadence variable),
    applique la rotation des métadonnées et limite la résolution.
    Les vidéos de calibration ET de capture doivent passer par ici
    avec les mêmes réglages, sinon les paramètres optiques ne correspondent plus."""
    if (not force and os.path.exists(dst)
            and os.path.getmtime(dst) >= os.path.getmtime(src)):
        return dst
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    m = int(max_height)  # limite appliquée au petit côté (paysage comme portrait)
    vf = (f"fps={fps},scale='if(gt(iw\\,ih)\\,-2\\,min({m}\\,iw))'"
          f":'if(gt(iw\\,ih)\\,min({m}\\,ih)\\,-2)'")
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", src, "-vf", vf,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", dst]
    subprocess.run(cmd, check=True)
    return dst


def read_audio(path, sr=16000):
    """Piste audio mono en float32 (vide si pas d'audio)."""
    cmd = [ffmpeg_exe(), "-loglevel", "error", "-i", path, "-vn", "-ac", "1",
           "-ar", str(sr), "-f", "s16le", "-"]
    out = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0


def video_info(path):
    import cv2
    cap = cv2.VideoCapture(path)
    info = dict(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return info


def iter_frames(path, step=1):
    import cv2
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield i, frame
        i += 1
    cap.release()


STANDARD_FPS = (24, 25, 30, 48, 50, 60, 90, 100, 120, 240)


def native_fps(path):
    """Cadence moyenne de la vidéo d'origine, arrondie à une cadence standard
    (les téléphones filment en cadence légèrement variable : 59,94 -> 60, 29,8 -> 30)."""
    import cv2
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    dur = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0) / 1000
    cap.release()
    if dur > 1 and n > 10:
        fps = n / dur if not fps or abs(n / dur - fps) / fps > 0.1 else fps
    if not fps or not np.isfinite(fps):
        return 30
    return min(STANDARD_FPS, key=lambda s: abs(s - fps))


def sample_frames(path, n=30):
    """n images réparties sur toute la vidéo."""
    import cv2
    info = video_info(path)
    total = max(info["frames"], 1)
    out = []
    cap = cv2.VideoCapture(path)
    for i in np.linspace(0, total - 1, min(n, total)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            out.append(f)
    cap.release()
    return out


def median_frame(path, n=25):
    """Image médiane sur n images réparties : fait disparaître la personne
    qui bouge, il ne reste que le décor (et les boules au sol)."""
    import cv2
    info = video_info(path)
    total = max(info["frames"], 1)
    idx = np.linspace(0, total - 1, min(n, total)).astype(int)
    cap = cv2.VideoCapture(path)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"Impossible de lire {path}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def find_camera_videos(folder, camera_names):
    """Trouve <nom_camera>.<ext> dans le dossier de la prise."""
    found = {}
    for f in sorted(os.listdir(folder)):
        base, ext = os.path.splitext(f)
        if ext in VIDEO_EXT and base.lower() in [c.lower() for c in camera_names]:
            name = [c for c in camera_names if c.lower() == base.lower()][0]
            found[name] = os.path.join(folder, f)
    missing = [c for c in camera_names if c not in found]
    if missing:
        raise FileNotFoundError(
            f"Vidéo manquante pour : {', '.join(missing)} dans {folder} "
            f"(nommez les fichiers comme les caméras, ex. iphone.mov, poco.mp4)")
    return found
