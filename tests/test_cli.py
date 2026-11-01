"""Test de la ligne de commande sur de vrais fichiers vidéo (avec son),
sans les modèles de détection : les résultats 2D de synthèse sont placés dans le cache."""
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import test_synthetique as ts  # noqa: E402
from mocapcam import video, pipeline  # noqa: E402

D = tempfile.mkdtemp(prefix="mocapcam_cli_")
studio, take = os.path.join(D, "studio"), os.path.join(D, "prise1")
os.makedirs(take)
ff = video.ffmpeg_exe()


def run(*args):
    r = subprocess.run([sys.executable, "-m", "mocapcam", *args], cwd=ROOT, capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1500:])
    assert r.returncode == 0, args
    return r.stdout


run("init", studio)
# pas de damier : les objectifs sont estimés automatiquement

mlist = [dict(m, position=[m["position"][0], m["position"][1], 0.01]) for m in pipeline.DEFAULT_STUDIO["markers"]]
sr, dur, fps = 48000, 5.0, 60
shift_s = 2 / 60.0                         # même décalage que les données 2D de synthèse
for ci, (cam, (R, t)) in enumerate(zip(("iphone", "poco"), ts.CAMS_TRUE)):
    raw = os.path.join(D, f"{cam}_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ts.W, ts.H))
    img = ts.render_floor(R, t, mlist, ts.K_TRUE[ci], ts.DIST_TRUE[ci])
    for i in range(int(dur * fps)):
        vw.write(img)
    vw.release()
    a = np.random.default_rng(ci).normal(0, 0.005, int(sr * dur))
    c0 = int((1.5 + (shift_s if ci == 1 else 0)) * sr)
    a[c0:c0 + 600] += np.random.default_rng(5).normal(0, 0.7, 600) * np.exp(-np.arange(600) / 120)
    wav = os.path.join(D, f"{cam}.raw")
    (np.clip(a, -1, 1) * 32767).astype(np.int16).tofile(wav)
    ext = ".mov" if cam == "iphone" else ".mp4"
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", raw, "-f", "s16le", "-ar", str(sr), "-ac", "1",
                    "-i", wav, "-c:v", "libx264", "-c:a", "aac", "-shortest", os.path.join(take, cam + ext)],
                   check=True)

out = run("marqueurs", studio, take)
assert "decalage" in out.lower() or "décalage" in out

# résultats 2D de synthèse dans le cache (remplace RTMW, dont les modèles ne sont pas téléchargeables ici)
X_gt = ts.make_ground_truth()
poses = ts.synth_2d(X_gt)
work = os.path.join(take, "mocapcam")
poses[1]["kp"] = np.concatenate([np.full_like(poses[1]["kp"][:2], np.nan), poses[1]["kp"]])
poses[1]["sc"] = np.concatenate([np.zeros_like(poses[1]["sc"][:2]), poses[1]["sc"]])
for cam, p in zip(("iphone", "poco"), poses):
    np.savez_compressed(os.path.join(work, f"pose2d_{cam}_60.npz"), kp=p["kp"], sc=p["sc"], width=ts.W, height=ts.H)
out = run("traiter", studio, take)
res = os.path.join(take, "prise1.mocap.json")
assert os.path.exists(res)
print("OK ligne de commande :", res, f"{os.path.getsize(res) / 1e6:.1f} Mo")
print("Dossier :", D)
