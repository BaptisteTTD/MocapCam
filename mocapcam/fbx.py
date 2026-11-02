"""Export FBX (squelette Mixamo animé) en lançant Blender sans interface.

Blender est trouvé tout seul (Program Files, Steam, PATH…) ; sinon, indiquez son chemin
dans studio.json, clé "blender" (ex. "C:\\\\Program Files\\\\Blender Foundation\\\\Blender 5.1\\\\blender.exe").
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys

ADDON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "blender_addon", "mocapcam_blender.py")


def _version_key(path):
    m = re.search(r"(\d+)\.(\d+)", path)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def find_blender(configured=None):
    """Chemin de blender(.exe), ou None."""
    cands = []
    for p in (configured, os.environ.get("BLENDER")):
        if p:
            cands.append(p)
    if sys.platform.startswith("win"):
        bases = [os.environ.get("ProgramFiles", r"C:\Program Files"),
                 os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                 os.environ.get("ProgramW6432", r"C:\Program Files"),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")]
        found = []
        for b in bases:
            if b:
                found += glob.glob(os.path.join(b, "Blender Foundation", "Blender*", "blender.exe"))
                found += glob.glob(os.path.join(b, "Blender*", "blender.exe"))
        cands += sorted(set(found), key=_version_key, reverse=True)
        cands.append(os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                                  "Steam", "steamapps", "common", "Blender", "blender.exe"))
    elif sys.platform == "darwin":
        cands.append("/Applications/Blender.app/Contents/MacOS/Blender")
    w = shutil.which("blender")
    if w:
        cands.append(w)
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def export(mocap_json, fbx_path=None, blender=None, root_mode="RELATIVE", log=print, timeout=900):
    """Crée <prise>.fbx à côté de la capture. Retourne le chemin, ou None si impossible."""
    exe = find_blender(blender)
    if not exe:
        log("  FBX non créé : Blender introuvable. Indiquez son chemin dans studio.json "
            '(clé "blender"), puis lancez : mocapcam fbx <dossier de la prise>.')
        return None
    if not fbx_path:
        d, f = os.path.split(os.path.abspath(mocap_json))
        if os.path.basename(d) == "mocapcam":            # données de travail -> FBX dans la prise
            d = os.path.dirname(d)
        fbx_path = os.path.join(d, re.sub(r"(\.mocap)?\.json$", "", f) + ".fbx")
    cmd = [exe, "-b", "--factory-startup", "--python-exit-code", "1", "--python", ADDON,
           "--", os.path.abspath(mocap_json), os.path.abspath(fbx_path), root_mode]
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, creationflags=flags)
    except (OSError, subprocess.TimeoutExpired) as e:
        log(f"  FBX non créé : Blender n'a pas pu tourner ({e})")
        return None
    out = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"MOCAPCAM_FBX_OK (\{.*\})", out)
    if r.returncode != 0 or not m or not os.path.exists(fbx_path):
        tail = "\n    ".join(out.strip().splitlines()[-12:])
        log(f"  FBX non créé : erreur de Blender ({exe}) :\n    {tail}")
        return None
    rep = json.loads(m.group(1))
    log(f"  FBX : {fbx_path} ({rep.get('bones')} os Mixamo, {rep.get('frames')} images à "
        f"{rep.get('fps'):g} i/s)")
    return fbx_path
