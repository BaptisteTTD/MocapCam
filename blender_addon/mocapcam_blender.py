bl_info = {
    "name": "MocapCam",
    "author": "MocapCam",
    "version": (0, 2, 0),
    "blender": (3, 6, 0),
    "location": "Fichier > Importer > MocapCam (.mocap.json)  |  Fichier > Exporter > FBX MocapCam (Mixamo)"
                "  |  Vue 3D > barre latérale (N) > MocapCam",
    "description": "Importe une capture MocapCam : squelette Mixamo (corps + doigts) "
                   "et application directe sur un personnage Mixamo",
    "category": "Animation",
}

"""
Principe de l'application sur un squelette :
pour chaque os, on construit un repère à partir des points capturés (direction de l'os +
un vecteur de référence pour la torsion), et le même repère à partir de la pose de repos
du squelette cible. La rotation monde de l'os = repère_capture · repère_repos⁻¹ · rotation_repos.
Cela marche quelle que soit la pose de repos (T-pose, A-pose), le roll des os, l'échelle
et l'orientation de l'objet armature (FBX Mixamo à 0.01, tourné de 90°, etc.).
"""

import json
import math
import os

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper
from mathutils import Matrix, Vector

PREFIX = "mixamorig:"

# --------------------------------------------------------------------------- hiérarchie Mixamo

FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")


def mixamo_hierarchy():
    """Liste ordonnée (os, parent) des 65 os Mixamo standard."""
    h = [("Hips", None), ("Spine", "Hips"), ("Spine1", "Spine"), ("Spine2", "Spine1"),
         ("Neck", "Spine2"), ("Head", "Neck"), ("HeadTop_End", "Head")]
    for s in ("Left", "Right"):
        h += [(f"{s}Shoulder", "Spine2"), (f"{s}Arm", f"{s}Shoulder"),
              (f"{s}ForeArm", f"{s}Arm"), (f"{s}Hand", f"{s}ForeArm")]
        for f in FINGERS:
            prev = f"{s}Hand"
            for k in range(1, 5):
                h.append((f"{s}Hand{f}{k}", prev))
                prev = f"{s}Hand{f}{k}"
    for s in ("Left", "Right"):
        h += [(f"{s}UpLeg", "Hips"), (f"{s}Leg", f"{s}UpLeg"), (f"{s}Foot", f"{s}Leg"),
              (f"{s}ToeBase", f"{s}Foot"), (f"{s}Toe_End", f"{s}ToeBase")]
    return h


# --------------------------------------------------------------------------- maths vectorisées

def nrm(v):
    v = np.asarray(v, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)


def dot(a, b):
    return np.sum(a * b, axis=-1, keepdims=True)


def make_frames(d, r):
    """Repère orthonormé : Y = direction de l'os, Z = référence (orthogonalisée)."""
    y = nrm(d)
    z = r - dot(r, y) * y
    zn = np.linalg.norm(z, axis=-1)
    z = nrm(z)
    x = np.cross(y, z)
    F = np.stack([x, y, z], axis=-1)
    dn = np.linalg.norm(np.asarray(d, float), axis=-1)
    valid = np.isfinite(F).all(axis=(-1, -2)) & (zn > 1e-6) & (dn > 1e-6)
    return F, valid


def hold_fill(F, valid):
    """Images invalides : on garde la dernière image valide (la première au tout début).
    Pas de bascule au milieu d'un trou : la pose reste figée puis reprend quand les
    données reviennent."""
    T = F.shape[0]
    if valid.all():
        return F
    idx = np.where(valid)[0]
    if idx.size == 0:
        return None
    pos = np.searchsorted(idx, np.arange(T), side="right") - 1
    pick = idx[np.clip(pos, 0, idx.size - 1)]
    return F[pick]


def mat_to_quat(M):
    """(T,3,3) -> (T,4) quaternions (w, x, y, z), continus dans le temps."""
    M = np.asarray(M)
    T = M.shape[0]
    q = np.zeros((T, 4))
    tr = M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]
    for i in range(T):
        m = M[i]
        if tr[i] > 0:
            s = math.sqrt(tr[i] + 1.0) * 2
            q[i] = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = nrm(q)
    for i in range(1, T):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def np_mat(m):
    return np.array([list(r) for r in m], float)


# --------------------------------------------------------------------------- points capturés

def load_motion(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    names = d["joint_names"]
    T = d["frame_count"]
    X = np.full((T, len(names), 3), np.nan)
    for i, fr in enumerate(d["positions"]):
        for j, p in enumerate(fr):
            if p is not None:
                X[i, j] = p
    J = {n: X[:, i] for i, n in enumerate(names)}
    return d, J


def derive(J):
    """Ajoute les points/axes dérivés utilisés par les repères des os."""
    J["mid_hip"] = (J["l_hip"] + J["r_hip"]) / 2
    J["mid_sh"] = (J["l_shoulder"] + J["r_shoulder"]) / 2
    J["mid_ear"] = (J["l_ear"] + J["r_ear"]) / 2
    J["lat_hip"] = J["l_hip"] - J["r_hip"]
    J["lat_sh"] = J["l_shoulder"] - J["r_shoulder"]
    J["lat_ear"] = J["l_ear"] - J["r_ear"]
    J["up_t"] = J["mid_sh"] - J["mid_hip"]
    J["fwd_t"] = np.cross(nrm(J["lat_hip"]) + nrm(J["lat_sh"]), J["up_t"])
    for s in ("l", "r"):
        J[f"{s}_toes"] = (J[f"{s}_big_toe"] + J[f"{s}_small_toe"]) / 2
    J["l_foot_lat"] = J["l_small_toe"] - J["l_big_toe"]
    J["r_foot_lat"] = J["r_big_toe"] - J["r_small_toe"]
    return J


def bone_specs():
    """os Mixamo -> (fonction direction, fonction référence de torsion)."""
    S = {
        "Hips": (lambda J: J["up_t"], lambda J: J["lat_hip"]),
        "Spine": (lambda J: J["up_t"], lambda J: 0.67 * nrm(J["lat_hip"]) + 0.33 * nrm(J["lat_sh"])),
        "Spine1": (lambda J: J["up_t"], lambda J: 0.33 * nrm(J["lat_hip"]) + 0.67 * nrm(J["lat_sh"])),
        "Spine2": (lambda J: J["up_t"], lambda J: J["lat_sh"]),
        "Neck": (lambda J: J["mid_ear"] - J["mid_sh"], lambda J: nrm(J["lat_sh"]) + nrm(J["lat_ear"])),
        "Head": (lambda J: np.cross(J["nose"] - J["mid_ear"], J["lat_ear"]), lambda J: J["lat_ear"]),
    }
    for s, side, h in (("l", "Left", "lh"), ("r", "Right", "rh")):
        S[f"{side}Shoulder"] = (lambda J, s=s: J[f"{s}_shoulder"] - J["mid_sh"],
                                lambda J: J["up_t"])
        S[f"{side}Arm"] = (
            lambda J, s=s: J[f"{s}_elbow"] - J[f"{s}_shoulder"],
            lambda J, s=s: np.cross(J[f"{s}_elbow"] - J[f"{s}_shoulder"],
                                    nrm(J[f"{s}_wrist"] - J[f"{s}_elbow"]) + 0.3 * nrm(J["fwd_t"])))
        S[f"{side}ForeArm"] = (lambda J, s=s: J[f"{s}_wrist"] - J[f"{s}_elbow"],
                               lambda J, h=h: J[f"{h}_5"] - J[f"{h}_17"])
        S[f"{side}Hand"] = (lambda J, h=h: J[f"{h}_9"] - J[f"{h}_0"],
                            lambda J, h=h: J[f"{h}_5"] - J[f"{h}_17"])
        for f, base in (("Index", 5), ("Middle", 9), ("Ring", 13), ("Pinky", 17)):
            for k in range(3):
                S[f"{side}Hand{f}{k + 1}"] = (
                    lambda J, h=h, a=base + k: J[f"{h}_{a + 1}"] - J[f"{h}_{a}"],
                    lambda J, h=h: J[f"{h}_5"] - J[f"{h}_17"])
        for k in range(3):
            S[f"{side}HandThumb{k + 1}"] = (
                lambda J, h=h, a=1 + k: J[f"{h}_{a + 1}"] - J[f"{h}_{a}"],
                lambda J, h=h: np.cross(J[f"{h}_9"] - J[f"{h}_0"], J[f"{h}_5"] - J[f"{h}_17"]))
        S[f"{side}UpLeg"] = (
            lambda J, s=s: J[f"{s}_knee"] - J[f"{s}_hip"],
            lambda J, s=s: np.cross(J[f"{s}_knee"] - J[f"{s}_hip"],
                                    nrm(J[f"{s}_ankle"] - J[f"{s}_knee"]) - 0.3 * nrm(J["fwd_t"])))
        S[f"{side}Leg"] = (lambda J, s=s: J[f"{s}_ankle"] - J[f"{s}_knee"],
                           lambda J, s=s: nrm(J[f"{s}_toes"] - J[f"{s}_heel"]) + 0.3 * nrm(J["fwd_t"]))
        S[f"{side}Foot"] = (lambda J, s=s: J[f"{s}_toes"] - J[f"{s}_ankle"],
                            lambda J, s=s: J[f"{s}_foot_lat"] + 0.5 * nrm(J["lat_hip"]))
    return S


# --------------------------------------------------------------------------- armature cible

def find_prefix(obj):
    for b in obj.data.bones:
        if b.name.endswith("Hips"):
            return b.name[: -len("Hips")]
    raise RuntimeError("Aucun os 'Hips' : ce n'est pas un squelette au format Mixamo.")


def _bone_frames(obj, prefix, posed):
    mw = obj.matrix_world
    out = {}
    src = obj.pose.bones if posed else obj.data.bones
    for b in src:
        if not b.name.startswith(prefix):
            continue
        m = mw @ (b.matrix if posed else b.matrix_local)
        tail = mw @ ((b.tail if posed else b.tail_local))
        out[b.name[len(prefix):]] = (np.array(m.translation), np_mat(m.to_3x3().normalized()),
                                     np.array(tail))
    return out


def virtual_joints(obj, prefix, posed=False):
    """Points équivalents aux 133 points capturés, calculés depuis les os
    (pose de repos, ou pose courante si posed=True)."""
    rest = _bone_frames(obj, prefix, False)
    cur = _bone_frames(obj, prefix, True) if posed else rest
    H = {k: v[0] for k, v in rest.items()}
    lat = nrm(H["LeftUpLeg"] - H["RightUpLeg"])
    up0 = H["Neck"] - H["Hips"]
    up = nrm(up0 - np.dot(up0, lat) * lat)
    fwd = np.cross(lat, up)
    leg = (np.linalg.norm(H["LeftUpLeg"] - H["LeftLeg"]) + np.linalg.norm(H["LeftLeg"] - H["LeftFoot"]))
    sc = leg / 0.85
    J = {}

    def attach(name, bone, off=np.zeros(3)):
        h_r, R_r, _ = rest[bone]
        h, R, _ = cur[bone]
        J[name] = h + R @ (R_r.T @ off)

    def tail_of(name, bone):
        h_r, R_r, t_r = rest[bone]
        attach(name, bone, t_r - h_r)

    attach("nose", "Head", sc * (0.09 * up + 0.10 * fwd))
    for s, sg in (("l", 1), ("r", -1)):
        attach(f"{s}_ear", "Head", sc * (0.07 * up + sg * 0.075 * lat))
        attach(f"{s}_eye", "Head", sc * (0.10 * up + 0.08 * fwd + sg * 0.032 * lat))
    for s, side, h, sg in (("l", "Left", "lh", 1), ("r", "Right", "rh", -1)):
        attach(f"{s}_shoulder", f"{side}Arm")
        attach(f"{s}_elbow", f"{side}ForeArm")
        attach(f"{s}_wrist", f"{side}Hand")
        attach(f"{s}_hip", f"{side}UpLeg")
        attach(f"{s}_knee", f"{side}Leg")
        attach(f"{s}_ankle", f"{side}Foot")
        out = sg * lat
        if f"{side}Toe_End" in rest:
            tip_bone, tip_off = f"{side}Toe_End", np.zeros(3)
        else:
            tip_bone = f"{side}ToeBase"
            tip_off = rest[tip_bone][2] - rest[tip_bone][0]
        attach(f"{s}_big_toe", tip_bone, tip_off - sc * 0.02 * out)
        attach(f"{s}_small_toe", tip_bone, tip_off + sc * (0.03 * out - 0.03 * fwd))
        attach(f"{s}_heel", f"{side}Foot", sc * (-0.065 * up - 0.05 * fwd))
        attach(f"{h}_0", f"{side}Hand")
        for f, base in (("Thumb", 1), ("Index", 5), ("Middle", 9), ("Ring", 13), ("Pinky", 17)):
            for k in range(4):
                bn = f"{side}Hand{f}{k + 1}"
                if bn in rest:
                    attach(f"{h}_{base + k}", bn)
                else:
                    tail_of(f"{h}_{base + k}", f"{side}Hand{f}{k}")
    return J


def _ordered_bones(arm):
    order = []

    def rec(b):
        order.append(b)
        for c in b.children:
            rec(c)
    for b in arm.bones:
        if b.parent is None:
            rec(b)
    return order


def _scale_from(J):
    leg = []
    for s in ("l", "r"):
        a = np.linalg.norm(J[f"{s}_hip"] - J[f"{s}_knee"], axis=-1)
        b = np.linalg.norm(J[f"{s}_knee"] - J[f"{s}_ankle"], axis=-1)
        leg.append(a + b)
    v = np.concatenate([np.atleast_1d(x) for x in leg])
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else None


def apply_motion(obj, J_mocap, fps, start_frame=1, root_mode="RELATIVE", action_name=None):
    """Calcule et écrit l'animation sur l'armature obj (noms Mixamo, préfixe quelconque)."""
    prefix = find_prefix(obj)
    Jm = derive(dict(J_mocap))
    Jr = derive(virtual_joints(obj, prefix, posed=False))
    specs = bone_specs()
    T = next(iter(J_mocap.values())).shape[0]
    mw = np_mat(obj.matrix_world)
    Rw = np_mat(obj.matrix_world.to_3x3().normalized())
    arm = obj.data
    rest_rot = {b.name: np_mat(b.matrix_local.to_3x3().normalized()) for b in arm.bones}
    M = {}
    channels = []
    report = {"animated": 0, "static": 0}

    for b in _ordered_bones(arm):
        short = b.name[len(prefix):] if b.name.startswith(prefix) else None
        Rb = rest_rot[b.name]
        p = b.parent
        Mb = None
        delta = None
        if short in specs:
            dfn, rfn = specs[short]
            with np.errstate(invalid="ignore"):
                dm, rm = dfn(Jm), rfn(Jm)
                # référence de torsion absente (ex. main cachée) : on garde la dernière
                # torsion connue, mais la direction de l'os continue de suivre la capture
                rv = np.isfinite(rm).all(-1) & (np.linalg.norm(np.nan_to_num(rm), axis=-1) > 1e-9)
                if rv.any() and not rv.all():
                    rm = hold_fill(rm, rv)
                Fm, valid = make_frames(dm, rm)
                Fr, vr = make_frames(dfn(Jr), rfn(Jr))
            Fm = hold_fill(Fm, valid) if bool(vr) else None
            if Fm is not None:
                delta = Fm @ Fr.T                       # rotation monde capture vs repos
                W = delta @ (Rw @ Rb)
                Mb = np.einsum("ij,tjk->tik", Rw.T, W)  # espace armature
        if Mb is None:
            report["static"] += 1
            if p is not None:
                Mb = M[p.name] @ (rest_rot[p.name].T @ Rb)
            else:
                Mb = np.repeat(Rb[None], T, 0)
        else:
            report["animated"] += 1
        M[b.name] = Mb
        if p is not None:
            B = np.einsum("ij,tjk->tik", Rb.T @ rest_rot[p.name], np.transpose(M[p.name], (0, 2, 1)) @ Mb)
        else:
            B = np.einsum("ij,tjk->tik", Rb.T, Mb)
        if delta is None and p is not None:
            continue  # os non piloté : on laisse sa pose de repos
        q = mat_to_quat(B)
        dp = f'pose.bones["{b.name}"].rotation_quaternion'
        for k in range(4):
            channels.append((dp, k, q[:, k], b.name))
        obj.pose.bones[b.name].rotation_mode = "QUATERNION"

        if short == "Hips" and delta is not None:
            # déplacement racine, mis à l'échelle du personnage
            s_m, s_t = _scale_from(Jm), _scale_from(Jr)
            s = (s_t / s_m) if (s_m and s_t) else 1.0
            mh = Jm["mid_hip"].copy()
            okm = np.isfinite(mh).all(1)
            if okm.any():
                idx = np.where(okm)[0]
                mh = np.stack([np.interp(np.arange(T), idx, mh[idx, k]) for k in range(3)], 1)
                h0 = np.array((obj.matrix_world @ b.matrix_local).translation)
                off = h0 - Jr["mid_hip"]
                if root_mode == "RELATIVE":
                    pos = np.empty_like(mh)
                    pos[:, :2] = Jr["mid_hip"][:2] + s * (mh[:, :2] - mh[idx[0], :2])
                    pos[:, 2] = mw[2, 3] + s * mh[:, 2]
                else:
                    pos = s * mh
                pos = pos + np.einsum("tij,j->ti", delta, off)
                inv = np.linalg.inv(mw)
                pa = (np.c_[pos, np.ones(T)] @ inv.T)[:, :3]   # espace armature
                rest4 = np_mat(b.matrix_local)
                if p is not None:
                    rest4 = np.linalg.inv(np_mat(p.matrix_local)) @ rest4
                    pa = (np.c_[pa, np.ones(T)] @ np.linalg.inv(np_mat(p.matrix_local)).T)[:, :3]
                Rinv = np.linalg.inv(rest4[:3, :3])
                loc = (pa - rest4[:3, 3]) @ Rinv.T
                dp = f'pose.bones["{b.name}"].location'
                for k in range(3):
                    channels.append((dp, k, loc[:, k], b.name))
            report["scale"] = s

    name = action_name or f"MocapCam_{obj.name}"
    write_action(obj, name, start_frame, channels)
    return report


def write_action(obj, name, start_frame, channels):
    ad = obj.animation_data or obj.animation_data_create()
    act = bpy.data.actions.new(name)
    ad.action = act
    frames = None
    for dp, idx, vals, group in channels:
        if hasattr(act, "fcurve_ensure_for_datablock"):      # Blender 4.4+
            fc = act.fcurve_ensure_for_datablock(obj, dp, index=idx, group_name=group)
        else:
            fc = act.fcurves.new(dp, index=idx, action_group=group)
        n = len(vals)
        if frames is None or len(frames) != n:
            frames = np.arange(start_frame, start_frame + n, dtype=float)
        fc.keyframe_points.add(n)
        co = np.empty(2 * n)
        co[0::2] = frames
        co[1::2] = vals
        fc.keyframe_points.foreach_set("co", co.tolist())
        fc.update()
    return act


# --------------------------------------------------------------------------- squelette généré

def _median_len(J, a, b, default):
    v = np.linalg.norm(J[a] - J[b], axis=-1)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size > 10 else default


def create_mixamo_armature(context, J, name="MocapCam_Squelette", obj_matrix=None,
                           pose="T"):
    """Crée un squelette aux 65 os Mixamo ('mixamorig:'), en T-pose,
    aux proportions mesurées sur la capture. Face à -Y, gauche du perso en +X."""
    Jd = derive(dict(J))
    m = lambda a, b, d: 0.5 * (_median_len(Jd, "l_" + a, "l_" + b, d) + _median_len(Jd, "r_" + a, "r_" + b, d))
    thigh, shin = m("hip", "knee", 0.43), m("knee", "ankle", 0.42)
    upper, fore = m("shoulder", "elbow", 0.28), m("elbow", "wrist", 0.26)
    hip_w = _median_len(Jd, "l_hip", "r_hip", 0.2)
    sh_w = _median_len(Jd, "l_shoulder", "r_shoulder", 0.36)
    torso = _median_len(Jd, "mid_hip", "mid_sh", 0.5)
    neck = _median_len(Jd, "mid_sh", "mid_ear", 0.2)
    foot = m("ankle", "toes", 0.18)
    sc = (thigh + shin) / 0.85

    def hand_len(a, b, d):
        return 0.5 * (_median_len(Jd, f"lh_{a}", f"lh_{b}", d) + _median_len(Jd, f"rh_{a}", f"rh_{b}", d))
    palm = hand_len(0, 9, 0.09 * sc)

    ankle_h = 0.07 * sc
    knee_z = ankle_h + shin
    hip_z = knee_z + thigh
    sh_z = hip_z + torso
    P = {}  # os -> (tête, queue) en coordonnées monde
    up = np.array([0, 0, 1.0])
    P["Hips"] = (np.array([0, 0, hip_z]), np.array([0, 0, hip_z + 0.1 * torso]))
    P["Spine"] = (np.array([0, 0, hip_z + 0.1 * torso]), np.array([0, 0, hip_z + 0.35 * torso]))
    P["Spine1"] = (P["Spine"][1], np.array([0, 0, hip_z + 0.6 * torso]))
    P["Spine2"] = (P["Spine1"][1], np.array([0, 0, hip_z + 0.92 * torso]))
    head_z = sh_z + neck - 0.07 * sc
    P["Neck"] = (P["Spine2"][1], np.array([0, 0, head_z]))
    P["Head"] = (np.array([0, 0, head_z]), np.array([0, 0, head_z + 0.2 * sc]))
    P["HeadTop_End"] = (P["Head"][1], P["Head"][1] + 0.04 * sc * up)
    for side, sg in (("Left", 1), ("Right", -1)):
        X = np.array([sg, 0, 0.0])
        sh = np.array([sg * sh_w / 2, 0, sh_z])
        P[f"{side}Shoulder"] = (np.array([sg * 0.03 * sc, 0, sh_z - 0.03 * sc]), sh)
        el = sh + upper * X
        wr = el + fore * X
        P[f"{side}Arm"] = (sh, el)
        P[f"{side}ForeArm"] = (el, wr)
        P[f"{side}Hand"] = (wr, wr + palm * X)
        bases = {"Index": wr + X * palm * 0.95 + np.array([0, -0.022 * sc, 0]),
                 "Middle": wr + X * palm,
                 "Ring": wr + X * palm * 0.95 + np.array([0, 0.02 * sc, 0]),
                 "Pinky": wr + X * palm * 0.88 + np.array([0, 0.04 * sc, 0])}
        fb = {"Thumb": 1, "Index": 5, "Middle": 9, "Ring": 13, "Pinky": 17}
        thumb_dir = nrm(np.array([sg * 1.0, -1.0, -0.35]))
        for f in FINGERS:
            b0 = fb[f]
            if f == "Thumb":
                pt = wr + sc * np.array([sg * 0.02, -0.02, -0.01])
                d = thumb_dir
                defaults = (0.04 * sc, 0.035 * sc, 0.03 * sc)
            else:
                pt = bases[f]
                d = X
                defaults = (0.04 * sc, 0.025 * sc, 0.022 * sc)
            for k in range(3):
                L = hand_len(b0 + k, b0 + k + 1, defaults[k])
                P[f"{side}Hand{f}{k + 1}"] = (pt, pt + L * d)
                pt = pt + L * d
            P[f"{side}Hand{f}4"] = (pt, pt + 0.015 * sc * d)
        hx = sg * hip_w / 2
        P[f"{side}UpLeg"] = (np.array([hx, 0, hip_z]), np.array([hx, 0, knee_z]))
        P[f"{side}Leg"] = (np.array([hx, 0, knee_z]), np.array([hx, 0, ankle_h]))
        ball = np.array([hx, -0.72 * foot, 0.02 * sc])
        tip = np.array([hx, -foot, 0.02 * sc])
        P[f"{side}Foot"] = (np.array([hx, 0, ankle_h]), ball)
        P[f"{side}ToeBase"] = (ball, tip)
        P[f"{side}Toe_End"] = (tip, tip + np.array([0, -0.03 * sc, 0]))

    if pose == "A":   # utile pour les tests : bras baissés de 45°
        for side, sg in (("Left", 1), ("Right", -1)):
            pivot = P[f"{side}Arm"][0]
            a = math.radians(-45 * sg)
            Rm = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
            for k, (h, t) in list(P.items()):
                if k.startswith(side) and not ("Leg" in k or "Foot" in k or "Toe" in k or "Shoulder" in k):
                    P[k] = (pivot + Rm @ (h - pivot), pivot + Rm @ (t - pivot))

    data = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, data)
    context.collection.objects.link(obj)
    if obj_matrix is not None:
        obj.matrix_world = obj_matrix
    inv = np.linalg.inv(np_mat(obj.matrix_world))
    to_local = lambda v: Vector((inv @ np.append(v, 1.0))[:3])
    for o in context.view_layer.objects:
        o.select_set(False)
    context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    eb = data.edit_bones
    for bname, parent in mixamo_hierarchy():
        h, t = P[bname]
        e = eb.new(PREFIX + bname)
        e.head, e.tail = to_local(h), to_local(t)
        e.roll = 0.0
        if parent:
            e.parent = eb[PREFIX + parent]
    bpy.ops.object.mode_set(mode="OBJECT")
    data.display_type = "STICK"
    obj.show_in_front = True
    return obj


# --------------------------------------------------------------------------- aides visuelles

def create_helpers(context, d, coll_name="MocapCam_Scene"):
    coll = bpy.data.collections.get(coll_name) or bpy.data.collections.new(coll_name)
    if coll.name not in context.scene.collection.children:
        context.scene.collection.children.link(coll)
    for c in d.get("cameras", []):
        cam = bpy.data.cameras.new(f"Cam_{c['name']}")
        cam.sensor_fit = "HORIZONTAL"
        cam.sensor_width = 36.0
        cam.lens = c["K"][0][0] * 36.0 / c["width"]
        o = bpy.data.objects.new(f"Cam_{c['name']}", cam)
        R = np.array(c["R"])
        t = np.array(c["t"])
        Rc = R.T @ np.diag([1.0, -1.0, -1.0])
        M = np.eye(4)
        M[:3, :3] = Rc
        M[:3, 3] = -R.T @ t
        o.matrix_world = Matrix(M.tolist())
        coll.objects.link(o)
    for m in d.get("markers", []):
        e = bpy.data.objects.new(f"Boule_{m['name']}_{m['color']}", None)
        e.empty_display_type = "SPHERE"
        e.empty_display_size = 0.04
        e.location = m["position"]
        coll.objects.link(e)
    v = d.get("volume")
    if v:
        lo, hi = np.array(v["min"]), np.array(v["max"])
        e = bpy.data.objects.new("Volume_capture", None)
        e.empty_display_type = "CUBE"
        e.location = ((lo + hi) / 2).tolist()
        e.scale = ((hi - lo) / 2).tolist()
        coll.objects.link(e)
    return coll


def create_point_empties(context, d, J, start_frame=1, coll_name="MocapCam_Points"):
    coll = bpy.data.collections.get(coll_name) or bpy.data.collections.new(coll_name)
    if coll.name not in context.scene.collection.children:
        context.scene.collection.children.link(coll)
    names = [n for n in d["joint_names"] if not n.startswith("face_")]
    for n in names:
        e = bpy.data.objects.new(f"pt_{n}", None)
        e.empty_display_type = "SPHERE"
        e.empty_display_size = 0.012 if n[:3] in ("lh_", "rh_") else 0.03
        coll.objects.link(e)
        X = J[n]
        ok = np.isfinite(X).all(1)
        if not ok.any():
            continue
        idx = np.where(ok)[0]
        Xf = np.stack([np.interp(np.arange(len(X)), idx, X[idx, k]) for k in range(3)], 1)
        write_action(e, f"pt_{n}", start_frame,
                     [("location", k, Xf[:, k], "Location") for k in range(3)])
    return coll


# --------------------------------------------------------------------------- export FBX

def export_fbx(context, obj, filepath):
    """Exporte l'armature et son animation en FBX, comme une animation Mixamo
    « sans skin » : os 'mixamorig:', Y vers le haut, personnage face à +Z, sans os de bout
    ajoutés, toutes les images gardées (pas de simplification)."""
    if context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for o in context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
    bpy.ops.export_scene.fbx(
        filepath=filepath, use_selection=True, object_types={"ARMATURE"},
        axis_forward="-Z", axis_up="Y", apply_unit_scale=True, global_scale=1.0,
        apply_scale_options="FBX_SCALE_NONE",
        add_leaf_bones=False, primary_bone_axis="Y", secondary_bone_axis="X",
        armature_nodetype="NULL", use_armature_deform_only=False,
        bake_anim=True, bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False, bake_anim_force_startend_keying=True,
        bake_anim_step=1.0, bake_anim_simplify_factor=0.0)
    return filepath


def capture_to_fbx(src, dst, root_mode="RELATIVE"):
    """Sans interface (blender -b) : capture .mocap.json -> FBX d'un squelette Mixamo animé."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    context = bpy.context
    d, J = load_motion(src)
    # l'animation commence au temps 0 du FBX (image 0), comme les animations Mixamo
    _setup_scene(context, d, 0)
    take = os.path.basename(src).replace(".mocap.json", "").replace(".json", "")
    context.scene.name = take            # nom de l'animation (« take ») dans le FBX
    arm = create_mixamo_armature(context, J, name="Armature")   # même nom que chez Mixamo
    rep = apply_motion(arm, J, d["fps"], 0, root_mode, action_name=take)
    export_fbx(context, arm, dst)
    rep.update(frames=int(d["frame_count"]), fps=float(d["fps"]), bones=len(arm.data.bones))
    return rep


def main_cli():
    """blender -b --factory-startup --python <ce fichier> -- capture.mocap.json sortie.fbx [RELATIVE|ABSOLUTE]"""
    import sys
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(argv) < 2:
        raise SystemExit("usage : -- capture.mocap.json sortie.fbx [RELATIVE|ABSOLUTE]")
    rep = capture_to_fbx(argv[0], argv[1], argv[2] if len(argv) > 2 else "RELATIVE")
    print(f"MOCAPCAM_FBX_OK {json.dumps(rep)}")


# --------------------------------------------------------------------------- interface

ROOT_ITEMS = [("RELATIVE", "Relatif", "Le perso démarre à sa position actuelle"),
              ("ABSOLUTE", "Absolu", "Positions de la zone de capture (origine = boule A)")]


def _setup_scene(context, d, start):
    sc = context.scene
    sc.render.fps = int(round(d["fps"]))
    sc.render.fps_base = 1.0
    sc.frame_start = start
    sc.frame_end = start + d["frame_count"] - 1


class MOCAPCAM_OT_import(bpy.types.Operator, ImportHelper):
    """Importer une capture MocapCam"""
    bl_idname = "mocapcam.import_capture"
    bl_label = "Importer MocapCam"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})
    create_skeleton: BoolProperty(name="Créer le squelette Mixamo", default=True)
    apply_active: BoolProperty(name="Appliquer aussi à l'armature active", default=False,
                               description="Si une armature Mixamo est sélectionnée")
    create_scene: BoolProperty(name="Caméras, boules et volume", default=True)
    create_points: BoolProperty(name="Points bruts (débogage)", default=False)
    root_mode: EnumProperty(name="Déplacement", items=ROOT_ITEMS, default="RELATIVE")
    start_frame: IntProperty(name="Image de départ", default=1)

    def execute(self, context):
        try:
            d, J = load_motion(self.filepath)
        except Exception as e:
            self.report({"ERROR"}, f"Lecture impossible : {e}")
            return {"CANCELLED"}
        target = context.active_object if self.apply_active else None
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        context.scene.mocapcam_filepath = self.filepath
        _setup_scene(context, d, self.start_frame)
        if self.create_scene:
            create_helpers(context, d)
        if self.create_points:
            create_point_empties(context, d, J, self.start_frame)
        if self.create_skeleton:
            name = os.path.basename(self.filepath).replace(".mocap.json", "").replace(".json", "")
            arm = create_mixamo_armature(context, J, name=f"MocapCam_{name}")
            rep = apply_motion(arm, J, d["fps"], self.start_frame, self.root_mode,
                               action_name=f"MocapCam_{name}")
            self.report({"INFO"}, f"Squelette créé : {rep['animated']} os animés")
        if target is not None and target.type == "ARMATURE":
            try:
                apply_motion(target, J, d["fps"], self.start_frame, self.root_mode)
            except RuntimeError as e:
                self.report({"WARNING"}, str(e))
        return {"FINISHED"}


class MOCAPCAM_OT_apply(bpy.types.Operator):
    """Appliquer la capture à l'armature active (noms d'os Mixamo)"""
    bl_idname = "mocapcam.apply_active"
    bl_label = "Appliquer à l'armature active"
    bl_options = {"REGISTER", "UNDO"}
    root_mode: EnumProperty(name="Déplacement", items=ROOT_ITEMS, default="RELATIVE")
    start_frame: IntProperty(name="Image de départ", default=1)

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type == "ARMATURE"

    def execute(self, context):
        path = bpy.path.abspath(context.scene.mocapcam_filepath)
        if not os.path.exists(path):
            self.report({"ERROR"}, "Choisissez d'abord un fichier .mocap.json")
            return {"CANCELLED"}
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        d, J = load_motion(path)
        _setup_scene(context, d, self.start_frame)
        try:
            rep = apply_motion(context.active_object, J, d["fps"], self.start_frame, self.root_mode)
        except RuntimeError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        self.report({"INFO"}, f"{rep['animated']} os animés (échelle {rep.get('scale', 1):.2f})")
        return {"FINISHED"}


class MOCAPCAM_OT_export_fbx(bpy.types.Operator, ExportHelper):
    """Exporter l'armature active et son animation en FBX (réglages Mixamo)"""
    bl_idname = "mocapcam.export_fbx"
    bl_label = "Exporter en FBX (Mixamo)"
    filename_ext = ".fbx"
    filter_glob: StringProperty(default="*.fbx", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type == "ARMATURE"

    def execute(self, context):
        try:
            export_fbx(context, context.active_object, self.filepath)
        except Exception as e:
            self.report({"ERROR"}, f"Export impossible : {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"FBX écrit : {self.filepath}")
        return {"FINISHED"}


class MOCAPCAM_PT_panel(bpy.types.Panel):
    bl_label = "MocapCam"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MocapCam"

    def draw(self, context):
        lay = self.layout
        lay.operator(MOCAPCAM_OT_import.bl_idname, icon="IMPORT")
        lay.separator()
        lay.prop(context.scene, "mocapcam_filepath", text="")
        lay.operator(MOCAPCAM_OT_apply.bl_idname, icon="ARMATURE_DATA")
        lay.operator(MOCAPCAM_OT_export_fbx.bl_idname, icon="EXPORT")
        o = context.active_object
        if o is not None and o.type == "ARMATURE":
            try:
                lay.label(text=f"Préfixe détecté : '{find_prefix(o)}'", icon="CHECKMARK")
            except RuntimeError:
                lay.label(text="Pas de noms d'os Mixamo", icon="ERROR")


def menu_import(self, context):
    self.layout.operator(MOCAPCAM_OT_import.bl_idname, text="MocapCam (.mocap.json)")


def menu_export(self, context):
    self.layout.operator(MOCAPCAM_OT_export_fbx.bl_idname, text="FBX MocapCam (Mixamo)")


CLASSES = (MOCAPCAM_OT_import, MOCAPCAM_OT_apply, MOCAPCAM_OT_export_fbx, MOCAPCAM_PT_panel)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.mocapcam_filepath = StringProperty(name="Capture", subtype="FILE_PATH")
    bpy.types.TOPBAR_MT_file_import.append(menu_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_export)
    bpy.types.TOPBAR_MT_file_import.remove(menu_import)
    del bpy.types.Scene.mocapcam_filepath
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    import sys as _sys
    if "--" in _sys.argv:      # lancé par MocapCam : blender -b ... --python mocapcam_blender.py -- ...
        main_cli()
    else:
        register()
