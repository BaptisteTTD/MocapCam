"""Définition des 133 points COCO-WholeBody (sortie de RTMW / rtmlib)."""

BODY = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
    "l_big_toe", "l_small_toe", "l_heel", "r_big_toe", "r_small_toe", "r_heel",
]
FACE = [f"face_{i}" for i in range(68)]
# Main : 0 poignet, 1-4 pouce (CMC, MCP, IP, bout), 5-8 index (MCP, PIP, DIP, bout),
# 9-12 majeur, 13-16 annulaire, 17-20 auriculaire
LEFT_HAND = [f"lh_{i}" for i in range(21)]
RIGHT_HAND = [f"rh_{i}" for i in range(21)]

JOINT_NAMES = BODY + FACE + LEFT_HAND + RIGHT_HAND
assert len(JOINT_NAMES) == 133
INDEX = {n: i for i, n in enumerate(JOINT_NAMES)}

BODY_IDX = list(range(0, 23))
FACE_IDX = list(range(23, 91))
LHAND_IDX = list(range(91, 112))
RHAND_IDX = list(range(112, 133))
HAND_IDX = LHAND_IDX + RHAND_IDX

# Points utilisés pour le choix de la personne / test de cohérence
CORE_IDX = [INDEX[n] for n in (
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle")]


def _build_swap():
    """Permutation gauche <-> droite (pour corriger les inversions du détecteur 2D)."""
    perm = list(range(133))
    for i, n in enumerate(JOINT_NAMES):
        if n.startswith("l_"):
            perm[i] = INDEX["r_" + n[2:]]
        elif n.startswith("r_"):
            perm[i] = INDEX["l_" + n[2:]]
        elif n.startswith("lh_"):
            perm[i] = INDEX["rh_" + n[3:]]
        elif n.startswith("rh_"):
            perm[i] = INDEX["lh_" + n[3:]]
    return perm


SWAP_LR = _build_swap()
