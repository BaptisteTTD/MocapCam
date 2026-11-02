"""Affinage global du mouvement (« bundle adjustment » du mouvement).

Au lieu de trianguler chaque image séparément puis de lisser, on cherche d'un seul coup la
trajectoire 3D des articulations sur toute la prise, qui à la fois :

* se reprojette au plus près des points 2D vus par *chaque* caméra (une articulation vue par
  une seule caméra pendant un instant reste guidée, au lieu d'être perdue) ;
* garde des os de longueur constante (cuisses, tibias, bras, phalanges, bassin, tête…) ;
* bouge de façon fluide (accélérations faibles), ce qui fait disparaître le bruit sans retard ;
  la pénalité est robuste : les vrais gestes vifs ne sont pas écrasés ;
* garde fixe un pied posé au sol (pas de glissement ni de vibration des jambes et du bassin) ;
* soude le poignet de la main à celui du bras.

Les trous sont comblés par la même occasion, en suivant la physique du mouvement.
Le corps est résolu d'abord, puis chaque main, accrochée au poignet trouvé pour le corps.
Méthode : Levenberg-Marquardt avec résolution creuse directe, pondération robuste (IRLS)
des points 2D et des accélérations.
"""
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu

from .skeleton import BODY_IDX, INDEX, LHAND_IDX, RHAND_IDX, SWAP_LR
from .triangulate import _thresholds, score_reference, score_weight, undistort_normalized


# ----------------------------------------------------------------------------- squelette

def _body_bones():
    """(a, b, poids) en indices globaux : os de longueur constante (poids 1 = rigide)."""
    B = []
    for s in ("l", "r"):
        B += [(f"{s}_hip", f"{s}_knee", 1.0), (f"{s}_knee", f"{s}_ankle", 1.0),
              (f"{s}_shoulder", f"{s}_elbow", 1.0), (f"{s}_elbow", f"{s}_wrist", 1.0),
              (f"{s}_ankle", f"{s}_heel", 1.0), (f"{s}_ankle", f"{s}_big_toe", 0.7),
              (f"{s}_ankle", f"{s}_small_toe", 0.7), (f"{s}_heel", f"{s}_big_toe", 0.5),
              (f"{s}_big_toe", f"{s}_small_toe", 0.5),
              (f"{s}_shoulder", f"{s}_hip", 0.08)]
    B += [("l_hip", "r_hip", 1.0), ("l_shoulder", "r_shoulder", 0.25),
          ("l_shoulder", "r_hip", 0.08), ("r_shoulder", "l_hip", 0.08),
          ("nose", "l_eye", 0.8), ("nose", "r_eye", 0.8), ("l_eye", "r_eye", 0.8),
          ("l_ear", "r_ear", 0.8), ("nose", "l_ear", 0.8), ("nose", "r_ear", 0.8),
          ("l_eye", "l_ear", 0.8), ("r_eye", "r_ear", 0.8)]
    return [(INDEX[a], INDEX[b], w) for a, b, w in B]


def _hand_bones(hidx):
    out = []
    for base in (1, 5, 9, 13, 17):              # chaînes des doigts depuis le poignet
        prev = 0
        for k in range(4):
            out.append((hidx[prev], hidx[base + k], 1.0 if prev else 0.8))
            prev = base + k
    for a, b in ((5, 9), (9, 13), (13, 17), (5, 17), (1, 5)):     # paume
        out.append((hidx[a], hidx[b], 0.8 if a != 1 else 0.3))
    return out


def _body_smooth():
    """Poids de la pénalité d'accélération, par articulation du corps."""
    k = {g: 300.0 for g in BODY_IDX}
    for n, w in (("l_hip", 1200), ("r_hip", 1200), ("l_shoulder", 800), ("r_shoulder", 800),
                 ("nose", 600), ("l_eye", 600), ("r_eye", 600), ("l_ear", 600), ("r_ear", 600),
                 ("l_knee", 500), ("r_knee", 500)):
        k[INDEX[n]] = float(w)
    return k


# ----------------------------------------------------------------------------- observations

def _observations(cams, poses, choices, frames, thr, joints):
    """Points 2D retenus (personne + gauche/droite choisis à la triangulation), pour les
    articulations `joints` (indices globaux) ; renvoie des indices locaux."""
    t_l, j_l, v_l, x_l, w_l = [], [], [], [], []
    norm = [undistort_normalized(p["kp"], c["K"], c["dist"]) for p, c in zip(poses, cams)]
    sref = score_reference(poses)
    sel = np.array(joints)
    for ti, t in enumerate(frames):
        if t not in choices:
            continue
        picked, flips = choices[t]
        for v, pk in enumerate(picked):
            if pk is None:
                continue
            jj, p = pk
            xn, sc = norm[v][jj, p], poses[v]["sc"][jj, p]
            if v > 0 and flips[v - 1]:
                xn, sc = xn[SWAP_LR], sc[SWAP_LR]
            xn, sc = xn[sel], sc[sel]
            w = score_weight(sc, sref)
            k = np.where(np.isfinite(xn).all(1) & (sc >= thr[sel]) & (w > 0))[0]
            t_l.append(np.full(len(k), ti))
            j_l.append(k)
            v_l.append(np.full(len(k), v))
            x_l.append(xn[k])
            w_l.append(w[k])
    if not t_l:
        return None
    return (np.concatenate(t_l), np.concatenate(j_l), np.concatenate(v_l), np.concatenate(x_l),
            np.concatenate(w_l))


def _contacts(X, fps, U, local):
    """Images où chaque point du pied est posé : bas (près de son point le plus bas) et
    presque immobile, pendant au moins 0,1 s. Renvoie (t, j local)."""
    out = []
    for s in ("l", "r"):
        for f in ("heel", "big_toe", "small_toe", "ankle"):
            j = local[INDEX[f"{s}_{f}"]]
            x = X[:, j]
            z = x[:, 2]
            if not np.isfinite(z).any():
                continue
            z0 = np.nanpercentile(z, 5)
            v = np.linalg.norm(np.gradient(x, axis=0), axis=1) * fps
            c = (z < z0 + 0.05 * U) & (v < 0.25 * U)
            edges = np.flatnonzero(np.diff(np.concatenate([[0], c.astype(int), [0]])))
            for a, b in zip(edges[::2], edges[1::2]):
                if b - a >= max(3, int(0.1 * fps)):
                    out += [(t, j) for t in range(a + 1, b)]
    return np.array(out, int).reshape(-1, 2)


# ----------------------------------------------------------------------------- problème

class _Problem:
    """Moindres carrés creux sur les positions X (T, J, 3) d'un groupe d'articulations."""

    def __init__(self, T, J, cams, obs, bones, L, kacc, anchors=(), contacts=None, k_contact=0.0,
                 c_obs=2.5, c_acc=3.0, rel_to=None):
        self.T, self.J = T, J
        self.R = np.stack([c["R"] for c in cams])
        self.tv = np.stack([np.asarray(c["t"], float).ravel() for c in cams])
        self.f = np.array([c["K"][0, 0] for c in cams])
        self.ot, self.oj, self.ov, self.ox, self.ow = obs
        self.ba = np.array([b[0] for b in bones], int)
        self.bb = np.array([b[1] for b in bones], int)
        self.bw = np.array([b[2] for b in bones], float)
        self.L = np.asarray(L, float)
        self.kacc = np.asarray(kacc, float)          # (J,)
        self.acc_j = np.where(self.kacc > 0)[0]
        self.anchors = list(anchors)                 # (j, cible (T,3), poids)
        self.cont = contacts if contacts is not None else np.zeros((0, 2), int)
        self.kc = k_contact
        self.c_obs, self.c_acc = c_obs, c_acc
        # accélérations mesurées par rapport à une articulation (le poignet pour la main) :
        # on lisse le mouvement des doigts dans la main, pas le déplacement de la main entière
        self.rel_to = rel_to
        if rel_to is not None:
            self.acc_j = self.acc_j[self.acc_j != rel_to]

    def vid(self, t, j, k):
        return (t * self.J + j) * 3 + k

    def residuals(self, x):
        X = x.reshape(self.T, self.J, 3)
        P = X[self.ot, self.oj]
        Pc = np.einsum("nij,nj->ni", self.R[self.ov], P) + self.tv[self.ov]
        z = np.where(np.abs(Pc[:, 2]) < 1e-6, 1e-6, Pc[:, 2])
        rep = ((Pc[:, :2] / z[:, None] - self.ox) * (self.f[self.ov] * self.ow)[:, None]).ravel()
        d = X[:, self.ba] - X[:, self.bb]
        bone = ((np.linalg.norm(d, axis=2) - self.L[None]) * self.bw[None]).ravel()
        Y = X[:, self.acc_j] if self.rel_to is None else X[:, self.acc_j] - X[:, self.rel_to][:, None]
        a = Y[:-2] - 2 * Y[1:-1] + Y[2:]
        acc = (a * self.kacc[self.acc_j][None, :, None]).ravel()
        out = [rep, bone, acc]
        for j, target, w in self.anchors:
            out.append(((X[:, j] - target) * w).ravel())
        if len(self.cont):
            c = self.cont
            out.append(((X[c[:, 0], c[:, 1]] - X[c[:, 0] - 1, c[:, 1]]) * self.kc).ravel())
        self.n_rep, self.n_bone, self.n_acc = len(rep), len(bone), len(acc)
        return np.concatenate(out)

    def jacobian(self, x):
        X = x.reshape(self.T, self.J, 3)
        T, J = self.T, self.J
        rows, cols, vals = [], [], []
        r0 = 0
        N = len(self.ot)
        P = X[self.ot, self.oj]
        Rv = self.R[self.ov]
        Pc = np.einsum("nij,nj->ni", Rv, P) + self.tv[self.ov]
        z = np.where(np.abs(Pc[:, 2]) < 1e-6, 1e-6, Pc[:, 2])
        xx, yy = Pc[:, 0] / z, Pc[:, 1] / z
        s = self.f[self.ov] * self.ow
        dx = (Rv[:, 0] - xx[:, None] * Rv[:, 2]) / z[:, None] * s[:, None]
        dy = (Rv[:, 1] - yy[:, None] * Rv[:, 2]) / z[:, None] * s[:, None]
        base = (self.ot * J + self.oj) * 3
        ar = np.arange(N)
        for k in range(3):
            rows += [r0 + 2 * ar, r0 + 2 * ar + 1]
            cols += [base + k, base + k]
            vals += [dx[:, k], dy[:, k]]
        r0 += 2 * N
        nB = len(self.ba)
        if nB:
            d = X[:, self.ba] - X[:, self.bb]
            n = np.linalg.norm(d, axis=2, keepdims=True)
            u = (d / np.where(n < 1e-9, 1e-9, n) * self.bw[None, :, None]).reshape(-1, 3)
            rr = r0 + np.arange(T * nB)
            tt = np.repeat(np.arange(T), nB)
            aa, bb = np.tile(self.ba, T), np.tile(self.bb, T)
            for k in range(3):
                rows += [rr, rr]
                cols += [self.vid(tt, aa, k), self.vid(tt, bb, k)]
                vals += [u[:, k], -u[:, k]]
            r0 += T * nB
        na = len(self.acc_j)
        if na and T > 2:
            ta = np.repeat(np.arange(T - 2), na * 3)
            ja = np.tile(np.repeat(self.acc_j, 3), T - 2)
            ka = np.tile(np.tile(np.arange(3), na), T - 2)
            ra = r0 + np.arange((T - 2) * na * 3)
            wa = np.tile(np.repeat(self.kacc[self.acc_j], 3), T - 2)
            for off, c in ((0, 1.0), (1, -2.0), (2, 1.0)):
                rows.append(ra)
                cols.append(self.vid(ta + off, ja, ka))
                vals.append(c * wa)
                if self.rel_to is not None:
                    rows.append(ra)
                    cols.append(self.vid(ta + off, np.full_like(ja, self.rel_to), ka))
                    vals.append(-c * wa)
            r0 += (T - 2) * na * 3
        for j, _target, w in self.anchors:
            rt = r0 + np.arange(T * 3)
            rows.append(rt)
            cols.append(self.vid(np.repeat(np.arange(T), 3), j, np.tile(np.arange(3), T)))
            vals.append(np.full(T * 3, w))
            r0 += T * 3
        if len(self.cont):
            c = self.cont
            nc = len(c)
            rc = r0 + np.arange(nc * 3)
            tc, jc, kc = np.repeat(c[:, 0], 3), np.repeat(c[:, 1], 3), np.tile(np.arange(3), nc)
            rows += [rc, rc]
            cols += [self.vid(tc, jc, kc), self.vid(tc - 1, jc, kc)]
            vals += [np.full(nc * 3, self.kc), np.full(nc * 3, -self.kc)]
            r0 += nc * 3
        return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(r0, T * J * 3)).tocsc()

    def weights(self, r):
        """Pondération robuste (IRLS, perte « soft L1 ») des points 2D et des accélérations ;
        os, ancrages et appuis restent quadratiques."""
        w = np.ones(len(r))
        a, b = 0, self.n_rep
        w[a:b] = (1 + (r[a:b] / self.c_obs) ** 2) ** -0.25
        a, b = self.n_rep + self.n_bone, self.n_rep + self.n_bone + self.n_acc
        w[a:b] = (1 + (r[a:b] / self.c_acc) ** 2) ** -0.25
        return w

    def cost(self, r):
        rho = r ** 2
        a, b = 0, self.n_rep
        rho[a:b] = 2 * self.c_obs ** 2 * (np.sqrt(1 + (r[a:b] / self.c_obs) ** 2) - 1)
        a, b = self.n_rep + self.n_bone, self.n_rep + self.n_bone + self.n_acc
        rho[a:b] = 2 * self.c_acc ** 2 * (np.sqrt(1 + (r[a:b] / self.c_acc) ** 2) - 1)
        return float(rho.sum())


def _solve(prob, x0, max_iter=30, tol=1e-4):
    """Levenberg-Marquardt, systèmes normaux creux résolus directement (LU)."""
    x = x0.copy()
    r = prob.residuals(x)
    cost = prob.cost(r)
    lam = 1e-3
    it = 0
    for it in range(1, max_iter + 1):
        Jm = prob.jacobian(x)
        w = prob.weights(r)
        Jw = diags(w) @ Jm
        A = (Jw.T @ Jw).tocsc()
        g = Jw.T @ (w * r)
        dA = A.diagonal()
        improved, rel = False, 0.0
        for _ in range(8):
            M = (A + diags(lam * np.maximum(dA, 1e-9))).tocsc()
            try:
                step = -splu(M, permc_spec="COLAMD").solve(g)
            except RuntimeError:
                lam *= 10
                continue
            xn = x + step
            rn = prob.residuals(xn)
            cn = prob.cost(rn)
            if cn < cost:
                rel = (cost - cn) / max(cost, 1e-12)
                x, r, cost = xn, rn, cn
                lam = max(lam / 3, 1e-7)
                improved = True
                break
            lam *= 10
        if not improved or rel < tol:
            break
    prob.residuals(x)
    return x, it


# ----------------------------------------------------------------------------- point d'entrée

def _fill(X):
    """Comble les trous par interpolation (valeurs du bord aux extrémités)."""
    X = X.copy()
    T = X.shape[0]
    t = np.arange(T)
    ok = np.isfinite(X[..., 0])
    for j in range(X.shape[1]):
        if ok[:, j].sum() >= 2:
            for k in range(3):
                X[:, j, k] = np.interp(t, t[ok[:, j]], X[ok[:, j], j, k])
    return X, ok.sum(0) >= 2


def _median_err(prob, x):
    r = prob.residuals(x)
    e = np.linalg.norm(r[:prob.n_rep].reshape(-1, 2), axis=1) / np.maximum(prob.ow, 1e-9)
    return float(np.median(e[prob.ow > 0.5]))


def optimize(X0, cams, poses, choices, fps, cfg=None, log=print, frames=None, max_iter=30):
    """X0 (T,133,3) : première estimation (triangulée, nettoyée, lissée). Retourne X affiné et
    des statistiques. frames : indices (base de temps commune) des lignes de X0."""
    cfg = cfg or {}
    T = X0.shape[0]
    frames = np.arange(T) if frames is None else np.asarray(frames)
    thr = _thresholds(cfg)
    out = X0.copy()
    stats = {}
    w_bone = cfg.get("bone_weight", 2000.0)

    # ---------------- corps
    joints = list(BODY_IDX)
    local = {g: i for i, g in enumerate(joints)}
    Xb, usable = _fill(X0[:, joints])
    if not usable[[local[INDEX[n]] for n in ("l_hip", "r_hip", "l_knee", "l_ankle", "r_knee", "r_ankle")]].all():
        log("  affinage global impossible (corps incomplet)")
        return X0, None
    legs = [np.linalg.norm(Xb[:, local[INDEX[f"{s}_hip"]]] - Xb[:, local[INDEX[f"{s}_knee"]]], axis=1)
            + np.linalg.norm(Xb[:, local[INDEX[f"{s}_knee"]]] - Xb[:, local[INDEX[f"{s}_ankle"]]], axis=1)
            for s in ("l", "r")]
    U = float(np.median(np.concatenate(legs)))          # unité : longueur de jambe
    obs = _observations(cams, poses, choices, frames, thr, joints)
    if obs is None:
        log("  affinage global impossible (pas d'observations)")
        return X0, None
    keep = usable[obs[1]]
    obs = tuple(o[keep] for o in obs)
    bones = [(local[a], local[b], w * w_bone / U) for a, b, w in _body_bones()
             if usable[local[a]] and usable[local[b]]]
    ia, ib = [b[0] for b in bones], [b[1] for b in bones]
    sm = _body_smooth()
    kacc = np.array([sm[g] * usable[local[g]] for g in joints]) * cfg.get("smooth_body", 2.0) / U
    cont = _contacts(Xb, fps, U, local)
    e0 = None
    for k in range(2):                         # longueurs d'os ré-estimées une fois
        L = np.median(np.linalg.norm(Xb[:, ia] - Xb[:, ib], axis=2), axis=0)
        prob = _Problem(T, len(joints), cams, obs, bones, L, kacc, contacts=cont,
                        k_contact=cfg.get("contact_weight", 2500.0) / U)
        e0 = _median_err(prob, Xb.ravel()) if e0 is None else e0
        x, its = _solve(prob, Xb.ravel(), max_iter if k == 0 else max_iter // 3)
        Xb = x.reshape(T, len(joints), 3)
    e1 = _median_err(prob, Xb.ravel())
    sel = [g for g in joints if usable[local[g]]]
    out[:, sel] = Xb[:, [local[g] for g in sel]]
    lens = np.linalg.norm(Xb[:, ia] - Xb[:, ib], axis=2)
    stats["body"] = dict(observations=int(len(obs[0])), contacts=int(len(cont)), iterations=its,
                         reproj_before_px=e0, reproj_after_px=e1,
                         bone_cv=float(np.median(np.std(lens, 0) / np.mean(lens, 0))))
    log(f"  affinage global du corps : {len(obs[0])} points 2D, {len(cont)} appuis au sol ; "
        f"erreur médiane {e0:.1f} -> {e1:.1f} px")

    # ---------------- mains, accrochées aux poignets du corps
    for side, hidx in (("l", LHAND_IDX), ("r", RHAND_IDX)):
        joints = list(hidx)
        local = {g: i for i, g in enumerate(joints)}
        Xh, usable = _fill(X0[:, joints])
        if usable.sum() < 10 or not usable[0]:
            continue
        obs = _observations(cams, poses, choices, frames, thr, joints)
        if obs is None:
            continue
        keep = usable[obs[1]]
        obs = tuple(o[keep] for o in obs)
        wrist = out[:, INDEX[f"{side}_wrist"]]
        Xh = Xh + (wrist - Xh[:, 0])[:, None, :]          # main posée sur le poignet du corps
        bones = [(local[a], local[b], w * w_bone / U) for a, b, w in _hand_bones(hidx)
                 if usable[local[a]] and usable[local[b]]]
        ia, ib = [b[0] for b in bones], [b[1] for b in bones]
        kacc = np.array([float(usable[local[g]]) for g in joints]) * 350.0 * cfg.get("smooth_hands", 8.0) / U
        e0 = None
        for k in range(2):
            L = np.median(np.linalg.norm(Xh[:, ia] - Xh[:, ib], axis=2), axis=0)
            prob = _Problem(T, len(joints), cams, obs, bones, L, kacc,
                            anchors=[(0, wrist, cfg.get("tie_weight", 3000.0) / U)], rel_to=0)
            e0 = _median_err(prob, Xh.ravel()) if e0 is None else e0
            x, its = _solve(prob, Xh.ravel(), max_iter if k == 0 else max_iter // 3)
            Xh = x.reshape(T, len(joints), 3)
        e1 = _median_err(prob, Xh.ravel())
        sel = [g for g in joints if usable[local[g]]]
        out[:, sel] = Xh[:, [local[g] for g in sel]]
        stats[f"{side}_hand"] = dict(observations=int(len(obs[0])), iterations=its,
                                     reproj_before_px=e0, reproj_after_px=e1)
        log(f"  affinage global main {'gauche' if side == 'l' else 'droite'} : {len(obs[0])} points 2D ; "
            f"erreur médiane {e0:.1f} -> {e1:.1f} px")
    stats["unit_m"] = U
    return out, stats
