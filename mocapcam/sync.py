"""Synchronisation des caméras par l'audio (clap de mains au début de la prise)."""
import numpy as np


def _onset_envelope(audio, sr, env_sr=1000):
    """Enveloppe d'attaque : met en évidence les sons brefs (clap)."""
    hop = sr // env_sr
    n = len(audio) // hop
    if n < 10:
        return np.zeros(1)
    frames = np.abs(audio[: n * hop]).reshape(n, hop).max(axis=1)
    loud = np.log1p(frames * 100.0)
    onset = np.maximum(np.diff(loud, prepend=loud[0]), 0.0)
    return onset - onset.mean()


def audio_delay(audio_ref, audio_other, sr=16000, max_lag_s=20.0, env_sr=1000):
    """Retourne (d, confiance) tel qu'un son au temps t dans la vidéo de
    référence apparaisse au temps t + d dans l'autre vidéo."""
    a = _onset_envelope(audio_ref, sr, env_sr)
    b = _onset_envelope(audio_other, sr, env_sr)
    if a.size < 10 or b.size < 10:
        return 0.0, 0.0
    n = 1 << int(np.ceil(np.log2(a.size + b.size)))
    corr = np.fft.irfft(np.conj(np.fft.rfft(a, n)) * np.fft.rfft(b, n), n)
    max_lag = min(int(max_lag_s * env_sr), n // 2 - 1, max(a.size, b.size))
    lags = np.concatenate([np.arange(0, max_lag + 1), np.arange(-max_lag, 0)])
    vals = np.concatenate([corr[: max_lag + 1], corr[n - max_lag:]])
    k = int(np.argmax(vals))
    best = vals[k]
    # confiance : pic principal vs second pic éloigné de plus de 50 ms
    mask = np.abs(lags - lags[k]) > 0.05 * env_sr
    second = vals[mask].max() if mask.any() else 0.0
    conf = float(best / (second + 1e-9)) if best > 0 else 0.0
    # affinage sous-échantillon (parabole)
    d = float(lags[k])
    if 0 < k < len(vals) - 1:
        y0, y1, y2 = vals[k - 1], vals[k], vals[k + 1]
        den = (y0 - 2 * y1 + y2)
        if abs(den) > 1e-12:
            d += 0.5 * (y0 - y2) / den
    return float(d / env_sr), float(conf)


# ----------------------------------------------------------------------------- chirp
# Signal sonore joué par MocapCam Studio (haut-parleurs du PC) juste après le départ
# de l'enregistrement. Le même signal est généré dans l'interface (Web Audio).
CHIRP = dict(f0=1500.0, f1=6000.0, dur=0.30)


def chirp_signal(sr, f0=CHIRP["f0"], f1=CHIRP["f1"], dur=CHIRP["dur"]):
    t = np.arange(int(dur * sr)) / sr
    k = (f1 - f0) / dur
    s = np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))
    return (s * np.hanning(len(t))).astype(np.float32)


def find_chirp(audio, sr=16000, search=None):
    """Position (s) du début du chirp dans la piste audio, et confiance
    (pic / bruit de corrélation). search = (début, fin) en secondes pour limiter la recherche."""
    if audio.size < sr // 2:
        return None, 0.0
    off = 0
    if search is not None:
        a, b = max(0, int(search[0] * sr)), min(audio.size, int(search[1] * sr))
        if b - a < sr // 2:
            return None, 0.0
        audio, off = audio[a:b], a
    tpl = chirp_signal(sr)
    n = 1 << int(np.ceil(np.log2(audio.size + tpl.size)))
    corr = np.fft.irfft(np.fft.rfft(audio, n) * np.conj(np.fft.rfft(tpl, n)), n)[: audio.size - tpl.size + 1]
    # normalisation par l'énergie locale (insensible au volume)
    e2 = np.convolve(audio.astype(np.float64) ** 2, np.ones(tpl.size), "valid")
    # plancher d'énergie : un passage de silence numérique (début de fichier sur certains
    # téléphones) ne doit pas créer de faux pic par division par ~0
    floor = 0.05 * np.median(e2[e2 > 0]) if np.any(e2 > 0) else 1.0
    e = np.sqrt(np.maximum(e2, floor)) + 1e-9
    c = np.abs(corr) / e / np.sqrt(np.sum(tpl.astype(np.float64) ** 2))
    k = int(np.argmax(c))
    noise = np.median(c) + 1e-9
    conf = float(c[k] / noise)
    t = float(k)
    if 0 < k < c.size - 1:
        y0, y1, y2 = c[k - 1], c[k], c[k + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            t += 0.5 * (y0 - y2) / den
    return (t + off) / sr, conf


def find_chirps(audio, sr=16000, max_n=4, min_conf=12.0, min_sep=1.0):
    """Tous les bips de la piste (plusieurs par prise : départ et arrêt), triés par temps."""
    if audio.size < sr // 2:
        return []
    tpl = chirp_signal(sr)
    n = 1 << int(np.ceil(np.log2(audio.size + tpl.size)))
    corr = np.fft.irfft(np.fft.rfft(audio, n) * np.conj(np.fft.rfft(tpl, n)), n)[: audio.size - tpl.size + 1]
    e2 = np.convolve(audio.astype(np.float64) ** 2, np.ones(tpl.size), "valid")
    floor = 0.05 * np.median(e2[e2 > 0]) if np.any(e2 > 0) else 1.0
    c = np.abs(corr) / (np.sqrt(np.maximum(e2, floor)) + 1e-9) / np.sqrt(np.sum(tpl.astype(np.float64) ** 2))
    noise = np.median(c) + 1e-9
    out = []
    c2 = c.copy()
    for _ in range(max_n):
        k = int(np.argmax(c2))
        conf = float(c2[k] / noise)
        if conf < min_conf:
            break
        t = float(k)
        if 0 < k < c.size - 1:
            y0, y1, y2 = c[k - 1], c[k], c[k + 1]
            den = y0 - 2 * y1 + y2
            if abs(den) > 1e-12:
                t += 0.5 * (y0 - y2) / den
        out.append((t / sr, conf))
        a, b = max(0, k - int(min_sep * sr)), min(c2.size, k + int(min_sep * sr))
        c2[a:b] = 0
    return sorted(out)


def match_chirps(peaks, offsets_s, tol=0.08):
    """Associe les bips trouvés dans une piste aux bips joués par le PC (écarts connus
    offsets_s, en s, par rapport au premier). Renvoie {indice du bip: temps dans la piste}."""
    best = {}
    best_score = -1.0
    for t, _ in peaks:
        for k0, d0 in enumerate(offsets_s):
            o = t - d0                       # instant du 1er bip dans cette piste
            m, score = {}, 0.0
            for k, dk in enumerate(offsets_s):
                cand = [(abs(pt - (o + dk)), pt, pc) for pt, pc in peaks if abs(pt - (o + dk)) < tol]
                if cand:
                    _, pt, pc = min(cand)
                    m[k] = pt
                    score += 1000 + pc
            if score > best_score:
                best, best_score = m, score
    return best
