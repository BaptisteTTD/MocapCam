"""MocapCam Studio : pilotage des téléphones depuis le PC.

  python -m mocapcam studio            (ou double-clic sur « MocapCam Studio.bat »)

- les téléphones (app MocapCam) se connectent en scannant un QR code ;
- aperçu en direct, état (batterie, i/s, synchro d'horloge) ;
- vérification des boules au sol en direct ;
- enregistrement lancé en même temps sur tous les téléphones + signal sonore de synchro ;
- rapatriement automatique des vidéos, puis traitement de la prise.
"""
import asyncio
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import wave
import webbrowser
from dataclasses import dataclass, field

import numpy as np
from aiohttp import WSMsgType, web

from . import markers as mk, pipeline, sync

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "studio_ui")

# horloge du PC en ms (monotone, calée sur l'heure système au démarrage)
_EPOCH = time.time() * 1000 - time.monotonic() * 1000


def now_ms():
    return _EPOCH + time.monotonic() * 1000


def safe_name(s, default="x"):
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(s or "")).strip("_")
    return s[:40] or default


def lan_addresses():
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.append(info[4][0])
    except OSError:
        pass
    out = []
    for ip in ips:
        if ip not in out and not ip.startswith(("127.", "169.254.")):
            out.append(ip)
    return out or ["127.0.0.1"]


@dataclass
class Cam:
    id: str
    ws: web.WebSocketResponse
    name: str
    platform: str = ""
    model: str = ""
    status: dict = field(default_factory=dict)
    samples: list = field(default_factory=list)     # (rtt, offset)
    offset: float = 0.0                             # horloge téléphone - horloge PC (ms)
    rtt: float = float("nan")
    preview: bytes = b""
    preview_ts: float = 0.0
    check: dict = None
    check_jpg: bytes = b""
    take: dict = field(default_factory=dict)        # état pour la prise en cours
    locked: bool = False
    connected_at: float = field(default_factory=now_ms)

    def public(self):
        st = self.status
        return dict(
            id=self.id, name=self.name, platform=self.platform, model=self.model,
            battery=st.get("battery"), storage_free=st.get("storage_free"), fps=st.get("fps"),
            stabilization=st.get("stabilization"), exposure=st.get("exposure"),
            recording=st.get("recording", False), locked=st.get("locked", self.locked),
            snapshot_size=st.get("snapshot_size"), device=st.get("device"),
            device_type=st.get("device_type"), resolution=st.get("resolution"),
            res720=st.get("res720"), supports60=st.get("supports60"),
            sync_ms=None if np.isnan(self.rtt) else round(self.rtt / 2, 1),
            preview_ts=self.preview_ts, check=self.check, take=self.take,
        )


class Studio:
    def __init__(self, studio_dir, captures_dir, port):
        self.studio_dir = os.path.abspath(studio_dir)
        self.captures_dir = os.path.abspath(captures_dir)
        os.makedirs(self.captures_dir, exist_ok=True)
        if not os.path.exists(os.path.join(self.studio_dir, "studio.json")):
            pipeline.init_studio(self.studio_dir)
        self.port = port
        self.token = secrets.token_hex(4)
        self.cams: dict[str, Cam] = {}
        self.uis: set = set()
        self.take = None            # prise en cours
        self.proc = None            # traitement en cours
        self.proc_log = []
        self._n = 0
        self._dirty = asyncio.Event()
        self.chirp_wav = None
        self.prefer60 = True        # priorité à 60 i/s (720p si le téléphone l'exige)
        self.preview_fps = 3.0

    # ------------------------------------------------------------------ état -> interface
    def takes_list(self):
        out = []
        for d in sorted(os.listdir(self.captures_dir), reverse=True):
            p = os.path.join(self.captures_dir, d)
            if not os.path.isdir(p) or d.startswith("_"):
                continue
            vids = [f for f in os.listdir(p) if f.lower().endswith((".mp4", ".mov"))]
            res = os.path.join(p, f"{d}.mocap.json")
            meta = {}
            if os.path.exists(os.path.join(p, "take.json")):
                try:
                    with open(os.path.join(p, "take.json"), encoding="utf-8") as f:
                        meta = json.load(f)
                except (OSError, ValueError):
                    pass
            out.append(dict(name=d, videos=vids, processed=os.path.exists(res),
                            result=res if os.path.exists(res) else None,
                            created=meta.get("created"), duration=meta.get("duration_s")))
        return out[:50]

    def next_take_name(self):
        n = 1
        names = {t["name"] for t in self.takes_list()}
        while f"prise_{n:03d}" in names:
            n += 1
        return f"prise_{n:03d}"

    def state(self):
        return dict(
            type="state", cams=[c.public() for c in self.cams.values()],
            take=self.take, takes=self.takes_list(), next_take=self.next_take_name(),
            processing=None if self.proc is None else dict(take=self.proc["take"], running=self.proc["running"],
                                                          ok=self.proc.get("ok")),
            qr=self.qr_payload(), now=now_ms(), studio=self.studio_dir, captures=self.captures_dir,
            prefer60=self.prefer60, preview_fps=self.preview_fps,
        )

    def qr_payload(self):
        return dict(mocapcam=1, hosts=lan_addresses(), port=self.port, token=self.token)

    def changed(self):
        self._dirty.set()

    async def broadcast(self, msg):
        data = json.dumps(msg)
        for ws in list(self.uis):
            try:
                await ws.send_str(data)
            except (ConnectionError, RuntimeError):
                self.uis.discard(ws)

    async def state_pusher(self):
        while True:
            await self._dirty.wait()
            self._dirty.clear()
            await self.broadcast(self.state())
            await asyncio.sleep(0.2)

    async def toast(self, msg, level="info"):
        await self.broadcast(dict(type="toast", msg=msg, level=level))

    # ------------------------------------------------------------------ téléphones
    async def send(self, cam, msg):
        try:
            await cam.ws.send_str(json.dumps(msg))
        except (ConnectionError, RuntimeError):
            pass

    async def ping_loop(self, cam):
        n = 0
        while not cam.ws.closed:
            await self.send(cam, dict(t="ping", s0=now_ms()))
            n += 1
            await asyncio.sleep(0.1 if n < 15 else 1.0)

    def on_pong(self, cam, msg):
        s1 = now_ms()
        s0 = float(msg["s0"])
        rtt = s1 - s0
        off = float(msg["c"]) - (s0 + s1) / 2
        cam.samples.append((rtt, off, s1))
        cam.samples = [x for x in cam.samples if s1 - x[2] < 60000][-60:]
        best = min(cam.samples, key=lambda x: x[0])
        cam.rtt, cam.offset = best[0], best[1]

    def unique_name(self, name, cam_id=None):
        base = safe_name(name, "camera").lower()
        taken = {c.name for c in self.cams.values() if c.id != cam_id}
        n, out = 2, base
        while out in taken:
            out = f"{base}_{n}"
            n += 1
        return out

    async def handle_phone(self, request):
        ws = web.WebSocketResponse(heartbeat=10, max_msg_size=32 * 1024 * 1024)
        await ws.prepare(request)
        cam = None
        pinger = None
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        m = json.loads(msg.data)
                    except ValueError:
                        continue
                    t = m.get("t")
                    if cam is None:
                        if t != "hello":
                            continue
                        if m.get("token") != self.token:
                            await ws.send_str(json.dumps(dict(t="error", message="QR code périmé : rescannez-le")))
                            await ws.close()
                            break
                        self._n += 1
                        cid = f"c{self._n}"
                        cam = Cam(id=cid, ws=ws, name=self.unique_name(m.get("name", "camera")),
                                  platform=m.get("platform", ""), model=m.get("model", ""))
                        self.cams[cid] = cam
                        await self.send(cam, dict(t="welcome", id=cid, name=cam.name,
                                                  prefer60=self.prefer60, preview=self.preview_fps))
                        pinger = asyncio.create_task(self.ping_loop(cam))
                        await self.toast(f"Caméra connectée : {cam.name}")
                        self.changed()
                        continue
                    await self.on_phone_message(cam, m)
                elif msg.type == WSMsgType.BINARY and cam is not None:
                    await self.on_phone_binary(cam, msg.data)
        finally:
            if pinger:
                pinger.cancel()
            if cam is not None:
                self.snapshot_cam(cam)
                self.cams.pop(cam.id, None)
                await self.check_take_complete()
                await self.toast(f"Caméra déconnectée : {cam.name}", "warn")
                self.changed()
        return ws

    async def on_phone_message(self, cam, m):
        t = m.get("t")
        if t == "pong":
            self.on_pong(cam, m)
            return
        if t == "status":
            cam.status = m
            self.changed()
        elif t == "recording":
            cam.take.update(state="recording", called_at_pc=m["called_at"] - cam.offset,
                            started_at_pc=m["started_at"] - cam.offset)
            self.snapshot_cam(cam)
            await self.maybe_chirp()
            self.changed()
        elif t == "recorded":
            cam.take.update(state="uploading", duration=m.get("duration"), size=m.get("size"))
            self.changed()
        elif t == "upload":
            cam.take.update(state="uploading", sent=m.get("sent"), total=m.get("total"))
            self.changed()
        elif t == "uploaded":
            cam.take["state"] = "done"
            self.snapshot_cam(cam)
            await self.check_take_complete()
            self.changed()
        elif t in ("upload_error", "error"):
            if t == "upload_error":
                cam.take["state"] = "error"
            await self.toast(f"{cam.name} : {m.get('message')}", "error")
            self.changed()
        elif t == "locked":
            cam.locked = bool(m.get("ok"))
            if m.get("message"):
                await self.toast(f"{cam.name} : {m['message']}", "warn")
            self.changed()

    async def on_phone_binary(self, cam, data):
        if not data:
            return
        kind, jpg = data[0], bytes(data[1:])
        if kind == 1:
            cam.preview, cam.preview_ts = jpg, now_ms()
            await self.broadcast(dict(type="preview", cam=cam.id, ts=cam.preview_ts))
        elif kind == 2:
            loop = asyncio.get_running_loop()
            res, dbg = await loop.run_in_executor(None, self.check_markers, cam.name, jpg)
            cam.check, cam.check_jpg = res, dbg
            self.changed()

    # ------------------------------------------------------------------ boules au sol
    def check_markers(self, name, jpg):
        import cv2
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return dict(ok=False, message="image illisible"), b""
        d = os.path.join(self.captures_dir, "_controle")
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, f"{name}.jpg"), img)
        try:
            cfg = pipeline.load_studio(self.studio_dir)
            mlist = [dict(m, position=m["position_3d"]) for m in cfg["markers"]]
            vol = mk.volume_box(mlist, cfg["volume"]["margin"], cfg["volume"]["height"])
            pose = mk.solve_camera_pose(img, mlist)
            dbg = mk.draw_debug(img, mlist, pose["K"], pose["dist"], pose, vol)
            res = dict(ok=True, rms=round(pose["rms"], 2), seen=len(pose["used"]), total=len(mlist),
                       missing=pose["missing"], focal=round(float(pose["K"][0, 0]), 0),
                       center=[round(v, 2) for v in pose["center"]], size=[img.shape[1], img.shape[0]])
        except Exception as e:  # noqa: BLE001
            dbg = img.copy()
            cv2.putText(dbg, "Boules non trouvees", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3)
            res = dict(ok=False, message=str(e), size=[img.shape[1], img.shape[0]])
        ok, enc = cv2.imencode(".jpg", dbg, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return res, enc.tobytes() if ok else b""

    # ------------------------------------------------------------------ enregistrement
    async def start_take(self, name):
        if self.take and self.take["state"] in ("starting", "recording", "stopping"):
            await self.toast("Un enregistrement est déjà en cours", "warn")
            return
        cams = list(self.cams.values())
        if not cams:
            await self.toast("Aucune caméra connectée", "error")
            return
        name = safe_name(name, self.next_take_name())
        d = os.path.join(self.captures_dir, name)
        if os.path.exists(d) and any(f.lower().endswith((".mp4", ".mov")) for f in os.listdir(d)):
            await self.toast(f"La prise {name} existe déjà : choisissez un autre nom", "error")
            return
        os.makedirs(d, exist_ok=True)
        self.take = dict(name=name, state="starting", dir=d, cams=[c.id for c in cams],
                         created=time.strftime("%Y-%m-%d %H:%M:%S"), chirp_at_pc=None)
        for c in cams:
            c.take = dict(name=name, state="armed")
            await self.send(c, dict(t="arm", take=name))
        await asyncio.sleep(0.6)
        at = now_ms() + 1500
        self.take["at_pc"] = at
        for c in cams:
            await self.send(c, dict(t="start", take=name, at=at + c.offset))
        self.changed()
        # le signal sonore part quand tout le monde enregistre (au plus tard 4 s après)
        asyncio.get_running_loop().call_later(5.5, lambda: asyncio.ensure_future(self.maybe_chirp(force=True)))

    async def maybe_chirp(self, force=False):
        tk = self.take
        if not tk or tk.get("chirp_at_pc") or tk["state"] not in ("starting", "recording"):
            return
        cams = [self.cams[i] for i in tk["cams"] if i in self.cams]
        if not force and not all(c.take.get("state") == "recording" for c in cams):
            return
        tk["state"] = "recording"
        # certains téléphones (Android) commencent réellement à filmer ~0,5 s après avoir
        # annoncé le départ : on laisse de la marge avant le bip
        await asyncio.sleep(1.2)
        await self.chirp()
        self.changed()

    async def chirp(self):
        tk = self.take
        t = now_ms()
        tk.setdefault("chirps_at_pc", []).append(t)
        if not tk.get("chirp_at_pc"):
            tk["chirp_at_pc"] = t
        if not self.play_chirp_local():
            await self.broadcast(dict(type="play_chirp"))   # joué par l'interface

    def play_chirp_local(self):
        """Joue le signal par les haut-parleurs (Windows). Sinon l'interface s'en charge."""
        try:
            import winsound
        except ImportError:
            return False
        if self.chirp_wav is None:
            sr = 48000
            s = sync.chirp_signal(sr)
            pcm = (np.concatenate([np.zeros(int(0.05 * sr)), s, np.zeros(int(0.05 * sr))]) * 0.9 * 32767).astype(np.int16)
            path = os.path.join(tempfile.gettempdir(), "mocapcam_chirp.wav")
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(pcm.tobytes())
            self.chirp_wav = path
        winsound.PlaySound(self.chirp_wav, winsound.SND_FILENAME | winsound.SND_ASYNC)
        return True

    async def stop_take(self):
        tk = self.take
        if not tk or tk["state"] not in ("starting", "recording"):
            return
        tk["state"] = "stopping"
        self.changed()
        # second bip avant l'arrêt : synchro vérifiée deux fois + mesure de dérive
        if tk.get("chirp_at_pc"):
            await self.chirp()
            await asyncio.sleep(0.9)
        tk["stopped_at_pc"] = now_ms()
        for cid in tk["cams"]:
            if cid in self.cams:
                await self.send(self.cams[cid], dict(t="stop"))
        self.changed()

    async def check_take_complete(self):
        tk = self.take
        if not tk or tk["state"] != "stopping":
            return
        cams = [self.cams[i] for i in tk["cams"] if i in self.cams]
        if all(c.take.get("state") in ("done", "error") for c in cams):
            self.write_take_json()
            n = len([f for f in os.listdir(tk["dir"]) if f.lower().endswith((".mp4", ".mov"))])
            tk["state"] = "ready"
            await self.toast(f"Prise {tk['name']} rapatriée ({n} vidéo(s))")
            self.changed()

    def snapshot_cam(self, c):
        """Mémorise l'état de la caméra pour la prise en cours (survit à une déconnexion)."""
        tk = self.take
        if not tk or c.take.get("name") != tk["name"]:
            return
        f = c.take.get("file")
        tk.setdefault("meta", {})[c.name] = dict(
            file=os.path.basename(f) if f and os.path.exists(f) else None,
            started_at_pc=c.take.get("started_at_pc"), called_at_pc=c.take.get("called_at_pc"),
            clock_offset_ms=c.offset, sync_rtt_ms=c.rtt, platform=c.platform, model=c.model,
            fps=c.status.get("fps"), stabilization=c.status.get("stabilization"),
            resolution=c.status.get("resolution"), exposure=c.status.get("exposure"), locked=c.locked)

    def write_take_json(self):
        tk = self.take
        meta = dict(take=tk["name"], created=tk["created"], chirp_at_pc=tk.get("chirp_at_pc"),
                    chirps_at_pc=tk.get("chirps_at_pc", []),
                    chirp=sync.CHIRP, at_pc=tk.get("at_pc"),
                    duration_s=round((tk.get("stopped_at_pc", now_ms()) - tk.get("at_pc", now_ms())) / 1000, 1),
                    cameras={})
        for cid in tk["cams"]:
            if cid in self.cams:
                self.snapshot_cam(self.cams[cid])
        meta["cameras"] = dict(tk.get("meta", {}))
        # vidéos présentes dans le dossier mais inconnues (caméra déconnectée trop tôt…)
        for f in sorted(os.listdir(tk["dir"])):
            n, ext = os.path.splitext(f)
            if ext.lower() in (".mp4", ".mov") and not f.endswith(".part"):
                meta["cameras"].setdefault(n, {})["file"] = f
        with open(os.path.join(tk["dir"], "take.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)

    async def handle_upload(self, request):
        q = request.rel_url.query
        if q.get("token") != self.token:
            return web.Response(status=403, text="token")
        cam_name = safe_name(q.get("cam"), "camera").lower()
        take = safe_name(q.get("take"), "prise")
        ext = "mov" if q.get("ext") == "mov" else "mp4"
        d = os.path.join(self.captures_dir, take)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{cam_name}.{ext}")
        tmp = path + ".part"
        size = 0
        with open(tmp, "wb") as f:
            async for chunk in request.content.iter_chunked(1 << 20):
                f.write(chunk)
                size += len(chunk)
        os.replace(tmp, path)
        meta_path = os.path.join(d, "take.json")
        if os.path.exists(meta_path) and not (self.take and self.take["name"] == take
                                              and self.take["state"] != "ready"):
            try:
                with open(meta_path, encoding="utf-8") as fh:
                    meta = json.load(fh)
                meta.setdefault("cameras", {}).setdefault(cam_name, {})["file"] = os.path.basename(path)
                with open(meta_path, "w", encoding="utf-8") as fh:
                    json.dump(meta, fh, indent=2, ensure_ascii=False)
                await self.toast(f"Vidéo de {cam_name} ajoutée à {take}")
            except (OSError, ValueError):
                pass
        for c in self.cams.values():
            if c.name == cam_name and c.take.get("name") == take:
                c.take["file"] = path
                c.take["bytes"] = size
        self.changed()
        return web.Response(text="ok")

    # ------------------------------------------------------------------ traitement
    async def process(self, take):
        if self.proc and self.proc["running"]:
            await self.toast("Un traitement est déjà en cours", "warn")
            return
        d = os.path.join(self.captures_dir, safe_name(take))
        if not os.path.isdir(d):
            return
        cmd = [sys.executable, "-u", "-m", "mocapcam", "traiter", self.studio_dir, d]
        env = dict(os.environ, PYTHONPATH=ROOT, PYTHONIOENCODING="utf-8")
        self.proc = dict(take=take, running=True, ok=None)
        await self.broadcast(dict(type="log", line=f"=== Traitement de {take} ===", reset=True))
        self.changed()
        p = await asyncio.create_subprocess_exec(*cmd, cwd=ROOT, env=env, stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.STDOUT)
        self.proc["pid"] = p.pid
        async for line in p.stdout:
            await self.broadcast(dict(type="log", line=line.decode("utf-8", "replace").rstrip()))
        rc = await p.wait()
        self.proc.update(running=False, ok=(rc == 0))
        await self.toast(f"Traitement de {take} terminé" if rc == 0 else f"Échec du traitement de {take}",
                         "info" if rc == 0 else "error")
        self.changed()

    def open_folder(self, take=None):
        d = os.path.join(self.captures_dir, safe_name(take)) if take else self.captures_dir
        if hasattr(os, "startfile") and os.path.isdir(d):
            os.startfile(d)  # noqa: S606 (Windows)

    # ------------------------------------------------------------------ interface
    async def handle_ui(self, request):
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self.uis.add(ws)
        await ws.send_str(json.dumps(self.state()))
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    m = json.loads(msg.data)
                except ValueError:
                    continue
                await self.on_ui(m)
        finally:
            self.uis.discard(ws)
        return ws

    async def on_ui(self, m):
        a = m.get("a")
        cams = list(self.cams.values())
        if a == "record":
            await self.start_take(m.get("take"))
        elif a == "stop":
            await self.stop_take()
        elif a == "lock":
            for c in cams:
                await self.send(c, dict(t="lock", fast_shutter=bool(m.get("fast"))))
        elif a == "unlock":
            for c in cams:
                await self.send(c, dict(t="unlock"))
        elif a == "check":
            for c in cams:
                c.check = dict(pending=True)
                await self.send(c, dict(t="snapshot"))
            self.changed()
        elif a == "identify" and m.get("cam") in self.cams:
            await self.send(self.cams[m["cam"]], dict(t="identify"))
        elif a == "rename" and m.get("cam") in self.cams:
            c = self.cams[m["cam"]]
            c.name = self.unique_name(m.get("name", c.name), c.id)
            await self.send(c, dict(t="rename", name=c.name))
            self.changed()
        elif a == "kick" and m.get("cam") in self.cams:
            await self.cams[m["cam"]].ws.close()
        elif a == "preview":
            self.preview_fps = float(m.get("fps", 3))
            for c in cams:
                await self.send(c, dict(t="preview", fps=self.preview_fps))
            self.changed()
        elif a == "prefer60":
            self.prefer60 = bool(m.get("value"))
            for c in cams:
                await self.send(c, dict(t="config", prefer60=self.prefer60))
            self.changed()
        elif a == "process":
            asyncio.ensure_future(self.process(m.get("take")))
        elif a == "open":
            self.open_folder(m.get("take"))
        elif a == "new_qr":
            self.token = secrets.token_hex(4)
            self.changed()

    async def handle_preview(self, request):
        c = self.cams.get(request.match_info["cam"])
        if not c or not c.preview:
            raise web.HTTPNotFound()
        return web.Response(body=c.preview, content_type="image/jpeg", headers={"Cache-Control": "no-store"})

    async def handle_check(self, request):
        c = self.cams.get(request.match_info["cam"])
        if not c or not c.check_jpg:
            raise web.HTTPNotFound()
        return web.Response(body=c.check_jpg, content_type="image/jpeg", headers={"Cache-Control": "no-store"})

    async def handle_qr(self, request):
        import io
        import segno
        buf = io.BytesIO()
        segno.make(json.dumps(self.qr_payload(), separators=(",", ":")), error="m").save(
            buf, kind="svg", scale=6, dark="#000", light="#fff", border=2, xmldecl=True)
        return web.Response(body=buf.getvalue(), content_type="image/svg+xml", headers={"Cache-Control": "no-store"})

    async def handle_index(self, request):
        return web.FileResponse(os.path.join(UI_DIR, "index.html"), headers={"Cache-Control": "no-store"})

    def app(self):
        app = web.Application(client_max_size=8 * 1024 ** 3)
        app.router.add_get("/", self.handle_index)
        app.router.add_get("/ws/ui", self.handle_ui)
        app.router.add_get("/ws/phone", self.handle_phone)
        app.router.add_post("/upload", self.handle_upload)
        app.router.add_get("/preview/{cam}", self.handle_preview)
        app.router.add_get("/check/{cam}", self.handle_check)
        app.router.add_get("/qr.svg", self.handle_qr)

        async def on_startup(app):
            app["pusher"] = asyncio.create_task(self.state_pusher())

        app.on_startup.append(on_startup)
        return app


def open_window(url):
    """Ouvre l'interface dans une fenêtre d'application Edge (Windows), sinon le navigateur."""
    if sys.platform.startswith("win"):
        try:
            subprocess.Popen(["cmd", "/c", "start", "", "msedge", f"--app={url}", "--window-size=1500,950"],
                             creationflags=0x08000000)
            return
        except OSError:
            pass
    webbrowser.open(url)


def run(studio_dir, captures_dir, port=8765, open_ui=True):
    st = Studio(studio_dir, captures_dir, port)
    url = f"http://localhost:{port}"
    print(f"MocapCam Studio : {url}")
    print(f"  adresses pour les téléphones : {', '.join(lan_addresses())} (port {port})")
    print(f"  prises enregistrées dans : {st.captures_dir}")
    print("  Laissez cette fenêtre ouverte. Ctrl+C pour quitter.")
    if open_ui:
        asyncio.get_event_loop_policy()
        threading_open = __import__("threading").Timer(1.0, open_window, args=(url,))
        threading_open.start()
    web.run_app(st.app(), host="0.0.0.0", port=port, print=None, access_log=None)
