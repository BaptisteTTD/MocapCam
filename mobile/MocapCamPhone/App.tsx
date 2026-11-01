import { StatusBar } from 'expo-status-bar'
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ActivityIndicator,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
  type LayoutChangeEvent,
} from 'react-native'
import { useKeepAwake } from 'expo-keep-awake'
import * as Battery from 'expo-battery'
import { File, Paths, UploadType } from 'expo-file-system'
import {
  Camera,
  CommonDynamicRanges,
  CommonResolutions,
  isScannedCode,
  useCameraDevice,
  useCameraPermission,
  useMicrophonePermission,
  useObjectOutput,
  useVideoOutput,
  type CameraRef,
  type CameraSessionConfig,
  type Constraint,
  type Recorder,
  type ScannedObject,
  type ScannedObjectType,
} from 'react-native-vision-camera'
import { parseQr, StudioClient, type ConnectInfo, type ConnState, type StudioCommand } from './src/studio'
import { decodeQr } from './src/qr'
import { cleanName, loadSettings, saveSettings, type Settings } from './src/settings'

const APP_VERSION = '1.0.1'

// Erreurs hors affichage (minuteries, callbacks natifs) : on les montre à l'écran
// au lieu de laisser l'app se fermer.
const errorListeners = new Set<(e: Error) => void>()
const g = globalThis as any
if (g.ErrorUtils?.setGlobalHandler) {
  g.ErrorUtils.setGlobalHandler((e: Error) => {
    if (errorListeners.size === 0) console.error(e)
    errorListeners.forEach((l) => l(e))
  })
}
const QR_TYPES: ScannedObjectType[] = ['qr']

// ============================================================================ App

export default function App() {
  return (
    <ErrorBoundary>
      <AppInner />
    </ErrorBoundary>
  )
}

/** Affiche l'erreur à l'écran au lieu de fermer l'app (plus simple à diagnostiquer). */
class ErrorBoundary extends React.Component<{ children: React.ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null }
  static getDerivedStateFromError(error: Error) {
    return { error }
  }
  private onGlobal = (error: Error) => this.setState({ error })
  componentDidMount() {
    errorListeners.add(this.onGlobal)
  }
  componentWillUnmount() {
    errorListeners.delete(this.onGlobal)
  }
  render() {
    if (this.state.error) {
      return (
        <View style={styles.center}>
          <StatusBar hidden />
          <Text style={styles.title}>MocapCam a rencontré une erreur</Text>
          <Text style={[styles.text, { fontSize: 13 }]} selectable>
            {String(this.state.error?.message ?? this.state.error)}
          </Text>
          <Text style={styles.small}>Faites une capture d'écran de ce message pour le diagnostic.</Text>
          <Btn label="Réessayer" onPress={() => this.setState({ error: null })} />
        </View>
      )
    }
    return this.props.children
  }
}

function AppInner() {
  const [settings, setSettings] = useState<Settings>(() => loadSettings())
  const [info, setInfo] = useState<ConnectInfo | null>(null)
  const cam = useCameraPermission()
  const mic = useMicrophonePermission()

  const updateSettings = useCallback((s: Settings) => {
    setSettings(s)
    saveSettings(s)
  }, [])

  if (!cam.hasPermission || !mic.hasPermission) {
    return (
      <View style={styles.center}>
        <StatusBar hidden />
        <Text style={styles.title}>MocapCam</Text>
        <Text style={styles.text}>L'app a besoin de la caméra et du micro (le son sert à synchroniser les téléphones).</Text>
        <Btn
          label="Autoriser"
          onPress={async () => {
            if (!cam.hasPermission) await cam.requestPermission()
            if (!mic.hasPermission) await mic.requestPermission()
          }}
        />
      </View>
    )
  }

  if (info == null) {
    return (
      <ConnectScreen
        settings={settings}
        onSettings={updateSettings}
        onConnect={(i, raw) => {
          updateSettings({ ...settings, lastQr: raw })
          setInfo(i)
        }}
      />
    )
  }
  return (
    <CaptureScreen
      info={info}
      settings={settings}
      onSettings={updateSettings}
      onDisconnect={() => setInfo(null)}
    />
  )
}

// ============================================================================ Connexion

function ConnectScreen({
  settings,
  onSettings,
  onConnect,
}: {
  settings: Settings
  onSettings: (s: Settings) => void
  onConnect: (i: ConnectInfo, raw: string) => void
}) {
  const [manual, setManual] = useState('')
  const [scanning, setScanning] = useState(true)
  const [manualError, setManualError] = useState<string | null>(null)
  const done = useRef(false)

  const onCode = useCallback(
    (value: string) => {
      if (done.current) return
      const i = parseQr(value)
      if (i && i.token) {
        done.current = true
        setScanning(false)
        onConnect(i, value)
      }
    },
    [onConnect],
  )
  const lastInfo = settings.lastQr ? parseQr(settings.lastQr) : null

  return (
    <View style={styles.row}>
      <StatusBar hidden />
      <View style={styles.scanBox}>
        {scanning ? (
          Platform.OS === 'ios' ? <IosQrScanner onCode={onCode} /> : <SnapshotQrScanner onCode={onCode} />
        ) : null}
        <View style={styles.scanFrame} pointerEvents="none" />
        <Text style={styles.scanHint}>Visez le QR code affiché par MocapCam Studio sur le PC</Text>
      </View>
      <View style={styles.side}>
        <Text style={styles.title}>MocapCam</Text>
        <Text style={styles.label}>Nom de cette caméra</Text>
        <TextInput
          style={styles.input}
          value={settings.name}
          autoCapitalize="none"
          autoCorrect={false}
          onChangeText={(t) => onSettings({ ...settings, name: t })}
          onEndEditing={() => onSettings({ ...settings, name: cleanName(settings.name) })}
        />
        <Text style={styles.small}>Il donne son nom à la vidéo : iphone, poco…</Text>
        {lastInfo ? (
          <Btn
            label="Reconnecter au dernier PC"
            onPress={() => {
              done.current = true
              onConnect(lastInfo, settings.lastQr!)
            }}
          />
        ) : null}
        <Text style={[styles.label, { marginTop: 16 }]}>Ou adresse et code affichés dans Studio</Text>
        <TextInput
          style={styles.input}
          placeholder="192.168.1.20:8765 ab12cd34"
          placeholderTextColor="#777"
          value={manual}
          autoCapitalize="none"
          keyboardType="numbers-and-punctuation"
          onChangeText={setManual}
        />
        <Btn
          label="Se connecter"
          onPress={() => {
            const i = parseQr(manual)
            if (!i || !i.token) {
              setManualError('Format attendu : 192.168.1.20:8765 ab12cd34')
              return
            }
            done.current = true
            setScanning(false)
            onConnect(i, manual)
          }}
        />
        {manualError ? <Text style={[styles.small, { color: '#ff8a80' }]}>{manualError}</Text> : null}
      </View>
    </View>
  )
}

/** iPhone : lecteur de codes natif de VisionCamera. */
function IosQrScanner({ onCode }: { onCode: (v: string) => void }) {
  const onScanned = useCallback(
    (objects: ScannedObject[]) => {
      for (const o of objects) {
        if (isScannedCode(o) && o.value) onCode(o.value)
      }
    },
    [onCode],
  )
  const objectOutput = useObjectOutput({ types: QR_TYPES, onObjectsScanned: onScanned })
  return <Camera style={StyleSheet.absoluteFill} device="back" isActive outputs={[objectOutput]} />
}

/** Android : VisionCamera n'y a pas de lecteur de codes, on décode les images de l'aperçu. */
function SnapshotQrScanner({ onCode }: { onCode: (v: string) => void }) {
  const ref = useRef<CameraRef>(null)
  const [ready, setReady] = useState(false)
  const onCodeRef = useRef(onCode)
  onCodeRef.current = onCode
  useEffect(() => {
    if (!ready) return
    let alive = true
    const loop = async () => {
      while (alive) {
        try {
          const cam = ref.current
          if (cam) {
            let img = await cam.takeSnapshot()
            const scale = 800 / Math.max(img.width, img.height)
            if (scale < 1) img = await img.resizeAsync(Math.round(img.width * scale), Math.round(img.height * scale))
            const raw = await img.toRawPixelDataAsync()
            const value = decodeQr(raw)
            if (value && alive) onCodeRef.current(value)
          }
        } catch {
          // aperçu pas encore prêt
        }
        await new Promise((r) => setTimeout(r, 200))
      }
    }
    void loop()
    return () => {
      alive = false
    }
  }, [ready])
  return (
    <Camera
      ref={ref}
      style={StyleSheet.absoluteFill}
      device="back"
      isActive
      onPreviewStarted={() => setReady(true)}
    />
  )
}

// ============================================================================ Capture

interface CamStatus {
  fps?: number
  stabilization?: string
  binned?: boolean
}

function CaptureScreen({
  info,
  settings,
  onSettings,
  onDisconnect,
}: {
  info: ConnectInfo
  settings: Settings
  onSettings: (s: Settings) => void
  onDisconnect: () => void
}) {
  useKeepAwake()
  const device = useCameraDevice('back', { physicalDevices: ['wide-angle'] })
  const videoOutput = useVideoOutput({
    targetResolution: CommonResolutions.FHD_16_9,
    enableAudio: true,
    fileType: 'mp4',
  })
  const constraints = useMemo<Constraint[]>(
    () => [
      { fps: 60 },
      { videoStabilizationMode: 'off' },
      { videoDynamicRange: CommonDynamicRanges.ANY_SDR },
      { resolutionBias: videoOutput },
    ],
    [videoOutput],
  )
  const cameraRef = useRef<CameraRef>(null)
  const viewSize = useRef({ width: 1, height: 1 })
  const [conn, setConn] = useState<{ state: ConnState; detail?: string }>({ state: 'connecting' })
  const [camStatus, setCamStatus] = useState<CamStatus>({})
  const [recording, setRecording] = useState<{ take: string; since: number } | null>(null)
  const [upload, setUpload] = useState<{ take: string; pct: number } | null>(null)
  const [locked, setLocked] = useState(false)
  const [banner, setBanner] = useState<string | null>(null)
  const [identify, setIdentify] = useState(false)
  const [now, setNow] = useState(Date.now())

  const nameRef = useRef(settings.name)
  nameRef.current = settings.name
  const recorderRef = useRef<Recorder | null>(null)
  const takeRef = useRef<string>('')
  const previewFps = useRef(3)
  const busySnap = useRef(false)
  const lastSnapSize = useRef<{ w: number; h: number } | null>(null)
  const clientRef = useRef<StudioClient | null>(null)
  const camStatusRef = useRef<CamStatus>({})
  camStatusRef.current = camStatus
  const lockedRef = useRef(false)
  lockedRef.current = locked

  const flash = (msg: string) => {
    setBanner(msg)
    setTimeout(() => setBanner((b) => (b === msg ? null : b)), 4000)
  }

  // ---------------------------------------------------------------- envoi de la vidéo
  const uploadFile = useCallback(async (filePath: string, take: string) => {
    const client = clientRef.current
    if (!client?.baseUrl) {
      flash("Pas de connexion : la vidéo reste sur le téléphone")
      return
    }
    const uri = filePath.startsWith('file://') ? filePath : `file://${filePath}`
    const file = new File(uri)
    const url =
      `${client.baseUrl}/upload?token=${encodeURIComponent(client.token)}` +
      `&cam=${encodeURIComponent(nameRef.current)}&take=${encodeURIComponent(take)}&ext=mp4`
    let lastSent = 0
    const task = file.createUploadTask(url, {
      httpMethod: 'POST',
      uploadType: UploadType.BINARY_CONTENT,
      headers: { 'Content-Type': 'video/mp4' },
    })
    const sub = task.addListener('progress', ({ bytesSent, totalBytes }) => {
      const t = Date.now()
      if (t - lastSent > 300 || bytesSent === totalBytes) {
        lastSent = t
        setUpload({ take, pct: totalBytes ? bytesSent / totalBytes : 0 })
        client.send({ t: 'upload', take, sent: bytesSent, total: totalBytes })
      }
    })
    try {
      const res = await task.uploadAsync()
      if (res.status === 200) {
        client.send({ t: 'uploaded', take })
        try {
          file.delete()
        } catch {
          // pas grave
        }
        flash('Vidéo envoyée au PC')
      } else {
        client.send({ t: 'upload_error', take, message: `HTTP ${res.status}` })
        flash(`Erreur d'envoi (HTTP ${res.status})`)
      }
    } catch (e) {
      client.send({ t: 'upload_error', take, message: String(e) })
      flash("Erreur d'envoi : la vidéo reste sur le téléphone")
    } finally {
      sub.remove()
      setUpload(null)
    }
  }, [])

  // ---------------------------------------------------------------- enregistrement
  const arm = useCallback(async () => {
    if (recorderRef.current && !recorderRef.current.isRecording) return recorderRef.current
    recorderRef.current = await videoOutput.createRecorder({})
    return recorderRef.current
  }, [videoOutput])

  const startAt = useCallback(
    async (take: string, at: number) => {
      const client = clientRef.current
      try {
        if (recorderRef.current?.isRecording) {
          client?.send({ t: 'error', message: 'déjà en enregistrement' })
          return
        }
        const rec = await arm()
        takeRef.current = take
        const wait = at - Date.now()
        if (wait > 0) await new Promise((r) => setTimeout(r, wait))
        const called = Date.now()
        await rec.startRecording(
          (filePath) => {
            setRecording(null)
            recorderRef.current = null
            client?.send({
              t: 'recorded',
              take,
              duration: rec.recordedDuration,
              size: rec.recordedFileSize,
            })
            void uploadFile(filePath, take)
          },
          (error) => {
            setRecording(null)
            recorderRef.current = null
            client?.send({ t: 'error', message: `enregistrement : ${error.message}` })
            flash(`Erreur d'enregistrement : ${error.message}`)
          },
        )
        const started = Date.now()
        setRecording({ take, since: started })
        client?.send({ t: 'recording', take, called_at: called, started_at: started })
      } catch (e) {
        client?.send({ t: 'error', message: `démarrage : ${String(e)}` })
        flash(`Impossible de démarrer : ${String(e)}`)
      }
    },
    [arm, uploadFile],
  )

  const stop = useCallback(async () => {
    const rec = recorderRef.current
    if (rec?.isRecording) {
      try {
        await rec.stopRecording()
      } catch (e) {
        clientRef.current?.send({ t: 'error', message: `arrêt : ${String(e)}` })
      }
    }
  }, [])

  // ---------------------------------------------------------------- réglages d'image
  const lock = useCallback(async (fastShutter: boolean) => {
    const ref = cameraRef.current
    const client = clientRef.current
    if (!ref) return
    try {
      const { width, height } = viewSize.current
      await ref.focusTo(
        { x: width / 2, y: height / 2 },
        { adaptiveness: 'locked', autoResetAfter: null, responsiveness: 'snappy' },
      )
      const c = ref.controller
      if (fastShutter && Platform.OS === 'ios' && c && c.device.supportsExposureLocking) {
        // temps de pose court (1/500 s) pour éviter le flou de bougé, ISO compensé
        await new Promise((r) => setTimeout(r, 600))
        const d = c.exposureDuration
        const iso = c.iso
        const target = Math.max(c.minExposureDuration, Math.min(1 / 500, d))
        const newIso = Math.min(c.maxISO, Math.max(c.minISO, (iso * d) / target))
        await c.setExposureLocked(target, newIso)
      }
      setLocked(true)
      client?.send({ t: 'locked', ok: true })
    } catch (e) {
      client?.send({ t: 'locked', ok: false, message: String(e) })
      flash(`Blocage impossible : ${String(e)}`)
    }
  }, [])

  const unlock = useCallback(async () => {
    try {
      await cameraRef.current?.resetFocus()
    } catch {
      // ignore
    }
    setLocked(false)
    clientRef.current?.send({ t: 'locked', ok: false })
  }, [])

  // ---------------------------------------------------------------- images vers le PC
  const sendSnapshot = useCallback(async (kind: 1 | 2) => {
    const ref = cameraRef.current
    const client = clientRef.current
    if (!ref || !client || busySnap.current) return
    busySnap.current = true
    try {
      let img = await ref.takeSnapshot()
      lastSnapSize.current = { w: img.width, h: img.height }
      if (kind === 1) {
        const scale = 640 / Math.max(img.width, img.height)
        img = await img.resizeAsync(Math.round(img.width * scale), Math.round(img.height * scale))
      }
      const enc = await img.toEncodedImageDataAsync('jpg', kind === 1 ? 55 : 92)
      client.sendBinary(kind, enc.buffer)
    } catch {
      // caméra pas prête
    } finally {
      busySnap.current = false
    }
  }, [])

  // ---------------------------------------------------------------- commandes du PC
  const onCommand = useCallback(
    (cmd: StudioCommand) => {
      switch (cmd.t) {
        case 'welcome':
          if (cmd.name && cmd.name !== nameRef.current) onSettings({ ...settings, name: cmd.name })
          break
        case 'arm':
          void arm()
          break
        case 'start':
          void startAt(cmd.take, cmd.at)
          break
        case 'stop':
          void stop()
          break
        case 'lock':
          void lock(!!cmd.fast_shutter)
          break
        case 'unlock':
          void unlock()
          break
        case 'snapshot':
          void sendSnapshot(2)
          break
        case 'preview':
          previewFps.current = cmd.fps
          break
        case 'identify':
          setIdentify(true)
          setTimeout(() => setIdentify(false), 3000)
          break
        case 'rename':
          onSettings({ ...settings, name: cleanName(cmd.name) })
          break
        case 'error':
          flash(cmd.message)
          break
      }
    },
    [arm, startAt, stop, lock, unlock, sendSnapshot, onSettings, settings],
  )
  const onCommandRef = useRef(onCommand)
  onCommandRef.current = onCommand

  // ---------------------------------------------------------------- connexion
  useEffect(() => {
    const client = new StudioClient(
      info,
      () => ({
        name: nameRef.current,
        platform: Platform.OS,
        model: (Platform.constants as any)?.Model ?? (Platform.constants as any)?.model ?? '',
        os: String(Platform.Version),
        app: APP_VERSION,
      }),
      (cmd) => onCommandRef.current(cmd),
      (state, detail) => setConn({ state, detail }),
    )
    clientRef.current = client
    client.start()
    return () => {
      client.stop()
      clientRef.current = null
    }
  }, [info])

  // état envoyé chaque seconde + horloge de l'écran
  useEffect(() => {
    const id = setInterval(async () => {
      setNow(Date.now())
      const client = clientRef.current
      if (!client) return
      try {
      let battery = -1
      try {
        battery = await Battery.getBatteryLevelAsync()
      } catch {
        // indisponible
      }
      const c = cameraRef.current?.controller
      const rec = recorderRef.current
      client.send({
        t: 'status',
        name: nameRef.current,
        battery,
        storage_free: (() => {
          try {
            return Paths.availableDiskSpace
          } catch {
            return null
          }
        })(),
        fps: camStatusRef.current.fps ?? null,
        stabilization: camStatusRef.current.stabilization ?? null,
        binned: camStatusRef.current.binned ?? null,
        snapshot_size: lastSnapSize.current,
        exposure: c
          ? { duration: c.exposureDuration, iso: c.iso, mode: c.exposureMode, focus: c.focusMode }
          : null,
        locked: lockedRef.current,
        recording: !!rec?.isRecording,
        recorded_duration: rec?.isRecording ? rec.recordedDuration : 0,
        take: rec?.isRecording ? takeRef.current : null,
        device: c?.device.localizedName ?? null,
      })
      } catch (e) {
        client.send({ t: 'error', message: `état : ${String(e)}` })
      }
    }, 1000)
    return () => clearInterval(id)
  }, [])

  // aperçu envoyé au PC
  useEffect(() => {
    let alive = true
    const loop = async () => {
      while (alive) {
        const fps = recorderRef.current?.isRecording ? Math.min(previewFps.current, 1) : previewFps.current
        if (fps > 0 && clientRef.current?.baseUrl) await sendSnapshot(1)
        await new Promise((r) => setTimeout(r, fps > 0 ? 1000 / fps : 500))
      }
    }
    void loop()
    return () => {
      alive = false
    }
  }, [sendSnapshot])

  const onConfig = useCallback((c: CameraSessionConfig) => {
    setCamStatus({
      fps: c.selectedFPS,
      stabilization: c.selectedVideoStabilizationMode,
      binned: c.isBinned,
    })
  }, [])

  const onLayout = (e: LayoutChangeEvent) => {
    viewSize.current = { width: e.nativeEvent.layout.width, height: e.nativeEvent.layout.height }
  }

  const recSecs = recording ? Math.max(0, (now - recording.since) / 1000) : 0
  const connColor = conn.state === 'connected' ? '#3ddc84' : conn.state === 'error' ? '#ff5252' : '#ffb300'

  return (
    <View style={styles.full} onLayout={onLayout}>
      <StatusBar hidden />
      {device ? (
        <Camera
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          device={device}
          isActive
          outputs={[videoOutput]}
          constraints={constraints}
          orientationSource="interface"
          onSessionConfigSelected={onConfig}
          onError={(e) => flash(`Caméra : ${e.message}`)}
        />
      ) : (
        <View style={styles.center}>
          <ActivityIndicator color="#fff" />
        </View>
      )}

      <View style={styles.topBar}>
        <View style={[styles.dot, { backgroundColor: connColor }]} />
        <Text style={styles.camName}>{settings.name}</Text>
        <Text style={styles.small}>
          {conn.state === 'connected' ? `connecté à ${conn.detail}` : conn.detail ?? 'connexion…'}
        </Text>
        <View style={{ flex: 1 }} />
        {recording ? (
          <View style={styles.rec}>
            <View style={styles.recDot} />
            <Text style={styles.recText}>REC {recSecs.toFixed(0)} s</Text>
          </View>
        ) : null}
      </View>

      <View style={styles.bottomBar}>
        <Text style={styles.small}>
          {camStatus.fps ? `${camStatus.fps} i/s` : '… i/s'} · stabilisation {camStatus.stabilization ?? '?'} ·{' '}
          {locked ? 'expo/mise au point bloquées' : 'auto'}
        </Text>
        <View style={{ flex: 1 }} />
        {upload ? <Text style={styles.small}>Envoi {Math.round(upload.pct * 100)} %</Text> : null}
        <Pressable
          style={styles.smallBtn}
          onPress={() => {
            void stop()
            onDisconnect()
          }}
        >
          <Text style={styles.smallBtnText}>Déconnecter</Text>
        </Pressable>
      </View>

      {banner ? (
        <View style={styles.banner}>
          <Text style={styles.text}>{banner}</Text>
        </View>
      ) : null}
      {identify ? (
        <View style={styles.identify}>
          <Text style={styles.identifyText}>{settings.name}</Text>
        </View>
      ) : null}
    </View>
  )
}

// ============================================================================ UI

function Btn({ label, onPress }: { label: string; onPress: () => void }) {
  return (
    <Pressable style={({ pressed }) => [styles.btn, pressed && { opacity: 0.7 }]} onPress={onPress}>
      <Text style={styles.btnText}>{label}</Text>
    </Pressable>
  )
}

const styles = StyleSheet.create({
  full: { flex: 1, backgroundColor: '#000' },
  row: { flex: 1, flexDirection: 'row', backgroundColor: '#111' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#111', padding: 24 },
  scanBox: { flex: 3, backgroundColor: '#000', alignItems: 'center', justifyContent: 'center' },
  scanFrame: { width: 220, height: 220, borderWidth: 3, borderColor: '#fff', borderRadius: 16, opacity: 0.8 },
  scanHint: { position: 'absolute', bottom: 16, color: '#fff', fontSize: 15, backgroundColor: '#0008', padding: 6 },
  side: { flex: 2, padding: 20, justifyContent: 'center' },
  title: { color: '#fff', fontSize: 28, fontWeight: '700', marginBottom: 12 },
  label: { color: '#bbb', fontSize: 14, marginBottom: 4 },
  input: {
    color: '#fff',
    fontSize: 18,
    borderWidth: 1,
    borderColor: '#555',
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 8,
    marginBottom: 6,
  },
  text: { color: '#fff', fontSize: 16, textAlign: 'center', marginBottom: 16 },
  small: { color: '#ddd', fontSize: 13 },
  btn: { backgroundColor: '#2f6fed', borderRadius: 10, paddingVertical: 12, paddingHorizontal: 18, marginTop: 10, alignItems: 'center' },
  btnText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  topBar: {
    position: 'absolute', top: 0, left: 0, right: 0, flexDirection: 'row', alignItems: 'center',
    paddingHorizontal: 16, paddingVertical: 8, backgroundColor: '#0009', gap: 10,
  },
  bottomBar: {
    position: 'absolute', bottom: 0, left: 0, right: 0, flexDirection: 'row', alignItems: 'center',
    paddingHorizontal: 16, paddingVertical: 6, backgroundColor: '#0009', gap: 12,
  },
  dot: { width: 12, height: 12, borderRadius: 6 },
  camName: { color: '#fff', fontSize: 20, fontWeight: '700' },
  rec: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  recDot: { width: 14, height: 14, borderRadius: 7, backgroundColor: '#ff3b30' },
  recText: { color: '#ff3b30', fontSize: 18, fontWeight: '700' },
  smallBtn: { borderWidth: 1, borderColor: '#aaa', borderRadius: 6, paddingHorizontal: 10, paddingVertical: 4 },
  smallBtnText: { color: '#fff', fontSize: 13 },
  banner: { position: 'absolute', top: 60, alignSelf: 'center', backgroundColor: '#000c', borderRadius: 8, padding: 10 },
  identify: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: '#2f6fedcc', alignItems: 'center', justifyContent: 'center' },
  identifyText: { color: '#fff', fontSize: 96, fontWeight: '800' },
})
