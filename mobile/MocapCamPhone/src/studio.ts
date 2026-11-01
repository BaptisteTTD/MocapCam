// Connexion au PC (MocapCam Studio) : WebSocket + reconnexion automatique.

export type StudioCommand =
  | { t: 'welcome'; id: string; name: string }
  | { t: 'ping'; s0: number }
  | { t: 'arm'; take: string }
  | { t: 'start'; take: string; at: number }
  | { t: 'stop' }
  | { t: 'lock'; fast_shutter?: boolean }
  | { t: 'unlock' }
  | { t: 'snapshot' }
  | { t: 'identify' }
  | { t: 'rename'; name: string }
  | { t: 'preview'; fps: number }
  | { t: 'error'; message: string }

export interface ConnectInfo {
  hosts: string[]
  port: number
  token: string
}

export type ConnState = 'idle' | 'connecting' | 'connected' | 'error'

export function parseQr(value: string): ConnectInfo | null {
  try {
    const d = JSON.parse(value)
    if (d && d.mocapcam && Array.isArray(d.hosts) && d.port && d.token) {
      return { hosts: d.hosts.map(String), port: Number(d.port), token: String(d.token) }
    }
  } catch {
    // pas du JSON : adresse saisie à la main "192.168.1.20:8765"
  }
  // saisie à la main : "192.168.1.20:8765 ab12cd34" (adresse + code affichés dans Studio)
  const m = /^\s*([\w.\-]+)(?::(\d+))?(?:[\s,;]+([0-9a-fA-F]{4,16}))?\s*$/.exec(value)
  if (m) return { hosts: [m[1]], port: Number(m[2] ?? 8765), token: (m[3] ?? '').toLowerCase() }
  return null
}

export class StudioClient {
  private ws: WebSocket | null = null
  private closedByUser = false
  private retry: ReturnType<typeof setTimeout> | null = null
  host: string | null = null

  constructor(
    private info: ConnectInfo,
    private getHello: () => Record<string, unknown>,
    private onCommand: (cmd: StudioCommand) => void,
    private onState: (s: ConnState, detail?: string) => void,
  ) {}

  get baseUrl(): string | null {
    return this.host ? `http://${this.host}:${this.info.port}` : null
  }

  get token(): string {
    return this.info.token
  }

  start() {
    this.closedByUser = false
    this.tryHosts(0)
  }

  stop() {
    this.closedByUser = true
    if (this.retry) clearTimeout(this.retry)
    this.ws?.close()
    this.ws = null
    this.onState('idle')
  }

  private tryHosts(i: number) {
    if (this.closedByUser) return
    const hosts = this.host ? [this.host] : this.info.hosts
    if (i >= hosts.length) {
      this.host = null
      this.onState('error', 'PC introuvable sur le Wi-Fi, nouvel essai…')
      this.retry = setTimeout(() => this.tryHosts(0), 2000)
      return
    }
    const host = hosts[i]
    this.onState('connecting', host)
    const ws = new WebSocket(`ws://${host}:${this.info.port}/ws/phone`)
    ws.binaryType = 'arraybuffer'
    let opened = false
    const timeout = setTimeout(() => {
      if (!opened) ws.close()
    }, 2500)
    ws.onopen = () => {
      opened = true
      clearTimeout(timeout)
      this.host = host
      this.ws = ws
      ws.send(JSON.stringify({ t: 'hello', token: this.info.token, ...this.getHello() }))
      this.onState('connected', host)
    }
    ws.onmessage = (ev) => {
      if (typeof ev.data !== 'string') return
      let cmd: StudioCommand
      try {
        cmd = JSON.parse(ev.data)
      } catch {
        return
      }
      if (cmd.t === 'ping') {
        // réponse immédiate pour la mesure d'horloge
        ws.send(JSON.stringify({ t: 'pong', s0: cmd.s0, c: Date.now() }))
        return
      }
      this.onCommand(cmd)
    }
    ws.onclose = () => {
      clearTimeout(timeout)
      if (this.ws === ws) this.ws = null
      if (this.closedByUser) return
      if (!opened) {
        this.tryHosts(i + 1)
      } else {
        this.onState('connecting', 'connexion perdue, reconnexion…')
        this.retry = setTimeout(() => this.tryHosts(0), 1000)
      }
    }
    ws.onerror = () => {
      // onclose suit toujours
    }
  }

  send(msg: Record<string, unknown>) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg))
  }

  sendBinary(kind: number, jpeg: ArrayBuffer) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return
    // évite d'accumuler des images si le Wi-Fi est lent
    if (kind === 1 && (this.ws as any).bufferedAmount > 400_000) return
    const out = new Uint8Array(jpeg.byteLength + 1)
    out[0] = kind
    out.set(new Uint8Array(jpeg), 1)
    this.ws.send(out.buffer)
  }
}
