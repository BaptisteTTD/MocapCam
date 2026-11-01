import { File, Paths } from 'expo-file-system'
import { Platform } from 'react-native'

export interface PendingUpload {
  take: string
  path: string
  cam: string
}

export interface Settings {
  name: string
  lastQr?: string
  pending?: PendingUpload[]
}

const file = () => new File(Paths.document, 'mocapcam_settings.json')

export const DEFAULT_SETTINGS: Settings = {
  name: Platform.OS === 'ios' ? 'iphone' : 'android',
}

export function loadSettings(): Settings {
  try {
    const f = file()
    if (f.exists) return { ...DEFAULT_SETTINGS, ...JSON.parse(f.textSync()) }
  } catch {
    // fichier absent ou illisible
  }
  return { ...DEFAULT_SETTINGS }
}

export function saveSettings(s: Settings) {
  try {
    const f = file()
    if (!f.exists) f.create()
    f.writeSync(JSON.stringify(s))
  } catch {
    // pas bloquant
  }
}

export function cleanName(n: string): string {
  return n.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '_').slice(0, 24) || DEFAULT_SETTINGS.name
}

/** Vidéos enregistrées mais pas encore envoyées au PC (renvoyées à la reconnexion). */
export function addPending(p: PendingUpload) {
  const s = loadSettings()
  s.pending = [...(s.pending ?? []).filter((x) => x.take !== p.take), p]
  saveSettings(s)
}

export function removePending(take: string) {
  const s = loadSettings()
  s.pending = (s.pending ?? []).filter((x) => x.take !== take)
  saveSettings(s)
}

export function getPending(): PendingUpload[] {
  return loadSettings().pending ?? []
}
