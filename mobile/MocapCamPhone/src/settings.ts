import { File, Paths } from 'expo-file-system'
import { Platform } from 'react-native'

export interface Settings {
  name: string
  lastQr?: string
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
