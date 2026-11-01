// Lecture de QR code en JavaScript à partir d'images de l'aperçu
// (sur Android, VisionCamera 5 n'a pas de lecteur de codes natif).
import jsQR from 'jsqr'

export interface RawImage {
  buffer: ArrayBuffer
  width: number
  height: number
  pixelFormat: string
}

// position des canaux R, G, B dans chaque pixel, et nombre d'octets par pixel
const LAYOUT: Record<string, [number, number, number, number]> = {
  RGBA: [0, 1, 2, 4],
  RGBX: [0, 1, 2, 4],
  BGRA: [2, 1, 0, 4],
  BGRX: [2, 1, 0, 4],
  ARGB: [1, 2, 3, 4],
  XRGB: [1, 2, 3, 4],
  ABGR: [3, 2, 1, 4],
  XBGR: [3, 2, 1, 4],
  RGB: [0, 1, 2, 3],
  BGR: [2, 1, 0, 3],
}

/** Convertit n'importe quel format de pixels en RGBA (attendu par jsQR). */
export function toRGBA(img: RawImage): Uint8ClampedArray {
  const { width: w, height: h } = img
  const src = new Uint8Array(img.buffer)
  const [r, g, b, bppDefault] = LAYOUT[img.pixelFormat] ?? LAYOUT.RGBA
  const stride = Math.floor(src.length / h)
  const bpp = stride >= w * 4 ? bppDefault : 3
  const out = new Uint8ClampedArray(w * h * 4)
  let o = 0
  for (let y = 0; y < h; y++) {
    let p = y * stride
    for (let x = 0; x < w; x++) {
      out[o] = src[p + r]
      out[o + 1] = src[p + g]
      out[o + 2] = src[p + b]
      out[o + 3] = 255
      o += 4
      p += bpp
    }
  }
  return out
}

export function decodeQr(img: RawImage): string | null {
  const res = jsQR(toRGBA(img), img.width, img.height, { inversionAttempts: 'dontInvert' })
  return res?.data ?? null
}
