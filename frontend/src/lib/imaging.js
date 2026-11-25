// All pixel work happens here, in the browser: decoding the PNGs returned by the
// API, thresholding the probability map, statistics, and rendering the three views.

/** Canvas resolution used for display. The 256x256 maps are upsampled (bilinear)
 *  to this size so the mask edge and outline look smooth on large screens. */
export const DISPLAY_SIZE = 512

/** Minimum mask size (pixels at 256x256) for a "tumor detected" verdict. */
export const MIN_TUMOR_PIXELS = 30

const OVERLAY_RGB = [255, 56, 72] // semi-transparent fill colour
const OUTLINE_RGB = [255, 92, 104] // thin outline colour
const OVERLAY_ALPHA = 0.42
const HEATMAP_ALPHA = 0.5

// Inferno colormap, sampled every 0.1 from matplotlib; interpolated into a 256-entry LUT.
export const INFERNO_STOPS = [
  [0, 0, 4],
  [22, 11, 57],
  [66, 10, 104],
  [106, 23, 110],
  [147, 38, 103],
  [188, 55, 84],
  [221, 81, 58],
  [243, 120, 25],
  [252, 165, 10],
  [246, 215, 70],
  [252, 255, 164],
]

const INFERNO_LUT = (() => {
  const lut = new Uint8Array(256 * 3)
  const segments = INFERNO_STOPS.length - 1
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * segments
    const k = Math.min(Math.floor(t), segments - 1)
    const f = t - k
    for (let c = 0; c < 3; c++) {
      lut[i * 3 + c] = Math.round(INFERNO_STOPS[k][c] * (1 - f) + INFERNO_STOPS[k + 1][c] * f)
    }
  }
  return lut
})()

/** CSS gradient matching the heatmap colormap (for the legend). */
export const INFERNO_CSS = `linear-gradient(to right, ${INFERNO_STOPS.map(
  (c, i) => `rgb(${c.join(',')}) ${(i / (INFERNO_STOPS.length - 1)) * 100}%`,
).join(', ')})`

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('Could not decode an image returned by the API.'))
    img.src = src
  })
}

/** Draw an image into an offscreen canvas of the given size and read back RGBA pixels. */
function readPixels(img, width, height, smooth = true) {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  ctx.imageSmoothingEnabled = smooth
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(img, 0, 0, width, height)
  return ctx.getImageData(0, 0, width, height).data
}

/** Keep only the red channel of an RGBA buffer (the prob map PNG is grayscale). */
function redChannel(rgba) {
  const out = new Uint8Array(rgba.length / 4)
  for (let i = 0, j = 0; j < out.length; i += 4, j++) out[j] = rgba[i]
  return out
}

/**
 * Turn the /predict JSON into pixel buffers.
 *  - prob: native-resolution probabilities (0..255) used for all statistics
 *  - probDisplay / mriDisplay: upsampled buffers used for rendering
 */
export async function decodePrediction(resp) {
  const [mriImg, probImg] = await Promise.all([
    loadImage(`data:image/png;base64,${resp.image}`),
    loadImage(`data:image/png;base64,${resp.prob_map}`),
  ])
  const width = probImg.naturalWidth
  const height = probImg.naturalHeight
  const prob = redChannel(readPixels(probImg, width, height, false))
  let peak = 0
  for (let i = 0; i < prob.length; i++) if (prob[i] > peak) peak = prob[i]

  return {
    width,
    height,
    prob,
    peak: peak / 255,
    probDisplay: redChannel(readPixels(probImg, DISPLAY_SIZE, DISPLAY_SIZE, true)),
    mriDisplay: readPixels(mriImg, DISPLAY_SIZE, DISPLAY_SIZE, true),
    mriUrl: `data:image/png;base64,${resp.image}`,
    brainPixels: resp.brain_pixels,
    defaultThreshold: resp.default_threshold ?? 0.5,
    inferenceMs: resp.inference_ms,
    filename: resp.filename,
  }
}

/** Mask statistics at a threshold, computed on the native 256x256 probability map. */
export function computeStats(decoded, threshold) {
  const { prob, brainPixels } = decoded
  const cut = threshold * 255
  let count = 0
  let sum = 0
  for (let i = 0; i < prob.length; i++) {
    const v = prob[i]
    if (v > cut) {
      count++
      sum += v
    }
  }
  return {
    maskPixels: count,
    brainPixels,
    areaPct: brainPixels > 0 ? (count / brainPixels) * 100 : 0,
    meanConfidence: count > 0 ? sum / count / 255 : null,
    detected: count >= MIN_TUMOR_PIXELS,
  }
}

/** Render one of the three views ("overlay" | "heatmap" | "original") into a canvas. */
export function renderView(canvas, decoded, view, threshold) {
  const size = DISPLAY_SIZE
  if (canvas.width !== size) canvas.width = size
  if (canvas.height !== size) canvas.height = size
  const ctx = canvas.getContext('2d')
  const out = ctx.createImageData(size, size)
  const px = out.data
  const mri = decoded.mriDisplay
  const prob = decoded.probDisplay
  px.set(mri)

  if (view === 'heatmap') {
    for (let j = 0, i = 0; j < prob.length; j++, i += 4) {
      const l = prob[j] * 3
      px[i] = mri[i] * (1 - HEATMAP_ALPHA) + INFERNO_LUT[l] * HEATMAP_ALPHA
      px[i + 1] = mri[i + 1] * (1 - HEATMAP_ALPHA) + INFERNO_LUT[l + 1] * HEATMAP_ALPHA
      px[i + 2] = mri[i + 2] * (1 - HEATMAP_ALPHA) + INFERNO_LUT[l + 2] * HEATMAP_ALPHA
    }
  } else if (view === 'overlay') {
    const cut = threshold * 255
    const inMask = (j) => prob[j] > cut
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const j = y * size + x
        if (!inMask(j)) continue
        const i = j * 4
        // Outline: a mask pixel with a nearby 4-neighbour outside the mask.
        // Checked at distance 1 and 2 so the outline is ~2 display pixels wide.
        const edge =
          x < 2 || y < 2 || x >= size - 2 || y >= size - 2 ||
          !inMask(j - 1) || !inMask(j + 1) || !inMask(j - size) || !inMask(j + size) ||
          !inMask(j - 2) || !inMask(j + 2) || !inMask(j - 2 * size) || !inMask(j + 2 * size)
        if (edge) {
          px[i] = OUTLINE_RGB[0]
          px[i + 1] = OUTLINE_RGB[1]
          px[i + 2] = OUTLINE_RGB[2]
        } else {
          px[i] = mri[i] * (1 - OVERLAY_ALPHA) + OVERLAY_RGB[0] * OVERLAY_ALPHA
          px[i + 1] = mri[i + 1] * (1 - OVERLAY_ALPHA) + OVERLAY_RGB[1] * OVERLAY_ALPHA
          px[i + 2] = mri[i + 2] * (1 - OVERLAY_ALPHA) + OVERLAY_RGB[2] * OVERLAY_ALPHA
        }
      }
    }
  }
  ctx.putImageData(out, 0, 0)
}

const ACCEPTED_TYPES = ['image/png', 'image/jpeg', 'image/tiff', 'image/bmp', 'image/webp']
const ACCEPTED_EXT = /\.(png|jpe?g|tiff?|bmp|webp)$/i
export const ACCEPT_ATTR = '.png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp,image/*'

/** Client-side file check before uploading. */
export function isAcceptedImage(file) {
  return ACCEPTED_TYPES.includes(file.type) || ACCEPTED_EXT.test(file.name || '')
}
