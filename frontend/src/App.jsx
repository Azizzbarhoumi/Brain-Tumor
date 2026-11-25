// NeuroSeg: upload an MRI slice, get a tumor mask, heatmap, verdict and area,
// with a threshold slider that recomputes everything live in the browser.
import { useCallback, useEffect, useRef, useState } from 'react'
import ErrorBanner from './components/ErrorBanner.jsx'
import Header from './components/Header.jsx'
import ResultPanel from './components/ResultPanel.jsx'
import UploadPanel, { sampleUrl } from './components/UploadPanel.jsx'
import { API_URL, ApiError, fetchHealth, predict } from './lib/api.js'
import { decodePrediction, isAcceptedImage } from './lib/imaging.js'

const HEALTH_POLL_MS = 10000

export default function App() {
  const [source, setSource] = useState(null) // { name, previewUrl, fallbackUrl, sample }
  const [decoded, setDecoded] = useState(null) // pixel buffers from the last prediction
  const [threshold, setThreshold] = useState(0.5)
  const [view, setView] = useState('overlay')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [api, setApi] = useState({ status: 'checking', device: null })

  const requestId = useRef(0) // ignore responses that arrive after a newer request
  const objectUrl = useRef(null)

  // ---- backend health, polled so the header reflects uvicorn starting/stopping
  useEffect(() => {
    let cancelled = false
    const check = async () => {
      try {
        const h = await fetchHealth()
        if (!cancelled) setApi({ status: h.model_loaded ? 'ready' : 'no-model', device: h.device })
      } catch {
        if (!cancelled) setApi({ status: 'offline', device: null })
      }
    }
    check()
    const id = setInterval(check, HEALTH_POLL_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  // Release the last object URL when the app unmounts.
  useEffect(() => () => objectUrl.current && URL.revokeObjectURL(objectUrl.current), [])

  const analyze = useCallback(async (file, meta = {}) => {
    if (!isAcceptedImage(file)) {
      setError({
        title: 'Unsupported file type',
        message: `"${file.name}" is not a supported image. Please upload a PNG, JPG or TIF MRI slice.`,
      })
      return
    }

    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
    const previewUrl = meta.previewUrl ?? (objectUrl.current = URL.createObjectURL(file))
    if (meta.previewUrl) objectUrl.current = null

    const id = ++requestId.current
    setSource({ name: file.name, previewUrl, fallbackUrl: null, sample: meta.sample ?? null })
    setError(null)
    setLoading(true)

    try {
      const resp = await predict(file)
      const result = await decodePrediction(resp)
      if (id !== requestId.current) return
      setDecoded(result)
      setThreshold(result.defaultThreshold)
      // Formats the browser can't preview (e.g. TIF) fall back to the API's resized PNG.
      setSource((s) => (s ? { ...s, fallbackUrl: result.mriUrl } : s))
      setApi((a) => (a.status === 'ready' ? a : { ...a, status: 'ready' }))
    } catch (err) {
      if (id !== requestId.current) return
      setDecoded(null)
      if (err instanceof ApiError) {
        setError({ title: err.title, message: err.message, hint: err.hint })
        if (err.kind === 'unreachable') setApi({ status: 'offline', device: null })
        if (err.kind === 'no-model') setApi((a) => ({ ...a, status: 'no-model' }))
      } else {
        setError({ title: 'Something went wrong', message: err.message || String(err) })
      }
    } finally {
      if (id === requestId.current) setLoading(false)
    }
  }, [])

  const analyzeSample = useCallback(
    async (sample) => {
      const url = sampleUrl(sample.file)
      try {
        const res = await fetch(url)
        if (!res.ok) throw new Error()
        const blob = await res.blob()
        const file = new File([blob], sample.file, { type: blob.type || 'image/png' })
        analyze(file, { previewUrl: url, sample: sample.file })
      } catch {
        setError({
          title: 'Sample not found',
          message: `${sample.file} is missing from frontend/public/samples/.`,
          hint: 'Run  python train.py  in the backend folder to export the sample slices.',
        })
      }
    },
    [analyze],
  )

  return (
    <div className="relative min-h-screen overflow-x-hidden">
      {/* Background: faint grid plus a soft accent glow */}
      <div className="pointer-events-none fixed inset-0 bg-grid [mask-image:radial-gradient(ellipse_at_top,black_30%,transparent_75%)]" />
      <div className="pointer-events-none fixed -top-48 left-1/2 h-[480px] w-[900px] -translate-x-1/2 rounded-full bg-accent/[0.07] blur-[120px]" />

      <div className="relative mx-auto flex max-w-[1500px] flex-col gap-8 px-5 py-8 sm:px-8 lg:py-10 2xl:px-12">
        <Header apiStatus={api.status} device={api.device} />

        <ErrorBanner error={error} onDismiss={() => setError(null)} />

        <main className="grid items-start gap-6 lg:grid-cols-[minmax(300px,380px)_minmax(0,1fr)] lg:gap-8 2xl:grid-cols-[420px_minmax(0,1fr)]">
          <UploadPanel source={source} busy={loading} onFile={(f) => analyze(f)} onSample={analyzeSample} />
          <ResultPanel
            decoded={decoded}
            loading={loading}
            view={view}
            onViewChange={setView}
            threshold={threshold}
            onThresholdChange={setThreshold}
          />
        </main>

        <footer className="flex flex-col gap-2 border-t border-white/[0.05] pt-6 text-xs text-slate-500 sm:flex-row sm:justify-between">
          <span>
            U-Net · ResNet34 encoder · trained on the LGG MRI Segmentation dataset (Buda et al., TCGA/TCIA).
          </span>
          <span className="font-mono">API {API_URL}</span>
        </footer>
      </div>
    </div>
  )
}
