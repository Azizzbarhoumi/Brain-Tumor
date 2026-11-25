// Right column: canvas viewer with Overlay / Heatmap / Original tabs, verdict,
// statistics and the live threshold slider. Everything is recomputed in the
// browser from the probability map whenever the threshold or the view changes.
import { useEffect, useMemo, useRef } from 'react'
import { DISPLAY_SIZE, INFERNO_CSS, MIN_TUMOR_PIXELS, computeStats, renderView } from '../lib/imaging.js'

const VIEWS = [
  { id: 'overlay', label: 'Overlay' },
  { id: 'heatmap', label: 'Heatmap' },
  { id: 'original', label: 'Original' },
]

const THRESHOLD_MIN = 0.05
const THRESHOLD_MAX = 0.95
const THRESHOLD_STEP = 0.05

/* ------------------------------------------------------------------------- */
function Tabs({ view, onChange }) {
  return (
    <div role="tablist" aria-label="Result view" className="inline-flex self-start rounded-xl border border-white/[0.06] bg-ink-950/70 p-1">
      {VIEWS.map((v) => {
        const active = v.id === view
        return (
          <button
            key={v.id}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(v.id)}
            className={`rounded-lg px-3.5 py-1.5 text-sm font-medium transition duration-200
              ${active ? 'bg-ink-700 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
          >
            {v.label}
          </button>
        )
      })}
    </div>
  )
}

/* Corner brackets drawn over the viewer for a scanner-like frame. */
function Frame() {
  const corner = 'absolute h-5 w-5 border-accent/70'
  return (
    <div className="pointer-events-none absolute inset-2.5">
      <span className={`${corner} left-0 top-0 rounded-tl-md border-l-2 border-t-2`} />
      <span className={`${corner} right-0 top-0 rounded-tr-md border-r-2 border-t-2`} />
      <span className={`${corner} bottom-0 left-0 rounded-bl-md border-b-2 border-l-2`} />
      <span className={`${corner} bottom-0 right-0 rounded-br-md border-b-2 border-r-2`} />
    </div>
  )
}

function Viewer({ decoded, view, threshold }) {
  const canvasRef = useRef(null)
  useEffect(() => {
    if (canvasRef.current) renderView(canvasRef.current, decoded, view, threshold)
  }, [decoded, view, threshold])

  return (
    <div className="relative overflow-hidden rounded-xl border border-white/[0.06] bg-black">
      <canvas
        ref={canvasRef}
        width={DISPLAY_SIZE}
        height={DISPLAY_SIZE}
        className="block aspect-square w-full"
        aria-label={`MRI slice, ${view} view`}
      />
      <Frame />
      <div className="pointer-events-none absolute left-5 top-4 font-mono text-[11px] uppercase tracking-widest text-slate-300/80">
        {view}
      </div>
      <div className="pointer-events-none absolute right-5 top-4 font-mono text-[11px] text-slate-400/80">
        {decoded.width}×{decoded.height}
      </div>
      {view === 'overlay' && (
        <div className="pointer-events-none absolute bottom-4 left-5 font-mono text-[11px] text-slate-300/80">
          p &gt; {threshold.toFixed(2)}
        </div>
      )}
      {view === 'heatmap' && (
        <div className="absolute inset-x-5 bottom-4">
          <HeatmapLegend threshold={threshold} />
        </div>
      )}
    </div>
  )
}

function HeatmapLegend({ threshold }) {
  return (
    <div className="rounded-lg bg-black/60 px-3 py-2 backdrop-blur-sm">
      <div className="mb-1 flex justify-between font-mono text-[10px] text-slate-300">
        <span>tumor probability</span>
        <span className="text-slate-400">threshold {threshold.toFixed(2)}</span>
      </div>
      <div className="relative h-2 rounded-full" style={{ background: INFERNO_CSS }}>
        <span
          className="absolute -top-1 h-4 w-0.5 -translate-x-1/2 rounded bg-white shadow"
          style={{ left: `${threshold * 100}%` }}
        />
      </div>
      <div className="mt-1 flex justify-between font-mono text-[10px] text-slate-400">
        <span>0.0</span>
        <span>0.5</span>
        <span>1.0</span>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------------- */
function VerdictCard({ stats, threshold }) {
  const detected = stats.detected
  return (
    <div
      className={`relative overflow-hidden rounded-2xl border p-5 transition-colors duration-300
        ${detected ? 'border-lesion/30 bg-lesion/[0.08]' : 'border-clear/25 bg-clear/[0.07]'}`}
    >
      <div className={`absolute -right-10 -top-10 h-32 w-32 rounded-full blur-3xl ${detected ? 'bg-lesion/25' : 'bg-clear/20'}`} />
      <span className="eyebrow">Verdict</span>
      <div className="mt-2 flex items-center gap-3">
        <span className={`relative flex h-3 w-3`}>
          <span className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${detected ? 'bg-lesion' : 'bg-clear'}`} />
          <span className={`relative inline-flex h-3 w-3 rounded-full ${detected ? 'bg-lesion' : 'bg-clear'}`} />
        </span>
        <p
          key={String(detected)}
          className={`animate-fade-up text-xl font-bold tracking-wide xl:text-[22px] ${detected ? 'text-lesion' : 'text-clear'}`}
        >
          {detected ? 'TUMOR DETECTED' : 'NO TUMOR DETECTED'}
        </p>
      </div>
      <p className="mt-2 text-xs leading-relaxed text-slate-400">
        {detected
          ? `${stats.maskPixels.toLocaleString()} pixels exceed p > ${threshold.toFixed(2)} (minimum ${MIN_TUMOR_PIXELS}).`
          : `Fewer than ${MIN_TUMOR_PIXELS} pixels exceed p > ${threshold.toFixed(2)}.`}
      </p>
    </div>
  )
}

function StatRow({ label, value }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-t border-white/[0.05] py-2.5 first:border-t-0">
      <span className="text-sm text-slate-400">{label}</span>
      <span className="font-mono text-sm tabular-nums text-slate-100">{value}</span>
    </div>
  )
}

function StatsCard({ stats, decoded }) {
  return (
    <div className="panel p-5">
      <span className="eyebrow">Tumor area</span>
      <p className="mt-2 flex items-baseline gap-2">
        <span className="font-mono text-4xl font-semibold tabular-nums text-white">{stats.areaPct.toFixed(2)}%</span>
        <span className="text-sm text-slate-400">of brain</span>
      </p>
      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-ink-700">
        <div
          className="h-full rounded-full bg-lesion transition-[width] duration-300"
          style={{ width: `${Math.min(100, stats.areaPct * 4)}%` }}
          title="Scaled x4 for visibility"
        />
      </div>
      <div className="mt-4">
        <StatRow label="Mask pixels" value={stats.maskPixels.toLocaleString()} />
        <StatRow label="Brain pixels" value={stats.brainPixels.toLocaleString()} />
        <StatRow
          label="Mean confidence in mask"
          value={stats.meanConfidence == null ? '—' : `${(stats.meanConfidence * 100).toFixed(1)}%`}
        />
        <StatRow label="Peak probability" value={`${(decoded.peak * 100).toFixed(1)}%`} />
      </div>
    </div>
  )
}

function ThresholdSlider({ value, defaultValue, onChange }) {
  const pct = ((value - THRESHOLD_MIN) / (THRESHOLD_MAX - THRESHOLD_MIN)) * 100
  return (
    <div className="panel p-5">
      <div className="flex items-center justify-between">
        <label htmlFor="threshold" className="eyebrow">Threshold</label>
        <button
          type="button"
          onClick={() => onChange(defaultValue)}
          disabled={Math.abs(value - defaultValue) < 1e-9}
          className="font-mono text-[11px] text-slate-500 transition hover:text-accent disabled:opacity-0"
        >
          reset
        </button>
      </div>
      <p className="mt-2 font-mono text-3xl font-semibold tabular-nums text-accent">{value.toFixed(2)}</p>
      <input
        id="threshold"
        type="range"
        min={THRESHOLD_MIN}
        max={THRESHOLD_MAX}
        step={THRESHOLD_STEP}
        value={value}
        onChange={(e) => onChange(Math.round(parseFloat(e.target.value) * 100) / 100)}
        className="range mt-2"
        style={{ '--pct': `${pct}%` }}
        aria-valuetext={`Threshold ${value.toFixed(2)}`}
      />
      <div className="flex justify-between font-mono text-[10px] text-slate-500">
        <span>{THRESHOLD_MIN.toFixed(2)} · sensitive</span>
        <span>strict · {THRESHOLD_MAX.toFixed(2)}</span>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------------- */
function Skeleton() {
  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_320px]" aria-busy="true" aria-label="Analyzing">
      <div className="flex flex-col gap-4">
        <div className="skeleton h-10 w-64" />
        <div className="skeleton relative aspect-square w-full overflow-hidden">
          <div className="absolute inset-x-0 h-px animate-scan bg-accent/60 shadow-[0_0_14px_2px_rgba(59,196,255,0.5)]" />
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="font-mono text-xs uppercase tracking-[0.25em] text-slate-500">Segmenting…</span>
          </div>
        </div>
      </div>
      <div className="flex flex-col gap-4 xl:pt-14">
        <div className="skeleton h-28" />
        <div className="skeleton h-60" />
        <div className="skeleton h-32" />
      </div>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="flex min-h-[420px] flex-col items-center justify-center gap-5 text-center lg:min-h-[560px]">
      <div className="relative h-28 w-28">
        <div className="absolute inset-0 rounded-full border border-accent/20" />
        <div className="absolute inset-4 rounded-full border border-dashed border-accent/30" />
        <div className="absolute inset-9 rounded-full border border-accent/50" />
        <div className="absolute left-1/2 top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 animate-pulse-soft rounded-full bg-accent" />
      </div>
      <div>
        <p className="text-lg font-medium text-slate-200">No slice analyzed yet</p>
        <p className="mt-1 max-w-sm text-sm leading-relaxed text-slate-500">
          Upload a brain MRI slice or pick a sample. The tumor mask, heatmap and statistics will appear here.
        </p>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------------- */
export default function ResultPanel({ decoded, loading, view, onViewChange, threshold, onThresholdChange }) {
  const stats = useMemo(() => (decoded ? computeStats(decoded, threshold) : null), [decoded, threshold])

  return (
    <section className="panel min-w-0 p-5 sm:p-6">
      <div className="mb-5 flex items-center justify-between">
        <span className="eyebrow">02 · Segmentation</span>
        {decoded && !loading && decoded.inferenceMs != null && (
          <span className="font-mono text-[11px] text-slate-500">inference {decoded.inferenceMs} ms</span>
        )}
      </div>

      {loading ? (
        <Skeleton />
      ) : !decoded ? (
        <EmptyState />
      ) : (
        <div key={decoded.mriUrl} className="grid animate-fade-up gap-6 xl:grid-cols-[minmax(0,1fr)_320px]">
          <div className="flex min-w-0 flex-col gap-4">
            <Tabs view={view} onChange={onViewChange} />
            <Viewer decoded={decoded} view={view} threshold={threshold} />
          </div>
          <div className="flex flex-col gap-4 xl:pt-14">
            <VerdictCard stats={stats} threshold={threshold} />
            <StatsCard stats={stats} decoded={decoded} />
            <ThresholdSlider value={threshold} defaultValue={decoded.defaultThreshold} onChange={onThresholdChange} />
          </div>
        </div>
      )}
    </section>
  )
}
