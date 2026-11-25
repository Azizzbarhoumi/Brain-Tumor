// Left column: drag-and-drop / click-to-browse upload zone with preview, plus sample slices.
import { useRef, useState } from 'react'
import { ACCEPT_ATTR } from '../lib/imaging.js'

// Written to frontend/public/samples/ by backend/train.py (from TEST patients).
export const SAMPLES = [
  { file: 'sample_tumor_1.png', label: 'Case 01', truth: 'tumor' },
  { file: 'sample_tumor_2.png', label: 'Case 02', truth: 'tumor' },
  { file: 'sample_tumor_3.png', label: 'Case 03', truth: 'tumor' },
  { file: 'sample_no_tumor.png', label: 'Case 04', truth: 'clear' },
]

export const sampleUrl = (file) => `${import.meta.env.BASE_URL}samples/${file}`

function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
      <path d="M12 16V4m0 0-4.5 4.5M12 4l4.5 4.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" strokeLinecap="round" />
    </svg>
  )
}

function SampleTile({ sample, active, disabled, onPick }) {
  const [missing, setMissing] = useState(false)
  const unavailable = missing || disabled
  return (
    <button
      type="button"
      onClick={() => !missing && onPick(sample)}
      disabled={unavailable}
      className={`group relative overflow-hidden rounded-xl border text-left transition duration-200
        ${active ? 'border-accent ring-2 ring-accent/30' : 'border-white/[0.06] hover:border-accent/50'}
        ${unavailable ? 'cursor-not-allowed opacity-60' : 'hover:-translate-y-0.5'}`}
      title={missing ? 'Run backend/train.py to export sample slices' : `Analyze ${sample.label}`}
    >
      <div className="aspect-square bg-black">
        {missing ? (
          <div className="flex h-full items-center justify-center p-2 text-center font-mono text-[10px] leading-tight text-slate-500">
            not exported yet
          </div>
        ) : (
          <img
            src={sampleUrl(sample.file)}
            alt={sample.label}
            onError={() => setMissing(true)}
            className="h-full w-full object-cover transition duration-300 group-hover:scale-[1.04]"
            draggable={false}
          />
        )}
      </div>
      <div className="flex items-center justify-between gap-1 bg-ink-850 px-2 py-1.5">
        <span className="font-mono text-[11px] text-slate-300">{sample.label}</span>
        <span
          className={`h-1.5 w-1.5 shrink-0 rounded-full ${sample.truth === 'tumor' ? 'bg-lesion/80' : 'bg-clear/80'}`}
          title={sample.truth === 'tumor' ? 'Ground truth: tumor' : 'Ground truth: no tumor'}
        />
      </div>
    </button>
  )
}

export default function UploadPanel({ source, busy, onFile, onSample }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)
  // Remember which preview URL the browser failed to decode (e.g. TIF); a new upload resets it.
  const [failedUrl, setFailedUrl] = useState(null)
  const previewFailed = Boolean(source?.previewUrl) && failedUrl === source.previewUrl

  const openPicker = () => inputRef.current?.click()

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) onFile(file)
  }

  const preview = previewFailed ? source?.fallbackUrl : source?.previewUrl

  return (
    <section className="flex flex-col gap-6">
      <div className="panel p-5">
        <div className="mb-4 flex items-center justify-between">
          <span className="eyebrow">01 · Input slice</span>
          {source && (
            <span className="max-w-[60%] truncate font-mono text-[11px] text-slate-500" title={source.name}>
              {source.name}
            </span>
          )}
        </div>

        <div
          role="button"
          tabIndex={0}
          onClick={openPicker}
          onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), openPicker())}
          onDragEnter={(e) => (e.preventDefault(), setDragging(true))}
          onDragOver={(e) => e.preventDefault()}
          onDragLeave={(e) => e.currentTarget.contains(e.relatedTarget) || setDragging(false)}
          onDrop={handleDrop}
          aria-label="Upload an MRI slice"
          className={`relative flex aspect-square w-full cursor-pointer flex-col items-center justify-center overflow-hidden rounded-xl border border-dashed outline-none transition duration-200
            focus-visible:ring-2 focus-visible:ring-accent/60
            ${dragging ? 'border-accent bg-accent/[0.06]' : 'border-white/15 bg-ink-950/60 hover:border-accent/60 hover:bg-ink-850/60'}`}
        >
          {preview ? (
            <>
              <img
                key={preview}
                src={preview}
                alt="Selected MRI slice"
                onError={() => preview === source?.previewUrl && setFailedUrl(preview)}
                className="absolute inset-0 h-full w-full animate-fade-up object-contain"
                draggable={false}
              />
              {busy && (
                <div className="pointer-events-none absolute inset-0">
                  <div className="absolute inset-x-0 h-16 -translate-y-1/2 animate-scan bg-gradient-to-b from-transparent via-accent/25 to-transparent" />
                  <div className="absolute inset-x-0 h-px animate-scan bg-accent/80 shadow-[0_0_12px_2px_rgba(59,196,255,0.6)]" />
                </div>
              )}
              <div className="absolute inset-x-0 bottom-0 flex justify-center bg-gradient-to-t from-black/80 to-transparent pb-3 pt-10 opacity-0 transition hover:opacity-100">
                <span className="font-mono text-xs text-slate-200">Click or drop to replace</span>
              </div>
            </>
          ) : source && previewFailed ? (
            <p className="px-6 text-center text-sm text-slate-400">
              This format can't be previewed in the browser.
              <br />
              <span className="text-slate-500">It will appear after analysis.</span>
            </p>
          ) : (
            <div className="flex flex-col items-center gap-3 px-6 text-center">
              <div className={`rounded-2xl border p-3.5 transition ${dragging ? 'border-accent text-accent' : 'border-white/10 text-slate-400'}`}>
                <UploadIcon />
              </div>
              <div>
                <p className="font-medium text-slate-200">
                  {dragging ? 'Drop to analyze' : 'Drop an MRI slice here'}
                </p>
                <p className="mt-1 text-sm text-slate-500">
                  or <span className="text-accent underline-offset-4 hover:underline">browse files</span>
                </p>
              </div>
              <p className="font-mono text-[11px] text-slate-600">PNG · JPG · TIF  ·  resized to 256×256</p>
            </div>
          )}
        </div>

        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT_ATTR}
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) onFile(file)
            e.target.value = '' // allow re-selecting the same file
          }}
        />
      </div>

      <div className="panel p-5">
        <div className="mb-4 flex items-center justify-between">
          <span className="eyebrow">Sample slices</span>
          <span className="flex items-center gap-3 font-mono text-[10px] text-slate-500">
            <span className="flex items-center gap-1"><span className="h-1.5 w-1.5 rounded-full bg-lesion/80" />tumor</span>
            <span className="flex items-center gap-1"><span className="h-1.5 w-1.5 rounded-full bg-clear/80" />clear</span>
          </span>
        </div>
        <div className="grid grid-cols-2 gap-2.5 min-[480px]:grid-cols-4">
          {SAMPLES.map((s) => (
            <SampleTile
              key={s.file}
              sample={s}
              active={source?.sample === s.file}
              disabled={busy}
              onPick={onSample}
            />
          ))}
        </div>
        <p className="mt-3 text-xs leading-relaxed text-slate-500">
          Slices from held-out test patients the model never saw during training.
        </p>
      </div>
    </section>
  )
}
