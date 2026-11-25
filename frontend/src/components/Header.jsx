// Top bar: brand, subtitle, disclaimer badge and live API status.

function Logo() {
  return (
    <svg viewBox="0 0 40 40" className="h-10 w-10" aria-hidden="true">
      <rect x="0.5" y="0.5" width="39" height="39" rx="11" className="fill-ink-850 stroke-white/10" />
      <circle cx="20" cy="20" r="11" fill="none" className="stroke-accent" strokeWidth="2" />
      <circle cx="20" cy="20" r="6.5" fill="none" className="stroke-accent/40" strokeWidth="1.5" strokeDasharray="2 3" />
      <circle cx="23.5" cy="16.5" r="3.4" className="fill-lesion" />
    </svg>
  )
}

const STATUS = {
  checking: { dot: 'bg-slate-500 animate-pulse-soft', text: 'Checking API…' },
  ready: { dot: 'bg-accent', text: 'API online · model ready' },
  'no-model': { dot: 'bg-amber-400', text: 'API online · model not trained' },
  offline: { dot: 'bg-lesion animate-pulse-soft', text: 'API offline' },
}

export default function Header({ apiStatus, device }) {
  const s = STATUS[apiStatus] ?? STATUS.checking
  return (
    <header className="flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex items-center gap-4">
        <Logo />
        <div>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <h1 className="text-2xl font-semibold tracking-tight text-white sm:text-[28px]">
              Neuro<span className="text-accent">Seg</span>
            </h1>
            <span className="rounded-full border border-amber-300/25 bg-amber-300/[0.07] px-2.5 py-0.5 text-[11px] font-medium text-amber-200/90">
              Educational demo, not a medical device
            </span>
          </div>
          <p className="mt-0.5 text-sm text-slate-400">
            Brain MRI tumor segmentation with a U-Net (ResNet34 encoder)
          </p>
        </div>
      </div>

      <div
        className="flex items-center gap-2.5 self-start rounded-full border border-white/[0.06] bg-ink-900/80 px-3.5 py-1.5 sm:self-auto"
        title={device ? `Inference device: ${device}` : undefined}
      >
        <span className={`h-2 w-2 rounded-full ${s.dot}`} />
        <span className="font-mono text-xs text-slate-300">{s.text}</span>
        {device && apiStatus === 'ready' && (
          <span className="hidden font-mono text-xs text-slate-500 md:inline">· {device}</span>
        )}
      </div>
    </header>
  )
}
