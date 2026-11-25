// Dismissible error message with an optional command hint.

export default function ErrorBanner({ error, onDismiss }) {
  if (!error) return null
  return (
    <div
      role="alert"
      className="flex animate-fade-up items-start gap-4 rounded-2xl border border-lesion/30 bg-lesion/[0.07] px-5 py-4"
    >
      <svg viewBox="0 0 24 24" className="mt-0.5 h-5 w-5 shrink-0 text-lesion" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7.5v5.5M12 16.5v.01" strokeLinecap="round" />
      </svg>
      <div className="min-w-0 flex-1">
        <p className="font-medium text-red-100">{error.title}</p>
        <p className="mt-0.5 break-words text-sm text-red-100/70">{error.message}</p>
        {error.hint && (
          <code className="mt-2 inline-block max-w-full overflow-x-auto whitespace-pre rounded-md bg-black/40 px-2.5 py-1 font-mono text-xs text-slate-200">
            {error.hint}
          </code>
        )}
      </div>
      <button
        type="button"
        onClick={onDismiss}
        className="rounded-md p-1 text-red-100/60 transition hover:bg-white/5 hover:text-red-100"
        aria-label="Dismiss error"
      >
        <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
          <path d="M6 6l12 12M18 6L6 18" strokeLinecap="round" />
        </svg>
      </button>
    </div>
  )
}
