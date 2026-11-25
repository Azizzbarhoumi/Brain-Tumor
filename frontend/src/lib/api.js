// Thin client for the NeuroSeg FastAPI backend.

// Configure with VITE_API_URL in frontend/.env; defaults to the local uvicorn server.
export const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/+$/, '')

/** Error with a short title, a message and an optional fix hint, ready for the UI. */
export class ApiError extends Error {
  constructor(kind, title, message, hint) {
    super(message)
    this.kind = kind
    this.title = title
    this.hint = hint
  }
}

const unreachable = () =>
  new ApiError(
    'unreachable',
    'Backend unreachable',
    `Could not connect to the NeuroSeg API at ${API_URL}.`,
    'Start it from the backend folder:  uvicorn app:app --port 8000',
  )

/** GET /health. Resolves to the JSON body, or throws ApiError('unreachable'). */
export async function fetchHealth(signal) {
  let res
  try {
    res = await fetch(`${API_URL}/health`, { signal })
  } catch (err) {
    if (err.name === 'AbortError') throw err
    throw unreachable()
  }
  if (!res.ok) throw unreachable()
  return res.json()
}

/** POST /predict with a File/Blob. Resolves to the prediction JSON. */
export async function predict(file, signal) {
  const body = new FormData()
  body.append('file', file, file.name || 'upload.png')

  let res
  try {
    res = await fetch(`${API_URL}/predict`, { method: 'POST', body, signal })
  } catch (err) {
    if (err.name === 'AbortError') throw err
    throw unreachable()
  }

  if (res.ok) return res.json()

  // FastAPI errors look like {"detail": "..."}
  let detail = `Request failed with status ${res.status}.`
  try {
    const data = await res.json()
    if (typeof data.detail === 'string') detail = data.detail
    else if (Array.isArray(data.detail)) detail = data.detail.map((d) => d.msg).join('; ')
  } catch {
    /* body was not JSON; keep the generic message */
  }

  if (res.status === 503) {
    throw new ApiError('no-model', 'Model not trained yet', detail, 'In the backend folder run:  python train.py')
  }
  if (res.status === 400 || res.status === 413 || res.status === 415 || res.status === 422) {
    throw new ApiError('bad-file', 'Could not read this file', detail)
  }
  throw new ApiError('server', 'Server error', detail, 'Check the uvicorn terminal for the full traceback.')
}
