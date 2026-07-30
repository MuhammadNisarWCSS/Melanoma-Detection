import { useEffect, useState } from 'react'
import { Activity, Github, ExternalLink } from 'lucide-react'
import { checkApiHealth } from '../api/client'

type ApiStatus = 'checking' | 'online' | 'offline' | 'no-model'

export default function Navbar() {
  const [status, setStatus] = useState<ApiStatus>('checking')

  useEffect(() => {
    checkApiHealth()
      .then((d) => setStatus(d.model_loaded ? 'online' : 'no-model'))
      .catch(() => setStatus('offline'))
  }, [])

  const statusConfig: Record<ApiStatus, { color: string; label: string; pulse: boolean }> = {
    online: { color: 'bg-emerald-500', label: 'API · Model Ready', pulse: true },
    'no-model': { color: 'bg-amber-400', label: 'API Online · No Model', pulse: false },
    offline: { color: 'bg-red-500', label: 'API Offline', pulse: false },
    checking: { color: 'bg-slate-500', label: 'Connecting…', pulse: true },
  }

  const s = statusConfig[status]

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-ink-600/60 bg-[#070b14]/85 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-5 sm:px-8 h-14 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="relative flex h-8 w-8 items-center justify-center rounded-lg border border-teal-400/25 bg-teal-400/8">
            <Activity className="h-4 w-4 text-teal-400" strokeWidth={2.5} />
            <span className="absolute -top-0.5 -right-0.5 h-2 w-2 rounded-full bg-teal-400 ring-2 ring-[#070b14]" />
          </div>
          <span className="text-[15px] font-semibold tracking-tight text-slate-100">
            Derm<span className="text-teal-400">AI</span>
          </span>
          <span className="hidden sm:inline-flex items-center gap-1.5 rounded-md border border-ink-600 bg-ink-800/60 px-2.5 py-0.5 font-mono text-[11px] text-slate-500">
            EfficientNet-B4
            <span className="text-ink-500">·</span>
            ISIC 2020
          </span>
        </div>

        {/* Right side */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className={`relative flex h-2 w-2`}>
              {s.pulse && (
                <span className={`absolute inline-flex h-full w-full animate-ping rounded-full ${s.color} opacity-60`} />
              )}
              <span className={`relative inline-flex h-2 w-2 rounded-full ${s.color}`} />
            </span>
            <span className="hidden sm:block text-[12px] text-slate-500">{s.label}</span>
          </div>

          <div className="h-4 w-px bg-ink-600" />

          <a
            href={
              import.meta.env.VITE_MLFLOW_URL
                ? `${String(import.meta.env.VITE_MLFLOW_URL).replace(/\/$/, '')}/`
                : 'http://18.219.3.159:5000'
            }
            target="_blank"
            rel="noopener noreferrer"
            className="hidden sm:flex items-center gap-1.5 text-[12px] text-slate-500 hover:text-teal-400 transition-colors"
          >
            MLflow
            <ExternalLink className="h-3 w-3" />
          </a>

          <a
            href="https://github.com/MuhammadNisarWCSS/Melanoma-Detection"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="GitHub"
            className="text-slate-600 hover:text-slate-300 transition-colors"
          >
            <Github className="h-5 w-5" />
          </a>
        </div>
      </div>
    </nav>
  )
}
