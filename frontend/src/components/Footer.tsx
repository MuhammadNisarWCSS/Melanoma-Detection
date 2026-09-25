import { Activity } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="border-t border-ink-600/60 py-10 px-5">
      <div className="max-w-7xl mx-auto flex flex-col items-center gap-4 sm:flex-row sm:justify-between">
        <div className="flex items-center gap-2.5 text-slate-500">
          <Activity className="h-4 w-4 text-teal-400/60" strokeWidth={2} />
          <span className="font-display text-[13px]">
            Melanoma Detection <span className="text-teal-400/70">AI</span>
          </span>
        </div>

        <div className="flex flex-col items-center gap-1 text-center sm:items-end">
          <p className="text-[11px] text-slate-700">
            A student project, not medical advice. If a spot worries you, see a dermatologist.
          </p>
          <div className="mt-1 flex items-center gap-3 text-[11px] text-slate-700">
            <a
              href="https://github.com/MuhammadNisarWCSS/Melanoma-Detection"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-teal-400 transition-colors"
            >
              View on GitHub
            </a>
            <span className="text-ink-500">·</span>
            <a
              href={
                import.meta.env.VITE_MLFLOW_URL
                  ? `${String(import.meta.env.VITE_MLFLOW_URL).replace(/\/$/, '')}/`
                  : 'http://3.18.225.100:5000'
              }
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-teal-400 transition-colors"
            >
              MLflow tracking
            </a>
          </div>
        </div>
      </div>
    </footer>
  )
}
