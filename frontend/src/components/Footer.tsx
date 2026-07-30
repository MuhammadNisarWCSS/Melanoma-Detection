import { Activity } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="border-t border-ink-600/60 py-10 px-5">
      <div className="max-w-7xl mx-auto flex flex-col items-center gap-4 sm:flex-row sm:justify-between">
        <div className="flex items-center gap-2.5 text-slate-500">
          <Activity className="h-4 w-4 text-teal-400/60" strokeWidth={2} />
          <span className="text-[13px]">
            Derm<span className="text-teal-400/70">AI</span>
          </span>
        </div>

        <div className="flex flex-col items-center gap-1 text-center sm:items-end">
          <p className="text-[12px] text-slate-600">
            EfficientNet-B4 · PyTorch Lightning · MLflow · FastAPI
          </p>
          <p className="text-[11px] text-slate-700">
            For research use only — not intended for clinical diagnosis
          </p>
          <a
            href="https://github.com/MuhammadNisarWCSS/Melanoma-Detection"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[11px] text-slate-700 hover:text-teal-400 transition-colors"
          >
            View on GitHub
          </a>
        </div>
      </div>
    </footer>
  )
}
