import { useEffect } from 'react'
import { AlertTriangle } from 'lucide-react'
import { motion } from 'framer-motion'

interface Props {
  confirmLabel: string
  onConfirm: () => void
  onCancel?: () => void
}

export default function DisclaimerModal({ confirmLabel, onConfirm, onCancel }: Props) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && onCancel) onCancel()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onCancel])

  return (
    <motion.div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 p-5 backdrop-blur-sm"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      role="dialog"
      aria-modal="true"
      aria-labelledby="disclaimer-title"
    >
      <motion.div
        className="w-full max-w-md rounded-2xl border border-ink-600/70 bg-[#fbf6f2] p-7 shadow-2xl"
        initial={{ opacity: 0, y: 16, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 8 }}
        transition={{ duration: 0.2 }}
      >
        <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-full bg-amber-400/15">
          <AlertTriangle className="h-5 w-5 text-amber-500" strokeWidth={2} />
        </div>
        <h2
          id="disclaimer-title"
          className="font-display text-[20px] font-semibold tracking-tight text-slate-100"
        >
          Before you continue
        </h2>
        <div className="mt-3 space-y-3 text-[14px] leading-relaxed text-slate-400">
          <p>
            This site is a student machine-learning project. It is not a medical device and
            it isn't a doctor, so nothing it says should be treated as medical advice or a
            diagnosis.
          </p>
          <p>
            The model gets things wrong. It can call a melanoma harmless, and it can flag a
            harmless mole. If a spot is changing, bleeding, itching, or just worrying you, see a
            dermatologist, whatever result you get here.
          </p>
        </div>
        <div className="mt-6 flex gap-2">
          {onCancel && (
            <button
              type="button"
              onClick={onCancel}
              className="rounded-lg border border-ink-600/70 px-4 py-3 text-[14px] font-medium text-slate-400 transition-colors hover:border-ink-500 hover:text-slate-200"
            >
              Cancel
            </button>
          )}
          <button
            type="button"
            autoFocus
            onClick={onConfirm}
            className="flex-1 rounded-lg bg-teal-400 py-3 text-[14px] font-semibold text-white transition-colors hover:bg-teal-500"
          >
            {confirmLabel}
          </button>
        </div>
      </motion.div>
    </motion.div>
  )
}
