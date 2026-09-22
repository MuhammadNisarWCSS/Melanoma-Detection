import { useState } from 'react'
import { ChevronDown, FlaskConical } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import TestMetricsSection from './TestMetrics'
import ModelStats from './ModelStats'

export default function TechnicalDetails() {
  const [open, setOpen] = useState(false)

  return (
    <section className="relative border-t border-ink-600/50 py-16 px-5 sm:px-8">
      <div className="mx-auto max-w-7xl">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="flex w-full items-center justify-between gap-4 rounded-2xl border border-ink-600/70 bg-ink-800/50 px-6 py-5 text-left transition-colors hover:border-ink-500"
        >
          <div className="flex items-center gap-3">
            <FlaskConical className="h-4 w-4 shrink-0 text-teal-400" />
            <div>
              <div className="text-[14px] font-semibold text-slate-200">
                Model performance &amp; methodology
              </div>
              <div className="mt-0.5 text-[12px] text-slate-500">
                For the technically curious — evaluation metrics, training runs, and architecture
              </div>
            </div>
          </div>
          <ChevronDown
            className={`h-4 w-4 shrink-0 text-slate-500 transition-transform ${open ? 'rotate-180' : ''}`}
          />
        </button>

        <AnimatePresence>
          {open && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="overflow-hidden"
            >
              <div className="pt-10">
                <TestMetricsSection />
                <ModelStats />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  )
}
