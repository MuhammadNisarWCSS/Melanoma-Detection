import { useEffect, useState } from 'react'
import { ChevronDown } from 'lucide-react'
import { motion } from 'framer-motion'
import { fetchApiMetadata, fetchMLflowStats } from '../api/client'

interface LiveStats {
  auroc: string
  aurocIsTest: boolean
  tta: string
  architecture: string
}

function useLiveHeroStats(): LiveStats {
  const [stats, setStats] = useState<LiveStats>({
    auroc: '0.000',
    aurocIsTest: false,
    tta: '0×',
    architecture: '—',
  })

  useEffect(() => {
    let cancelled = false

    Promise.allSettled([fetchMLflowStats(), fetchApiMetadata()]).then(
      ([mlflowResult, metaResult]) => {
        if (cancelled) return
        setStats((prev) => {
          const next = { ...prev }
          if (mlflowResult.status === 'fulfilled') {
            const { runs } = mlflowResult.value
            if (runs.length > 0) {
              // Prefer test AUROC from the champion (best-val) run; fall back to val
              // AUROC — and track which one, so the label below never claims a val
              // number is a held-out test result.
              const champion = runs[0]
              const auroc = champion.test_auroc ?? champion.val_auroc
              if (auroc != null) {
                next.auroc = auroc.toFixed(3)
                next.aurocIsTest = champion.test_auroc != null
              }
              if (champion.backbone && champion.backbone !== 'unknown') {
                const raw = champion.backbone.toLowerCase()
                const match = raw.match(/b(\d+)/)
                next.architecture = match ? `B${match[1]}+Meta` : champion.backbone
              }
            }
          }
          if (metaResult.status === 'fulfilled') {
            next.tta = `${metaResult.value.tta_passes}×`
          }
          return next
        })
      },
    )

    return () => { cancelled = true }
  }, [])

  return stats
}

export default function Hero() {
  const liveStats = useLiveHeroStats()

  const STATS = [
    {
      value: liveStats.auroc,
      unit: 'AUROC',
      label: liveStats.aurocIsTest ? 'Held-out test AUROC' : 'Validation AUROC',
    },
    { value: liveStats.tta, unit: 'TTA', label: 'Test-time augmentation' },
    { value: liveStats.architecture, unit: '', label: 'Architecture' },
    { value: '384²', unit: 'px', label: 'Input resolution' },
  ]

  return (
    <section className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden px-5 pt-14">
      {/* Layered background */}
      <div className="absolute inset-0 bg-radial-teal" />
      <div className="absolute inset-0 bg-dot-pattern bg-dot-grid opacity-[0.35]" />

      {/* Decorative rings — microscope aperture metaphor */}
      <div className="pointer-events-none absolute right-[-180px] top-[-140px] h-[640px] w-[640px] animate-drift">
        <div className="absolute inset-0 rounded-full border border-teal-400/5" />
        <div className="absolute inset-[60px] rounded-full border border-teal-400/8" />
        <div className="absolute inset-[120px] rounded-full border border-teal-400/10" />
        <div className="absolute inset-[180px] rounded-full border border-teal-400/12" />
        <div className="absolute inset-[220px] rounded-full border-2 border-teal-400/6" />
        <div
          className="absolute inset-[240px] rounded-full"
          style={{ background: 'radial-gradient(circle, rgba(0,212,170,0.06) 0%, transparent 70%)' }}
        />
      </div>

      {/* Content */}
      <motion.div
        className="relative z-10 max-w-4xl mx-auto text-center"
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
      >
        {/* Eyebrow badge */}
        <div className="mb-8 inline-flex items-center gap-2.5 rounded-full border border-teal-400/20 bg-teal-400/5 px-4 py-1.5 text-[13px] text-teal-400">
          <span className="h-1.5 w-1.5 rounded-full bg-teal-400 animate-pulse" />
          ISIC 2020 Dermoscopy · Research Grade
        </div>

        {/* Heading */}
        <h1 className="mb-5 text-[52px] font-bold leading-[1.05] tracking-tight text-slate-100 sm:text-[68px] lg:text-[80px]">
          Clinical Melanoma
          <br />
          <span className="text-gradient-teal">Detection</span>
        </h1>

        <p className="mx-auto mb-14 max-w-2xl text-[17px] leading-relaxed text-slate-400">
          Multimodal deep learning combining dermoscopy imaging with patient metadata —
          EfficientNet-B4 backbone with calibrated threshold and GradCAM explainability.
        </p>

        {/* Stats row */}
        <motion.div
          className="mb-16 flex flex-wrap items-center justify-center gap-3"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.25, duration: 0.5 }}
        >
          {STATS.map(({ value, unit, label }) => (
            <div
              key={label}
              className="flex items-center gap-3 rounded-xl border border-ink-600/70 bg-ink-800/80 px-5 py-3 backdrop-blur-sm"
            >
              <div className="text-left">
                <div className="font-mono text-[22px] font-bold leading-none text-slate-100">
                  {value}
                  {unit && <span className="ml-0.5 text-[13px] text-teal-400">{unit}</span>}
                </div>
                <div className="mt-0.5 text-[11px] text-slate-500">{label}</div>
              </div>
            </div>
          ))}
        </motion.div>

        {/* CTA */}
        <a
          href="#analyze"
          className="inline-flex items-center gap-1.5 text-[13px] text-slate-500 transition-colors hover:text-teal-400"
        >
          Begin Analysis
          <ChevronDown className="h-4 w-4 animate-bounce" />
        </a>
      </motion.div>

      {/* Bottom fade */}
      <div className="absolute bottom-0 left-0 right-0 h-40 bg-gradient-to-t from-[#070b14] to-transparent" />
    </section>
  )
}
