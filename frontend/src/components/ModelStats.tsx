import { useEffect, useState } from 'react'
import {
  BarChart3,
  Layers,
  Cpu,
  TrendingUp,
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  RefreshCw,
} from 'lucide-react'
import { motion } from 'framer-motion'
import { fetchMLflowStats, fetchApiMetadata, type MLflowStats, type MLflowRun } from '../api/client'

// ─── Static benchmark data (from README) ─────────────────────────────────────

const BENCHMARKS = [
  { model: 'EfficientNet-B4 + Metadata', auroc: 0.89, highlight: true },
  { model: 'EfficientNet-B2 + Metadata', auroc: 0.87, highlight: false },
  { model: 'EfficientNet-B4 (image only)', auroc: 0.87, highlight: false },
  { model: 'ResNet-50 + Metadata', auroc: 0.85, highlight: false },
]

const AUROC_MIN = 0.82
const AUROC_MAX = 0.92

function aurocBar(auroc: number): number {
  return (auroc - AUROC_MIN) / (AUROC_MAX - AUROC_MIN)
}

// ─── Architecture diagram ─────────────────────────────────────────────────────

function ArchDiagram() {
  const lines = [
    { text: '// MelanomaClassifier — multimodal fusion', teal: true },
    { text: '' },
    { text: 'Input: Dermoscopy Image (384 × 384 px)' },
    { text: '  → EfficientNet-B4  (ImageNet pretrained, head removed)' },
    { text: '  → AdaptiveAvgPool  →  1792-d visual features' },
    { text: '' },
    { text: 'Input: Patient Metadata  (age, sex, anatomical site)' },
    { text: '  → MetadataMLP: Linear(3 → 64) → BN → ReLU → Dropout(0.3)' },
    { text: '  → Linear(64 → 32) → ReLU  →  32-d metadata features' },
    { text: '' },
    { text: '// Fusion', teal: true },
    { text: 'concat([1792-d, 32-d])  →  1824-d joint representation' },
    { text: '  → Linear(1824 → 512) → ReLU → Dropout(0.5)' },
    { text: '  → Linear(512 → 1)    → sigmoid  →  malignancy probability' },
    { text: '' },
    { text: '// Training details', teal: true },
    { text: 'Loss: Focal Loss  (γ=2.0, α=0.25)' },
    { text: 'Opt:  AdamW  (lr=1e-3, wd=1e-4)' },
    { text: 'Sched: CosineAnnealingLR  ·  EarlyStopping on val AUROC' },
    { text: 'Infer: 8-pass TTA  ·  Post-hoc threshold calibration' },
  ]

  return (
    <div className="overflow-x-auto rounded-xl bg-[#050810] p-5">
      <pre className="font-mono text-[12px] leading-6 text-slate-500">
        {lines.map((l, i) =>
          l.text === '' ? (
            <div key={i} className="h-3" />
          ) : (
            <div key={i} className={l.teal ? 'text-teal-500' : ''}>
              {l.text}
            </div>
          )
        )}
      </pre>
    </div>
  )
}

// ─── Run table ────────────────────────────────────────────────────────────────

function formatDuration(ms: number | null): string {
  if (!ms) return '—'
  const s = Math.floor(ms / 1000)
  const m = Math.floor(s / 60)
  const h = Math.floor(m / 60)
  if (h > 0) return `${h}h ${m % 60}m`
  if (m > 0) return `${m}m ${s % 60}s`
  return `${s}s`
}

function RunTable({ runs }: { runs: MLflowRun[] }) {
  const shown = runs.slice(0, 12)

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[620px] text-[13px]">
        <thead>
          <tr className="border-b border-ink-600/60">
            {['Run ID', 'Backbone', 'Highest Val AUROC', 'Val F1', 'Duration', 'Status'].map((h) => (
              <th
                key={h}
                className="px-4 py-3 text-left font-mono text-[10px] uppercase tracking-wider text-slate-600"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-ink-600/30">
          {shown.map((run, i) => (
            <tr
              key={run.run_id}
              className={`transition-colors hover:bg-ink-700/30 ${i === 0 ? 'bg-teal-400/3' : ''}`}
            >
              <td className="px-4 py-3 font-mono text-[11px] text-slate-600">
                {run.run_id.slice(0, 8)}…
              </td>
              <td className="px-4 py-3 text-slate-400">{run.backbone}</td>
              <td className="px-4 py-3">
                <span className={`font-mono ${i === 0 ? 'font-semibold text-teal-400' : 'text-slate-400'}`}>
                  {run.val_auroc != null ? run.val_auroc.toFixed(4) : '—'}
                </span>
              </td>
              <td className="px-4 py-3 font-mono text-slate-500">
                {run.val_f1 != null ? run.val_f1.toFixed(4) : '—'}
              </td>
              <td className="px-4 py-3 text-slate-600">
                <span className="flex items-center gap-1.5">
                  <Clock className="h-3 w-3" />
                  {formatDuration(run.duration_ms)}
                </span>
              </td>
              <td className="px-4 py-3">
                {run.status === 'FINISHED' ? (
                  <span className="inline-flex items-center gap-1 rounded-full border border-emerald-500/20 bg-emerald-500/8 px-2 py-0.5 text-[10px] font-medium text-emerald-400">
                    <CheckCircle2 className="h-2.5 w-2.5" />
                    Finished
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 rounded-full border border-amber-400/20 bg-amber-400/8 px-2 py-0.5 text-[10px] font-medium text-amber-400">
                    <XCircle className="h-2.5 w-2.5" />
                    {run.status}
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────

type LoadState = 'idle' | 'loading' | 'success' | 'error'

export default function ModelStats() {
  const [loadState, setLoadState] = useState<LoadState>('idle')
  const [stats, setStats] = useState<MLflowStats | null>(null)
  const [ttaPasses, setTtaPasses] = useState<number | null>(null)

  const load = () => {
    setLoadState('loading')

    fetchApiMetadata()
      .then((m) => setTtaPasses(m.tta_passes))
      .catch(() => { /* keep null — will show fallback */ })

    fetchMLflowStats()
      .then((s) => { setStats(s); setLoadState('success') })
      .catch(() => setLoadState('error'))
  }

  useEffect(() => { load() }, [])

  const bestAuroc = stats?.best_auroc ?? 0
  const totalRuns = stats?.total_runs ?? null
  const hasLiveRuns = loadState === 'success' && (stats?.runs?.length ?? 0) > 0

  // Derive architecture label from the best MLflow run's backbone param.
  const bestBackbone = stats?.runs?.[0]?.backbone
  let architectureLabel = '—'
  if (bestBackbone && bestBackbone !== 'unknown') {
    const match = bestBackbone.toLowerCase().match(/b(\d+)/)
    architectureLabel = match ? `B${match[1]}+Meta` : bestBackbone
  }

  const ttaLabel = ttaPasses != null ? `${ttaPasses}×` : '—'
  const ttaSub = ttaPasses != null ? 'Live from API' : 'Test-time augmentation'

  const statCards = [
    {
      icon: BarChart3,
      label: 'Best Val AUROC',
      value: bestAuroc.toFixed(3),
      sub: loadState === 'success' ? 'Live from MLflow' : 'Validated benchmark',
    },
    {
      icon: TrendingUp,
      label: 'Training Runs',
      value: totalRuns != null ? String(totalRuns) : '—',
      sub: 'Tracked experiments',
    },
    {
      icon: Layers,
      label: 'Architecture',
      value: architectureLabel,
      sub: hasLiveRuns ? 'Best run backbone' : 'No runs yet',
    },
    {
      icon: Cpu,
      label: 'TTA Passes',
      value: ttaLabel,
      sub: ttaSub,
    },
  ]

  return (
    <section
      id="stats"
      className="relative border-t border-ink-600/50 py-24 px-5 sm:px-8"
    >
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <motion.div
          className="mb-12 flex items-start justify-between"
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.45 }}
        >
          <div>
            <p className="mb-2 font-mono text-[11px] uppercase tracking-[0.2em] text-teal-400">
              Performance
            </p>
            <h2 className="text-[32px] font-bold tracking-tight text-slate-100">
              Model Statistics
            </h2>
            <div className="mt-2 flex items-center gap-2.5">
              <p className="text-[15px] text-slate-400">
                {loadState === 'success'
                  ? hasLiveRuns
                    ? `Live MLflow data · ${stats!.runs.length} training runs tracked`
                    : 'MLflow reachable · No runs recorded yet'
                  : loadState === 'error'
                  ? 'MLflow offline · Showing validated benchmarks'
                  : 'Connecting to MLflow tracking server…'}
              </p>
              <span
                className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] ${
                  loadState === 'success'
                    ? 'border-emerald-500/25 bg-emerald-500/8 text-emerald-400'
                    : loadState === 'error'
                    ? 'border-slate-700 bg-ink-800/60 text-slate-600'
                    : 'border-amber-400/25 bg-amber-400/8 text-amber-400'
                }`}
              >
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    loadState === 'success'
                      ? 'bg-emerald-500 animate-pulse'
                      : loadState === 'error'
                      ? 'bg-slate-700'
                      : 'bg-amber-400 animate-pulse'
                  }`}
                />
                {loadState === 'success' ? 'Live' : loadState === 'error' ? 'Offline' : 'Connecting'}
              </span>
            </div>
          </div>

          {loadState === 'error' && (
            <button
              onClick={load}
              className="flex items-center gap-1.5 rounded-lg border border-ink-600/70 bg-ink-800/50 px-3 py-2 text-[12px] text-slate-500 transition-colors hover:border-ink-500 hover:text-slate-300"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              Retry
            </button>
          )}
          {loadState === 'loading' && (
            <Loader2 className="mt-2 h-4 w-4 animate-spin text-slate-600" />
          )}
        </motion.div>

        {/* Stat cards */}
        <div className="mb-12 grid grid-cols-2 gap-4 lg:grid-cols-4">
          {statCards.map(({ icon: Icon, label, value, sub }, i) => (
            <motion.div
              key={label}
              className="rounded-2xl border border-ink-600/70 bg-ink-800/60 p-5"
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: i * 0.07 }}
            >
              <div className="mb-4 flex items-center justify-between">
                <span className="font-mono text-[10px] uppercase tracking-wider text-slate-600">
                  {label}
                </span>
                <Icon className="h-3.5 w-3.5 text-teal-400/50" />
              </div>
              <div className="font-mono text-[30px] font-bold leading-none text-slate-100">
                {value}
              </div>
              <div className="mt-1.5 text-[11px] text-slate-600">{sub}</div>
            </motion.div>
          ))}
        </div>

        {/* Benchmark table */}
        <motion.div
          className="mb-8 overflow-hidden rounded-2xl border border-ink-600/70 bg-ink-800/50"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <div className="border-b border-ink-600/60 px-6 py-4">
            <h3 className="text-[15px] font-semibold text-slate-200">Architecture Comparison</h3>
            <p className="mt-0.5 text-[12px] text-slate-600">
              Validation AUROC on held-out ISIC 2020 split
            </p>
          </div>
          <div className="divide-y divide-ink-600/30">
            {BENCHMARKS.map((b, i) => (
              <div
                key={b.model}
                className={`flex items-center gap-6 px-6 py-4 ${
                  b.highlight ? 'bg-teal-400/[0.03]' : ''
                }`}
              >
                <div className="flex flex-1 items-center gap-3">
                  {b.highlight && (
                    <span className="shrink-0 rounded border border-teal-400/20 bg-teal-400/10 px-1.5 py-0.5 font-mono text-[9px] font-semibold uppercase tracking-wider text-teal-400">
                      BEST
                    </span>
                  )}
                  <span
                    className={`text-[14px] ${b.highlight ? 'font-medium text-slate-200' : 'text-slate-500'}`}
                  >
                    {b.model}
                  </span>
                </div>
                <div className="flex items-center gap-4">
                  <span
                    className={`w-14 text-right font-mono text-[14px] ${
                      b.highlight ? 'font-semibold text-teal-400' : 'text-slate-500'
                    }`}
                  >
                    {b.auroc.toFixed(2)}
                  </span>
                  <div className="hidden w-28 sm:block">
                    <div className="h-[3px] overflow-hidden rounded-full bg-ink-600">
                      <motion.div
                        className="h-full rounded-full"
                        style={{
                          backgroundColor: b.highlight ? '#00d4aa' : '#253548',
                          boxShadow: b.highlight ? '0 0 6px rgba(0,212,170,0.4)' : 'none',
                        }}
                        initial={{ width: 0 }}
                        whileInView={{ width: `${aurocBar(b.auroc) * 100}%` }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.8, delay: i * 0.1, ease: 'easeOut' }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Live MLflow runs */}
        {hasLiveRuns && (
          <motion.div
            className="mb-8 overflow-hidden rounded-2xl border border-ink-600/70 bg-ink-800/50"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            <div className="flex items-center justify-between border-b border-ink-600/60 px-6 py-4">
              <div>
                <h3 className="text-[15px] font-semibold text-slate-200">Live Training Runs</h3>
                <p className="mt-0.5 text-[12px] text-slate-600">
                  melanoma-detection experiment · MLflow tracking server
                </p>
              </div>
              <span className="font-mono text-[11px] text-teal-400 border border-teal-400/20 rounded px-2 py-0.5">
                {stats!.runs.length} runs
              </span>
            </div>
            <RunTable runs={stats!.runs} />
          </motion.div>
        )}

        {/* Architecture diagram */}
        <motion.div
          className="overflow-hidden rounded-2xl border border-ink-600/70 bg-ink-800/50"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <div className="border-b border-ink-600/60 px-6 py-4">
            <h3 className="text-[15px] font-semibold text-slate-200">Architecture Overview</h3>
            <p className="mt-0.5 text-[12px] text-slate-600">
              Multimodal EfficientNet-B4 + patient metadata fusion
            </p>
          </div>
          <div className="p-5">
            <ArchDiagram />
          </div>
        </motion.div>
      </div>
    </section>
  )
}
