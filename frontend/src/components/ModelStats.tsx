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
    { text: 'Loss: Focal Loss  (γ=2.0, α=0.5)' },
    { text: 'Opt:  AdamW  (lr=3e-4, wd=1e-3)' },
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

// ─── Run table (model selection) ─────────────────────────────────────────────

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
      <table className="w-full min-w-[720px] text-[13px]">
        <thead>
          <tr className="border-b border-ink-600/60">
            {['Run ID', 'Backbone', 'Val AUROC', 'Test AUROC', 'Test Specificity', 'Duration', 'Status'].map((h) => (
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
              <td className="px-4 py-3 font-mono text-slate-400">
                {run.test_auroc != null ? run.test_auroc.toFixed(4) : '—'}
              </td>
              <td className="px-4 py-3 font-mono text-slate-500">
                {run.test_specificity != null ? `${(run.test_specificity * 100).toFixed(1)}%` : '—'}
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

// ─── Architecture comparison (real runs from MLflow) ─────────────────────────

function ArchComparison({ runs }: { runs: MLflowRun[] }) {
  const scored = runs.filter((r) => r.test_auroc != null)
  if (scored.length === 0) return null

  const aurocVals = scored.map((r) => r.test_auroc!).filter((v) => v > 0)
  const MIN = Math.max(0, Math.min(...aurocVals) - 0.02)
  const MAX = Math.min(1, Math.max(...aurocVals) + 0.01)

  return (
    <div className="divide-y divide-ink-600/30">
      {scored.map((run, i) => {
        const barWidth = MAX > MIN ? (run.test_auroc! - MIN) / (MAX - MIN) : 0
        const isChampion = i === 0
        return (
          <div
            key={run.run_id}
            className={`flex items-center gap-4 px-6 py-4 ${isChampion ? 'bg-teal-400/[0.03]' : ''}`}
          >
            <div className="flex flex-1 items-center gap-3 min-w-0">
              {isChampion && (
                <span
                  className="shrink-0 rounded border border-teal-400/20 bg-teal-400/10 px-1.5 py-0.5 font-mono text-[9px] font-semibold uppercase tracking-wider text-teal-400"
                  title="Highest validation AUROC — test AUROC shown alongside is that run's held-out score, not necessarily the highest test AUROC in the list"
                >
                  CHAMPION (val)
                </span>
              )}
              <span className={`truncate text-[14px] ${isChampion ? 'font-medium text-slate-200' : 'text-slate-500'}`}>
                {run.backbone}
              </span>
            </div>
            <div className="flex items-center gap-3 shrink-0">
              <div className="text-right">
                <div className={`font-mono text-[13px] ${isChampion ? 'font-semibold text-teal-400' : 'text-slate-500'}`}>
                  {run.test_auroc!.toFixed(4)}
                </div>
                <div className="text-[10px] text-slate-700">test AUROC</div>
              </div>
              {run.test_specificity != null && (
                <div className="text-right hidden sm:block">
                  <div className="font-mono text-[12px] text-slate-500">
                    {(run.test_specificity * 100).toFixed(1)}%
                  </div>
                  <div className="text-[10px] text-slate-700">specificity</div>
                </div>
              )}
              <div className="hidden w-24 sm:block">
                <div className="h-[3px] overflow-hidden rounded-full bg-ink-600">
                  <motion.div
                    className="h-full rounded-full"
                    style={{
                      backgroundColor: isChampion ? '#00d4aa' : '#253548',
                      boxShadow: isChampion ? '0 0 6px rgba(0,212,170,0.4)' : 'none',
                    }}
                    initial={{ width: 0 }}
                    whileInView={{ width: `${barWidth * 100}%` }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.8, delay: i * 0.1, ease: 'easeOut' }}
                  />
                </div>
              </div>
            </div>
          </div>
        )
      })}
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

  const bestTestAuroc = stats?.runs?.[0]?.test_auroc ?? null
  const bestValAuroc = stats?.best_auroc ?? 0
  const totalRuns = stats?.total_runs ?? null
  const hasLiveRuns = loadState === 'success' && (stats?.runs?.length ?? 0) > 0

  const bestBackbone = stats?.runs?.[0]?.backbone
  let architectureLabel = '—'
  if (bestBackbone && bestBackbone !== 'unknown') {
    const match = bestBackbone.toLowerCase().match(/b(\d+)/)
    architectureLabel = match ? `B${match[1]}+Meta` : bestBackbone
  }

  const ttaLabel = ttaPasses != null ? `${ttaPasses}×` : '—'
  const ttaSub = ttaPasses != null ? 'Live from API' : 'Test-time augmentation'

  // Test AUROC of the champion (best-val) run when available — not necessarily the
  // highest test AUROC across all runs, since champion selection is by val AUROC.
  const displayAuroc = bestTestAuroc ?? bestValAuroc
  const displayAurocLabel = bestTestAuroc != null ? 'Champion Test AUROC' : 'Best Val AUROC'
  const displayAurocSub = bestTestAuroc != null
    ? (loadState === 'success' ? 'Live from MLflow (test)' : 'Held-out test set')
    : (loadState === 'success' ? 'Live from MLflow (val)' : 'Validated benchmark')

  const statCards = [
    {
      icon: BarChart3,
      label: displayAurocLabel,
      value: displayAuroc.toFixed(3),
      sub: displayAurocSub,
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

  const hasArchComparison = hasLiveRuns && (stats?.runs?.some((r) => r.test_auroc != null) ?? false)

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
              Experiments
            </p>
            <h2 className="text-[32px] font-bold tracking-tight text-slate-100">
              Model Selection
            </h2>
            <div className="mt-2 flex items-center gap-2.5">
              <p className="text-[15px] text-slate-400">
                {loadState === 'success'
                  ? hasLiveRuns
                    ? `Live MLflow data · ${stats!.runs.length} training runs tracked`
                    : 'MLflow reachable · No runs recorded yet'
                  : loadState === 'error'
                  ? 'MLflow offline · Showing static data'
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
            <p className="mt-2 text-[12px] text-slate-600">
              Val AUROC drove early stopping, checkpoint selection, and threshold calibration.
              Final numbers are from the held-out test set above.
            </p>
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

        {/* Architecture comparison — real runs from MLflow */}
        {hasArchComparison && (
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
                Held-out test AUROC &amp; specificity · runs from MLflow experiment
              </p>
            </div>
            <ArchComparison runs={stats!.runs} />
          </motion.div>
        )}

        {/* Live MLflow runs (model selection table) */}
        {hasLiveRuns && (
          <motion.div
            className="mb-8 overflow-hidden rounded-2xl border border-ink-600/70 bg-ink-800/50"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            <div className="flex items-center justify-between border-b border-ink-600/60 px-6 py-4">
              <div>
                <h3 className="text-[15px] font-semibold text-slate-200">Model Selection (Validation)</h3>
                <p className="mt-0.5 text-[12px] text-slate-600">
                  Runs sorted by val AUROC · used for early stopping &amp; checkpoint selection · final
                  evaluation on held-out test set
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
