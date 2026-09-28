import { useEffect, useState } from 'react'
import {
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  RefreshCw,
  Image as ImageIcon,
  UserRound,
  Scan,
  BrainCircuit,
  Merge,
  ArrowDown,
  Percent,
} from 'lucide-react'
import { motion } from 'framer-motion'
import {
  fetchMLflowStats,
  fetchApiMetadata,
  fetchTestMetrics,
  type MLflowStats,
  type MLflowRun,
} from '../api/client'

// ─── Architecture diagram ─────────────────────────────────────────────────────

function FlowStep({
  icon: Icon,
  title,
  desc,
}: {
  icon: typeof ImageIcon
  title: string
  desc: string
}) {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-ink-600/70 bg-ink-800/60 p-4">
      <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-teal-400/10">
        <Icon className="h-4 w-4 text-teal-500" />
      </div>
      <div>
        <div className="text-[13px] font-semibold text-slate-200">{title}</div>
        <div className="mt-0.5 text-[12px] leading-relaxed text-slate-500">{desc}</div>
      </div>
    </div>
  )
}

function FlowArrow() {
  return (
    <div className="flex justify-center py-1">
      <ArrowDown className="h-4 w-4 text-slate-600" />
    </div>
  )
}

function ArchDiagram() {
  return (
    <div className="mx-auto max-w-2xl">
      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <FlowStep
            icon={ImageIcon}
            title="Dermoscopy photo"
            desc="A close-up photo of the skin lesion, 384 by 384 pixels."
          />
          <FlowArrow />
          <FlowStep
            icon={Scan}
            title="EfficientNet-B4"
            desc="A general purpose image network that turns the photo into a list of 1,792 numbers describing what it sees."
          />
        </div>
        <div>
          <FlowStep
            icon={UserRound}
            title="Patient details"
            desc="Age, sex, and where on the body the lesion is."
          />
          <FlowArrow />
          <FlowStep
            icon={BrainCircuit}
            title="Small neural network"
            desc="A much smaller network that turns those three fields into a list of 32 numbers."
          />
        </div>
      </div>

      <FlowArrow />

      <FlowStep
        icon={Merge}
        title="Fusion layers"
        desc="The two lists of numbers are joined into one and passed through a couple more layers, so the final score can weigh what the photo shows alongside who the patient is."
      />

      <FlowArrow />

      <FlowStep
        icon={Percent}
        title="Malignancy probability"
        desc="A single number from 0 to 1: the model's estimate of how likely the lesion is to be melanoma."
      />
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
            {[
              'Run ID',
              'Backbone',
              'Val AUROC',
              'Test AUROC',
              'Test Specificity',
              'Duration',
              'Status',
            ].map((h) => (
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
                <span
                  className={`font-mono ${i === 0 ? 'font-semibold text-teal-400' : 'text-slate-400'}`}
                >
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
                  title="Highest validation AUROC. The test AUROC shown alongside is that run's held-out score, not necessarily the highest test AUROC in the list."
                >
                  CHAMPION (val)
                </span>
              )}
              <span
                className={`truncate text-[14px] ${isChampion ? 'font-medium text-slate-200' : 'text-slate-500'}`}
              >
                {run.backbone}
              </span>
            </div>
            <div className="flex items-center gap-3 shrink-0">
              <div className="text-right">
                <div
                  className={`font-mono text-[13px] ${isChampion ? 'font-semibold text-teal-400' : 'text-slate-500'}`}
                >
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
                      backgroundColor: isChampion ? '#c1683f' : '#d8c2a8',
                      boxShadow: isChampion ? '0 0 6px rgba(193,104,63,0.4)' : 'none',
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
  const [deployedAuroc, setDeployedAuroc] = useState<number | null>(null)
  const [deployedBackbone, setDeployedBackbone] = useState<string | null>(null)

  const load = () => {
    setLoadState('loading')

    fetchApiMetadata()
      .then((m) => setTtaPasses(m.tta_passes))
      .catch(() => {
        /* keep null — will show fallback */
      })

    // The deployed model's own AUROC — the same number shown in "Does it
    // actually work?" above — so this section can never quote a stale
    // per-run metric (e.g. a test AUROC logged before the patient-disjoint
    // split fix) that contradicts it.
    fetchTestMetrics()
      .then((m) => {
        setDeployedAuroc(m.auroc)
        setDeployedBackbone(m.backbone)
      })
      .catch(() => {
        /* keep null — will fall back to MLflow's best val AUROC */
      })

    fetchMLflowStats()
      .then((s) => {
        setStats(s)
        setLoadState('success')
      })
      .catch(() => setLoadState('error'))
  }

  useEffect(() => {
    load()
  }, [])

  const bestValAuroc = stats?.best_auroc ?? 0
  const totalRuns = stats?.total_runs ?? null
  const hasLiveRuns = loadState === 'success' && (stats?.runs?.length ?? 0) > 0

  const bestBackbone = deployedBackbone ?? stats?.runs?.[0]?.backbone
  let architectureLabel = '—'
  if (bestBackbone && bestBackbone !== 'unknown') {
    const match = bestBackbone.toLowerCase().match(/b(\d+)/)
    architectureLabel = match ? `B${match[1]}+Meta` : bestBackbone
  }

  const ttaLabel = ttaPasses != null ? `${ttaPasses}×` : '—'

  // The deployed model's held-out test AUROC when available (matches the Test Set
  // Results section exactly); falls back to the best tracked validation AUROC.
  const displayAuroc = deployedAuroc ?? bestValAuroc
  const displayAurocLabel = deployedAuroc != null ? 'Deployed Test AUROC' : 'Best Val AUROC'

  const hasArchComparison = hasLiveRuns && (stats?.runs?.some((r) => r.test_auroc != null) ?? false)

  return (
    <section id="stats" className="relative border-t border-ink-600/50 py-24 px-5 sm:px-8">
      <div className="mx-auto max-w-5xl">
        {/* Header */}
        <motion.div
          className="mb-4 flex flex-wrap items-start justify-between gap-4"
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.45 }}
        >
          <div>
            <p className="mb-2 font-mono text-[11px] uppercase tracking-[0.2em] text-teal-400">
              Experiments
            </p>
            <h2 className="font-display text-[34px] font-semibold tracking-tight text-slate-100">
              How the model was chosen
            </h2>
          </div>

          <div className="flex items-center gap-3">
            <span
              className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] ${
                loadState === 'success'
                  ? 'border-emerald-500/25 bg-emerald-500/8 text-emerald-400'
                  : loadState === 'error'
                    ? 'border-slate-700 bg-ink-800/60 text-slate-600'
                    : 'border-amber-400/25 bg-amber-400/8 text-amber-400'
              }`}
            >
              {loadState === 'loading' ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    loadState === 'success'
                      ? 'bg-emerald-500 animate-pulse'
                      : loadState === 'error'
                        ? 'bg-slate-700'
                        : 'bg-amber-400 animate-pulse'
                  }`}
                />
              )}
              {loadState === 'success'
                ? 'Live'
                : loadState === 'error'
                  ? 'Offline'
                  : 'Connecting'}
            </span>
            {loadState === 'error' && (
              <button
                onClick={load}
                className="flex items-center gap-1.5 rounded-lg border border-ink-600/70 bg-ink-800/50 px-3 py-2 text-[12px] text-slate-500 transition-colors hover:border-ink-500 hover:text-slate-300"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                Retry
              </button>
            )}
          </div>
        </motion.div>

        {/* Narrative summary — replaces the old boxed stat cards */}
        <motion.div
          className="border-b border-ink-600/40 pb-10 pt-2 text-[15px] leading-relaxed text-slate-400"
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
        >
          <p>
            {totalRuns != null ? (
              <>
                <span className="font-semibold text-slate-200">{totalRuns} runs</span> were tracked
                in MLflow (a tool that logs every training attempt so results can be compared
                honestly later), each one{' '}
              </>
            ) : (
              'A series of runs were tracked in MLflow, each one '
            )}
            testing a different backbone, learning rate, or loss setting. A backbone is the core
            neural network that does the actual image recognition, and it can be swapped out like a
            part, which is exactly how these runs differ. The champion is a{' '}
            <span className="font-semibold text-slate-200">{architectureLabel}</span> network,
            shorthand for an EfficientNet backbone (the number after B marks how large that network
            is) fused with the patient metadata branch, and it reached{' '}
            <span className="font-semibold text-teal-500">
              {displayAuroc.toFixed(3)} {displayAurocLabel.toLowerCase()}
            </span>
            . At inference it runs{' '}
            <span className="font-semibold text-slate-200">
              {ttaLabel} test-time augmentation (TTA)
            </span>
            , averaging the score across flips and rotations of the same photo so a prediction does
            not depend on how the picture happened to be oriented.
          </p>
          <p className="mt-3 text-[13px] text-slate-500">
            While a model trains, validation AUROC (the score on data set aside for tuning, not for
            the final grade) does three jobs: it tells training when to stop early instead of
            running a fixed number of passes and overfitting, it decides which pass through the data
            gets saved as the final checkpoint, and it is what the classification threshold gets
            calibrated against. The final numbers reported above come from the held-out test set
            instead, a slice none of that tuning ever touched.
          </p>
        </motion.div>

        {/* Architecture comparison — real runs from MLflow */}
        {hasArchComparison && (
          <motion.div
            className="border-b border-ink-600/40 py-10"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            <h3 className="text-[15px] font-semibold text-slate-200">Comparing architectures</h3>
            <p className="mt-1 max-w-2xl text-[13px] leading-relaxed text-slate-500">
              Different model designs, all scored on the same held-out photos. Higher AUROC means
              better ranking of malignant vs. benign; higher specificity means fewer false alarms.
            </p>
            <div className="mt-5 -mx-2">
              <ArchComparison runs={stats!.runs} />
            </div>
          </motion.div>
        )}

        {/* Live MLflow runs (model selection table) */}
        {hasLiveRuns && (
          <motion.div
            className="border-b border-ink-600/40 py-10"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="text-[15px] font-semibold text-slate-200">Every experiment</h3>
                <p className="mt-1 text-[13px] leading-relaxed text-slate-500">
                  Sorted by validation AUROC, which drove checkpoint selection; test AUROC and
                  specificity are the honest, held-out numbers.
                </p>
              </div>
              <span className="font-mono text-[11px] text-teal-500 border border-teal-400/30 rounded px-2 py-0.5">
                {stats!.runs.length} runs
              </span>
            </div>
            <div className="mt-5 overflow-hidden rounded-xl border border-ink-600/60">
              <RunTable runs={stats!.runs} />
            </div>
          </motion.div>
        )}

        {/* Architecture diagram */}
        <motion.div
          className="pt-10"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <h3 className="text-[15px] font-semibold text-slate-200">Under the hood</h3>
          <p className="mt-1 max-w-2xl text-[13px] leading-relaxed text-slate-500">
            Here is the path a photo and a patient's details take to become one score.
          </p>
          <div className="mt-5">
            <ArchDiagram />
          </div>
          <p className="mt-5 max-w-2xl text-[13px] leading-relaxed text-slate-500">
            A few more details on how training itself works: it uses focal loss, a version of the
            usual error measure that pays extra attention to the rare melanoma cases instead of
            letting the far more common benign cases dominate, the AdamW optimizer to adjust the
            network's weights after each batch of photos, and a learning rate schedule that
            gradually takes smaller steps as training goes on so it settles instead of overshooting.
          </p>
        </motion.div>
      </div>
    </section>
  )
}
