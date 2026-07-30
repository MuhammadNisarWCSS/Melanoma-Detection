import { useEffect, useRef, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  RefreshCw,
  Shield,
  ShieldOff,
} from 'lucide-react'
import { motion } from 'framer-motion'
import { fetchTestMetrics, type TestMetrics } from '../api/client'

// ─── Helpers ─────────────────────────────────────────────────────────────────

function fmt(v: number, digits = 4) {
  return v.toFixed(digits)
}

function pct(v: number, digits = 1) {
  return `${(v * 100).toFixed(digits)}%`
}

// ─── Inline SVG charts ───────────────────────────────────────────────────────

function RocChart({
  fpr,
  tpr,
  opFpr,
  opTpr,
}: {
  fpr: number[]
  tpr: number[]
  opFpr: number
  opTpr: number
}) {
  const W = 260
  const H = 260
  const PAD = 30

  const sx = (v: number) => PAD + v * (W - PAD * 2)
  const sy = (v: number) => H - PAD - v * (H - PAD * 2)

  const pts = fpr.map((x, i) => `${sx(x)},${sy(tpr[i])}`).join(' ')

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-[260px]">
      {/* Grid lines */}
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <line
          key={t}
          x1={sx(0)} y1={sy(t)} x2={sx(1)} y2={sy(t)}
          stroke="#1e2d40" strokeWidth="1"
        />
      ))}
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <line
          key={t}
          x1={sx(t)} y1={sy(0)} x2={sx(t)} y2={sy(1)}
          stroke="#1e2d40" strokeWidth="1"
        />
      ))}

      {/* Chance diagonal */}
      <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(1)} stroke="#334155" strokeWidth="1" strokeDasharray="4 3" />

      {/* ROC curve */}
      <polyline points={pts} fill="none" stroke="#00d4aa" strokeWidth="1.8" strokeLinejoin="round" />

      {/* Operating point */}
      <circle cx={sx(opFpr)} cy={sy(opTpr)} r="4" fill="#f59e0b" stroke="#070b14" strokeWidth="1.5" />

      {/* Axes labels */}
      <text x={W / 2} y={H - 4} textAnchor="middle" fontSize="9" fill="#475569">FPR (1 − Specificity)</text>
      <text
        x={10}
        y={H / 2}
        textAnchor="middle"
        fontSize="9"
        fill="#475569"
        transform={`rotate(-90, 10, ${H / 2})`}
      >TPR (Sensitivity)</text>

      {/* Axis ticks */}
      {[0, 0.5, 1].map((t) => (
        <text key={t} x={sx(t)} y={H - 16} textAnchor="middle" fontSize="7" fill="#334155">
          {t.toFixed(1)}
        </text>
      ))}
      {[0, 0.5, 1].map((t) => (
        <text key={t} x={PAD - 4} y={sy(t) + 3} textAnchor="end" fontSize="7" fill="#334155">
          {t.toFixed(1)}
        </text>
      ))}
    </svg>
  )
}

function ReliabilityChart({
  meanPred,
  meanTrue,
}: {
  meanPred: number[]
  meanTrue: number[]
}) {
  const W = 260
  const H = 260
  const PAD = 30

  const sx = (v: number) => PAD + v * (W - PAD * 2)
  const sy = (v: number) => H - PAD - v * (H - PAD * 2)

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-[260px]">
      {/* Grid */}
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <line key={t} x1={sx(0)} y1={sy(t)} x2={sx(1)} y2={sy(t)} stroke="#1e2d40" strokeWidth="1" />
      ))}
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <line key={t} x1={sx(t)} y1={sy(0)} x2={sx(t)} y2={sy(1)} stroke="#1e2d40" strokeWidth="1" />
      ))}

      {/* Perfect calibration diagonal */}
      <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(1)} stroke="#334155" strokeWidth="1" strokeDasharray="4 3" />

      {/* Calibration bars */}
      {meanPred.map((x, i) => (
        <rect
          key={i}
          x={sx(x) - 5}
          y={sy(meanTrue[i])}
          width={10}
          height={sy(0) - sy(meanTrue[i])}
          fill="#00d4aa33"
          stroke="#00d4aa"
          strokeWidth="1"
        />
      ))}

      {/* Dots */}
      {meanPred.map((x, i) => (
        <circle key={i} cx={sx(x)} cy={sy(meanTrue[i])} r="3" fill="#00d4aa" />
      ))}

      <text x={W / 2} y={H - 4} textAnchor="middle" fontSize="9" fill="#475569">Mean predicted probability</text>
      <text
        x={10}
        y={H / 2}
        textAnchor="middle"
        fontSize="9"
        fill="#475569"
        transform={`rotate(-90, 10, ${H / 2})`}
      >Fraction of positives</text>
    </svg>
  )
}

// ─── Threshold slider ─────────────────────────────────────────────────────────

type SweepPoint = TestMetrics['sweep'][0]

function ThresholdSlider({ sweep, currentThreshold }: { sweep: SweepPoint[]; currentThreshold: number }) {
  const [idx, setIdx] = useState<number>(() => {
    let best = 0
    let bestDist = Infinity
    sweep.forEach((pt, i) => {
      const d = Math.abs(pt.threshold - currentThreshold)
      if (d < bestDist) { bestDist = d; best = i }
    })
    return best
  })

  const pt = sweep[idx]

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <span className="font-mono text-[11px] uppercase tracking-wider text-slate-600">Threshold</span>
        <span className="font-mono text-[13px] text-teal-400">{pt.threshold.toFixed(4)}</span>
      </div>
      <input
        type="range"
        min={0}
        max={sweep.length - 1}
        value={idx}
        onChange={(e) => setIdx(Number(e.target.value))}
        className="w-full accent-teal-400"
      />
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          { label: 'Sensitivity', value: pct(pt.sensitivity) },
          { label: 'Specificity', value: pct(pt.specificity) },
          { label: 'PPV', value: pct(pt.ppv) },
          { label: 'False alarms', value: `${pt.fp.toFixed(0)} FP` },
        ].map(({ label, value }) => (
          <div key={label} className="rounded-lg border border-ink-600/60 bg-ink-800/60 p-3 text-center">
            <div className="font-mono text-[15px] font-semibold text-slate-100">{value}</div>
            <div className="mt-0.5 text-[10px] text-slate-600">{label}</div>
          </div>
        ))}
      </div>
      {/* Mini confusion matrix */}
      <div className="mt-2 grid grid-cols-2 gap-1 text-center text-[11px]">
        <div className="rounded border border-emerald-500/20 bg-emerald-500/8 px-2 py-1.5 font-mono text-emerald-400">
          TP {pt.tp.toFixed(0)}
        </div>
        <div className="rounded border border-red-400/20 bg-red-400/8 px-2 py-1.5 font-mono text-red-400">
          FP {pt.fp.toFixed(0)}
        </div>
        <div className="rounded border border-amber-400/20 bg-amber-400/8 px-2 py-1.5 font-mono text-amber-400">
          FN {pt.fn.toFixed(0)}
        </div>
        <div className="rounded border border-slate-600/40 bg-slate-700/30 px-2 py-1.5 font-mono text-slate-400">
          TN {pt.tn.toFixed(0)}
        </div>
      </div>
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────

type LoadState = 'idle' | 'loading' | 'success' | 'error'

export default function TestMetricsSection() {
  const [loadState, setLoadState] = useState<LoadState>('idle')
  const [data, setData] = useState<TestMetrics | null>(null)
  const hasFetched = useRef(false)

  const load = () => {
    setLoadState('loading')
    fetchTestMetrics()
      .then((d) => { setData(d); setLoadState('success') })
      .catch(() => setLoadState('error'))
  }

  useEffect(() => {
    if (!hasFetched.current) {
      hasFetched.current = true
      load()
    }
  }, [])

  // ─── Static fallback card when API is unavailable ─────────────────────────
  const FALLBACK = {
    auroc: 0.0, auroc_lo: 0.0, auroc_hi: 0.0,
    pauc: 0.0, sensitivity: 0.0, sensitivity_lo: 0.0, sensitivity_hi: 0.0,
    specificity: 0.0, ppv: 0.0, npv: 0.0, ece: 0.0,
    tp: 0, fp: 0, tn: 0, fn: 0, n_test: 0, n_positive: 0,
  }

  const auroc = data?.auroc ?? FALLBACK.auroc
  const pauc = data?.pauc ?? FALLBACK.pauc
  const sensitivity = data?.sensitivity ?? FALLBACK.sensitivity
  const specificity = data?.specificity ?? FALLBACK.specificity
  const ppv = data?.ppv ?? FALLBACK.ppv
  const npv = data?.npv ?? FALLBACK.npv
  const ece = data?.ece ?? FALLBACK.ece
  const tp = data?.tp ?? FALLBACK.tp
  const fp = data?.fp ?? FALLBACK.fp
  const tn = data?.tn ?? FALLBACK.tn
  const fn = data?.fn ?? FALLBACK.fn
  const nTest = data?.n_test ?? FALLBACK.n_test
  const nPos = data?.n_positive ?? FALLBACK.n_positive
  const threshold = data?.threshold ?? 0.5

  const auroc_lo = data?.ci?.auroc?.lo ?? FALLBACK.auroc_lo
  const auroc_hi = data?.ci?.auroc?.hi ?? FALLBACK.auroc_hi
  const sens_lo = data?.ci?.sensitivity?.lo ?? FALLBACK.sensitivity_lo
  const sens_hi = data?.ci?.sensitivity?.hi ?? FALLBACK.sensitivity_hi
  const spec_lo = data?.ci?.specificity?.lo ?? null
  const spec_hi = data?.ci?.specificity?.hi ?? null

  // FPR at operating point = fp / (fp + tn)
  const opFpr = (fp + tn) > 0 ? fp / (fp + tn) : 0
  const opTpr = sensitivity

  const backbone = data?.backbone ?? 'efficientnet_b4'
  const valAuroc = data?.val_auroc

  const headlineCards = [
    {
      label: 'AUROC',
      value: fmt(auroc, 4),
      ci: `[${fmt(auroc_lo, 3)}, ${fmt(auroc_hi, 3)}]`,
      sub: 'Area under ROC curve',
      icon: Activity,
      highlight: true,
    },
    {
      label: 'pAUC ≥80% TPR',
      value: fmt(pauc, 4),
      ci: null,
      sub: 'ISIC 2020 competition metric',
      icon: Activity,
    },
    {
      label: 'Sensitivity',
      value: pct(sensitivity),
      ci: `[${pct(sens_lo, 1)}, ${pct(sens_hi, 1)}]`,
      sub: `${tp}/${tp + fn} cancers caught`,
      icon: Shield,
    },
    {
      label: 'Specificity',
      value: pct(specificity),
      ci: spec_lo != null ? `[${pct(spec_lo, 1)}, ${pct(spec_hi!, 1)}]` : null,
      sub: `${fp} false positives of ${fp + tn}`,
      icon: CheckCircle2,
    },
  ]

  return (
    <section
      id="test-metrics"
      className="relative border-t border-ink-600/50 py-24 px-5 sm:px-8"
    >
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <motion.div
          className="mb-10 flex flex-wrap items-start justify-between gap-4"
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.45 }}
        >
          <div>
            <p className="mb-2 font-mono text-[11px] uppercase tracking-[0.2em] text-teal-400">
              Held-out evaluation
            </p>
            <h2 className="text-[32px] font-bold tracking-tight text-slate-100">
              Test Set Results
            </h2>
            {loadState === 'success' && data ? (
              <>
                <p className="mt-1.5 text-[14px] text-slate-400">
                  {nTest.toLocaleString()} images &middot; {nPos} malignant &middot;{' '}
                  {pct(nPos / nTest)} prevalence &middot; threshold {threshold.toFixed(4)}
                  {backbone && ` \u00b7 ${backbone}`}
                </p>
                {valAuroc != null && (
                  <p className="mt-1 text-[12px] text-slate-500">
                    Val AUROC {fmt(valAuroc, 4)} &rarr; Test AUROC {fmt(auroc, 4)}{' '}
                    <span
                      className={auroc >= valAuroc - 0.02 ? 'text-emerald-400' : 'text-amber-400'}
                    >
                      {auroc >= valAuroc - 0.02 ? '(no overfitting)' : '(gap vs. validation)'}
                    </span>
                  </p>
                )}
              </>
            ) : (
              <p className="mt-1.5 text-[14px] text-slate-400">
                Generated by <span className="text-slate-300">scripts/evaluate.py</span> on the
                held-out test split and served from the API.
              </p>
            )}
          </div>

          <div className="flex items-center gap-3">
            <span
              className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] ${
                loadState === 'success'
                  ? 'border-emerald-500/25 bg-emerald-500/8 text-emerald-400'
                  : loadState === 'error'
                  ? 'border-amber-400/25 bg-amber-400/8 text-amber-400'
                  : 'border-amber-400/25 bg-amber-400/8 text-amber-400'
              }`}
            >
              {loadState === 'loading' ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    loadState === 'success' ? 'bg-emerald-500' : 'bg-amber-400'
                  }`}
                />
              )}
              {loadState === 'success' ? 'Live from API' : loadState === 'error' ? 'Static fallback' : 'Loading…'}
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

      {loadState !== 'success' || !data ? (
        <motion.div
          className="flex flex-col items-center gap-3 rounded-2xl border border-ink-600/70 bg-ink-800/40 px-6 py-16 text-center"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
        >
          <ShieldOff className="h-8 w-8 text-slate-600" />
          <p className="max-w-md text-[13px] text-slate-500">
            {loadState === 'loading'
              ? 'Loading test metrics…'
              : 'Test metrics are unavailable right now — the API returns them once a run has logged test_metrics.json.'}
          </p>
          {loadState === 'error' && (
            <button
              onClick={load}
              className="flex items-center gap-1.5 rounded-lg border border-ink-600/70 bg-ink-800/50 px-3 py-2 text-[12px] text-slate-500 transition-colors hover:border-ink-500 hover:text-slate-300"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              Retry
            </button>
          )}
        </motion.div>
      ) : (
      <>
        {/* Honest-data disclaimer */}
        <motion.div
          className="mb-8 flex gap-3 rounded-xl border border-amber-400/15 bg-amber-400/5 px-4 py-3"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
        >
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400/70" />
          <p className="text-[12px] leading-relaxed text-slate-500">
            Validation metrics are selection-biased — they guided early stopping, checkpoint selection, and
            threshold calibration. These test numbers come from{' '}
            <span className="text-slate-400">data/processed/test.csv</span>, which no part of training or
            calibration has seen. With only {nPos} malignant cases the confidence intervals are wide; see
            below.
          </p>
        </motion.div>

        {/* Headline stat cards */}
        <div className="mb-10 grid grid-cols-2 gap-4 lg:grid-cols-4">
          {headlineCards.map(({ label, value, ci, sub, icon: Icon, highlight }, i) => (
            <motion.div
              key={label}
              className={`rounded-2xl border bg-ink-800/60 p-5 ${
                highlight ? 'border-teal-400/30' : 'border-ink-600/70'
              }`}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: i * 0.07 }}
            >
              <div className="mb-3 flex items-center justify-between">
                <span className="font-mono text-[10px] uppercase tracking-wider text-slate-600">{label}</span>
                <Icon className={`h-3.5 w-3.5 ${highlight ? 'text-teal-400/70' : 'text-slate-600'}`} />
              </div>
              <div
                className={`font-mono text-[28px] font-bold leading-none ${
                  highlight ? 'text-teal-400' : 'text-slate-100'
                }`}
              >
                {value}
              </div>
              {ci && (
                <div className="mt-1 font-mono text-[10px] text-slate-600">
                  95% CI {ci}
                </div>
              )}
              <div className="mt-1.5 text-[11px] text-slate-600">{sub}</div>
            </motion.div>
          ))}
        </div>

        {/* Confusion matrix + clinical framing */}
        <div className="mb-8 grid gap-6 lg:grid-cols-2">
          <motion.div
            className="rounded-2xl border border-ink-600/70 bg-ink-800/50 p-6"
            initial={{ opacity: 0, x: -12 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.4 }}
          >
            <h3 className="mb-1 text-[15px] font-semibold text-slate-200">Confusion Matrix</h3>
            <p className="mb-4 text-[12px] text-slate-600">
              At calibrated threshold {threshold.toFixed(4)} &middot; target sensitivity ≥80%
            </p>
            <div className="grid grid-cols-2 gap-2 text-center text-[13px]">
              <div className="rounded-xl border border-emerald-500/25 bg-emerald-500/8 p-4">
                <div className="font-mono text-[26px] font-bold text-emerald-400">{tp}</div>
                <div className="mt-1 text-[10px] font-medium uppercase tracking-wider text-emerald-400/70">TP</div>
                <div className="mt-0.5 text-[10px] text-slate-600">True positive</div>
              </div>
              <div className="rounded-xl border border-red-400/25 bg-red-400/8 p-4">
                <div className="font-mono text-[26px] font-bold text-red-400">{fp}</div>
                <div className="mt-1 text-[10px] font-medium uppercase tracking-wider text-red-400/70">FP</div>
                <div className="mt-0.5 text-[10px] text-slate-600">False alarm</div>
              </div>
              <div className="rounded-xl border border-amber-400/25 bg-amber-400/8 p-4">
                <div className="font-mono text-[26px] font-bold text-amber-400">{fn}</div>
                <div className="mt-1 text-[10px] font-medium uppercase tracking-wider text-amber-400/70">FN</div>
                <div className="mt-0.5 text-[10px] text-slate-600">Missed cancer</div>
              </div>
              <div className="rounded-xl border border-slate-600/40 bg-slate-700/20 p-4">
                <div className="font-mono text-[26px] font-bold text-slate-300">{tn}</div>
                <div className="mt-1 text-[10px] font-medium uppercase tracking-wider text-slate-500">TN</div>
                <div className="mt-0.5 text-[10px] text-slate-600">True negative</div>
              </div>
            </div>
          </motion.div>

          <motion.div
            className="rounded-2xl border border-ink-600/70 bg-ink-800/50 p-6"
            initial={{ opacity: 0, x: 12 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.4 }}
          >
            <h3 className="mb-4 text-[15px] font-semibold text-slate-200">Clinical Framing</h3>
            <div className="space-y-3 text-[13px]">
              <div className="flex items-center justify-between border-b border-ink-600/40 pb-2">
                <span className="text-slate-400">PPV (precision)</span>
                <span className="font-mono font-semibold text-slate-200">{pct(ppv)}</span>
              </div>
              <div className="flex items-center justify-between border-b border-ink-600/40 pb-2">
                <span className="text-slate-400">NPV</span>
                <span className="font-mono font-semibold text-slate-200">{pct(npv)}</span>
              </div>
              <div className="flex items-center justify-between border-b border-ink-600/40 pb-2">
                <span className="text-slate-400">ECE (calibration)</span>
                <span className={`font-mono font-semibold ${ece < 0.05 ? 'text-emerald-400' : 'text-amber-400'}`}>
                  {fmt(ece, 4)}
                </span>
              </div>
              <div className="flex items-center justify-between border-b border-ink-600/40 pb-2">
                <span className="text-slate-400">False alarms per cancer</span>
                <span className="font-mono font-semibold text-slate-200">
                  {(fp / (tp + 1e-8)).toFixed(1)}×
                </span>
              </div>
              <p className="mt-3 rounded-lg border border-ink-600/50 bg-ink-700/40 p-3 text-[12px] leading-relaxed text-slate-400">
                Catches <span className="text-slate-200">{tp} of {tp + fn} melanomas</span> at the cost of{' '}
                <span className="text-slate-200">{fp} benign referrals</span>. PPV ({pct(ppv)}) is low because
                prevalence is only {pct(nPos / nTest)} — expected behaviour in screening contexts.
              </p>
            </div>
          </motion.div>
        </div>

        {/* ROC + Reliability diagrams */}
        {data?.roc && data?.reliability && (
          <div className="mb-8 grid gap-6 lg:grid-cols-2">
            <motion.div
              className="rounded-2xl border border-ink-600/70 bg-ink-800/50 p-6"
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4 }}
            >
              <h3 className="mb-0.5 text-[15px] font-semibold text-slate-200">ROC Curve</h3>
              <p className="mb-4 text-[12px] text-slate-600">
                AUROC {fmt(auroc, 4)} · amber dot = operating point
              </p>
              <div className="flex justify-center">
                <RocChart
                  fpr={data.roc.fpr}
                  tpr={data.roc.tpr}
                  opFpr={opFpr}
                  opTpr={opTpr}
                />
              </div>
            </motion.div>

            <motion.div
              className="rounded-2xl border border-ink-600/70 bg-ink-800/50 p-6"
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: 0.05 }}
            >
              <h3 className="mb-0.5 text-[15px] font-semibold text-slate-200">Reliability Diagram</h3>
              <p className="mb-4 text-[12px] text-slate-600">
                ECE {fmt(ece, 4)} · dashed = perfect calibration
              </p>
              <div className="flex justify-center">
                <ReliabilityChart
                  meanPred={data.reliability.mean_pred}
                  meanTrue={data.reliability.mean_true}
                />
              </div>
            </motion.div>
          </div>
        )}

        {/* Threshold slider */}
        {data?.sweep && data.sweep.length > 0 && (
          <motion.div
            className="rounded-2xl border border-ink-600/70 bg-ink-800/50 p-6"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.4, delay: 0.1 }}
          >
            <h3 className="mb-0.5 text-[15px] font-semibold text-slate-200">Threshold Explorer</h3>
            <p className="mb-5 text-[12px] text-slate-600">
              Drag to see how sensitivity/specificity/PPV trade off at different decision thresholds.
            </p>
            <ThresholdSlider sweep={data.sweep} currentThreshold={threshold} />
          </motion.div>
        )}
      </>
      )}
      </div>
    </section>
  )
}
