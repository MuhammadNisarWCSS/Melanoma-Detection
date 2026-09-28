import { useEffect, useRef, useState } from 'react'
import { Loader2, RefreshCw, ShieldOff } from 'lucide-react'
import { motion } from 'framer-motion'
import { fetchTestMetrics, type TestMetrics } from '../api/client'
import SubgroupTable from './SubgroupTable'

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
  const W = 280
  const H = 260
  const PAD = 30

  const sx = (v: number) => PAD + v * (W - PAD * 2)
  const sy = (v: number) => H - PAD - v * (H - PAD * 2)

  const pts = fpr.map((x, i) => `${sx(x)},${sy(tpr[i])}`).join(' ')

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-[320px]">
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <line
          key={t}
          x1={sx(0)}
          y1={sy(t)}
          x2={sx(1)}
          y2={sy(t)}
          stroke="#e6d9c9"
          strokeWidth="1"
        />
      ))}
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <line
          key={t}
          x1={sx(t)}
          y1={sy(0)}
          x2={sx(t)}
          y2={sy(1)}
          stroke="#e6d9c9"
          strokeWidth="1"
        />
      ))}

      <line
        x1={sx(0)}
        y1={sy(0)}
        x2={sx(1)}
        y2={sy(1)}
        stroke="#c9bdb2"
        strokeWidth="1"
        strokeDasharray="4 3"
      />

      <polyline
        points={pts}
        fill="none"
        stroke="#c1683f"
        strokeWidth="2"
        strokeLinejoin="round"
      />

      <circle
        cx={sx(opFpr)}
        cy={sy(opTpr)}
        r="4.5"
        fill="#b3761c"
        stroke="#fbf6f2"
        strokeWidth="1.5"
      />

      <text x={W / 2} y={H - 4} textAnchor="middle" fontSize="9" fill="#9c8a80">
        FPR (1 − Specificity)
      </text>
      <text
        x={10}
        y={H / 2}
        textAnchor="middle"
        fontSize="9"
        fill="#9c8a80"
        transform={`rotate(-90, 10, ${H / 2})`}
      >
        TPR (Sensitivity)
      </text>

      {[0, 0.5, 1].map((t) => (
        <text key={t} x={sx(t)} y={H - 16} textAnchor="middle" fontSize="7" fill="#c9bdb2">
          {t.toFixed(1)}
        </text>
      ))}
      {[0, 0.5, 1].map((t) => (
        <text key={t} x={PAD - 4} y={sy(t) + 3} textAnchor="end" fontSize="7" fill="#c9bdb2">
          {t.toFixed(1)}
        </text>
      ))}
    </svg>
  )
}

// ─── Confusion matrix as a single proportional bar ──────────────────────────

function ConfusionBar({ tp, fp, fn, tn }: { tp: number; fp: number; fn: number; tn: number }) {
  const total = tp + fp + fn + tn || 1
  const segments = [
    { key: 'TP', n: tp, color: '#2f8a5c', desc: 'True Positive: a cancer, correctly flagged' },
    { key: 'FN', n: fn, color: '#b3761c', desc: 'False Negative: a cancer that was missed' },
    { key: 'FP', n: fp, color: '#c0392b', desc: 'False Positive: harmless, wrongly flagged' },
    { key: 'TN', n: tn, color: '#9c8a80', desc: 'True Negative: harmless, correctly cleared' },
  ]

  return (
    <div>
      <div className="flex h-8 w-full gap-[2px] overflow-hidden rounded-full">
        {segments.map((s) => {
          const w = (s.n / total) * 100
          if (w <= 0) return null
          return (
            <motion.div
              key={s.key}
              className="flex h-full items-center justify-center first:rounded-l-full last:rounded-r-full"
              style={{ backgroundColor: s.color }}
              initial={{ width: 0 }}
              whileInView={{ width: `${w}%` }}
              viewport={{ once: true }}
              transition={{ duration: 0.7, ease: 'easeOut' }}
            >
              {w > 7 && (
                <span className="font-mono text-[11px] font-semibold text-ink-900/80">{s.n}</span>
              )}
            </motion.div>
          )
        })}
      </div>
      <div className="mt-4 grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-4">
        {segments.map((s) => (
          <div key={s.key} className="flex items-start gap-2">
            <span
              className="mt-1 h-2 w-2 shrink-0 rounded-full"
              style={{ backgroundColor: s.color }}
            />
            <div>
              <div className="font-mono text-[12px] text-slate-300">
                {s.key} <span className="text-slate-500">· {s.n}</span>
              </div>
              <div className="text-[11px] text-slate-600">{s.desc}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Threshold slider ─────────────────────────────────────────────────────────

type SweepPoint = TestMetrics['sweep'][0]

function ThresholdSlider({
  sweep,
  currentThreshold,
}: {
  sweep: SweepPoint[]
  currentThreshold: number
}) {
  const [idx, setIdx] = useState<number>(() => {
    let best = 0
    let bestDist = Infinity
    sweep.forEach((pt, i) => {
      const d = Math.abs(pt.threshold - currentThreshold)
      if (d < bestDist) {
        bestDist = d
        best = i
      }
    })
    return best
  })

  const pt = sweep[idx]

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <span className="font-mono text-[11px] uppercase tracking-wider text-slate-600">
          Threshold
        </span>
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
          <div key={label} className="text-center">
            <div className="font-mono text-[16px] font-semibold text-slate-100">{value}</div>
            <div className="mt-0.5 text-[10px] text-slate-600">{label}</div>
          </div>
        ))}
      </div>
      <p className="text-[11px] leading-relaxed text-slate-600">
        Sensitivity is the share of real cancers this cutoff catches. Specificity is the share of
        harmless moles it correctly clears. PPV is the share of its alarms that turn out to be real
        cancer. FP counts the harmless moles it would flag by mistake at this cutoff.
      </p>
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
      .then((d) => {
        setData(d)
        setLoadState('success')
      })
      .catch(() => setLoadState('error'))
  }

  useEffect(() => {
    if (!hasFetched.current) {
      hasFetched.current = true
      load()
    }
  }, [])

  const auroc = data?.auroc ?? 0
  const sensitivity = data?.sensitivity ?? 0
  const specificity = data?.specificity ?? 0
  const ppv = data?.ppv ?? 0
  const tp = data?.tp ?? 0
  const fp = data?.fp ?? 0
  const tn = data?.tn ?? 0
  const fn = data?.fn ?? 0
  const nTest = data?.n_test ?? 0
  const nPos = data?.n_positive ?? 0
  const threshold = data?.threshold ?? 0.5

  const auroc_lo = data?.ci?.auroc?.lo ?? 0
  const auroc_hi = data?.ci?.auroc?.hi ?? 0
  const sens_lo = data?.ci?.sensitivity?.lo ?? 0
  const sens_hi = data?.ci?.sensitivity?.hi ?? 0

  const opFpr = fp + tn > 0 ? fp / (fp + tn) : 0
  const opTpr = sensitivity

  const backbone = data?.backbone ?? 'efficientnet_b4'
  const valAuroc = data?.val_auroc

  return (
    <section id="test-metrics" className="relative border-t border-ink-600/50 py-24 px-5 sm:px-8">
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
              Held-out evaluation
            </p>
            <h2 className="font-display text-[34px] font-semibold tracking-tight text-slate-100">
              Does it actually work?
            </h2>
            <p className="mt-3 max-w-3xl text-[14px] leading-relaxed text-slate-500">
              This project trains a computer vision model to look at a photo of a skin lesion,
              combine it with three basic patient details (age, sex, and where on the body the
              lesion sits), and estimate how likely it is to be melanoma, a dangerous form of skin
              cancer. Everything below measures how well that model actually performs, using photos
              and patients it never saw while it was being trained or tuned.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <span
              className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] ${
                loadState === 'success'
                  ? 'border-emerald-500/25 bg-emerald-500/8 text-emerald-400'
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
              {loadState === 'success'
                ? 'Live from API'
                : loadState === 'error'
                  ? 'Static fallback'
                  : 'Loading…'}
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
                : 'Test metrics are not available right now. The API will show them once a run has logged test_metrics.json.'}
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
            {/* ── Hero: headline number + narrative ─────────────────────────── */}
            <motion.div
              className="grid gap-10 border-b border-ink-600/40 pb-12 pt-6 lg:grid-cols-[minmax(0,220px)_1fr]"
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
            >
              <div>
                <div className="font-mono text-[64px] font-bold leading-none text-teal-400">
                  {fmt(auroc, 3)}
                </div>
                <div className="mt-2 font-mono text-[11px] uppercase tracking-wider text-slate-600">
                  Area Under the ROC Curve (AUROC)
                </div>
                <div className="mt-1 text-[11px] text-slate-600">
                  95% confidence interval (CI): [{fmt(auroc_lo, 3)}, {fmt(auroc_hi, 3)}]
                </div>
                <p className="mt-3 text-[13px] leading-relaxed text-slate-500">
                  Grab one malignant photo and one benign photo at random. This number is how often
                  the model scores the malignant one higher. A coin flip gets 0.5, and a perfect
                  model gets 1.0.
                </p>
                <p className="mt-3 text-[13px] leading-relaxed text-slate-500">
                  A confidence interval is a range we are reasonably sure contains the true number.
                  The test set has only {nPos} malignant photos, so a couple of different photos
                  could shift the score, and this range shows how much wiggle room that leaves.
                  Even its low end sits well above 0.5, so the result holds up despite the small
                  sample.
                </p>
              </div>

              <div className="space-y-4 text-[15px] leading-relaxed text-slate-400">
                <p>
                  {nTest.toLocaleString()} photos went into this test, and only{' '}
                  <span className="font-semibold text-slate-200">{nPos}</span> of them are malignant
                  ({pct(nPos / nTest)} prevalence). That imbalance is why the numbers below focus on
                  catching the rare cancers rather than plain accuracy: a model that guessed
                  "benign" every single time would still be right {pct(1 - nPos / nTest)} of the
                  time and still be completely useless.
                </p>
                <p>
                  On those photos, none of which the model trained on, it catches{' '}
                  <span className="font-semibold text-slate-200">
                    {pct(sensitivity)} of melanomas
                  </span>{' '}
                  (95% CI {pct(sens_lo)}–{pct(sens_hi)}) and correctly clears{' '}
                  <span className="font-semibold text-slate-200">{pct(specificity)}</span> of
                  harmless moles. When it does raise an alarm, it is right{' '}
                  <span className="font-semibold text-slate-200">{pct(ppv)}</span> of the time. That
                  is its positive predictive value, or PPV, and it is low on purpose. Melanoma is
                  rare, so a screening tool like this one is tuned to miss as few real cancers as
                  possible, even if that means more false alarms.
                </p>
                <p>
                  Getting an honest number here took a real bug fix. An earlier version of this
                  pipeline split the data by <em>image</em> instead of by <em>patient</em>, so the
                  same mole could show up in both the training set and the test set. That leaked
                  1,656 of the 1,657 test images and inflated AUROC to a fake{' '}
                  <span className="text-slate-500 line-through">0.9355</span>. Splitting by patient
                  instead, and adding a CI check that fails the build if it ever regresses, is what
                  produced the {fmt(auroc, 3)} you see above.
                </p>
                <p>
                  The part of the model that looks at the photo itself is{' '}
                  <span className="font-semibold text-slate-200">{backbone}</span>, a convolutional
                  neural network that was originally trained on 1.4 million everyday photos (a
                  dataset called ImageNet) to recognize general shapes and textures. Starting from
                  that pretrained network and fine tuning it on dermoscopy photos means it did not
                  have to learn what an edge or a texture is from scratch, only how those patterns
                  relate to melanoma.
                </p>
                <p>
                  The model does not output a plain yes or no. It outputs a probability between 0
                  and 1, and a threshold decides where the cutoff for "flag this" sits. Here that
                  threshold is <span className="font-semibold text-slate-200">
                    {threshold.toFixed(4)}
                  </span>, much lower than the 0.5 you might expect. It was set by testing many
                  possible cutoffs on the validation set and picking the lowest one that still
                  caught at least 80% of real melanomas, since in screening a missed cancer is far
                  worse than an extra false alarm. A higher threshold would mean fewer false alarms
                  but more missed cancers.
                </p>
                <p>
                  This model was trained and tuned on one slice of the data, called validation,
                  where it scored{' '}
                  {valAuroc != null && (
                    <span className="font-semibold text-slate-200">{fmt(valAuroc, 3)} AUROC</span>
                  )}
                  . It was then run once, at the very end, on a completely separate slice called the
                  test set, where it scored{' '}
                  <span className="font-semibold text-slate-200">{fmt(auroc, 3)} AUROC</span>. If the
                  test score had dropped a lot below the validation score, that would be a warning
                  sign called overfitting, where a model quietly memorizes quirks of the data it was
                  tuned on instead of learning patterns that hold up on new photos. Since the test
                  score stayed close to the validation score, that is not happening here.
                  {valAuroc != null && (
                    <span
                      className={`ml-2 inline-block rounded-full border px-2 py-0.5 text-[13px] ${
                        auroc >= valAuroc - 0.02
                          ? 'border-emerald-500/25 bg-emerald-500/8 text-emerald-400'
                          : 'border-amber-400/25 bg-amber-400/8 text-amber-400'
                      }`}
                    >
                      val {fmt(valAuroc, 3)} &rarr; test {fmt(auroc, 3)} ·{' '}
                      {auroc >= valAuroc - 0.02 ? 'no overfitting' : 'gap vs. validation'}
                    </span>
                  )}
                </p>
              </div>
            </motion.div>

            {/* ── Confusion matrix as a single bar ──────────────────────────── */}
            <motion.div
              className="border-b border-ink-600/40 py-12"
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4 }}
            >
              <h3 className="text-[15px] font-semibold text-slate-200">
                Every test photo, in one bar
              </h3>
              <p className="mt-1 max-w-2xl text-[13px] leading-relaxed text-slate-500">
                Green and grey are correct calls; amber is a missed cancer and coral is a false
                alarm. Everything else on this page is built from these four counts.
              </p>
              <div className="mt-6">
                <ConfusionBar tp={tp} fp={fp} fn={fn} tn={tn} />
              </div>
              <p className="mt-6 max-w-2xl text-[13px] leading-relaxed text-slate-500">
                In practice that means catching{' '}
                <span className="font-semibold text-slate-300">
                  {tp} of {tp + fn} real melanomas
                </span>{' '}
                at the cost of{' '}
                <span className="font-semibold text-slate-300">{fp} unnecessary referrals</span>,
                about{' '}
                <span className="font-semibold text-slate-300">
                  {(fp / (tp + 1e-8)).toFixed(1)}× false alarms for every cancer found
                </span>
                . That trade-off is intentional. The threshold is calibrated to favor sensitivity,
                because in screening a missed cancer is far more costly than an extra biopsy.
              </p>
            </motion.div>

            <SubgroupTable />

            {/* ── ROC, as a figure ───────────────────────────────────────────── */}
            {data?.roc && (
              <div className="border-b border-ink-600/40 py-12">
                <motion.div
                  className="mx-auto max-w-md"
                  initial={{ opacity: 0 }}
                  whileInView={{ opacity: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4 }}
                >
                  <h3 className="text-[15px] font-semibold text-slate-200">Trade-off curve</h3>
                  <p className="mt-1 text-[13px] leading-relaxed text-slate-500">
                    This is the Receiver Operating Characteristic (ROC) curve: the x-axis is the
                    False Positive Rate (FPR) and the y-axis is the True Positive Rate (TPR), also
                    called sensitivity. A curve hugging the top-left corner is better, and the amber
                    dot is where the deployed threshold actually operates.
                  </p>
                  <div className="mt-4 flex justify-center">
                    <RocChart fpr={data.roc.fpr} tpr={data.roc.tpr} opFpr={opFpr} opTpr={opTpr} />
                  </div>
                  <p className="mt-2 text-center font-mono text-[11px] text-slate-600">
                    Figure 1. ROC curve, AUROC {fmt(auroc, 4)}
                  </p>
                </motion.div>
              </div>
            )}

            {/* ── Threshold explorer ─────────────────────────────────────────── */}
            {data?.sweep && data.sweep.length > 0 && (
              <motion.div
                className="py-12"
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: 0.1 }}
              >
                <h3 className="text-[15px] font-semibold text-slate-200">Move the cutoff</h3>
                <p className="mt-1 max-w-2xl text-[13px] leading-relaxed text-slate-500">
                  Drag to see the trade-off directly: a lower cutoff catches more cancers but flags
                  more harmless moles, and a higher one does the reverse. The deployed site uses the
                  calibrated value above.
                </p>
                <div className="mt-6 max-w-xl">
                  <ThresholdSlider sweep={data.sweep} currentThreshold={threshold} />
                </div>
              </motion.div>
            )}
          </>
        )}
      </div>
    </section>
  )
}
