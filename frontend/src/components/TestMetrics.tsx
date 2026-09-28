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

// ─── Confusion matrix as a single proportional bar ──────────────────────────

function ConfusionBar({ tp, fp, fn, tn }: { tp: number; fp: number; fn: number; tn: number }) {
  const total = tp + fp + fn + tn || 1
  const segments = [
    {
      key: 'TP',
      name: 'True Positive',
      n: tp,
      color: '#2f8a5c',
      desc: 'A cancer, correctly flagged',
    },
    {
      key: 'FN',
      name: 'False Negative',
      n: fn,
      color: '#c0392b',
      desc: 'A cancer that was missed',
    },
    {
      key: 'FP',
      name: 'False Positive',
      n: fp,
      color: '#c1683f',
      desc: 'Harmless, wrongly flagged',
    },
    {
      key: 'TN',
      name: 'True Negative',
      n: tn,
      color: '#9c8a80',
      desc: 'Harmless, correctly cleared',
    },
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
                {s.name} <span className="text-slate-500">· {s.n}</span>
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
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {[
          { label: 'Sensitivity', value: pct(pt.sensitivity) },
          { label: 'Specificity', value: pct(pt.specificity) },
          { label: 'PPV', value: pct(pt.ppv) },
          { label: 'NPV', value: pct(pt.tn / (pt.tn + pt.fn + 1e-8)) },
          { label: 'False positives', value: `${pt.fp.toFixed(0)} FP` },
          { label: 'Missed cancers', value: `${pt.fn.toFixed(0)} FN` },
        ].map(({ label, value }) => (
          <div key={label} className="text-center">
            <div className="font-mono text-[16px] font-semibold text-teal-400">{value}</div>
            <div className="mt-0.5 text-[10px] text-slate-600">{label}</div>
          </div>
        ))}
      </div>
      <p className="text-[11px] leading-relaxed text-slate-600">
        Sensitivity is the share of real cancers this cutoff catches. Specificity is the share of
        harmless moles it correctly clears. PPV is the share of its alarms that turn out to be real
        cancer, and NPV is the share of its clearances that turn out to be genuinely harmless. FP
        counts the harmless moles it would flag by mistake at this cutoff, and FN counts the real
        melanomas it would miss.
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
              Held out test evaluation
            </p>
            <h2 className="font-display text-[34px] font-semibold tracking-tight text-slate-100">
              Model performance and metrics
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
            {/* ── Hero: headline number + narrative, single column so each
                 explanation sits right under the number it explains ────────── */}
            <motion.div
              className="max-w-3xl border-b border-ink-600/40 pb-12 pt-6"
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
            >
              <h3 className="mb-1.5 text-[12px] font-semibold uppercase tracking-wide text-slate-300">
                AUROC
              </h3>
              <div className="font-mono text-[64px] font-bold leading-none text-teal-400">
                {fmt(auroc, 3)}
              </div>
              <div className="mt-2 font-mono text-[11px] uppercase tracking-wider text-slate-600">
                Area Under the ROC Curve (AUROC)
              </div>
              <p className="mt-3 text-[15px] leading-relaxed text-slate-400">
                Grab one malignant photo and one benign photo at random. This number is how often
                the model scores the malignant one higher. A coin flip gets 0.5, and a perfect
                model gets 1.0.
              </p>

              <h3 className="mb-1.5 mt-6 text-[12px] font-semibold uppercase tracking-wide text-slate-300">
                Confidence interval
              </h3>
              <div className="font-mono text-[13px] text-teal-400">
                95% confidence interval: [{fmt(auroc_lo, 3)}, {fmt(auroc_hi, 3)}]
              </div>
              <p className="mt-2 text-[15px] leading-relaxed text-slate-400">
                A confidence interval is a range, rather than a single number, that is likely to
                contain a model's true performance. It exists because any test score carries some
                uncertainty: this test set includes only {nPos} malignant photos, so evaluating the
                model on a different random sample of test photos would likely produce a somewhat
                different AUROC. Rather than treating {fmt(auroc, 3)} as an exact measurement, it
                should be interpreted as an estimate, with the model's true performance more
                precisely bounded between {fmt(auroc_lo, 3)} and {fmt(auroc_hi, 3)}. Even the lower
                bound of this range is well above 0.5, the score random guessing would achieve.
              </p>

              <h3 className="mb-1.5 mt-6 text-[12px] font-semibold uppercase tracking-wide text-slate-300">
                Test set composition
              </h3>
              <p className="text-[15px] leading-relaxed text-slate-400">
                {nTest.toLocaleString()} photos went into this test, and only{' '}
                <span className="font-semibold text-teal-400">{nPos}</span> of them are malignant
                ({pct(nPos / nTest)} prevalence). That imbalance is why the numbers below focus on
                catching the rare cancers rather than plain accuracy: a model that guessed
                "benign" every single time would still be right {pct(1 - nPos / nTest)} of the
                time and still be completely useless.
              </p>

              <h3 className="mb-1.5 mt-6 text-[12px] font-semibold uppercase tracking-wide text-slate-300">
                Sensitivity, specificity, and PPV
              </h3>
              <p className="text-[15px] leading-relaxed text-slate-400">
                These test photos were set aside before training and the model never saw them,
                so the results reflect how it performs on genuinely new cases. Of the melanomas
                in this set, it correctly identifies{' '}
                <span className="font-semibold text-teal-400">{pct(sensitivity)}</span>, a
                measure called sensitivity. Of the harmless moles, it correctly clears{' '}
                <span className="font-semibold text-teal-400">{pct(specificity)}</span>, a
                measure called specificity.
              </p>
              <p className="mt-2 text-[15px] leading-relaxed text-slate-400">
                When the model does raise an alarm, it is correct{' '}
                <span className="font-semibold text-teal-400">{pct(ppv)}</span> of the time.
                This figure is called positive predictive value, or PPV, and it is intentionally
                low: because melanoma is rare, the model is tuned to miss as few real cancers as
                possible, even at the cost of more false positives.
              </p>

              <h3 className="mb-1.5 mt-6 text-[12px] font-semibold uppercase tracking-wide text-slate-300">
                Patient-level data split
              </h3>
              <p className="text-[15px] leading-relaxed text-slate-400">
                Splits are made by <em>patient</em>, not by <em>image</em>, so the same mole can
                never appear in both the training and test sets. An earlier version of the
                pipeline split by image instead, which let 1,656 of the 1,657 test images leak
                into training and inflated the reported AUROC to a misleading{' '}
                <span className="text-slate-500">0.9355</span>. A
                continuous-integration check now fails the build if this ever recurs, and the{' '}
                {fmt(auroc, 3)} AUROC shown above reflects the corrected split.
              </p>

              <h3 className="mb-1.5 mt-6 text-[12px] font-semibold uppercase tracking-wide text-slate-300">
                Model architecture
              </h3>
              <p className="text-[15px] leading-relaxed text-slate-400">
                The part of the model that looks at the photo itself is{' '}
                <span className="font-semibold text-slate-200">{backbone}</span>, a convolutional
                neural network that was originally trained on 1.4 million everyday photos (a
                dataset called ImageNet) to recognize general shapes and textures. Starting from
                that pretrained network and fine tuning it on dermoscopy photos means it did not
                have to learn what an edge or a texture is from scratch, only how those patterns
                relate to melanoma.
              </p>

              <h3 className="mb-1.5 mt-6 text-[12px] font-semibold uppercase tracking-wide text-slate-300">
                Classification threshold
              </h3>
              <p className="text-[15px] leading-relaxed text-slate-400">
                The model does not produce a binary yes-or-no classification. It outputs a
                probability between 0 and 1, and a threshold determines the cutoff at which a
                result is flagged. That threshold is currently set to{' '}
                <span className="font-semibold text-teal-400">{threshold.toFixed(4)}</span>. It
                was determined by evaluating a range of candidate cutoffs on the validation set
                and selecting the lowest value that still identified at least 80% of true
                melanomas, since in a
                screening context a missed cancer carries a far greater cost than an unnecessary
                false positive. A higher threshold would reduce false positives but increase the
                number of missed cancers.
              </p>

              {data?.sweep && data.sweep.length > 0 && (
                <div className="mt-6">
                  <h3 className="text-[15px] font-semibold text-slate-200">Move the cutoff</h3>
                  <p className="mt-1 text-[13px] leading-relaxed text-slate-500">
                    Drag to see the trade-off directly: a lower cutoff catches more cancers but
                    flags more harmless moles, and a higher one does the reverse. The deployed
                    site uses the calibrated value above.
                  </p>
                  <div className="mt-6 max-w-xl">
                    <ThresholdSlider sweep={data.sweep} currentThreshold={threshold} />
                  </div>
                </div>
              )}

              <h3 className="mb-1.5 mt-6 text-[12px] font-semibold uppercase tracking-wide text-slate-300">
                Validation vs. test
              </h3>
              <p className="text-[15px] leading-relaxed text-slate-400">
                This model was trained and tuned on one portion of the data, referred to as the
                validation set, where it achieved a score of{' '}
                {valAuroc != null && (
                  <span className="font-semibold text-teal-400">{fmt(valAuroc, 3)} AUROC</span>
                )}
                . It was then evaluated once, at the conclusion of development, on a wholly
                separate portion known as the test set, where it achieved{' '}
                <span className="font-semibold text-teal-400">{fmt(auroc, 3)} AUROC</span>. A
                substantial decline between the two scores would indicate overfitting, a failure
                mode in which a model memorizes idiosyncrasies of the data it was tuned on rather
                than learning patterns that generalize to new photos. Because the test score
                remained close to the validation score, there is no evidence of overfitting here.
              </p>
            </motion.div>

            {/* ── Confusion matrix as a single bar ──────────────────────────── */}
            <motion.div
              className="border-b border-ink-600/40 py-12"
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4 }}
            >
              <h3 className="text-[15px] font-semibold text-slate-200">Test set outcomes</h3>
              <p className="mt-1 max-w-2xl text-[13px] leading-relaxed text-slate-500">
                The bar below classifies every photo in the test set into one of four outcomes:
                true positive, false negative, false positive, or true negative. Every statistic
                on this page is derived from these four counts.
              </p>
              <div className="mt-6">
                <ConfusionBar tp={tp} fp={fp} fn={fn} tn={tn} />
              </div>
              <p className="mt-6 max-w-2xl text-[13px] leading-relaxed text-slate-500">
                In practice, this means correctly identifying{' '}
                <span className="font-semibold text-teal-400">
                  {tp} of {tp + fn} real melanomas
                </span>{' '}
                at the cost of{' '}
                <span className="font-semibold text-teal-400">
                  {fp} unnecessary follow-up evaluations
                </span>
                , or approximately{' '}
                <span className="font-semibold text-teal-400">
                  {(fp / (tp + 1e-8)).toFixed(1)}× false positives for every cancer detected
                </span>
                . This trade-off is intentional: the threshold is calibrated to favor sensitivity,
                since in a screening context a missed cancer is far more costly than an
                unnecessary follow-up evaluation.
              </p>
            </motion.div>

            <SubgroupTable />
          </>
        )}
      </div>
    </section>
  )
}
