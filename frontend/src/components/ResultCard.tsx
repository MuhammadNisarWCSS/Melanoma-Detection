import { AlertTriangle, CheckCircle, Info, ChevronDown, ChevronUp } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { useState } from 'react'
import type { PredictResponse } from '../api/client'

interface Props {
  result: PredictResponse
  imageSrc: string | null
  inputs?: { age: number; sex: string; site: string }
}

function ProbabilityGauge({
  probability,
  isMalignant,
}: {
  probability: number
  isMalignant: boolean
}) {
  const radius = 54
  const stroke = 7
  const normalizedR = radius - stroke / 2
  const circumference = 2 * Math.PI * normalizedR
  const offset = circumference - probability * circumference
  const color = isMalignant ? '#c0392b' : '#c1683f'
  const pct = Math.round(probability * 100)

  return (
    <div className="flex flex-col items-center gap-2">
      <svg
        width={radius * 2 + stroke}
        height={radius * 2 + stroke}
        viewBox={`0 0 ${radius * 2 + stroke} ${radius * 2 + stroke}`}
      >
        {/* Track */}
        <circle
          cx={radius + stroke / 2}
          cy={radius + stroke / 2}
          r={normalizedR}
          fill="none"
          stroke="#1e2d3d"
          strokeWidth={stroke}
        />
        {/* Progress */}
        <circle
          cx={radius + stroke / 2}
          cy={radius + stroke / 2}
          r={normalizedR}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform={`rotate(-90 ${radius + stroke / 2} ${radius + stroke / 2})`}
          style={{
            filter: `drop-shadow(0 0 8px ${color}55)`,
            transition: 'stroke-dashoffset 1s ease-out',
          }}
        />
        {/* Label */}
        <text
          x={radius + stroke / 2}
          y={radius + stroke / 2 - 6}
          textAnchor="middle"
          dominantBaseline="middle"
          fill={color}
          fontSize="22"
          fontWeight="700"
          fontFamily="JetBrains Mono, monospace"
        >
          {pct}%
        </text>
        <text
          x={radius + stroke / 2}
          y={radius + stroke / 2 + 13}
          textAnchor="middle"
          fill="#9c8a80"
          fontSize="9"
          fontFamily="Inter, sans-serif"
          letterSpacing="1"
        >
          MALIGNANCY
        </text>
      </svg>
    </div>
  )
}

function DetailRow({
  term,
  value,
  note,
  warn = false,
}: {
  term: string
  value: string
  note: string
  warn?: boolean
}) {
  return (
    <div className="px-5 py-3.5">
      <div className="flex items-baseline justify-between gap-4">
        <dt className="text-slate-500">{term}</dt>
        <dd className={`text-right font-medium ${warn ? 'text-amber-400' : 'text-slate-200'}`}>
          {value}
        </dd>
      </div>
      <p className="mt-1 text-[11px] leading-snug text-slate-600">{note}</p>
    </div>
  )
}

export default function ResultCard({ result, imageSrc, inputs }: Props) {
  const [showGradcam, setShowGradcam] = useState(false)
  const [showDetails, setShowDetails] = useState(false)
  const isMalignant = result.label === 1
  const isHighUncertainty = result.tta_std > 0.1
  const isOod = Boolean(result.out_of_distribution)

  return (
    <motion.div
      className="space-y-4 animate-fade-in"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: 'easeOut' }}
    >
      {/* ── OOD Warning (takes precedence over a confident benign/malignant banner) ── */}
      {isOod && (
        <motion.div
          initial={{ opacity: 0, scale: 0.97 }}
          animate={{ opacity: 1, scale: 1 }}
          className="flex items-start gap-3 rounded-2xl border border-amber-400/30 bg-amber-400/8 p-5"
        >
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-400" />
          <div>
            <div className="text-[15px] font-semibold text-amber-300">
              Image does not look like a dermoscopy photo
            </div>
            <p className="mt-1 text-[13px] leading-relaxed text-slate-400">
              This upload sits outside the model&apos;s training distribution (contact dermatoscope
              images from ISIC 2020). The {result.label_str} label below is unreliable — clinical
              photos, screenshots, and heavily recompressed web images are outside the intended use.
              {result.ood_distance != null && (
                <>
                  {' '}
                  <span className="font-mono text-slate-500">
                    distance={result.ood_distance.toFixed(1)}
                  </span>
                </>
              )}
            </p>
          </div>
        </motion.div>
      )}

      {/* ── Diagnosis Banner ── */}
      <div
        className={`flex items-center gap-4 rounded-2xl border p-5 ${
          isOod
            ? 'border-ink-500/40 bg-ink-800/60 opacity-70'
            : isMalignant
              ? 'border-red-500/25 bg-red-500/6'
              : 'border-emerald-500/25 bg-emerald-500/6'
        }`}
      >
        <div
          className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-xl border ${
            isOod
              ? 'border-ink-500/40 bg-ink-700/40'
              : isMalignant
                ? 'border-red-500/30 bg-red-500/10'
                : 'border-emerald-500/30 bg-emerald-500/10'
          }`}
        >
          {isMalignant ? (
            <AlertTriangle className={`h-6 w-6 ${isOod ? 'text-slate-500' : 'text-red-400'}`} />
          ) : (
            <CheckCircle className={`h-6 w-6 ${isOod ? 'text-slate-500' : 'text-emerald-400'}`} />
          )}
        </div>
        <div>
          <div
            className={`text-[22px] font-bold uppercase tracking-[0.1em] ${
              isOod ? 'text-slate-400' : isMalignant ? 'text-red-400' : 'text-emerald-400'
            }`}
          >
            {result.label_str}
          </div>
          <div className="mt-0.5 text-[13px] text-slate-400">
            {isOod
              ? 'Label shown for transparency — do not act on it'
              : isMalignant
                ? 'Malignant features detected — clinical review recommended'
                : 'No significant malignant features identified'}
          </div>
        </div>
      </div>

      {/* ── Probability ── */}
      <div className="flex flex-col items-center justify-center rounded-2xl border border-ink-600/70 bg-ink-800/70 py-6 px-4">
        <div className="mb-3 font-mono text-[10px] uppercase tracking-widest text-slate-600">
          Probability
        </div>
        <ProbabilityGauge probability={result.probability} isMalignant={isMalignant} />
      </div>

      {/* ── Technical details (collapsed by default) ── */}
      <div className="overflow-hidden rounded-2xl border border-ink-600/70 bg-ink-800/70">
        <button
          type="button"
          onClick={() => setShowDetails((p) => !p)}
          aria-expanded={showDetails}
          className="flex w-full items-center justify-between px-5 py-3.5 text-left transition-colors hover:bg-ink-700/40"
        >
          <div>
            <span className="text-[13px] font-medium text-slate-300">
              How this score was worked out
            </span>
            <span className="ml-3 text-[11px] text-slate-600">Optional technical details</span>
          </div>
          {showDetails ? (
            <ChevronUp className="h-4 w-4 text-slate-600" />
          ) : (
            <ChevronDown className="h-4 w-4 text-slate-600" />
          )}
        </button>
        <AnimatePresence>
          {showDetails && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.25 }}
              className="overflow-hidden"
            >
              <dl className="divide-y divide-ink-600/40 border-t border-ink-600/50 text-[12px]">
                <DetailRow
                  term="Score vs. cutoff"
                  value={`${result.probability.toFixed(3)} ${
                    result.probability >= result.threshold_used ? '≥' : '<'
                  } ${result.threshold_used.toFixed(3)}`}
                  note="The cutoff is set so the model catches about 80% of melanomas in testing. A score at or above it is flagged."
                />
                {result.n_views != null && result.views_flagged != null && (
                  <DetailRow
                    term="Views flagged"
                    value={`${result.views_flagged} of ${result.n_views}`}
                    note="Your photo is checked 8 times, flipped and rotated. The score is the average. If the views split, treat the result with more caution."
                    warn={isHighUncertainty}
                  />
                )}
                <DetailRow
                  term="Photo check"
                  value={
                    result.out_of_distribution == null
                      ? 'Not checked'
                      : isOod
                        ? 'Unusual photo'
                        : 'Looks like a dermoscopy image'
                  }
                  note="Compares your photo with the kind of images the model was trained on."
                  warn={isOod}
                />
                {inputs && (
                  <DetailRow
                    term="Details used"
                    value={`Age ${inputs.age}, ${inputs.sex.toLowerCase()}, ${inputs.site.toLowerCase()}`}
                    note="Changing these can change the score."
                  />
                )}
                <p className="px-5 py-3.5 text-[11px] leading-relaxed text-slate-600">
                  In testing this model caught about 77% of melanomas and raised many false alarms,
                  so a result here is a nudge, not an answer.
                </p>
              </dl>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ── GradCAM ── */}
      {result.gradcam_heatmap_b64 && (
        <div className="overflow-hidden rounded-2xl border border-ink-600/70 bg-ink-800/70">
          <button
            onClick={() => setShowGradcam((p) => !p)}
            className="flex w-full items-center justify-between px-5 py-3.5 text-left transition-colors hover:bg-ink-700/40"
          >
            <div>
              <span className="font-mono text-[11px] uppercase tracking-widest text-slate-500">
                GradCAM Heatmap
              </span>
              <span className="ml-3 text-[11px] text-slate-600">Attention visualization</span>
            </div>
            {showGradcam ? (
              <ChevronUp className="h-4 w-4 text-slate-600" />
            ) : (
              <ChevronDown className="h-4 w-4 text-slate-600" />
            )}
          </button>
          <AnimatePresence>
            {showGradcam && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.25 }}
                className="overflow-hidden"
              >
                <div className="border-t border-ink-600/50">
                  <img
                    src={`data:image/png;base64,${result.gradcam_heatmap_b64}`}
                    alt="GradCAM attention heatmap"
                    className="w-full object-contain"
                    style={{ maxHeight: 320 }}
                  />
                </div>
                <div className="px-5 py-2.5 text-[11px] text-slate-600">
                  Warm regions drove the model&apos;s {result.label_str} prediction.
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* ── Uncertainty Warning ── */}
      {isHighUncertainty && (
        <motion.div
          initial={{ opacity: 0, scale: 0.97 }}
          animate={{ opacity: 1, scale: 1 }}
          className="flex items-start gap-3 rounded-xl border border-amber-400/20 bg-amber-400/5 p-4"
        >
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
          <p className="text-[13px] leading-relaxed text-slate-400">
            <span className="font-medium text-amber-400">The views disagreed</span>. The 8 versions
            of your photo gave scores that varied by{' '}
            <span className="font-mono">{result.tta_std.toFixed(2)}</span>, so treat this result
            with extra caution. A sharper, better-lit photo may help.
          </p>
        </motion.div>
      )}

      {/* ── Disclaimer ── */}
      <p className="px-2 text-center text-[11px] leading-relaxed text-slate-700">
        For research purposes only. Results do not constitute medical advice. All findings should be
        reviewed by a qualified dermatologist.
      </p>
    </motion.div>
  )
}
