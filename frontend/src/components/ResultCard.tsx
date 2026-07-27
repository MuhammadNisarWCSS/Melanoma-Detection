import { AlertTriangle, CheckCircle, Info, ChevronDown, ChevronUp } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { useState } from 'react'
import type { PredictResponse } from '../api/client'

interface Props {
  result: PredictResponse
  imageSrc: string | null
}

function ProbabilityGauge({ probability, isMalignant }: { probability: number; isMalignant: boolean }) {
  const radius = 54
  const stroke = 7
  const normalizedR = radius - stroke / 2
  const circumference = 2 * Math.PI * normalizedR
  const offset = circumference - probability * circumference
  const color = isMalignant ? '#ef4444' : '#00d4aa'
  const pct = Math.round(probability * 100)

  return (
    <div className="flex flex-col items-center gap-2">
      <svg width={radius * 2 + stroke} height={radius * 2 + stroke} viewBox={`0 0 ${radius * 2 + stroke} ${radius * 2 + stroke}`}>
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
          style={{ filter: `drop-shadow(0 0 8px ${color}55)`, transition: 'stroke-dashoffset 1s ease-out' }}
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
          fill="#475569"
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

function MetricBar({
  label,
  value,
  fill,
  color,
  warn = false,
}: {
  label: string
  value: string
  fill: number
  color: string
  warn?: boolean
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-[11px] font-mono uppercase tracking-widest text-slate-500">{label}</span>
        <span className={`text-[13px] font-mono font-medium ${warn ? 'text-amber-400' : 'text-slate-200'}`}>{value}</span>
      </div>
      <div className="h-[3px] overflow-hidden rounded-full bg-ink-600">
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: color, boxShadow: `0 0 6px ${color}50` }}
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(fill * 100, 100)}%` }}
          transition={{ duration: 0.8, ease: 'easeOut', delay: 0.15 }}
        />
      </div>
    </div>
  )
}

export default function ResultCard({ result, imageSrc }: Props) {
  const [showGradcam, setShowGradcam] = useState(true)
  const isMalignant = result.label === 1
  const isHighUncertainty = result.tta_std > 0.1
  const confidence = result.confidence
  const pct = Math.round(result.probability * 100)

  return (
    <motion.div
      className="space-y-4 animate-fade-in"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: 'easeOut' }}
    >
      {/* ── Diagnosis Banner ── */}
      <div
        className={`flex items-center gap-4 rounded-2xl border p-5 ${
          isMalignant
            ? 'border-red-500/25 bg-red-500/6'
            : 'border-emerald-500/25 bg-emerald-500/6'
        }`}
      >
        <div
          className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-xl border ${
            isMalignant ? 'border-red-500/30 bg-red-500/10' : 'border-emerald-500/30 bg-emerald-500/10'
          }`}
        >
          {isMalignant ? (
            <AlertTriangle className="h-6 w-6 text-red-400" />
          ) : (
            <CheckCircle className="h-6 w-6 text-emerald-400" />
          )}
        </div>
        <div>
          <div
            className={`text-[22px] font-bold uppercase tracking-[0.1em] ${
              isMalignant ? 'text-red-400' : 'text-emerald-400'
            }`}
          >
            {result.label_str}
          </div>
          <div className="mt-0.5 text-[13px] text-slate-400">
            {isMalignant
              ? 'Malignant features detected — clinical review recommended'
              : 'No significant malignant features identified'}
          </div>
        </div>
        <div className="ml-auto text-right hidden sm:block">
          <div className="font-mono text-[11px] text-slate-600">threshold</div>
          <div className="font-mono text-[14px] text-slate-400">{result.threshold_used.toFixed(3)}</div>
        </div>
      </div>

      {/* ── Gauge + Metrics ── */}
      <div className="grid grid-cols-2 gap-4">
        {/* Gauge card */}
        <div className="flex flex-col items-center justify-center rounded-2xl border border-ink-600/70 bg-ink-800/70 py-6 px-4">
          <div className="mb-3 font-mono text-[10px] uppercase tracking-widest text-slate-600">Probability</div>
          <ProbabilityGauge probability={result.probability} isMalignant={isMalignant} />
          <div className="mt-3 font-mono text-[11px] text-slate-600">
            raw: <span className="text-slate-400">{result.probability.toFixed(4)}</span>
          </div>
        </div>

        {/* Metric bars */}
        <div className="flex flex-col justify-center space-y-5 rounded-2xl border border-ink-600/70 bg-ink-800/70 p-5">
          <MetricBar
            label="Confidence"
            value={`${Math.round(confidence * 100)}%`}
            fill={confidence}
            color="#00d4aa"
          />
          <MetricBar
            label="TTA Std"
            value={result.tta_std.toFixed(4)}
            fill={Math.min(result.tta_std * 8, 1)}
            color={isHighUncertainty ? '#f59e0b' : '#00d4aa'}
            warn={isHighUncertainty}
          />
          <div className="border-t border-ink-600 pt-3">
            <div className="flex justify-between">
              <span className="font-mono text-[10px] uppercase tracking-widest text-slate-600">Decision</span>
              <span
                className={`rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase ${
                  isMalignant ? 'bg-red-500/10 text-red-400' : 'bg-emerald-500/10 text-emerald-400'
                }`}
              >
                {pct}% ≥ {Math.round(result.threshold_used * 100)}%?{' '}
                {result.probability >= result.threshold_used ? 'yes' : 'no'}
              </span>
            </div>
          </div>
        </div>
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
            <span className="font-medium text-amber-400">High prediction uncertainty</span> — TTA std{' '}
            <span className="font-mono">{result.tta_std.toFixed(4)}</span> &gt; 0.10. Consider
            improving image quality or acquiring multiple views.
          </p>
        </motion.div>
      )}

      {/* ── Disclaimer ── */}
      <p className="px-2 text-center text-[11px] leading-relaxed text-slate-700">
        For research purposes only. Results do not constitute medical advice.
        All findings should be reviewed by a qualified dermatologist.
      </p>
    </motion.div>
  )
}
