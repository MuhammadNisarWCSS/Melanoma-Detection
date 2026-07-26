import {
  useState,
  useRef,
  useCallback,
  type DragEvent,
  type ChangeEvent,
} from 'react'
import {
  Upload,
  Loader2,
  AlertCircle,
  User,
  MapPin,
  RefreshCw,
  ImageIcon,
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { predict, type PredictResponse } from '../api/client'
import ResultCard from './ResultCard'

const SITES = [
  { value: 'torso', label: 'Torso' },
  { value: 'lower extremity', label: 'Lower Extremity' },
  { value: 'upper extremity', label: 'Upper Extremity' },
  { value: 'head/neck', label: 'Head / Neck' },
  { value: 'palms/soles', label: 'Palms / Soles' },
  { value: 'oral/genital', label: 'Oral / Genital' },
  { value: 'unknown', label: 'Unknown / Not Specified' },
]

function EmptyPanel() {
  return (
    <div className="flex min-h-[480px] flex-col items-center justify-center rounded-2xl border border-dashed border-ink-600/60 p-10 text-center">
      <div className="relative mb-5 h-20 w-20">
        <div className="absolute inset-0 rounded-full border border-ink-500/60" />
        <div className="absolute inset-4 rounded-full border border-ink-500/40" />
        <div className="absolute inset-[34px] rounded-full border border-teal-400/20" />
        <div
          className="absolute inset-[38px] rounded-full bg-teal-400/10"
          style={{ boxShadow: '0 0 20px rgba(0,212,170,0.08)' }}
        />
      </div>
      <p className="text-[14px] text-slate-400">Analysis results will appear here</p>
      <p className="mt-1 text-[12px] text-slate-600">
        Upload an image and fill in patient details, then click Run Analysis
      </p>
    </div>
  )
}

function LoadingPanel() {
  return (
    <div className="flex min-h-[480px] flex-col items-center justify-center rounded-2xl border border-ink-600/60 bg-ink-800/50 p-10">
      {/* Animated rings */}
      <div className="relative mb-8 h-24 w-24">
        <div className="absolute inset-0 rounded-full border-2 border-ink-500/30" />
        <div className="absolute inset-0 rounded-full border-2 border-t-teal-400 border-r-transparent border-b-transparent border-l-transparent animate-spin" />
        <div className="absolute inset-3 rounded-full border border-teal-400/15 animate-spin-slow" style={{ animationDirection: 'reverse' }} />
        <div className="absolute inset-[38px] rounded-full bg-teal-400/10 animate-pulse-teal" />
      </div>
      <p className="text-[15px] font-medium text-slate-200">Analyzing image…</p>
      <p className="mt-1.5 text-[13px] text-slate-500">Running 8-pass TTA inference</p>
      <div className="mt-6 flex gap-1.5">
        {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
          <motion.div
            key={i}
            className="h-1 w-5 rounded-full bg-teal-400/30"
            animate={{ opacity: [0.3, 1, 0.3] }}
            transition={{ duration: 1.2, delay: i * 0.15, repeat: Infinity }}
          />
        ))}
      </div>
      <p className="mt-3 text-[11px] font-mono text-slate-600">
        pass <motion.span animate={{ opacity: [0, 1] }} transition={{ duration: 0.4, repeat: Infinity }}>_</motion.span>
      </p>
    </div>
  )
}

export default function AnalysisTool() {
  const [image, setImage] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const [age, setAge] = useState(45)
  const [sex, setSex] = useState<string>('unknown')
  const [site, setSite] = useState<string>('unknown')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const loadFile = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) return
    setImage(file)
    setResult(null)
    setError(null)
    const url = URL.createObjectURL(file)
    setPreview(url)
  }, [])

  const handleDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      setDragging(false)
      const file = e.dataTransfer.files[0]
      if (file) loadFile(file)
    },
    [loadFile]
  )

  const handleFileChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) loadFile(file)
    },
    [loadFile]
  )

  const handleSubmit = async () => {
    if (!image) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await predict({
        image,
        age_approx: age,
        sex,
        anatom_site: site,
        return_gradcam: true,
      })
      setResult(res)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Prediction failed'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setImage(null)
    setPreview(null)
    setResult(null)
    setError(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  return (
    <section id="analyze" className="relative py-24 px-5 sm:px-8">
      {/* Section separator line */}
      <div className="mx-auto max-w-7xl">
        {/* Section header */}
        <motion.div
          className="mb-12"
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.45 }}
        >
          <p className="mb-2 font-mono text-[11px] uppercase tracking-[0.2em] text-teal-400">
            Analysis Tool
          </p>
          <h2 className="text-[32px] font-bold tracking-tight text-slate-100">
            Dermoscopy Analysis
          </h2>
          <p className="mt-2 text-[15px] text-slate-400">
            Upload a dermoscopy image and provide patient metadata for classification.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
          {/* ── Left Panel ── */}
          <div className="space-y-5">
            {/* Drop zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`relative cursor-pointer overflow-hidden rounded-2xl border-2 border-dashed transition-all duration-200
                ${dragging
                  ? 'border-teal-400 bg-teal-400/5 shadow-teal'
                  : preview
                    ? 'border-ink-500/70 hover:border-teal-400/40'
                    : 'border-ink-600/70 bg-ink-800/40 hover:border-teal-400/40 hover:bg-ink-800/60'
                }`}
              style={{ minHeight: 260 }}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleFileChange}
              />

              {preview ? (
                <div className="relative" style={{ minHeight: 260 }}>
                  <img
                    src={preview}
                    alt="Preview"
                    className="w-full object-cover"
                    style={{ minHeight: 260, maxHeight: 380 }}
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-[#070b14]/70 via-transparent to-transparent" />
                  <div className="absolute bottom-3 left-4 flex items-center gap-2">
                    <ImageIcon className="h-3.5 w-3.5 text-slate-400" />
                    <span className="text-[12px] font-medium text-slate-300 truncate max-w-[200px]">
                      {image?.name}
                    </span>
                  </div>
                  <div className="absolute right-3 top-3 rounded-lg border border-ink-600/70 bg-ink-900/80 px-2.5 py-1 text-[11px] text-slate-400 backdrop-blur-sm">
                    Click to replace
                  </div>
                </div>
              ) : (
                <div
                  className="flex flex-col items-center justify-center p-12"
                  style={{ minHeight: 260 }}
                >
                  <div
                    className={`mb-4 flex h-16 w-16 items-center justify-center rounded-full border-2 transition-colors ${
                      dragging ? 'border-teal-400 bg-teal-400/10' : 'border-ink-500/60'
                    }`}
                  >
                    <Upload
                      className={`h-7 w-7 transition-colors ${dragging ? 'text-teal-400' : 'text-slate-600'}`}
                    />
                  </div>
                  <p className="mb-1 text-[15px] font-medium text-slate-300">
                    {dragging ? 'Release to upload' : 'Drop dermoscopy image here'}
                  </p>
                  <p className="text-[13px] text-slate-500">or click to browse · JPEG, PNG, BMP, TIFF</p>
                </div>
              )}
            </div>

            {/* Metadata form */}
            <div className="rounded-2xl border border-ink-600/70 bg-ink-800/50 p-5 space-y-5">
              <div className="flex items-center gap-2 text-[13px] font-semibold text-slate-300">
                <User className="h-4 w-4 text-teal-400" strokeWidth={2} />
                Patient Metadata
              </div>

              {/* Age */}
              <div>
                <div className="mb-2 flex items-center justify-between">
                  <label className="font-mono text-[10px] uppercase tracking-widest text-slate-500">
                    Patient Age
                  </label>
                  <span className="font-mono text-[13px] font-semibold text-teal-400">{age} yrs</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={90}
                  step={1}
                  value={age}
                  onChange={(e) => setAge(Number(e.target.value))}
                  className="w-full"
                />
                <div className="mt-1 flex justify-between font-mono text-[10px] text-slate-700">
                  <span>0</span>
                  <span>45</span>
                  <span>90</span>
                </div>
              </div>

              {/* Sex */}
              <div>
                <label className="mb-2 block font-mono text-[10px] uppercase tracking-widest text-slate-500">
                  Biological Sex
                </label>
                <div className="flex gap-2">
                  {(['male', 'female', 'unknown'] as const).map((s) => (
                    <button
                      key={s}
                      type="button"
                      onClick={() => setSex(s)}
                      className={`flex-1 rounded-lg border py-2 text-[13px] font-medium capitalize transition-all duration-150 ${
                        sex === s
                          ? 'border-teal-400/40 bg-teal-400/10 text-teal-400'
                          : 'border-ink-600/70 bg-ink-700/40 text-slate-500 hover:border-ink-500 hover:text-slate-400'
                      }`}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>

              {/* Site */}
              <div>
                <label className="mb-2 flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-widest text-slate-500">
                  <MapPin className="h-3 w-3" />
                  Anatomical Site
                </label>
                <select
                  value={site}
                  onChange={(e) => setSite(e.target.value)}
                  className="w-full rounded-lg border border-ink-600/70 bg-ink-700/60 px-3 py-2.5 pr-9 text-[13px] text-slate-300 transition-colors hover:border-ink-500 focus:border-teal-400/50"
                >
                  {SITES.map((s) => (
                    <option key={s.value} value={s.value}>
                      {s.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Error */}
            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="flex items-start gap-3 rounded-xl border border-red-500/25 bg-red-500/6 p-4"
                >
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-400" />
                  <div>
                    <p className="text-[13px] font-semibold text-red-400">Prediction Error</p>
                    <p className="mt-0.5 text-[12px] text-slate-400">{error}</p>
                    {error.includes('503') || error.includes('not loaded') ? (
                      <p className="mt-1 text-[11px] text-slate-600">
                        Ensure the FastAPI server is running and a model is loaded.
                      </p>
                    ) : null}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Action buttons */}
            <div className="flex gap-3">
              <button
                type="button"
                onClick={handleSubmit}
                disabled={!image || loading}
                className={`flex flex-1 items-center justify-center gap-2 rounded-xl py-3.5 text-[14px] font-semibold transition-all duration-200 ${
                  !image || loading
                    ? 'cursor-not-allowed border border-ink-600/50 bg-ink-800/40 text-slate-700'
                    : 'bg-teal-400 text-[#070b14] hover:bg-teal-300 shadow-teal hover:shadow-teal'
                }`}
              >
                {loading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Analyzing…
                  </>
                ) : (
                  <>
                    Run Analysis
                    <span className="ml-0.5 font-mono text-[11px] opacity-60">8× TTA</span>
                  </>
                )}
              </button>

              {(image || result) && (
                <button
                  type="button"
                  onClick={handleReset}
                  title="Reset"
                  className="flex h-[50px] w-[50px] items-center justify-center rounded-xl border border-ink-600/70 bg-ink-800/50 text-slate-500 transition-colors hover:border-ink-500 hover:text-slate-300"
                >
                  <RefreshCw className="h-4 w-4" />
                </button>
              )}
            </div>
          </div>

          {/* ── Right Panel ── */}
          <div>
            {loading ? (
              <LoadingPanel />
            ) : result ? (
              <ResultCard result={result} imageSrc={preview} />
            ) : (
              <EmptyPanel />
            )}
          </div>
        </div>
      </div>
    </section>
  )
}
