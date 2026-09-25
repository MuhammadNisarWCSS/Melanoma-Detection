import { useState, useRef, useCallback, type DragEvent, type ChangeEvent } from 'react'
import {
  Upload,
  Loader2,
  AlertCircle,
  User,
  MapPin,
  RefreshCw,
  ImageIcon,
  ShieldCheck,
  Stethoscope,
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { predict, type PredictResponse } from '../api/client'
import ResultCard from './ResultCard'
import DisclaimerModal from './DisclaimerModal'

const SITES = [
  { value: 'torso', label: 'Chest, back, or torso' },
  { value: 'lower extremity', label: 'Leg or foot' },
  { value: 'upper extremity', label: 'Arm or hand' },
  { value: 'head/neck', label: 'Head or neck' },
  { value: 'palms/soles', label: 'Palm or sole' },
  { value: 'oral/genital', label: 'Mouth or genital area' },
  { value: 'unknown', label: 'Not sure' },
]

const SEX_LABELS: Record<'male' | 'female' | 'unknown', string> = {
  male: 'Male',
  female: 'Female',
  unknown: 'Prefer not to say',
}

const MAX_AGE = 100

const TRUST_POINTS = [
  {
    icon: ShieldCheck,
    title: 'Your data is not saved',
    detail: 'Your photo and details are used for this one check and then discarded.',
  },
  {
    icon: Stethoscope,
    title: 'Not a diagnosis',
    detail: "It's a rough guide to help you decide whether to see a doctor.",
  },
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
          style={{ boxShadow: '0 0 20px rgba(193,104,63,0.12)' }}
        />
      </div>
      <p className="text-[14px] text-slate-400">Your result will show up here</p>
      <p className="mt-1 text-[12px] text-slate-600">
        Add a photo and a few details, then press Check my skin
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
        <div
          className="absolute inset-3 rounded-full border border-teal-400/15 animate-spin-slow"
          style={{ animationDirection: 'reverse' }}
        />
        <div className="absolute inset-[38px] rounded-full bg-teal-400/10 animate-pulse-teal" />
      </div>
      <p className="text-[15px] font-medium text-slate-200">Taking a close look…</p>
      <p className="mt-1.5 text-[13px] text-slate-500">
        Looking at the photo from eight orientations and averaging them
      </p>
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
    </div>
  )
}

export default function AnalysisTool() {
  const [image, setImage] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const [ageText, setAgeText] = useState('50')
  const [showDisclaimer, setShowDisclaimer] = useState<'intro' | 'submit' | null>('intro')
  const [sex, setSex] = useState<string>('unknown')
  const [site, setSite] = useState<string>('unknown')
  const [loading, setLoading] = useState(false)
  const [submitted, setSubmitted] = useState({ age: 50, sex: 'unknown', site: 'unknown' })
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
    [loadFile],
  )

  const handleFileChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) loadFile(file)
    },
    [loadFile],
  )

  const age = Math.min(MAX_AGE, Math.max(0, parseInt(ageText, 10) || 0))

  const handleAgeText = (e: ChangeEvent<HTMLInputElement>) => {
    const digits = e.target.value.replace(/\D/g, '').slice(0, 3)
    setAgeText(digits === '' ? '' : String(Math.min(MAX_AGE, Number(digits))))
  }

  const runPrediction = async () => {
    if (!image) return
    setSubmitted({ age, sex, site })
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

  const handleSubmit = () => {
    if (!image || loading) return
    setShowDisclaimer('submit')
  }

  const handleReset = () => {
    setImage(null)
    setPreview(null)
    setResult(null)
    setError(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  return (
    <section id="analyze" className="relative pt-28 pb-20 px-5 sm:px-8 lg:pt-32">
      <div className="mx-auto max-w-6xl">
        {/* Intro — the tool is the hero, not a marketing banner above it */}
        <motion.div
          className="mb-10 max-w-2xl"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <p className="mb-3 text-[13px] font-medium text-teal-400">
            A free second look at a spot you're unsure about
          </p>
          <h1 className="font-display text-[36px] font-semibold leading-[1.15] tracking-tight text-slate-100 sm:text-[44px]">
            Is that mole worth getting checked?
          </h1>
          <p className="mt-4 text-[16px] leading-relaxed text-slate-400">
            Upload a close-up photo and tell us your age, sex and where the spot is. A model trained
            on about 33,000 dermoscopy images will say whether it looks benign or worth showing a
            dermatologist, and highlight the part of the photo that drove its answer.
          </p>
          <div className="mt-7 grid grid-cols-1 gap-4 sm:grid-cols-2">
            {TRUST_POINTS.map(({ icon: Icon, title, detail }) => (
              <div key={title} className="flex items-start gap-3">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-teal-400/10">
                  <Icon className="h-4 w-4 text-teal-400" strokeWidth={2} />
                </div>
                <div>
                  <div className="text-[13px] font-semibold leading-8 text-slate-200 sm:leading-8">
                    {title}
                  </div>
                  <p className="text-[12px] leading-relaxed text-slate-500">{detail}</p>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* ── Merged input card: big photo pane + dark details strip ── */}
        <div className="grid grid-cols-1 overflow-hidden rounded-2xl border border-ink-600/70 lg:grid-cols-[1.6fr_1fr]">
          {/* Photo pane */}
          <div
            onDragOver={(e) => {
              e.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`relative cursor-pointer overflow-hidden transition-colors duration-200
              ${dragging ? 'bg-teal-400/5' : preview ? '' : 'bg-ink-800/40 hover:bg-ink-800/60'}`}
            style={{ minHeight: 440 }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFileChange}
            />

            {preview ? (
              <div className="relative h-full" style={{ minHeight: 440 }}>
                <img
                  src={preview}
                  alt="Preview"
                  className="h-full w-full object-cover"
                  style={{ minHeight: 440 }}
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/45 via-transparent to-transparent" />
                <div className="absolute bottom-4 left-5 flex items-center gap-2">
                  <ImageIcon className="h-3.5 w-3.5 text-slate-100" />
                  <span className="text-[12px] font-medium text-white truncate max-w-[240px]">
                    {image?.name}
                  </span>
                </div>
                <div className="absolute right-4 top-4 rounded-lg border border-white/20 bg-black/40 px-2.5 py-1 text-[11px] text-white backdrop-blur-sm">
                  Click to replace
                </div>
              </div>
            ) : (
              <div
                className={`flex h-full flex-col items-center justify-center border-2 border-dashed p-12 text-center transition-colors duration-200 ${
                  dragging ? 'border-teal-400' : 'border-ink-600/70'
                }`}
                style={{ minHeight: 440 }}
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
                <p className="mb-1 text-[16px] font-medium text-slate-300">
                  {dragging ? 'Release to upload' : 'Drop a photo of the spot here'}
                </p>
                <p className="text-[13px] text-slate-500">
                  or click to browse. Sharp, well-lit close-ups work best. JPEG, PNG, BMP or TIFF.
                </p>
              </div>
            )}
          </div>

          {/* Dark details strip — camera-app feel */}
          <div className="flex flex-col bg-[#2b2320] p-7">
            <div className="mb-5 flex items-center gap-2 text-[11px] font-mono uppercase tracking-widest text-[#a89686]">
              <User className="h-3.5 w-3.5" strokeWidth={2} />
              Details
            </div>

            {/* Age */}
            <div className="mb-6">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-[12px] text-[#a89686]">Your age</span>
                <div className="flex items-center gap-1.5">
                  <input
                    type="text"
                    inputMode="numeric"
                    aria-label="Age in years"
                    value={ageText}
                    onChange={handleAgeText}
                    onBlur={() => setAgeText(String(age))}
                    className="w-14 rounded-md border border-[#4a3f38] px-2 py-1 text-right font-mono text-[13px] font-semibold text-teal-400 focus:border-teal-400/50"
                    style={{ background: '#241d18' }}
                  />
                  <span className="text-[12px] text-[#a89686]">yrs</span>
                </div>
              </div>
              <input
                type="range"
                min={0}
                max={90}
                step={1}
                value={Math.min(age, 90)}
                onChange={(e) => setAgeText(e.target.value)}
                className="w-full"
                style={{ background: '#4a3f38' }}
              />
            </div>

            {/* Sex */}
            <div className="mb-6 border-b border-[#4a3f38] pb-6">
              <div className="mb-2 text-[12px] text-[#a89686]">Sex</div>
              <div className="flex gap-2">
                {(['male', 'female', 'unknown'] as const).map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => setSex(s)}
                    className={`flex-1 rounded-lg py-2 text-[12px] font-medium transition-all duration-150 ${
                      sex === s
                        ? 'bg-teal-400 text-white'
                        : 'border border-[#4a3f38] text-[#a89686] hover:border-[#655648]'
                    }`}
                  >
                    {SEX_LABELS[s]}
                  </button>
                ))}
              </div>
            </div>

            {/* Site */}
            <div className="mb-auto">
              <label className="mb-2 flex items-center gap-1.5 text-[12px] text-[#a89686]">
                <MapPin className="h-3 w-3" />
                Where is it?
              </label>
              <select
                value={site}
                onChange={(e) => setSite(e.target.value)}
                className="w-full rounded-lg border border-[#4a3f38] px-3 py-2.5 pr-9 text-[13px] font-medium text-[#f3e9df] transition-colors hover:border-[#655648] focus:border-teal-400/50"
                style={{ background: '#241d18' }}
              >
                {SITES.map((s) => (
                  <option key={s.value} value={s.value}>
                    {s.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Actions */}
            <div className="mt-6 flex gap-2">
              <button
                type="button"
                onClick={handleSubmit}
                disabled={!image || loading}
                className={`flex flex-1 items-center justify-center gap-2 rounded-lg py-3.5 text-[14px] font-semibold transition-all duration-200 ${
                  !image || loading
                    ? 'cursor-not-allowed bg-[#4a3f38] text-[#7a6d61]'
                    : 'bg-teal-400 text-white hover:bg-teal-500'
                }`}
              >
                {loading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Checking…
                  </>
                ) : (
                  'Check my skin'
                )}
              </button>

              {(image || result) && (
                <button
                  type="button"
                  onClick={handleReset}
                  title="Reset"
                  className="flex h-[50px] w-[50px] shrink-0 items-center justify-center rounded-lg border border-[#4a3f38] text-[#a89686] transition-colors hover:border-[#655648] hover:text-[#f3e9df]"
                >
                  <RefreshCw className="h-4 w-4" />
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Error */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="mt-5 flex items-start gap-3 rounded-xl border border-red-500/25 bg-red-500/6 p-4"
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

        {/* ── Result ── */}
        <div className="mx-auto mt-8 max-w-2xl">
          {loading ? (
            <LoadingPanel />
          ) : result ? (
            <ResultCard
              result={result}
              imageSrc={preview}
              inputs={{
                age: submitted.age,
                sex: SEX_LABELS[submitted.sex as keyof typeof SEX_LABELS] ?? submitted.sex,
                site: SITES.find((s) => s.value === submitted.site)?.label ?? submitted.site,
              }}
            />
          ) : (
            <EmptyPanel />
          )}
        </div>
      </div>

      <AnimatePresence>
        {showDisclaimer && (
          <DisclaimerModal
            key="disclaimer"
            confirmLabel={
              showDisclaimer === 'intro' ? 'I understand' : 'I understand, check my skin'
            }
            onConfirm={() => {
              const wasSubmit = showDisclaimer === 'submit'
              setShowDisclaimer(null)
              if (wasSubmit) void runPrediction()
            }}
            onCancel={showDisclaimer === 'submit' ? () => setShowDisclaimer(null) : undefined}
          />
        )}
      </AnimatePresence>
    </section>
  )
}
