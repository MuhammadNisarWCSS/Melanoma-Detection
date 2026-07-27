const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
// Strip trailing slash so `/mlflow/` + `/api/...` does not become `//api`.
const MLFLOW_BASE = (import.meta.env.VITE_MLFLOW_URL || 'http://18.219.3.159:5000').replace(
  /\/$/,
  '',
)
// Match configs/training/default.yaml — exclude melanoma-smoke and Default.
const MLFLOW_EXPERIMENT =
  import.meta.env.VITE_MLFLOW_EXPERIMENT || 'melanoma-detection'

// ─── Melanoma API ────────────────────────────────────────────────────────────

export interface PredictRequest {
  image: File
  age_approx: number
  sex: string
  anatom_site: string
  return_gradcam: boolean
}

export interface PredictResponse {
  probability: number
  label: number
  label_str: string
  confidence: number
  tta_std: number
  threshold_used: number
  gradcam_heatmap_b64: string | null
}

export async function predict(req: PredictRequest): Promise<PredictResponse> {
  const form = new FormData()
  form.append('image', req.image)
  form.append('age_approx', req.age_approx.toString())
  form.append('sex', req.sex)
  form.append('anatom_site', req.anatom_site)
  form.append('return_gradcam', req.return_gradcam.toString())

  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: form,
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }

  return res.json()
}

export async function checkApiHealth(): Promise<{ status: string; model_loaded: boolean }> {
  const res = await fetch(`${API_BASE}/health`, {
    signal: AbortSignal.timeout(3000),
  })
  return res.json()
}

export async function fetchApiMetadata(): Promise<{
  threshold: number
  tta_passes: number
  device: string
}> {
  const res = await fetch(`${API_BASE}/metadata`, {
    signal: AbortSignal.timeout(3000),
  })
  if (!res.ok) throw new Error('metadata unavailable')
  return res.json()
}

// ─── MLflow API ──────────────────────────────────────────────────────────────

export interface MLflowRun {
  run_id: string
  backbone: string
  /** Peak (max over epochs) validation AUROC for the run. */
  val_auroc: number | null
  /** Val F1 at the peak-AUROC epoch (not the final epoch). */
  val_f1: number | null
  /** Val loss at the peak-AUROC epoch (not the final epoch). */
  val_loss: number | null
  status: string
  start_time: number
  duration_ms: number | null
}

export interface MLflowStats {
  runs: MLflowRun[]
  best_auroc: number | null
  total_runs: number
}

interface MetricPoint {
  step: number
  value: number
}

const VAL_AUROC_KEYS = ['val/auroc', 'val_auroc', 'validation_auroc'] as const
const VAL_F1_KEYS = ['val/f1', 'val_f1', 'validation_f1'] as const
const VAL_LOSS_KEYS = ['val/loss', 'val_loss'] as const

function extractMetrics(run: Record<string, unknown>): Record<string, number> {
  const data = (run.data as Record<string, unknown>) || {}
  const metrics = (data.metrics as Array<{ key: string; value: number }>) || []
  return Object.fromEntries(metrics.map((m) => [m.key, m.value]))
}

function extractParams(run: Record<string, unknown>): Record<string, string> {
  const data = (run.data as Record<string, unknown>) || {}
  const params = (data.params as Array<{ key: string; value: string }>) || []
  return Object.fromEntries(params.map((p) => [p.key, p.value]))
}

function firstPresent(metrics: Record<string, number>, keys: readonly string[]): number | null {
  for (const key of keys) {
    if (metrics[key] != null && Number.isFinite(metrics[key])) return metrics[key]
  }
  return null
}

async function fetchMetricHistory(
  runId: string,
  metricKey: string,
): Promise<MetricPoint[]> {
  const url = new URL(`${MLFLOW_BASE}/api/2.0/mlflow/metrics/get-history`)
  url.searchParams.set('run_id', runId)
  url.searchParams.set('metric_key', metricKey)
  url.searchParams.set('max_results', '25000')

  const res = await fetch(url.toString(), {
    signal: AbortSignal.timeout(5000),
  })
  if (!res.ok) return []

  const data = await res.json()
  const points = (data.metrics as Array<{ step: number; value: number }>) || []
  return points
    .filter((p) => Number.isFinite(p.value))
    .map((p) => ({ step: Number(p.step), value: Number(p.value) }))
}

async function fetchMetricHistoryFirstKey(
  runId: string,
  metricKeys: readonly string[],
): Promise<MetricPoint[]> {
  for (const key of metricKeys) {
    try {
      const points = await fetchMetricHistory(runId, key)
      if (points.length > 0) return points
    } catch {
      // Try next key.
    }
  }
  return []
}

function valueAtStep(points: MetricPoint[], step: number): number | null {
  const exact = points.find((p) => p.step === step)
  if (exact) return exact.value

  // Fallback: nearest logged point at or before the peak step.
  let best: MetricPoint | null = null
  for (const p of points) {
    if (p.step <= step && (!best || p.step > best.step)) best = p
  }
  return best?.value ?? null
}

/**
 * Peak val AUROC over the run, plus F1 / loss from that same step.
 * MLflow run summaries only expose the *last* epoch's metrics; history is required
 * until training re-logs peak values into the run overview.
 */
async function fetchPeakValMetrics(
  runId: string,
  fallback: { auroc: number | null; f1: number | null; loss: number | null },
): Promise<{
  val_auroc: number | null
  val_f1: number | null
  val_loss: number | null
}> {
  const [aurocHist, f1Hist, lossHist] = await Promise.all([
    fetchMetricHistoryFirstKey(runId, VAL_AUROC_KEYS),
    fetchMetricHistoryFirstKey(runId, VAL_F1_KEYS),
    fetchMetricHistoryFirstKey(runId, VAL_LOSS_KEYS),
  ])

  if (aurocHist.length === 0) {
    return {
      val_auroc: fallback.auroc,
      val_f1: fallback.f1,
      val_loss: fallback.loss,
    }
  }

  let peak = aurocHist[0]
  for (const p of aurocHist) {
    if (p.value > peak.value) peak = p
  }

  return {
    val_auroc: peak.value,
    val_f1: valueAtStep(f1Hist, peak.step) ?? fallback.f1,
    val_loss: valueAtStep(lossHist, peak.step) ?? fallback.loss,
  }
}

export async function fetchMLflowStats(): Promise<MLflowStats> {
  // Only the full-training experiment — smoke runs live in melanoma-smoke and
  // would otherwise show up here while the MLflow UI is filtered to melanoma-detection.
  const expRes = await fetch(`${MLFLOW_BASE}/api/2.0/mlflow/experiments/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      max_results: 50,
      filter: `name = '${MLFLOW_EXPERIMENT}'`,
    }),
    signal: AbortSignal.timeout(5000),
  })

  if (!expRes.ok) throw new Error('MLflow unavailable')

  const expData = await expRes.json()
  const experiments =
    (expData.experiments as Array<{ experiment_id: string; lifecycle_stage?: string }>) || []
  const expIds = experiments
    .filter((e) => (e.lifecycle_stage ?? 'active') === 'active')
    .map((e) => e.experiment_id)

  if (expIds.length === 0) {
    return { runs: [], best_auroc: null, total_runs: 0 }
  }

  const runsRes = await fetch(`${MLFLOW_BASE}/api/2.0/mlflow/runs/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      experiment_ids: expIds,
      max_results: 25,
      // Summary metrics are last-epoch values; we re-rank by peak AUROC below.
      order_by: ['attributes.start_time DESC'],
      filter: "attributes.status != 'KILLED'",
    }),
    signal: AbortSignal.timeout(5000),
  })

  if (!runsRes.ok) throw new Error('MLflow runs unavailable')

  const runsData = await runsRes.json()
  const rawRuns = (runsData.runs as Array<Record<string, unknown>>) || []

  const runs: MLflowRun[] = await Promise.all(
    rawRuns.map(async (run) => {
      const info = (run.info as Record<string, unknown>) || {}
      const metrics = extractMetrics(run)
      const params = extractParams(run)
      const runId = String(info.run_id || '')

      const startTime = Number(info.start_time || 0)
      const endTime = Number(info.end_time || 0)

      const fallback = {
        auroc: firstPresent(metrics, VAL_AUROC_KEYS),
        f1: firstPresent(metrics, VAL_F1_KEYS),
        loss: firstPresent(metrics, VAL_LOSS_KEYS),
      }
      const peak = runId
        ? await fetchPeakValMetrics(runId, fallback)
        : {
            val_auroc: fallback.auroc,
            val_f1: fallback.f1,
            val_loss: fallback.loss,
          }

      return {
        run_id: runId,
        backbone:
          params['model/backbone'] ||
          params['model.backbone'] ||
          params['backbone'] ||
          'unknown',
        val_auroc: peak.val_auroc,
        val_f1: peak.val_f1,
        val_loss: peak.val_loss,
        status: String(info.status || 'UNKNOWN'),
        start_time: startTime,
        duration_ms: endTime > 0 && startTime > 0 ? endTime - startTime : null,
      }
    }),
  )

  runs.sort((a, b) => (b.val_auroc ?? -Infinity) - (a.val_auroc ?? -Infinity))

  const aurocs = runs.map((r) => r.val_auroc).filter((v): v is number => v !== null)
  const best_auroc = aurocs.length > 0 ? Math.max(...aurocs) : null

  return { runs, best_auroc, total_runs: runs.length }
}
