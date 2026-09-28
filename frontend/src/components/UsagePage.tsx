import { useEffect, useState } from 'react'

// Unlinked owner page at /usage. Open by design: the endpoint only returns aggregate counts.
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

interface UsageStats {
  total: number
  last_24h: number
  last_7d: number
  last_30d: number
  flagged: number
  out_of_distribution: number
  avg_latency_ms: number | null
  first: string | null
  last: string | null
  daily: { day: string; count: number }[]
}

function pct(n: number, total: number) {
  return total ? `${((n / total) * 100).toFixed(1)}%` : '–'
}

export default function UsagePage() {
  const [stats, setStats] = useState<UsageStats | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch(`${API_BASE}/usage`)
      .then((res) => {
        if (!res.ok) throw new Error(`Error ${res.status}`)
        return res.json()
      })
      .then(setStats)
      .catch((err) => setError(err instanceof Error ? err.message : 'Request failed'))
  }, [])

  const tiles = stats
    ? [
        ['Total analyses', stats.total],
        ['Last 24 hours', stats.last_24h],
        ['Last 7 days', stats.last_7d],
        ['Last 30 days', stats.last_30d],
        ['Flagged malignant', pct(stats.flagged, stats.total)],
        ['Out-of-distribution', pct(stats.out_of_distribution, stats.total)],
        ['Avg latency', stats.avg_latency_ms == null ? '–' : `${Math.round(stats.avg_latency_ms)} ms`],
      ]
    : []

  return (
    <div className="min-h-screen bg-[#fbf6f2] px-4 py-12 text-stone-800">
      <div className="mx-auto max-w-xl">
        <h1 className="mb-6 text-2xl font-semibold">Usage</h1>
        {error && <p className="mb-4 text-red-700">{error}</p>}
        {!stats && !error && <p className="text-stone-500">Loading…</p>}
        {stats && (
          <>
            <div className="mb-8 grid grid-cols-2 gap-3">
              {tiles.map(([label, value]) => (
                <div key={label} className="rounded border border-stone-200 bg-white p-4">
                  <div className="text-sm text-stone-500">{label}</div>
                  <div className="text-2xl font-semibold">{value}</div>
                </div>
              ))}
            </div>
            <h2 className="mb-2 font-semibold">Per day (UTC)</h2>
            <table className="w-full text-sm">
              <tbody>
                {stats.daily.map((d) => (
                  <tr key={d.day} className="border-b border-stone-200">
                    <td className="py-1">{d.day}</td>
                    <td className="py-1 text-right">{d.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {stats.first && (
              <p className="mt-4 text-xs text-stone-500">
                Since {stats.first} · latest {stats.last}
              </p>
            )}
          </>
        )}
      </div>
    </div>
  )
}
