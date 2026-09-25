import { Fragment, useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { fetchSubgroupMetrics, type SubgroupMetrics, type SubgroupRow } from '../api/client'

const SECTIONS: { key: 'sex' | 'age' | 'site'; title: string; order: string[] }[] = [
  { key: 'sex', title: 'Sex', order: ['female', 'male'] },
  { key: 'age', title: 'Age', order: ['<40', '40-59', '60+'] },
  {
    key: 'site',
    title: 'Body site',
    order: [
      'torso',
      'upper extremity',
      'lower extremity',
      'head/neck',
      'palms/soles',
      'oral/genital',
    ],
  },
]

const LABELS: Record<string, string> = {
  female: 'Female',
  male: 'Male',
  '<40': 'Under 40',
  '40-59': '40 to 59',
  '60+': '60 and over',
  torso: 'Torso',
  'upper extremity': 'Arm or hand',
  'lower extremity': 'Leg or foot',
  'head/neck': 'Head or neck',
  'palms/soles': 'Palm or sole',
  'oral/genital': 'Mouth or genital area',
}

const pct = (v: number | null) => (v == null ? '—' : `${Math.round(v * 100)}%`)

function caught(r: SubgroupRow) {
  if (r.n_malignant === 0 || r.sensitivity == null) return 'No cases to test'
  return `${Math.round(r.sensitivity * r.n_malignant)} of ${r.n_malignant}`
}

export default function SubgroupTable() {
  const [data, setData] = useState<SubgroupMetrics | null>(null)

  useEffect(() => {
    fetchSubgroupMetrics()
      .then(setData)
      .catch(() => setData(null))
  }, [])

  if (!data) return null

  return (
    <motion.div
      className="mb-8 overflow-hidden rounded-2xl border border-ink-600/70 bg-ink-800/50"
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4 }}
    >
      <div className="border-b border-ink-600/60 px-6 py-4">
        <h3 className="text-[15px] font-semibold text-slate-200">Where the model is weaker</h3>
        <p className="mt-1 text-[12px] leading-relaxed text-slate-500">
          The same test photos, split by patient group. An overall score can hide groups the model
          handles badly. Each group has only a handful of melanomas, so read these as warning signs,
          not precise estimates.
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-[12px]">
          <thead>
            <tr className="border-b border-ink-600/40 text-slate-600">
              <th className="px-6 py-2 font-medium">Group</th>
              <th className="px-3 py-2 font-medium">Photos</th>
              <th className="px-3 py-2 font-medium">Melanomas caught</th>
              <th className="px-3 py-2 font-medium">Harmless cleared</th>
            </tr>
          </thead>
          <tbody>
            {SECTIONS.map(({ key, title, order }) => (
              <Fragment key={key}>
                <tr>
                  <td
                    colSpan={4}
                    className="bg-ink-700/30 px-6 py-1.5 font-mono text-[10px] uppercase tracking-wider text-slate-500"
                  >
                    {title}
                  </td>
                </tr>
                {order
                  .filter((g) => data[key]?.[g])
                  .map((g) => {
                    const r = data[key][g]
                    const weak = r.sensitivity != null && r.n_malignant >= 3 && r.sensitivity < 0.5
                    return (
                      <tr key={`${key}-${g}`} className="border-b border-ink-600/30">
                        <td className="px-6 py-2 text-slate-300">{LABELS[g] ?? g}</td>
                        <td className="px-3 py-2 font-mono text-slate-400">{r.n}</td>
                        <td
                          className={`px-3 py-2 font-mono ${weak ? 'text-amber-400' : 'text-slate-300'}`}
                        >
                          {caught(r)}
                          {r.n_malignant > 0 && (
                            <span className="ml-2 text-slate-600">({pct(r.sensitivity)})</span>
                          )}
                        </td>
                        <td className="px-3 py-2 font-mono text-slate-400">{pct(r.specificity)}</td>
                      </tr>
                    )
                  })}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
      <p className="border-t border-ink-600/40 px-6 py-3 text-[12px] leading-relaxed text-slate-500">
        The weakest spots are patients under 40 and lesions on the legs, where the model caught only
        2 of 5 melanomas. Older patients get the highest catch rate but also more false alarms.
        Palms, soles and the mouth or genital area had no melanomas in the test set at all, so the
        model is untested there.
      </p>
    </motion.div>
  )
}
