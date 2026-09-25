import { Camera, Sparkles, FileCheck } from 'lucide-react'
import { motion } from 'framer-motion'

const STEPS = [
  {
    icon: Camera,
    title: 'Take a clear photo',
    detail: 'Good light, in focus, close enough that the spot fills most of the frame.',
  },
  {
    icon: Sparkles,
    title: 'Add a couple details',
    detail: 'Your age, sex, and where it is on your body help sharpen the read.',
  },
  {
    icon: FileCheck,
    title: 'Get your result',
    detail: "See whether it looks benign or worth a dermatologist's look, and what stood out.",
  },
]

export default function HowItWorks() {
  return (
    <section className="relative border-t border-ink-600/50 py-20 px-5 sm:px-8">
      <div className="mx-auto max-w-6xl">
        <motion.div
          className="mb-12 max-w-xl"
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.45 }}
        >
          <p className="mb-2 text-[13px] font-medium text-teal-400">How it works</p>
          <h2 className="font-display text-[28px] font-semibold tracking-tight text-slate-100">
            Three steps, usually under a minute.
          </h2>
        </motion.div>

        <div className="grid grid-cols-1 gap-6 sm:grid-cols-3">
          {STEPS.map(({ icon: Icon, title, detail }, i) => (
            <motion.div
              key={title}
              className="rounded-2xl border border-ink-600/70 bg-ink-800/60 p-6"
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: i * 0.08 }}
            >
              <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-full bg-teal-400/10">
                <Icon className="h-5 w-5 text-teal-400" strokeWidth={2} />
              </div>
              <div className="mb-1.5 text-[10px] font-mono uppercase tracking-widest text-slate-600">
                Step {i + 1}
              </div>
              <h3 className="mb-2 text-[16px] font-semibold text-slate-200">{title}</h3>
              <p className="text-[13px] leading-relaxed text-slate-500">{detail}</p>
            </motion.div>
          ))}
        </div>

        <motion.p
          className="mx-auto mt-10 max-w-xl text-center text-[13px] leading-relaxed text-slate-500"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4, delay: 0.2 }}
        >
          The model misses some melanomas and flags many harmless moles, so a "benign" result is
          not a clearance. Use it to help decide whether to book an appointment, not to skip one.
        </motion.p>
      </div>
    </section>
  )
}
