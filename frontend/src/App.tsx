import Navbar from './components/Navbar'
import Hero from './components/Hero'
import AnalysisTool from './components/AnalysisTool'
import ModelStats from './components/ModelStats'
import Footer from './components/Footer'

export default function App() {
  return (
    <div className="min-h-screen bg-[#070b14] text-slate-100 selection:bg-teal-400/20 selection:text-teal-300">
      <Navbar />
      <main>
        <Hero />
        <AnalysisTool />
        <ModelStats />
      </main>
      <Footer />
    </div>
  )
}
