import Navbar from './components/Navbar'
import AnalysisTool from './components/AnalysisTool'
import HowItWorks from './components/HowItWorks'
import TechnicalDetails from './components/TechnicalDetails'
import Footer from './components/Footer'

export default function App() {
  return (
    <div className="min-h-screen bg-[#fbf6f2] text-slate-100 selection:bg-teal-400/20 selection:text-teal-300">
      <Navbar />
      <main>
        <AnalysisTool />
        <HowItWorks />
        <TechnicalDetails />
      </main>
      <Footer />
    </div>
  )
}
