import React from 'react'
import Navbar from './components/Navbar'
import BusinessPulse from './components/BusinessPulse'
import DemandIntelligence from './components/DemandIntelligence'
import PricingIntelligence from './components/PricingIntelligence'
import SmartBundling from './components/SmartBundling'
import InventoryRisk from './components/InventoryRisk'
import Footer from './components/Footer'

function App() {
  return (
    <div className="min-h-screen bg-[#0F172A]">
      <Navbar />
      <main className="max-w-7xl mx-auto px-6 py-8 space-y-12">
        <BusinessPulse />
        <DemandIntelligence />
        <PricingIntelligence />
        <SmartBundling />
        <InventoryRisk />
      </main>
      <Footer />
    </div>
  )
}

export default App
