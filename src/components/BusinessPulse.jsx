import React from 'react'

export default function BusinessPulse() {
  const metrics = [
    { title: 'Revenue (This Week)', value: '₹2,10,000', subtext: '+14% vs last week', color: 'text-[#10B981]' },
    { title: 'Sell-through Rate', value: '68%', subtext: 'Healthy', color: 'text-[#10B981]' },
    { title: 'Dead Stock Risk', value: 'Moderate', subtext: '4 SKUs flagged', color: 'text-yellow-500' },
    { title: 'Festival Signal', value: '🔥 Rising', subtext: '5 days to predicted peak', color: 'text-[#F59E0B]' },
  ]

  return (
    <section>
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-3xl font-bold text-white">Store Overview</h2>
        <div className="bg-[#F59E0B]/20 border border-[#F59E0B]/40 rounded-full px-4 py-1.5">
          <span className="text-[#F59E0B] text-sm font-semibold">High Demand Mode: Active</span>
        </div>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 grid grid-cols-2 gap-4">
          {metrics.map((metric, idx) => (
            <div key={idx} className="bg-[#1E293B] rounded-lg p-6 hover:shadow-lg transition-shadow">
              <div className="text-gray-400 text-sm mb-2 font-medium">{metric.title}</div>
              <div className={`text-4xl font-bold ${metric.color} mb-1.5`}>{metric.value}</div>
              <div className="text-gray-500 text-sm">{metric.subtext}</div>
            </div>
          ))}
        </div>
        
        <div id="surge-alert" className="bg-gradient-to-br from-[#F59E0B] to-[#D97706] rounded-lg p-10 shadow-2xl lg:row-span-2 flex flex-col justify-center animate-pulse-slow min-h-[320px]">
          <div className="text-3xl font-bold text-white mb-5">🔥 Demand Signal Detected</div>
          <p className="text-white text-xl mb-6 leading-relaxed">
            Maroon Embroidered Sets expected to rise by 32% in Mumbai within 5–7 days.
          </p>
          <div className="space-y-3 text-white">
            <div className="flex justify-between text-lg">
              <span className="font-semibold">Confidence:</span>
              <span className="font-bold">87%</span>
            </div>
            <div className="border-t border-white/30 pt-3">
              <div className="font-semibold mb-2 text-lg">Recommended Action:</div>
              <div className="text-lg">Increase stock by 40 units</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
