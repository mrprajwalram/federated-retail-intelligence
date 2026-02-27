import React from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, Legend } from 'recharts'

export default function DemandIntelligence() {
  const historicalData = [
    { date: 'Jan 28', units: 12 },
    { date: 'Feb 4', units: 15 },
    { date: 'Feb 11', units: 18 },
    { date: 'Feb 18', units: 22 },
    { date: 'Feb 25', units: 28 },
  ]
  
  const predictedData = [
    { date: 'Feb 25', units: 28 },
    { date: 'Mar 4', units: 38 },
    { date: 'Mar 11', units: 45 },
  ]
  
  const confidenceBand = [
    { date: 'Feb 25', lower: 28, upper: 28 },
    { date: 'Mar 4', lower: 34, upper: 42 },
    { date: 'Mar 11', lower: 40, upper: 50 },
  ]

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      const isPredicted = predictedData.some(d => d.date === data.date)
      return (
        <div className="bg-[#1E293B] border border-gray-600 rounded-lg p-3 shadow-xl">
          <p className="text-[#F59E0B] font-semibold mb-1">{data.date}</p>
          <p className="text-white">
            {isPredicted ? 'Predicted' : 'Actual'} units: <span className="font-bold">{data.units}</span>
          </p>
          {isPredicted && <p className="text-gray-400 text-sm">Confidence: 87%</p>}
        </div>
      )
    }
    return null
  }

  return (
    <section>
      <h2 className="text-3xl font-bold text-white mb-6">Demand Signals</h2>
      <div className="bg-[#1E293B] rounded-lg p-8">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-xl font-semibold text-white">Demand Forecast</h3>
          <select className="bg-[#0F172A] text-white px-4 py-2 rounded-lg border border-gray-600 hover:border-[#F59E0B] transition-colors cursor-pointer font-medium">
            <option>Maroon Embroidered Set</option>
          </select>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className="lg:col-span-3 relative">
            <ResponsiveContainer width="100%" height={420}>
              <LineChart>
                <defs>
                  <linearGradient id="confidenceBand" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10B981" stopOpacity={0.2}/>
                    <stop offset="95%" stopColor="#10B981" stopOpacity={0.05}/>
                  </linearGradient>
                  <linearGradient id="predictedGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#F59E0B" stopOpacity={0.15}/>
                    <stop offset="95%" stopColor="#F59E0B" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" strokeOpacity={0.5} />
                <XAxis dataKey="date" stroke="#9CA3AF" style={{ fontSize: '12px' }} />
                <YAxis stroke="#9CA3AF" style={{ fontSize: '12px' }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend 
                  wrapperStyle={{ paddingTop: '20px' }}
                  iconType="line"
                  formatter={(value) => <span style={{ color: '#9CA3AF', fontSize: '13px' }}>{value}</span>}
                />
                <Area 
                  data={confidenceBand}
                  type="monotone" 
                  dataKey="upper" 
                  stroke="none"
                  fill="url(#confidenceBand)"
                />
                <Area 
                  data={confidenceBand}
                  type="monotone" 
                  dataKey="lower" 
                  stroke="none"
                  fill="#0F172A"
                />
                <Line 
                  data={historicalData}
                  type="monotone" 
                  dataKey="units" 
                  stroke="#10B981" 
                  strokeWidth={3}
                  dot={{ fill: '#10B981', r: 5 }}
                  name="Historical Data"
                />
                <Line 
                  data={predictedData}
                  type="monotone" 
                  dataKey="units" 
                  stroke="#F59E0B" 
                  strokeWidth={3}
                  strokeDasharray="8 4"
                  dot={{ fill: '#F59E0B', r: 5 }}
                  name="Predicted Demand"
                />
              </LineChart>
            </ResponsiveContainer>
            <div className="absolute top-1/2 left-[70%] transform -translate-y-1/2 pointer-events-none">
              <div className="border-l-2 border-dashed border-[#F59E0B]/40 h-72"></div>
              <div className="absolute -top-8 left-2 bg-[#F59E0B]/20 border border-[#F59E0B]/40 rounded px-2 py-1 text-xs text-[#F59E0B] font-semibold whitespace-nowrap">
                Predicted Surge Window
              </div>
            </div>
          </div>
          
          <div className="space-y-4">
            <div className="bg-[#0F172A] rounded-lg p-5">
              <div className="text-gray-400 text-sm mb-2 font-medium">Current Inventory</div>
              <div className="text-3xl font-bold text-white">120</div>
            </div>
            <div className="bg-[#0F172A] rounded-lg p-5">
              <div className="text-gray-400 text-sm mb-2 font-medium">Recommended Inventory</div>
              <div className="text-3xl font-bold text-[#F59E0B]">160</div>
            </div>
            <div className="bg-[#0F172A] rounded-lg p-5">
              <div className="text-gray-400 text-sm mb-2 font-medium">Stock-out Risk</div>
              <div className="text-3xl font-bold text-[#10B981]">Low</div>
            </div>
          </div>
        </div>
        
        <p className="text-gray-400 text-sm mt-4">
          Pattern observed across retailers like you.
        </p>
      </div>
    </section>
  )
}
