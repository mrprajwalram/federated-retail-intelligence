import React from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export default function PricingIntelligence() {
  const [hoveredRow, setHoveredRow] = React.useState(null)
  
  const pricingData = [
    { sku: 'Maroon Set', current: '₹1299', recommended: '₹1549', lift: '+12%', risk: 'Low', confidence: 'High', color: 'text-[#10B981]', confColor: 'bg-[#10B981]' },
    { sku: 'Beige Kurti', current: '₹1799', recommended: '₹1699', lift: '+6%', risk: 'Medium', confidence: 'Medium', color: 'text-yellow-500', confColor: 'bg-yellow-500' },
  ]

  const elasticityData = [
    { price: 1100, demand: 95 },
    { price: 1200, demand: 88 },
    { price: 1299, demand: 82, current: true },
    { price: 1400, demand: 78 },
    { price: 1549, demand: 72, recommended: true },
    { price: 1700, demand: 65 },
  ]
  
  const CustomDot = (props) => {
    const { cx, cy, payload } = props
    if (payload.current) {
      return (
        <g>
          <circle cx={cx} cy={cy} r={6} fill="#EF4444" stroke="#fff" strokeWidth={2} />
          <text x={cx} y={cy - 15} textAnchor="middle" fill="#EF4444" fontSize={12} fontWeight="bold">
            Current
          </text>
        </g>
      )
    }
    if (payload.recommended) {
      return (
        <g>
          <circle cx={cx} cy={cy} r={6} fill="#10B981" stroke="#fff" strokeWidth={2} />
          <text x={cx} y={cy - 15} textAnchor="middle" fill="#10B981" fontSize={12} fontWeight="bold">
            Optimal
          </text>
        </g>
      )
    }
    return <circle cx={cx} cy={cy} r={3} fill="#F59E0B" />
  }

  return (
    <section>
      <h2 className="text-3xl font-bold text-white mb-6">Pricing Signals</h2>
      <div className="bg-[#1E293B] rounded-lg p-8">
        <h3 className="text-xl font-semibold text-white mb-6">Recommended Pricing</h3>
        
        <div className="overflow-x-auto mb-6">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left text-gray-400 py-3 px-4 text-sm font-semibold">SKU</th>
                <th className="text-left text-gray-400 py-3 px-4 text-sm font-semibold">Current Price</th>
                <th className="text-left text-gray-400 py-3 px-4 text-sm font-semibold">Recommended</th>
                <th className="text-left text-gray-400 py-3 px-4 text-sm font-semibold">Revenue Lift</th>
                <th className="text-left text-gray-400 py-3 px-4 text-sm font-semibold">Confidence</th>
                <th className="text-left text-gray-400 py-3 px-4 text-sm font-semibold">Action</th>
              </tr>
            </thead>
            <tbody>
              {pricingData.map((item, idx) => (
                <tr 
                  key={idx} 
                  className="border-b border-gray-700/50 hover:bg-[#0F172A]/50 transition-all group cursor-pointer"
                  onMouseEnter={() => setHoveredRow(idx)}
                  onMouseLeave={() => setHoveredRow(null)}
                  style={{
                    boxShadow: hoveredRow === idx ? '0 4px 12px rgba(0,0,0,0.3)' : 'none',
                    transform: hoveredRow === idx ? 'translateY(-2px)' : 'translateY(0)',
                  }}
                >
                  <td className="text-white py-4 px-4 font-medium">{item.sku}</td>
                  <td className="text-gray-300 py-4 px-4 font-medium">{item.current}</td>
                  <td className="py-4 px-4">
                    <span className="text-[#F59E0B] font-bold text-lg bg-[#F59E0B]/10 px-2 py-1 rounded">
                      {item.recommended}
                    </span>
                  </td>
                  <td className="text-[#10B981] py-4 px-4 font-bold text-lg">
                    {item.lift} <span className="text-sm">↑</span>
                  </td>
                  <td className="py-4 px-4">
                    <span className={`${item.confColor} text-white px-2 py-1 rounded-full text-xs font-semibold`}>
                      {item.confidence}
                    </span>
                  </td>
                  <td className="py-4 px-4">
                    <button 
                      className={`text-[#F59E0B] text-sm font-semibold transition-opacity ${
                        hoveredRow === idx ? 'opacity-100' : 'opacity-0'
                      }`}
                    >
                      Apply Price →
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="bg-[#0F172A] rounded-lg p-6">
          <h4 className="text-white font-semibold mb-4 text-lg">Price Elasticity – Maroon Set</h4>
          <ResponsiveContainer width="100%" height={270}>
            <LineChart data={elasticityData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" strokeOpacity={0.5} />
              <XAxis 
                dataKey="price" 
                stroke="#9CA3AF" 
                label={{ value: 'Price (₹)', position: 'insideBottom', offset: -5, fill: '#9CA3AF', fontSize: 12 }} 
                style={{ fontSize: '12px' }}
              />
              <YAxis 
                stroke="#9CA3AF" 
                label={{ value: 'Demand', angle: -90, position: 'insideLeft', fill: '#9CA3AF', fontSize: 12 }} 
                style={{ fontSize: '12px' }}
              />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1E293B', border: '1px solid #374151', borderRadius: '8px' }}
                labelStyle={{ color: '#F59E0B' }}
              />
              <Line type="monotone" dataKey="demand" stroke="#F59E0B" strokeWidth={3} dot={<CustomDot />} />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-gray-400 text-sm mt-4">
            Similar retailers saw stable demand even after price increases.
          </p>
        </div>
      </div>
    </section>
  )
}
