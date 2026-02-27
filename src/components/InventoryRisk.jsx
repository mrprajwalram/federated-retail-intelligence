import React from 'react'

export default function InventoryRisk() {
  const risks = [
    { sku: 'Blue Printed Kurti', level: 'High', reason: 'Demand decline detected', action: 'Bundle or discount 5–8%', color: 'bg-[#EF4444]', borderColor: 'border-l-[#EF4444]', bgColor: 'bg-[#EF4444]/5', icon: '⚠️', badge: 'Action Recommended' },
    { sku: 'Yellow Festive Set', level: 'Medium', reason: 'Low cross-region demand', action: 'Promote locally', color: 'bg-yellow-500', borderColor: 'border-l-yellow-500', bgColor: '', icon: '⚡', badge: null },
  ]

  return (
    <section>
      <h2 className="text-3xl font-bold text-white mb-6">Stock Watch</h2>
      <div className="bg-[#1E293B] rounded-lg p-8">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left text-gray-400 py-3 px-4 text-sm font-semibold">SKU</th>
                <th className="text-left text-gray-400 py-3 px-4 text-sm font-semibold">Risk Level</th>
                <th className="text-left text-gray-400 py-3 px-4 text-sm font-semibold">Reason</th>
                <th className="text-left text-gray-400 py-3 px-4 text-sm font-semibold">Suggested Action</th>
              </tr>
            </thead>
            <tbody>
              {risks.map((risk, idx) => (
                <tr key={idx} className={`border-b border-gray-700/50 border-l-4 ${risk.borderColor} ${risk.bgColor}`}>
                  <td className="text-white py-5 px-4 font-medium">
                    <div className="flex items-center gap-2">
                      <span className="text-xl">{risk.icon}</span>
                      <span>{risk.sku}</span>
                      {risk.badge && (
                        <span className="ml-2 bg-[#EF4444] text-white px-2 py-0.5 rounded text-xs font-semibold">
                          {risk.badge}
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="py-5 px-4">
                    <span className={`${risk.color} text-white px-3 py-1.5 rounded-full text-sm font-semibold`}>
                      {risk.level}
                    </span>
                  </td>
                  <td className="text-gray-300 py-5 px-4 font-medium">{risk.reason}</td>
                  <td className="text-white py-5 px-4 font-semibold">{risk.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  )
}
