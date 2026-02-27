import React, { useState } from 'react'

export default function SmartBundling() {
  const [showWhy, setShowWhy] = useState(false)
  
  return (
    <section>
      <h2 className="text-3xl font-bold text-white mb-6">Smart Bundles</h2>
      <div className="bg-[#1E293B] rounded-lg p-8">
        <h3 className="text-xl font-semibold text-white mb-6">Bundle Opportunities</h3>
        
        <div className="bg-gradient-to-r from-[#10B981]/20 to-[#10B981]/5 border border-[#10B981]/30 rounded-lg p-6 mb-6">
          <div className="flex justify-between items-start mb-4">
            <div className="flex items-center gap-4">
              <div className="bg-[#0F172A] rounded-lg p-3 border border-gray-600">
                <div className="text-3xl">👗</div>
              </div>
              <div className="text-xl">+</div>
              <div className="bg-[#0F172A] rounded-lg p-3 border border-gray-600">
                <div className="text-3xl">💍</div>
              </div>
              <div className="ml-2">
                <div className="text-2xl font-bold text-white">
                  Maroon Set + Oxidized Earrings
                </div>
              </div>
            </div>
            <button 
              onClick={() => setShowWhy(!showWhy)}
              className="text-[#10B981] text-sm font-semibold hover:text-[#059669] transition-colors"
            >
              {showWhy ? '− Hide Details' : '+ Why this bundle?'}
            </button>
          </div>
          
          {showWhy && (
            <div className="bg-[#0F172A]/50 rounded p-4 mb-4 space-y-2 text-gray-300 text-sm">
              <div className="flex items-start gap-2">
                <span className="text-[#10B981]">•</span>
                <span>Retailers like you often sell these together</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-[#10B981]">•</span>
                <span>Peak pairing during festive weekends</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-[#10B981]">•</span>
                <span>High margin accessory pairing</span>
              </div>
            </div>
          )}
          
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div>
              <div className="text-gray-400 text-sm font-medium">Expected Conversion Lift</div>
              <div className="text-[#10B981] text-3xl font-bold">+18%</div>
            </div>
            <div>
              <div className="text-gray-400 text-sm font-medium">Margin Impact</div>
              <div className="text-[#10B981] text-3xl font-bold">+14%</div>
            </div>
            <div>
              <div className="text-gray-400 text-sm font-medium">Confidence</div>
              <div className="text-white text-3xl font-bold">High</div>
            </div>
          </div>
          
          <div className="bg-[#0F172A]/50 rounded p-4">
            <div className="text-gray-400 text-sm mb-2 font-medium">Margin Comparison</div>
            <div className="space-y-2">
              <div className="flex items-center gap-3">
                <div className="text-gray-300 text-sm w-32">Individual:</div>
                <div className="flex-1 bg-gray-700 rounded-full h-3 overflow-hidden">
                  <div className="bg-gray-500 h-full" style={{ width: '58%' }}></div>
                </div>
                <div className="text-gray-300 text-sm font-semibold">58%</div>
              </div>
              <div className="flex items-center gap-3">
                <div className="text-[#10B981] text-sm w-32 font-medium">Bundle:</div>
                <div className="flex-1 bg-gray-700 rounded-full h-3 overflow-hidden">
                  <div className="bg-[#10B981] h-full" style={{ width: '72%' }}></div>
                </div>
                <div className="text-[#10B981] text-sm font-bold">72%</div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-[#0F172A] rounded-lg p-8">
          <h4 className="text-white font-semibold mb-6 text-lg">Bundle Simulator</h4>
          <div className="grid grid-cols-3 gap-6 mb-6">
            <div>
              <div className="text-gray-400 text-sm mb-2 font-medium">Bundle Price</div>
              <div className="text-[#F59E0B] text-4xl font-bold">₹1,799</div>
            </div>
            <div>
              <div className="text-gray-400 text-sm mb-2 font-medium">Projected Revenue</div>
              <div className="text-white text-4xl font-bold">₹2,45,000</div>
            </div>
            <div>
              <div className="text-gray-400 text-sm mb-2 font-medium">Inventory Clearance Speed</div>
              <div className="text-[#10B981] text-4xl font-bold">2x Faster</div>
            </div>
          </div>
          <div className="border-t border-gray-700 pt-4 space-y-2">
            <p className="text-gray-400 text-sm">
              Popular pairing during festive season.
            </p>
            <div className="flex items-center gap-2 text-[#10B981] text-sm font-semibold">
              <span>💡</span>
              <span>Bundle reduces slow-moving accessory inventory by 37%</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
