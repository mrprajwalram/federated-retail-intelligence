import React, { useState } from 'react'

export default function Navbar() {
  const [isUpdating, setIsUpdating] = useState(false)
  const [showToast, setShowToast] = useState(false)
  
  const handleUpdate = () => {
    setIsUpdating(true)
    
    // Trigger surge card pulse
    const surgeCard = document.getElementById('surge-alert')
    if (surgeCard) {
      surgeCard.classList.add('animate-pulse')
    }
    
    setTimeout(() => {
      setIsUpdating(false)
      setShowToast(true)
      
      if (surgeCard) {
        surgeCard.classList.remove('animate-pulse')
      }
      
      setTimeout(() => setShowToast(false), 4000)
    }, 2000)
  }
  
  return (
    <>
      <nav className="bg-[#1E293B] border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-6 py-3 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <img src="/drishti-logo.png" alt="Drishti - Smarter, Together" className="h-14" onError={(e) => {e.target.style.display='none'; e.target.nextSibling.style.display='block'}} />
            <div style={{display: 'none'}}>
              <h1 className="text-2xl font-bold text-[#F59E0B]">Drishti</h1>
              <div className="text-gray-400 text-xs mt-0.5" style={{ opacity: 0.65 }}>Smarter Decisions, Together.</div>
            </div>
          </div>
          <div className="flex items-center gap-6">
            <div className="flex flex-col items-end gap-1">
              <div className="flex items-center gap-3">
                <div className="text-gray-300 font-medium">Savita Boutique — Mumbai</div>
                <div className="bg-[#10B981]/20 border border-[#10B981]/40 rounded-full px-3 py-1.5 flex items-center gap-2">
                  <span className="text-[#10B981] text-lg">🌐</span>
                  <span className="text-[#10B981] text-sm font-semibold">Connected to 43 similar boutiques</span>
                </div>
              </div>
            </div>
            <button 
              onClick={handleUpdate}
              disabled={isUpdating}
              className={`px-5 py-2.5 rounded-lg font-semibold transition-all flex items-center gap-2 ${
                isUpdating 
                  ? 'bg-[#F59E0B]/50 text-white cursor-wait' 
                  : 'bg-[#F59E0B] text-white hover:bg-[#D97706] hover:shadow-lg'
              }`}
            >
              {isUpdating ? (
                <>
                  <span className="animate-spin">⟳</span>
                  <span>Updating...</span>
                </>
              ) : (
                <>
                  <span>⟳</span>
                  <span>Refresh Insights</span>
                </>
              )}
            </button>
          </div>
        </div>
      </nav>
      
      {showToast && (
        <div className="fixed top-20 right-6 bg-[#10B981] text-white px-6 py-3 rounded-lg shadow-2xl animate-slide-in z-50">
          <div className="flex items-center gap-2">
            <span className="text-xl">✓</span>
            <span className="font-semibold">Network insights updated.</span>
          </div>
        </div>
      )}
    </>
  )
}
