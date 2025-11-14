const percent = (x) => `${(x * 100).toFixed(1)}%`

const CurrentAllocation = ({ dates, series }) => {
  if (!dates || !series) {
    return null
  }
  const lastIdx = dates.length - 1
  const current = Object.entries(series).map(([ticker, arr]) => ({
    ticker,
    weight: arr[lastIdx] ?? 0,
  }))
  const sorted = current.sort((a, b) => b.weight - a.weight)

  return (
    <div className="w-full bg-neutral-900 rounded-lg border border-neutral-800 p-4">
      <h3 className="text-lg font-medium mb-3">Asignación actual</h3>
      <div className="space-y-2">
        {sorted.map(({ ticker, weight }) => (
          <div key={ticker}>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-neutral-300">{ticker}</span>
              <span className="text-neutral-400">{percent(weight)}</span>
            </div>
            <div className="w-full h-2 bg-neutral-800 rounded">
              <div
                className="h-2 bg-sky-500 rounded"
                style={{ width: `${Math.max(0, Math.min(100, weight * 100))}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default CurrentAllocation


