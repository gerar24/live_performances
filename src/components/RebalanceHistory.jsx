const toPct = (x) => `${(x * 100).toFixed(1)}%`

const RebalanceHistory = ({ dates, series, rebalances, namesMap = {} }) => {
  const list = Array.isArray(rebalances) ? rebalances : []
  return (
    <div className="w-full bg-neutral-900 rounded-lg border border-neutral-800 p-4">
      <h3 className="text-lg font-medium mb-3">Historial de rebalanceos</h3>
      {list.length === 0 ? (
        <div className="text-neutral-400 text-sm">Sin cambios detectados.</div>
      ) : (
        <div className="space-y-3">
          {list.map((r) => (
            <div key={r.date || Math.random()} className="border border-neutral-800 rounded p-3">
              <div className="text-sm text-neutral-300 mb-1">{r.date}</div>
              {r.notes ? <div className="text-xs text-neutral-400 mb-2">{r.notes}</div> : null}
              <div className="flex flex-wrap gap-2">
                {Object.entries(r.allocation || {})
                  .filter(([, w]) => (w ?? 0) > 0)
                  .sort((a, b) => b[1] - a[1])
                  .map(([t, w]) => (
                    <span
                      key={t}
                      className="text-xs bg-neutral-800 border border-neutral-700 rounded px-2 py-1"
                    >
                      {namesMap[t] || t}: {toPct(w)}
                    </span>
                  ))}
              </div>
              {r.price && Object.keys(r.price).length > 0 ? (
                <div className="mt-2 text-xs text-neutral-400">
                  Precio usado: {Object.entries(r.price).map(([t, p]) => `${namesMap[t] || t}:${p}`).join('  ')}
                </div>
              ) : null}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default RebalanceHistory

