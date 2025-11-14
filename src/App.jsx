import { useEffect, useMemo, useState } from 'react'
import Header from './components/Header.jsx'
import EquityChart from './components/EquityChart.jsx'
import MetricsTable from './components/MetricsTable.jsx'
import PortfolioDetail from './components/PortfolioDetail.jsx'

const baseUrl = import.meta.env.BASE_URL || '/'
const withBase = (p) => {
  // Ensure single slash between base and path
  if (baseUrl.endsWith('/')) {
    return `${baseUrl}${p.replace(/^\//, '')}`
  }
  return `${baseUrl}/${p.replace(/^\//, '')}`
}

const fetchJson = async (path) => {
  const res = await fetch(withBase(path))
  if (!res.ok) throw new Error(`Failed to load ${path}`)
  return res.json()
}

function App() {
  const [equity, setEquity] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [allocations, setAllocations] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedPortfolio, setSelectedPortfolio] = useState('')
  const [enabledPortfolios, setEnabledPortfolios] = useState({})
  const [enabledBenchmarks, setEnabledBenchmarks] = useState({})
  const namesMap = useMemo(() => (equity?.names || {}), [equity])

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    Promise.all([
      fetchJson('data/equity_curve.json'),
      fetchJson('data/metrics.json'),
      fetchJson('data/allocation_history.json'),
    ])
      .then(([e, m, a]) => {
        if (cancelled) return
        setEquity(e)
        setMetrics(m)
        setAllocations(a)
        const firstPortfolio = Object.keys(a || {})[0] || ''
        setSelectedPortfolio(firstPortfolio)
        // Initialize toggles (all enabled)
        const ep = Object.keys(e?.portfolios || {}).reduce((acc, k) => {
          acc[k] = true
          return acc
        }, {})
        const eb = Object.keys(e?.benchmarks || {}).reduce((acc, k) => {
          acc[k] = true
          return acc
        }, {})
        setEnabledPortfolios(ep)
        setEnabledBenchmarks(eb)
      })
      .catch((err) => {
        if (!cancelled) setError(err.message || String(err))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const portfolioNames = useMemo(() => Object.keys(allocations || {}), [allocations])

  return (
    <div className="min-h-screen">
      <Header />
      <main className="max-w-6xl mx-auto px-4 space-y-6 pb-10">
        {loading ? (
          <div className="text-neutral-400">Cargando...</div>
        ) : error ? (
          <div className="text-red-400">Error: {error}</div>
        ) : (
          <>
            <section className="bg-neutral-900 border border-neutral-800 rounded-lg p-4 space-y-3">
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <div className="text-sm text-neutral-300 mb-2">Portfolios</div>
                  <div className="flex flex-wrap gap-3">
                    {Object.keys(equity?.portfolios || {}).map((name) => (
                      <label key={`ep-${name}`} className="flex items-center gap-2 text-sm">
                        <input
                          type="checkbox"
                          className="accent-sky-500"
                          checked={enabledPortfolios[name] ?? true}
                          onChange={(e) =>
                            setEnabledPortfolios((prev) => ({ ...prev, [name]: e.target.checked }))
                          }
                        />
                        <span>{name}</span>
                      </label>
                    ))}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-neutral-300 mb-2">Benchmarks</div>
                  <div className="flex flex-wrap gap-3">
                    {Object.keys(equity?.benchmarks || {}).map((name) => (
                      <label key={`eb-${name}`} className="flex items-center gap-2 text-sm">
                        <input
                          type="checkbox"
                          className="accent-sky-500"
                          checked={enabledBenchmarks[name] ?? true}
                          onChange={(e) =>
                            setEnabledBenchmarks((prev) => ({ ...prev, [name]: e.target.checked }))
                          }
                        />
                        <span>{namesMap[name] || name}</span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>
              <EquityChart
                data={equity}
                enabledPortfolios={enabledPortfolios}
                enabledBenchmarks={enabledBenchmarks}
                namesMap={namesMap}
              />
            </section>
            <MetricsTable data={metrics} namesMap={namesMap} />

            <section className="bg-neutral-900 border border-neutral-800 rounded-lg p-4">
              <div className="flex flex-col sm:flex-row sm:items-center gap-3 sm:gap-6">
                <label className="text-sm text-neutral-300">Selecciona un portfolio</label>
                <select
                  className="bg-neutral-800 border border-neutral-700 rounded px-3 py-2 text-sm"
                  value={selectedPortfolio}
                  onChange={(e) => setSelectedPortfolio(e.target.value)}
                >
                  {portfolioNames.map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
              </div>
            </section>

            <PortfolioDetail
              selectedPortfolio={selectedPortfolio}
              allocationData={{ ...allocations, names: namesMap }}
            />
          </>
        )}
      </main>
    </div>
  )
}

export default App

