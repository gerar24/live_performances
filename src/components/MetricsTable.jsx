const formatPct = (v) => {
  if (v === null || v === undefined || Number.isNaN(v)) return '-'
  return `${(v * 100).toFixed(2)}%`
}

const formatNum = (v, d = 2) => {
  if (v === null || v === undefined || Number.isNaN(v)) return '-'
  return Number(v).toFixed(d)
}

const MetricsTable = ({ data }) => {
  const portfolioEntries = Object.entries((data && data.portfolios) || {})
  const benchmarkEntries = Object.entries((data && data.benchmarks) || {})

  const Row = ({ name, m }) => (
    <tr className="border-b border-neutral-800">
      <td className="px-4 py-2">{name}</td>
      <td className="px-4 py-2">{formatPct(m?.cagr)}</td>
      <td className="px-4 py-2">{formatNum(m?.sharpe, 2)}</td>
      <td className="px-4 py-2">{formatPct(m?.volatility)}</td>
      <td className="px-4 py-2">{formatPct(m?.max_drawdown)}</td>
      <td className="px-4 py-2">{formatNum(m?.sortino, 2)}</td>
      <td className="px-4 py-2">{formatNum(m?.beta, 2)}{m?.beta_vs ? ` (${m.beta_vs})` : ''}</td>
      <td className="px-4 py-2">{formatPct(m?.alpha)}</td>
      <td className="px-4 py-2">
        {m?.correlations
          ? Object.entries(m.correlations)
              .map(([k, v]) => `${k}:${formatNum(v, 2)}`)
              .join(' ')
          : '-'}
      </td>
    </tr>
  )

  return (
    <div className="w-full bg-neutral-900 rounded-lg border border-neutral-800 overflow-hidden">
      <table className="w-full text-left">
        <thead className="bg-neutral-900/60 border-b border-neutral-800">
          <tr>
            <th className="px-4 py-3 font-medium">Nombre</th>
            <th className="px-4 py-3 font-medium">CAGR</th>
            <th className="px-4 py-3 font-medium">Sharpe</th>
            <th className="px-4 py-3 font-medium">Volatilidad</th>
            <th className="px-4 py-3 font-medium">Max Drawdown</th>
            <th className="px-4 py-3 font-medium">Sortino</th>
            <th className="px-4 py-3 font-medium">Beta</th>
            <th className="px-4 py-3 font-medium">Alpha (Ann.)</th>
            <th className="px-4 py-3 font-medium">Correlaciones</th>
          </tr>
        </thead>
        <tbody>
          {portfolioEntries.map(([name, m]) => (
            <Row key={`p-${name}`} name={name} m={m} />
          ))}
          {benchmarkEntries.map(([name, m]) => (
            <Row key={`b-${name}`} name={name} m={m} />
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default MetricsTable

