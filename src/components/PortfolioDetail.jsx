import AllocationChart from './AllocationChart.jsx'
import CurrentAllocation from './CurrentAllocation.jsx'
import RebalanceHistory from './RebalanceHistory.jsx'

const PortfolioDetail = ({ selectedPortfolio, allocationData }) => {
  if (!selectedPortfolio || !allocationData) return null
  const entry = allocationData[selectedPortfolio]
  if (!entry) return null

  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold">{selectedPortfolio}</h2>
      <AllocationChart dates={entry.dates} series={entry.series} namesMap={allocationData.names || {}} />
      <div className="grid md:grid-cols-2 gap-4">
        <CurrentAllocation dates={entry.dates} series={entry.series} namesMap={allocationData.names || {}} />
        <RebalanceHistory dates={entry.dates} series={entry.series} rebalances={entry.rebalances} namesMap={allocationData.names || {}} />
      </div>
    </section>
  )
}

export default PortfolioDetail

