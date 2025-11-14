import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js'
import { useMemo } from 'react'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler)

const primaryColors = [
  '#60a5fa', // blue-400
  '#34d399', // emerald-400
  '#f472b6', // pink-400
  '#f59e0b', // amber-500
  '#a78bfa', // violet-400
  '#22d3ee', // cyan-400
]

const secondaryColors = [
  '#a3a3a3', // neutral-400
  '#737373', // neutral-500
  '#525252', // neutral-600
  '#404040', // neutral-700
]

const EquityChart = ({ data, enabledPortfolios, enabledBenchmarks, namesMap = {} }) => {
  const chartData = useMemo(() => {
    if (!data || !data.dates) {
      return { labels: [], datasets: [] }
    }

    const labels = data.dates
    const datasets = []

    // Portfolios (solid)
    const portfolioEntries = Object.entries(data.portfolios || {})
    portfolioEntries.forEach(([name, series], idx) => {
      if (enabledPortfolios && enabledPortfolios[name] === false) return
      datasets.push({
        label: name,
        data: series,
        borderColor: primaryColors[idx % primaryColors.length],
        backgroundColor: primaryColors[idx % primaryColors.length],
        borderWidth: 2,
        tension: 0.25,
        pointRadius: 0,
      })
    })

    // Benchmarks (dashed, muted)
    const benchEntries = Object.entries(data.benchmarks || {})
    benchEntries.forEach(([name, series], idx) => {
      if (enabledBenchmarks && enabledBenchmarks[name] === false) return
      const display = namesMap[name] || name
      datasets.push({
        label: display,
        data: series,
        borderColor: secondaryColors[idx % secondaryColors.length],
        backgroundColor: secondaryColors[idx % secondaryColors.length],
        borderWidth: 1.5,
        tension: 0.25,
        pointRadius: 0,
        borderDash: [6, 4],
      })
    })

    return { labels, datasets }
  }, [data, enabledPortfolios, enabledBenchmarks, namesMap])

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: {
        position: 'top',
        labels: { color: '#d4d4d4', boxWidth: 12 },
      },
      tooltip: {
        callbacks: {
          label: (ctx) => `${ctx.dataset.label}: ${Number(ctx.parsed.y).toFixed(2)}`,
        },
      },
    },
    scales: {
      x: {
        ticks: { color: '#a3a3a3', maxTicksLimit: 8 },
        grid: { color: 'rgba(120, 120, 120, 0.15)' },
      },
      y: {
        ticks: { color: '#a3a3a3' },
        grid: { color: 'rgba(120, 120, 120, 0.15)' },
      },
    },
  }

  return (
    <div className="w-full h-96 bg-neutral-900 rounded-lg border border-neutral-800 p-4">
      <Line data={chartData} options={options} />
    </div>
  )
}

export default EquityChart

