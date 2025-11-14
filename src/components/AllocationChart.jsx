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

const colorPalette = [
  '#60a5fa',
  '#34d399',
  '#f472b6',
  '#f59e0b',
  '#a78bfa',
  '#22d3ee',
  '#f87171',
  '#4ade80',
  '#c084fc',
  '#facc15',
]

const AllocationChart = ({ dates, series }) => {
  const chartData = useMemo(() => {
    if (!dates || !series) return { labels: [], datasets: [] }
    const tickers = Object.keys(series)
    const datasets = tickers.map((t, idx) => ({
      label: t,
      data: series[t].map((w) => Math.round(w * 1000) / 10), // percentage
      borderColor: colorPalette[idx % colorPalette.length],
      backgroundColor: colorPalette[idx % colorPalette.length] + '33',
      fill: true,
      pointRadius: 0,
      tension: 0.2,
    }))
    return { labels: dates, datasets }
  }, [dates, series])

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    stacked: true,
    plugins: {
      legend: {
        position: 'top',
        labels: { color: '#d4d4d4', boxWidth: 12 },
      },
      tooltip: {
        callbacks: {
          label: (ctx) => `${ctx.dataset.label}: ${Number(ctx.parsed.y).toFixed(1)}%`,
        },
      },
    },
    scales: {
      x: {
        stacked: true,
        ticks: { color: '#a3a3a3', maxTicksLimit: 8 },
        grid: { color: 'rgba(120,120,120,0.15)' },
      },
      y: {
        stacked: true,
        min: 0,
        max: 100,
        ticks: { color: '#a3a3a3', callback: (v) => `${v}%` },
        grid: { color: 'rgba(120,120,120,0.15)' },
      },
    },
  }

  return (
    <div className="w-full h-80 bg-neutral-900 rounded-lg border border-neutral-800 p-4">
      <Line data={chartData} options={options} />
    </div>
  )
}

export default AllocationChart


