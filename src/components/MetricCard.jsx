import GlassCard from './GlassCard'

export default function MetricCard({ icon:Icon, value, label, progress }) {
  return (
    <GlassCard className="metric-card">
      <div className="metric-icon"><Icon size={22}/></div>
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
      <div className="progress-track" aria-label={`${label} ${value}`}>
        <span style={{ width: `${progress}%` }} />
      </div>
    </GlassCard>
  )
}
