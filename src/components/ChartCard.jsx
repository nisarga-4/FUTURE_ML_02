import GlassCard from './GlassCard'
export default function ChartCard({ title, subtitle, children, className='' }) {
  return (
    <GlassCard className={`chart-card ${className}`} hover={false}>
      <div className="chart-head"><div><h3>{title}</h3><span>{subtitle}</span></div></div>
      <div className="chart-body">{children}</div>
    </GlassCard>
  )
}
