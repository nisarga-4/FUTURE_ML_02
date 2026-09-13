import { BrainCircuit, ShieldCheck, Target, ArrowRight, Clock3, ListChecks } from 'lucide-react'
import { Link } from 'react-router-dom'
import MetricCard from '../components/MetricCard'
import GlassCard from '../components/GlassCard'
import PageHeader from '../components/PageHeader'
import { reviewTickets } from '../data/mockData'

export default function Dashboard() {
  return (
    <>
      <PageHeader
        title="Good afternoon, Avery."
        text="Your AI support operation is healthy. Here’s what needs attention today."
        actions={<Link className="primary-btn compact-btn" to="/new-ticket">New ticket <ArrowRight size={16}/></Link>}
      />
      <div className="metrics-grid">
        <MetricCard icon={BrainCircuit} value="93.23%" label="Category Routing" progress={93.23}/>
        <MetricCard icon={ShieldCheck} value="97.45%" label="Priority Accuracy" progress={97.45}/>
        <MetricCard icon={Target} value="98.97%" label="Selective Priority" progress={98.97}/>
      </div>

      <div className="dashboard-grid">
        <GlassCard className="overview-card" hover={false}>
          <div className="section-heading compact"><div><p className="eyebrow">TODAY</p><h2>Support overview</h2><span>Operational snapshot across your support workflow.</span></div></div>
          <div className="overview-stats">
            <div><strong>184</strong><span>Tickets classified</span></div>
            <div><strong>27</strong><span>Needs review</span></div>
            <div><strong>24m</strong><span>Median response</span></div>
            <div><strong>92.6%</strong><span>Auto-resolution eligible</span></div>
          </div>
          <div className="flow-banner">
            <div className="flow-icon"><BrainCircuit size={22}/></div>
            <div><strong>AI routing is performing above target.</strong><p>Only 6.8% of today’s tickets required human category correction.</p></div>
          </div>
        </GlassCard>

        <GlassCard className="attention-card" hover={false}>
          <div className="section-heading compact"><div><p className="eyebrow">QUEUE</p><h2>Needs attention</h2><span>Low-confidence and high-impact requests.</span></div><ListChecks size={20}/></div>
          <div className="attention-list">
            {reviewTickets.slice(0,4).map(t => (
              <div key={t.id} className="attention-item">
                <div className="ticket-dot"/>
                <div><strong>{t.subject}</strong><span>{t.id} · {t.category}</span></div>
                <span className={`priority-tag ${t.priority.toLowerCase()}`}>{t.priority}</span>
              </div>
            ))}
          </div>
          <Link className="text-link" to="/review-queue">Open review queue <ArrowRight size={15}/></Link>
        </GlassCard>
      </div>

      <GlassCard className="activity-strip" hover={false}>
        <div><Clock3 size={19}/><span>Latest model sync</span><strong>12 minutes ago</strong></div>
        <div><span>Routing model</span><strong>v3.8.2</strong></div>
        <div><span>Priority model</span><strong>v2.6.1</strong></div>
        <div><span>Inference health</span><strong className="success-text">Healthy</strong></div>
      </GlassCard>
    </>
  )
}
