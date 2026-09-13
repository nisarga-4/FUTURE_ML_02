import { useMemo, useState } from 'react'
import { Search, SlidersHorizontal } from 'lucide-react'
import PageHeader from '../components/PageHeader'
import GlassCard from '../components/GlassCard'
import { reviewTickets } from '../data/mockData'

export default function ReviewQueue() {
  const [q, setQ] = useState('')
  const [priority, setPriority] = useState('All')
  const rows = useMemo(() => reviewTickets.filter(t =>
    (priority === 'All' || t.priority === priority) &&
    `${t.id} ${t.subject} ${t.category} ${t.team}`.toLowerCase().includes(q.toLowerCase())
  ), [q, priority])

  return (
    <>
      <PageHeader title="Review Queue" text="Inspect uncertain, high-impact, or policy-sensitive AI decisions before final routing."/>
      <GlassCard className="table-card" hover={false}>
        <div className="table-toolbar">
          <label className="queue-search"><Search size={17}/><input value={q} onChange={e=>setQ(e.target.value)} placeholder="Search queue..."/></label>
          <label className="filter-select"><SlidersHorizontal size={17}/><select value={priority} onChange={e=>setPriority(e.target.value)}>
            <option>All</option><option>Critical</option><option>High</option><option>Medium</option><option>Low</option>
          </select></label>
        </div>
        <div className="responsive-table">
          <table>
            <thead><tr><th>Ticket</th><th>Category</th><th>Priority</th><th>Confidence</th><th>Assigned team</th><th>Timestamp</th><th>Status</th></tr></thead>
            <tbody>
              {rows.map(t => (
                <tr key={t.id}>
                  <td><strong>{t.subject}</strong><span>{t.id}</span></td>
                  <td><span className="category-pill">{t.category}</span></td>
                  <td><span className={`priority-tag ${t.priority.toLowerCase()}`}>{t.priority}</span></td>
                  <td><div className="mini-confidence"><span style={{width:`${t.confidence}%`}}/><b>{t.confidence}%</b></div></td>
                  <td>{t.team}</td><td>{t.time}</td>
                  <td><span className={`status-tag ${t.status === 'Approved' ? 'approved':''}`}>{t.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </GlassCard>
    </>
  )
}
