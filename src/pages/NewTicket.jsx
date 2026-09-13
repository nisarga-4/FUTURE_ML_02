import { useState } from 'react'
import TicketForm from '../components/TicketForm'
import DecisionPanel from '../components/DecisionPanel'
import PageHeader from '../components/PageHeader'
import { BrainCircuit, ShieldCheck, Target } from 'lucide-react'
import MetricCard from '../components/MetricCard'

export default function NewTicket() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  return (
    <>
      <PageHeader title="Intelligent ticket routing" text="Turn incoming support requests into structured, actionable decisions in seconds."/>
      <div className="metrics-grid compact-metrics">
        <MetricCard icon={BrainCircuit} value="93.23%" label="Category Routing" progress={93.23}/>
        <MetricCard icon={ShieldCheck} value="97.45%" label="Priority Accuracy" progress={97.45}/>
        <MetricCard icon={Target} value="98.97%" label="Selective Priority" progress={98.97}/>
      </div>
      <div className="workbench-grid">
        <TicketForm onResult={setResult} onLoading={setLoading}/>
        <DecisionPanel loading={loading} result={result}/>
      </div>
    </>
  )
}
