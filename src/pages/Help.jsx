import { BookOpen, MessageSquareText, ShieldQuestion, Workflow, ArrowRight } from 'lucide-react'
import PageHeader from '../components/PageHeader'
import GlassCard from '../components/GlassCard'

const cards = [
  [BookOpen,'Getting started','Learn the routing workflow, confidence system and review process.'],
  [Workflow,'Model decisions','Understand category prediction, priority scoring and selective automation.'],
  [ShieldQuestion,'Review policy','See when tickets are automatically routed versus sent for human review.'],
  [MessageSquareText,'Support','Find integration guidance and contact your internal platform administrator.']
]

export default function Help() {
  return (
    <>
      <PageHeader title="Help Center" text="Everything your support team needs to use SupportFlow AI confidently."/>
      <div className="help-grid">
        {cards.map(([Icon,title,text]) => (
          <GlassCard className="help-card" key={title}>
            <div className="help-icon"><Icon size={22}/></div>
            <h3>{title}</h3><p>{text}</p><button className="text-link">Open guide <ArrowRight size={15}/></button>
          </GlassCard>
        ))}
      </div>
      <GlassCard className="faq-card" hover={false}>
        <h2>Frequently asked</h2>
        {[
          ['What happens when model confidence is low?','The ticket can be diverted to Review Queue based on your configured threshold rather than automatically finalized.'],
          ['Can this connect to a Python ML backend?','Yes. The frontend already isolates inference inside src/services/api.js so you can replace the mock response with Flask or FastAPI calls.'],
          ['Is the interface accessible?','Interactive controls use semantic elements, labels, focus states, readable contrast and reduced-motion support.']
        ].map(([q,a])=><details key={q}><summary>{q}</summary><p>{a}</p></details>)}
      </GlassCard>
    </>
  )
}
