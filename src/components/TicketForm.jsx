import { useState } from 'react'
import { RotateCcw, Sparkles } from 'lucide-react'
import GlassCard from './GlassCard'
import { SelectField, Switch } from './Field'
import { classifyTicket } from '../services/api'

const initial = {
  description:'',
  affected:'',
  downtime:'',
  errorRate:'',
  productArea:'',
  customerTier:'',
  serviceDisruption:false,
  securityRelated:false,
  revenueImpact:false,
  vipCustomer:false
}

export default function TicketForm({ onResult, onLoading }) {
  const [form, setForm] = useState(initial)
  const [errors, setErrors] = useState({})

  const change = (e) => {
    const { name, value, checked, type } = e.target
    setForm(v => ({...v, [name]: type === 'checkbox' ? checked : value}))
    setErrors(v => ({...v, [name]: ''}))
  }

  const validate = () => {
    const next = {}
    if (form.description.trim().length < 20) next.description = 'Please provide at least 20 characters.'
    if (!form.affected) next.affected = 'Select the affected customer count.'
    if (!form.productArea) next.productArea = 'Select a product area.'
    setErrors(next)
    return Object.keys(next).length === 0
  }

  const submit = async (e) => {
    e.preventDefault()
    if (!validate()) return
    onLoading(true)
    onResult(null)
    try {
      const result = await classifyTicket(form)
      onResult(result)
    } finally {
      onLoading(false)
    }
  }

  return (
    <GlassCard className="ticket-form-card" strong hover={false}>
      <div className="section-heading">
        <div><p className="eyebrow">AI ROUTING</p><h2>New Support Request</h2>
        <span>Describe the issue and let AI classify and prioritize it.</span></div>
      </div>

      <form onSubmit={submit}>
        <label className="field textarea-field">
          <span>Ticket description <em>*</em></span>
          <textarea
            name="description"
            value={form.description}
            onChange={change}
            maxLength={1200}
            placeholder="Example: After the latest graphics driver update, 18 design laptops are showing intermittent screen flicker and two users can no longer use external displays..."
            aria-invalid={!!errors.description}
          />
          <div className="field-meta">
            <small className={errors.description ? 'error-text':''}>{errors.description || 'Include symptoms, timing, and business impact for better results.'}</small>
            <small>{form.description.length}/1200</small>
          </div>
        </label>

        <div className="form-grid">
          <SelectField label="Customers affected" name="affected" value={form.affected} onChange={change} required error={errors.affected}
            options={['1 customer','2–10 customers','11–50 customers','51–250 customers','250+ customers']} />
          <SelectField label="Estimated downtime" name="downtime" value={form.downtime} onChange={change}
            options={['No downtime','< 15 minutes','15–60 minutes','1–4 hours','4+ hours']} />
          <SelectField label="Error rate" name="errorRate" value={form.errorRate} onChange={change}
            options={['< 1%','1–5%','5–15%','15–40%','40%+']} />
          <SelectField label="Product area" name="productArea" value={form.productArea} onChange={change} required error={errors.productArea}
            options={['Hardware','Identity & Access','Storage','Messaging','HR Systems','Procurement','Other']} />
          <SelectField label="Customer tier" name="customerTier" value={form.customerTier} onChange={change}
            options={['Standard','Business','Enterprise','Strategic']} />
        </div>

        <div className="impact-block">
          <div className="impact-title"><strong>Impact indicators</strong><span>Optional signals improve priority prediction.</span></div>
          <div className="switch-grid">
            <Switch label="Service disruption" name="serviceDisruption" checked={form.serviceDisruption} onChange={change}/>
            <Switch label="Security related" name="securityRelated" checked={form.securityRelated} onChange={change}/>
            <Switch label="Revenue impact" name="revenueImpact" checked={form.revenueImpact} onChange={change}/>
            <Switch label="VIP customer" name="vipCustomer" checked={form.vipCustomer} onChange={change}/>
          </div>
        </div>

        <div className="form-actions">
          <button type="button" className="secondary-btn" onClick={() => {setForm(initial); setErrors({}); onResult(null)}}><RotateCcw size={17}/>Clear</button>
          <button className="primary-btn" type="submit"><Sparkles size={18}/>Classify with AI</button>
        </div>
      </form>
    </GlassCard>
  )
}
