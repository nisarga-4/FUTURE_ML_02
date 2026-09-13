import { useState } from 'react'
import PageHeader from '../components/PageHeader'
import GlassCard from '../components/GlassCard'
import { Switch } from '../components/Field'

export default function Settings() {
  const [state, setState] = useState({ autoRouting:true, selective:true, notifications:true, confidence:true })
  const change = e => setState(v=>({...v,[e.target.name]:e.target.checked}))
  return (
    <>
      <PageHeader title="Settings" text="Control AI decision policies, confidence thresholds and workspace behavior."/>
      <div className="settings-grid">
        <GlassCard hover={false} className="settings-card">
          <h3>AI decision policy</h3><p>Configure how SupportFlow handles predictions.</p>
          <Switch label="Automatic routing" name="autoRouting" checked={state.autoRouting} onChange={change}/>
          <Switch label="Selective priority" name="selective" checked={state.selective} onChange={change}/>
          <Switch label="Show confidence scores" name="confidence" checked={state.confidence} onChange={change}/>
          <label className="field"><span>Minimum automatic confidence</span><input type="range" defaultValue="86" min="60" max="99"/></label>
        </GlassCard>
        <GlassCard hover={false} className="settings-card">
          <h3>Workspace preferences</h3><p>Adjust experience and notification preferences.</p>
          <Switch label="Review notifications" name="notifications" checked={state.notifications} onChange={change}/>
          <label className="field"><span>Default queue view</span><select defaultValue="Confidence"><option>Confidence</option><option>Priority</option><option>Newest</option></select></label>
          <label className="field"><span>Timezone</span><select defaultValue="Asia/Kolkata"><option>Asia/Kolkata</option><option>UTC</option><option>America/New_York</option></select></label>
          <button className="primary-btn settings-save">Save changes</button>
        </GlassCard>
      </div>
    </>
  )
}
