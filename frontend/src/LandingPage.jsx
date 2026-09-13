// Public landing page and lightweight name-only workspace entry.
import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  BarChart3, BrainCircuit, Check, ChevronRight, Clock3, Database,
  LayoutDashboard, ShieldCheck, SlidersHorizontal, Sparkles, TicketCheck, X, Zap,
} from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import './landing.css'

const reveal = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0, transition: { duration: 0.65, ease: [0.2, 0.8, 0.2, 1] } },
}

function Brand() {
  return (
    <div className="landing-brand" aria-label="SupportFlow AI">
      <span className="landing-brand-mark"><i /><i /><i /></span>
      <span>SupportFlow <b>AI</b></span>
    </div>
  )
}

function GlassIcon({ icon: Icon, cyan = false }) {
  return <span className={`landing-glass-icon ${cyan ? 'cyan' : ''}`}><i /><Icon size={24} /></span>
}

function PredictionPreview() {
  return (
    <motion.div className="landing-preview" variants={reveal}>
      <div className="preview-glow" />
      <div className="preview-topline">
        <div><span className="preview-dot" />Live ML decision</div>
        <span>Automatic routing</span>
      </div>
      <div className="preview-request">
        <span>Incoming request</span>
        <p>Production payment service is unavailable. Around 500 customers cannot complete checkout and 90% of transactions are failing.</p>
      </div>
      <div className="preview-decision-grid">
        <div><GlassIcon icon={Database} /><span>Category</span><strong>Billing</strong></div>
        <div><GlassIcon icon={SlidersHorizontal} /><span>Assigned team</span><strong>Billing Support</strong></div>
        <div><GlassIcon icon={Zap} cyan /><span>Priority</span><strong className="high">High</strong></div>
        <div><GlassIcon icon={Clock3} cyan /><span>Response</span><strong>30 minutes</strong></div>
      </div>
      <div className="preview-confidence">
        <div><span>Category confidence</span><b>98%</b><i><em style={{ width: '98%' }} /></i></div>
        <div><span>Priority confidence</span><b>91%</b><i><em style={{ width: '91%' }} /></i></div>
      </div>
    </motion.div>
  )
}

const features = [
  { icon: BrainCircuit, title: 'Description-only intelligence', text: 'One customer statement becomes a category, priority, team and response target.' },
  { icon: SlidersHorizontal, title: 'Automatic ticket routing', text: 'Six operational categories connect each request to the right support team.' },
  { icon: BarChart3, title: 'Confidence-aware decisions', text: 'Calibrated scores make automatic decisions transparent and reviewable.' },
  { icon: ShieldCheck, title: 'Human-review safeguards', text: 'Low-confidence predictions can be held for review instead of routed blindly.' },
]

export default function LandingPage() {
  const navigate = useNavigate()
  const [showSignIn, setShowSignIn] = useState(false)
  const [firstName, setFirstName] = useState('')
  const [lastName, setLastName] = useState('')
  const openSignIn = () => setShowSignIn(true)
  const enterWorkspace = (event) => {
    event.preventDefault()
    const fullName = `${firstName.trim()} ${lastName.trim()}`.trim()
    if (!fullName) return
    localStorage.setItem('supportflow_user_name', fullName)
    navigate('/dashboard')
  }

  return (
    <div className="landing-page">
      <div className="landing-atmosphere" aria-hidden="true">
        <span className="landing-aurora one" /><span className="landing-aurora two" />
        <span className="landing-orb left"><i /></span><span className="landing-orb right"><i /></span>
        <span className="landing-grain" />
      </div>

      <header className="landing-nav">
        <Brand />
        <nav aria-label="Landing navigation">
          <a href="#capabilities">Capabilities</a>
          <a href="#performance">Performance</a>
          <a href="#workflow">Workflow</a>
        </nav>
        <button className="landing-nav-cta" onClick={openSignIn}>Sign in <ChevronRight size={16} /></button>
      </header>

      <main className="landing-main">
        <motion.section className="landing-hero" initial="hidden" animate="show" transition={{ staggerChildren: 0.11 }}>
          <div className="landing-hero-copy">
            <motion.div className="landing-status" variants={reveal}><span />Models online <b>Production ready</b></motion.div>
            <motion.h1 variants={reveal}>Support decisions,<br /><span>made beautifully clear.</span></motion.h1>
            <motion.p variants={reveal}>SupportFlow AI reads a customer’s statement and instantly classifies, prioritizes and routes the request—without making the user complete a long operational questionnaire.</motion.p>
            <motion.div className="landing-actions" variants={reveal}>
              <button className="landing-primary" onClick={openSignIn}><Sparkles size={18} />Launch SupportFlow <ChevronRight size={18} /></button>
              <a className="landing-secondary" href="#workflow">See how it works</a>
            </motion.div>
            <motion.div className="landing-proof" variants={reveal}>
              <span><Check size={15} />Two trained ML models</span>
              <span><Check size={15} />Flask inference API</span>
              <span><Check size={15} />Confidence safeguards</span>
            </motion.div>
          </div>
          <PredictionPreview />
        </motion.section>

        <motion.section id="performance" className="landing-metrics" initial="hidden" whileInView="show" viewport={{ once: true, amount: 0.3 }} transition={{ staggerChildren: 0.08 }}>
          <motion.div variants={reveal}><span>Category test accuracy</span><strong>99.73%</strong><small>Macro F1 99.76%</small></motion.div>
          <motion.div variants={reveal}><span>Priority test accuracy</span><strong>95.29%</strong><small>Macro F1 94.71%</small></motion.div>
          <motion.div variants={reveal}><span>Selective priority accuracy</span><strong>98.18%</strong><small>90.92% automated coverage</small></motion.div>
          <motion.div variants={reveal}><span>Routing categories</span><strong>6</strong><small>Purpose-built support taxonomy</small></motion.div>
        </motion.section>

        <section id="capabilities" className="landing-section">
          <div className="landing-section-heading"><span>Intelligent operations</span><h2>From one statement to the next best action.</h2><p>A focused workflow for support teams that need speed without losing visibility or control.</p></div>
          <div className="landing-feature-grid">
            {features.map(({ icon, title, text }, index) => <motion.article key={title} className="landing-feature" initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: index * 0.07 }}><GlassIcon icon={icon} cyan={index === 1 || index === 3} /><h3>{title}</h3><p>{text}</p><span>0{index + 1}</span></motion.article>)}
          </div>
        </section>

        <section id="workflow" className="landing-section landing-workflow">
          <div className="landing-section-heading"><span>Three-step workflow</span><h2>Operational intelligence without the friction.</h2></div>
          <div className="workflow-track">
            <div><b>01</b><GlassIcon icon={TicketCheck} /><h3>Describe</h3><p>Paste a natural customer support statement.</p></div>
            <i />
            <div><b>02</b><GlassIcon icon={BrainCircuit} cyan /><h3>Analyze</h3><p>Two models evaluate routing and business impact.</p></div>
            <i />
            <div><b>03</b><GlassIcon icon={LayoutDashboard} /><h3>Act</h3><p>Receive the team, priority, SLA and confidence.</p></div>
          </div>
        </section>

        <section className="landing-final-cta">
          <span className="cta-orb"><i /></span>
          <div><span>SupportFlow AI workspace</span><h2>Route the next request with confidence.</h2><p>Move from incoming text to a transparent operational decision in seconds.</p></div>
          <button className="landing-primary" onClick={openSignIn}>Open workspace <ChevronRight size={18} /></button>
        </section>
      </main>

      <footer className="landing-footer"><Brand /><p>AI-powered ticket classification and prioritization.</p><span>Internship ML Engineering Project</span></footer>

      <AnimatePresence>
        {showSignIn && (
          <motion.div className="landing-signin-backdrop" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onMouseDown={() => setShowSignIn(false)}>
            <motion.form className="landing-signin-card" initial={{ opacity: 0, y: 22, scale: .97 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: 12, scale: .98 }} onMouseDown={(event) => event.stopPropagation()} onSubmit={enterWorkspace}>
              <button type="button" className="landing-signin-close" onClick={() => setShowSignIn(false)} aria-label="Close"><X size={19} /></button>
              <Brand />
              <span className="landing-signin-eyebrow">Welcome</span>
              <h2>Enter the AI workspace</h2>
              <p>Tell us your name so the dashboard feels like your own.</p>
              <div className="landing-name-grid">
                <label>First name<input autoFocus required value={firstName} onChange={(event) => setFirstName(event.target.value)} placeholder="Jordan" /></label>
                <label>Last name<input required value={lastName} onChange={(event) => setLastName(event.target.value)} placeholder="Davis" /></label>
              </div>
              <button className="landing-primary landing-signin-submit" type="submit"><Sparkles size={18} />Continue to dashboard <ChevronRight size={18} /></button>
              <small>No password or account required.</small>
            </motion.form>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
