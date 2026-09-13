import { useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import {
  Activity, AlertCircle, BarChart3, Bell, BrainCircuit, Check, ChevronDown, ChevronRight,
  CircleHelp, Clock3, Command, Database, FilePlus2, Filter, Folder, Gauge, HardDrive,
  Headphones, HelpCircle, LayoutDashboard, LifeBuoy, Menu, MessageSquareText, PanelLeftClose,
  PanelLeftOpen, Plus, RefreshCw, Search, Settings, ShieldCheck, SlidersHorizontal, Sparkles,
  Star, TicketCheck, Users, WandSparkles, X, Zap,
} from 'lucide-react'
import { NavLink, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { categoryData, priorityData, responseData, tickets } from './data/mockData'
import { classifyTicket } from './services/api'
import LiquidGlassStage from './components/LiquidGlassStage'
import LandingPage from './LandingPage'

const navItems = [
  { label: 'Dashboard', to: '/dashboard', icon: LayoutDashboard },
  { label: 'New Ticket', to: '/new-ticket', icon: FilePlus2 },
  { label: 'Review Queue', to: '/review-queue', icon: SlidersHorizontal, badge: 12 },
  { label: 'Analytics', to: '/analytics', icon: BarChart3 },
  { label: 'Settings', to: '/settings', icon: Settings },
  { label: 'Help', to: '/help', icon: CircleHelp },
]

const pageTitles = {
  '/dashboard': ['Dashboard', 'A clear view of today’s support operations.'],
  '/new-ticket': ['New Support Request', 'Classify, prioritize, and route in one intelligent flow.'],
  '/review-queue': ['Review Queue', 'Confirm the decisions that deserve a human eye.'],
  '/analytics': ['Analytics', 'Model performance and operational signals in one place.'],
  '/settings': ['Settings', 'Tune routing, review thresholds, and team preferences.'],
  '/help': ['Help Center', 'Guidance for confident, consistent support decisions.'],
}

const rise = {
  hidden: { opacity: 0, y: 18 },
  show: { opacity: 1, y: 0, transition: { duration: .48, ease: [0.2, 0.8, 0.2, 1] } },
}

function Logo({ compact = false }) {
  return (
    <div className="brand" aria-label="SupportFlow AI">
      <span className="brand-mark"><i /><i /><i /></span>
      {!compact && <span className="brand-name">SupportFlow <b>AI</b></span>}
    </div>
  )
}

function LiquidBackground() {
  const ref = useRef(null)
  const reduced = useReducedMotion()
  useEffect(() => {
    if (reduced) return
    const move = (event) => {
      const x = (event.clientX / window.innerWidth - .5) * 14
      const y = (event.clientY / window.innerHeight - .5) * 10
      ref.current?.style.setProperty('--px', `${x}px`)
      ref.current?.style.setProperty('--py', `${y}px`)
    }
    window.addEventListener('pointermove', move, { passive: true })
    return () => window.removeEventListener('pointermove', move)
  }, [reduced])
  return (
    <div className="liquid-scene" ref={ref} aria-hidden="true">
      <div className="aurora aurora-a" />
      <div className="aurora aurora-b" />
      <div className="liquid-ribbon" />
      <div className="edge-orb edge-orb-left"><span /></div>
      <div className="edge-orb edge-orb-right"><span /></div>
      <div className="grain" />
    </div>
  )
}

function Sidebar({ collapsed, mobileOpen, closeMobile, toggle }) {
  return (
    <aside className={`sidebar glass-shell ${collapsed ? 'collapsed' : ''} ${mobileOpen ? 'mobile-open' : ''}`}>
      <div className="sidebar-head">
        <Logo compact={collapsed} />
        <button className="icon-button collapse-button" onClick={toggle} aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}>
          {collapsed ? <PanelLeftOpen size={19} /> : <PanelLeftClose size={19} />}
        </button>
        <button className="icon-button mobile-close" onClick={closeMobile} aria-label="Close menu"><X size={20} /></button>
      </div>
      <nav className="sidebar-nav" aria-label="Primary navigation">
        {navItems.map(({ label, to, icon: Icon, badge }) => (
          <NavLink key={to} to={to} end={to === '/'} onClick={closeMobile} className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <Icon size={20} strokeWidth={1.9} />
            {!collapsed && <span>{label}</span>}
            {!collapsed && badge && <b className="nav-badge">{badge}</b>}
          </NavLink>
        ))}
      </nav>
      <div className="sidebar-promo">
        <div className="promo-orb"><span /></div>
        {!collapsed && <><Sparkles size={18} /><p>Smarter support for brighter experiences.</p></>}
      </div>
    </aside>
  )
}

function Topbar({ openMobile }) {
  const [notifications, setNotifications] = useState(false)
  const [profile, setProfile] = useState(false)
  const [query, setQuery] = useState('')
  const navigate = useNavigate()
  const userName = localStorage.getItem('supportflow_user_name') || 'Jordan Davis'
  const initials = userName.split(/\s+/).filter(Boolean).slice(0, 2).map((part) => part[0].toUpperCase()).join('') || 'JD'
  const signOut = () => {
    localStorage.removeItem('supportflow_user_name')
    setProfile(false)
    navigate('/')
  }
  const submitSearch = (e) => {
    e.preventDefault()
    if (query.trim()) navigate(`/review-queue?q=${encodeURIComponent(query.trim())}`)
  }
  return (
    <header className="topbar glass-shell">
      <button className="icon-button menu-button" onClick={openMobile} aria-label="Open menu"><Menu size={21} /></button>
      <div className="model-status"><span />Models online</div>
      <form className="global-search" onSubmit={submitSearch} role="search">
        <Search size={18} />
        <input value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Search tickets and customers" placeholder="Search tickets, customers, or anything..." />
        <kbd>⌘ K</kbd>
      </form>
      <div className="top-actions">
        <div className="popover-wrap">
          <button className="icon-button notification-button" onClick={() => setNotifications(!notifications)} aria-expanded={notifications} aria-label="Notifications"><Bell size={20} /><i /></button>
          <AnimatePresence>{notifications && <motion.div className="popover glass-panel" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 8 }}>
            <strong>Notifications</strong><p>3 tickets need your review.</p><p>Weekly accuracy is up 1.4%.</p>
          </motion.div>}</AnimatePresence>
        </div>
        <div className="popover-wrap">
          <button className="profile-button" onClick={() => setProfile(!profile)} aria-expanded={profile}><span>{initials}</span><div><b>{userName}</b><small>Operations lead</small></div><ChevronDown size={16} /></button>
          <AnimatePresence>{profile && <motion.div className="popover profile-popover glass-panel" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 8 }}><button>View profile</button><button>Account settings</button><button onClick={signOut}>Sign out</button></motion.div>}</AnimatePresence>
        </div>
      </div>
    </header>
  )
}

function GlassCard({ className = '', children, ...props }) {
  return <motion.section variants={rise} className={`glass-panel ${className}`} {...props}>{children}</motion.section>
}

function OrbIcon({ icon: Icon, cyan = false }) {
  return <span className={`orb-icon ${cyan ? 'cyan' : ''}`}><span className="orb-liquid" /><Icon size={24} strokeWidth={1.8} /></span>
}

function MetricCard({ icon, label, value, progress, cyan }) {
  return (
    <GlassCard className="metric-card">
      <OrbIcon icon={icon} cyan={cyan} />
      <div className="metric-content"><span>{label}</span><strong>{value}</strong><div className={`liquid-progress ${cyan ? 'cyan' : ''}`}><i style={{ '--progress': `${progress}%` }} /></div></div>
      <span className="metric-delta">+1.4%</span>
    </GlassCard>
  )
}

function PageHeading() {
  const { pathname } = useLocation()
  const [title, subtitle] = pageTitles[pathname] || pageTitles['/dashboard']
  return <motion.div className="page-heading" initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }}><div><h1>{title}</h1><p>{subtitle}</p></div><div className="date-chip"><Activity size={16} /> Live operations</div></motion.div>
}

function DashboardPage() {
  const navigate = useNavigate()
  return (
    <motion.div className="page-grid" initial="hidden" animate="show" transition={{ staggerChildren: .07 }}>
      <div className="metrics-grid">
        <MetricCard icon={Folder} label="Category Routing" value="93.23%" progress={93.23} />
        <MetricCard icon={Zap} label="Priority Accuracy" value="97.45%" progress={97.45} cyan />
        <MetricCard icon={Database} label="Selective Priority" value="98.97%" progress={98.97} />
      </div>
      <div className="dashboard-main">
        <GlassCard className="overview-card">
          <div className="card-heading"><div><span className="eyebrow">Today</span><h2>Support operations</h2><p>1,135 tickets classified with stable model confidence.</p></div><button className="secondary-button" onClick={() => navigate('/analytics')}>View analytics <ChevronRight size={16} /></button></div>
          <div className="overview-stat-grid">
            <div><TicketCheck size={20} /><span>Auto-routed</span><strong>1,048</strong><small>92.3% of volume</small></div>
            <div><Users size={20} /><span>Human review</span><strong>87</strong><small>7.7% of volume</small></div>
            <div><Clock3 size={20} /><span>Median response</span><strong>23m</strong><small>6m faster this week</small></div>
          </div>
          <div className="mini-chart-wrap"><ResponsiveContainer width="100%" height={210}><AreaChart data={responseData}><defs><linearGradient id="dashFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#6C4DFF" stopOpacity={.42}/><stop offset="100%" stopColor="#6C4DFF" stopOpacity={0}/></linearGradient></defs><CartesianGrid stroke="rgba(78,53,211,.09)" vertical={false}/><XAxis dataKey="day" axisLine={false} tickLine={false}/><YAxis hide/><Tooltip content={<ChartTooltip />}/><Area type="monotone" dataKey="confidence" stroke="#6C4DFF" strokeWidth={3} fill="url(#dashFill)" /></AreaChart></ResponsiveContainer></div>
        </GlassCard>
        <GlassCard className="quick-classify">
          <OrbIcon icon={BrainCircuit} cyan />
          <span className="eyebrow">AI workspace</span><h2>Route the next request with confidence.</h2><p>Describe the issue, add its operational impact, and receive a clear category, team, priority, and response target.</p>
          <button className="primary-button" onClick={() => navigate('/new-ticket')}><WandSparkles size={18} /> New classification <ChevronRight size={17} /></button>
          <div className="signal-list"><span><Check size={15}/> Eight routing categories</span><span><Check size={15}/> Human review safeguards</span><span><Check size={15}/> Transparent confidence</span></div>
        </GlassCard>
      </div>
      <GlassCard className="recent-card"><div className="card-heading compact"><div><h2>Recent classification activity</h2><p>Latest routed tickets across your support teams.</p></div><button className="text-button" onClick={() => navigate('/review-queue')}>View all <ChevronRight size={15}/></button></div><TicketTable rows={tickets.slice(0, 4)} /></GlassCard>
    </motion.div>
  )
}

const initialForm = { description: '', customersAffected: '1', downtime: '0', errorRate: '0', productArea: 'Billing', customerTier: 'Standard', impacts: { disruption: false, security: false, revenue: false, vip: false } }
const initialResult = { category: 'Hardware', team: 'Hardware Support Team', priority: 'High', responseTarget: '30 minutes', categoryConfidence: 96, priorityConfidence: 89, automatic: true }

function NewTicketPage() {
  const [form, setForm] = useState(initialForm)
  const [errors, setErrors] = useState({})
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(initialResult)
  const [revealed, setRevealed] = useState(false)
  const update = (key, value) => setForm((old) => ({ ...old, [key]: value }))
  const toggle = (key) => setForm((old) => ({ ...old, impacts: { ...old.impacts, [key]: !old.impacts[key] } }))
  const clear = () => { setForm(initialForm); setErrors({}); setResult(initialResult); setRevealed(false) }
  const submit = async (event) => {
    event.preventDefault()
    const nextErrors = {}
    if (form.description.trim().length < 20) nextErrors.description = 'Please describe the issue in at least 20 characters.'
    setErrors(nextErrors)
    if (Object.keys(nextErrors).length) return
    setLoading(true); setRevealed(false)
    try {
      const prediction = await classifyTicket(form)
      setResult(prediction)
      setRevealed(true)
    } catch (error) {
      setErrors({ submit: error.message })
    } finally { setLoading(false) }
  }

  return (
    <motion.div className="ticket-workspace" initial="hidden" animate="show" transition={{ staggerChildren: .08 }}>
      <div className="metrics-grid compact-metrics">
        <MetricCard icon={Folder} label="Category Routing" value="93.23%" progress={93.23}/><MetricCard icon={Zap} label="Priority Accuracy" value="97.45%" progress={97.45} cyan/><MetricCard icon={Database} label="Selective Priority" value="98.97%" progress={98.97}/>
      </div>
      <div className="ticket-columns">
        <GlassCard className="form-card">
          <div className="panel-title"><OrbIcon icon={MessageSquareText}/><div><h2>Analyze Support Request</h2><p>Enter one complete statement. SupportFlow extracts the operational signals automatically.</p></div></div>
          <form onSubmit={submit} noValidate>
            <label className="field textarea-field statement-field"><span>Customer statement <b>*</b></span><textarea maxLength={2000} value={form.description} onChange={(e) => update('description', e.target.value)} placeholder="Example: Our production payment service has been unavailable for 60 minutes. Around 500 enterprise customers cannot complete checkout and 90% of transactions are failing..." aria-invalid={!!errors.description} aria-describedby="description-help"/><small className="counter">{form.description.length}/2000</small>{errors.description && <em id="description-help" className="field-error"><AlertCircle size={14}/>{errors.description}</em>}</label>
            <div className="form-grid">
              <SelectField label="Customers affected" required value={form.customersAffected} onChange={(v) => update('customersAffected', v)} options={['1','5','10','50','100','500','1000']}/>
              <SelectField label="Estimated downtime" value={form.downtime} onChange={(v) => update('downtime', v)} options={[['0','No downtime'],['15','15 minutes'],['30','30 minutes'],['60','1 hour'],['120','2+ hours']]}/>
              <SelectField label="Error rate" value={form.errorRate} onChange={(v) => update('errorRate', v)} options={[['0','0% (no errors)'],['10','Up to 10%'],['25','10–25%'],['50','25–50%'],['90','Above 50%']]}/>
              <SelectField label="Product area" required value={form.productArea} onChange={(v) => update('productArea', v)} options={['Hardware','Account & Access','Cloud Storage','Billing','People Operations','Internal Tools']} placeholder="Select product area" error={errors.productArea}/>
              <SelectField label="Customer tier" value={form.customerTier} onChange={(v) => update('customerTier', v)} options={['Standard','Growth','Business','Enterprise']}/>
            </div>
            <fieldset className="switch-fieldset"><legend>Impact indicators <HelpCircle size={15}/></legend><div className="switch-grid"><Switch label="Service disruption" checked={form.impacts.disruption} onChange={() => toggle('disruption')}/><Switch label="Security related" checked={form.impacts.security} onChange={() => toggle('security')}/><Switch label="Revenue impact" checked={form.impacts.revenue} onChange={() => toggle('revenue')}/><Switch label="VIP customer" checked={form.impacts.vip} onChange={() => toggle('vip')}/></div></fieldset>
            {errors.submit && <div className="submit-error"><AlertCircle size={16}/>{errors.submit}</div>}
            <div className="form-actions"><button type="button" className="secondary-button clear-button" onClick={clear}>Clear</button><button type="submit" className={`primary-button classify-button ${loading ? 'loading' : ''}`} disabled={loading}>{loading ? <><span className="liquid-loader"><i/></span>Analysing ticket...</> : <><WandSparkles size={19}/>Classify with AI<ChevronRight size={18}/></>}</button></div>
          </form>
        </GlassCard>
        <GlassCard className="decision-card">
          <div className="decision-head"><div className="panel-title"><OrbIcon icon={BrainCircuit} cyan/><div><h2>ML Decision</h2><p>AI analysis based on your input.</p></div></div><span className="automatic-badge"><Check size={17}/>Automatic decision<HelpCircle size={14}/></span></div>
          <div className="decision-stage">
            <AnimatePresence mode="wait">
              {loading ? <DecisionLoader key="loader"/> : revealed && <motion.div key={`${result.category}-${result.priority}`} initial={{ opacity: 0, y: 14, filter: 'blur(8px)' }} animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }} transition={{ duration: .55 }}>
                <div className="decision-list"><DecisionRow icon={Database} label="Predicted category" value={result.category}/><DecisionRow icon={Users} label="Assigned team" value={result.team}/><div className="decision-split"><DecisionRow icon={Star} label="Priority" value={result.priority} danger={result.priority === 'High'}/><DecisionRow icon={Clock3} label="Response target" value={result.responseTarget} cyan/></div></div>
                <div className="confidence-grid"><Confidence label="Category confidence" value={result.categoryConfidence}/><Confidence label="Priority confidence" value={result.priorityConfidence} cyan/></div>
                <div className="explanation-card"><span><Sparkles size={19}/></span><p>This ticket was automatically classified and prioritized by SupportFlow AI using the latest routing model.</p><div><b>High confidence</b><small>Pattern matched in similar tickets</small></div></div>
              </motion.div>}
            </AnimatePresence>
          </div>
        </GlassCard>
      </div>
    </motion.div>
  )
}

function SelectField({ label, value, onChange, options, placeholder, required, error }) {
  return <label className="field"><span>{label} {required && <b>*</b>}</span><select value={value} onChange={(e) => onChange(e.target.value)} aria-invalid={!!error}><option value="" disabled>{placeholder}</option>{options.map((option) => { const pair = Array.isArray(option) ? option : [option, option]; return <option key={pair[0]} value={pair[0]}>{pair[1]}</option> })}</select>{error && <em className="field-error"><AlertCircle size={14}/>{error}</em>}</label>
}

function Switch({ label, checked, onChange }) {
  return <label className="switch"><button type="button" role="switch" aria-checked={checked} onClick={onChange} className={checked ? 'on' : ''}><i/></button><span>{label}</span></label>
}

function DecisionRow({ icon: Icon, label, value, danger, cyan }) {
  return <div className={`decision-row ${danger ? 'danger' : ''} ${cyan ? 'cyan' : ''}`}><OrbIcon icon={Icon} cyan={cyan}/><div><span>{label}</span><strong>{value}</strong></div><ChevronRight size={19}/></div>
}

function Confidence({ label, value, cyan }) {
  return <div className="confidence"><div><span>{label}<HelpCircle size={14}/></span><strong>{value}%</strong></div><div className={`liquid-progress ${cyan ? 'cyan' : ''}`}><i style={{ '--progress': `${value}%` }}/></div></div>
}

function DecisionLoader() {
  return <motion.div className="decision-loader" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}><div className="analysis-vessel"><span/><BrainCircuit size={35}/></div><h3>Reading operational signals</h3><p>Comparing category patterns and calculating priority confidence.</p><div className="loading-track"><i/></div></motion.div>
}

function ReviewQueuePage() {
  const location = useLocation()
  const initialQ = new URLSearchParams(location.search).get('q') || ''
  const [search, setSearch] = useState(initialQ)
  const [priority, setPriority] = useState('All priorities')
  const [status, setStatus] = useState('All statuses')
  const filtered = useMemo(() => tickets.filter((ticket) => {
    const q = search.toLowerCase(); const matchesText = !q || `${ticket.id} ${ticket.subject} ${ticket.category} ${ticket.team}`.toLowerCase().includes(q)
    return matchesText && (priority === 'All priorities' || ticket.priority === priority) && (status === 'All statuses' || ticket.status === status)
  }), [search, priority, status])
  return <motion.div className="page-grid" initial="hidden" animate="show"><GlassCard className="queue-card"><div className="toolbar"><div className="inline-search"><Search size={18}/><input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search by ticket, category, or team" aria-label="Search review queue"/></div><label><Filter size={17}/><select value={priority} onChange={(e) => setPriority(e.target.value)}><option>All priorities</option><option>High</option><option>Medium</option><option>Low</option></select></label><label><select value={status} onChange={(e) => setStatus(e.target.value)}><option>All statuses</option><option>Needs review</option><option>Auto-routed</option></select></label></div><div className="queue-summary"><div><strong>{filtered.length}</strong><span>Visible tickets</span></div><div><strong>12</strong><span>Awaiting review</span></div><div><strong>6m</strong><span>Oldest wait</span></div></div><TicketTable rows={filtered} review/></GlassCard></motion.div>
}

function TicketTable({ rows, review = false }) {
  return <div className="table-scroll"><table><thead><tr><th>Ticket</th><th>Category</th><th>Priority</th><th>Confidence</th><th>Assigned team</th><th>Status</th>{review && <th aria-label="Actions"/>}</tr></thead><tbody>{rows.map((ticket) => <tr key={ticket.id}><td><b>{ticket.id}</b><span>{ticket.subject}</span><small>{ticket.time}</small></td><td><span className="category-chip">{ticket.category}</span></td><td><span className={`priority-chip ${ticket.priority.toLowerCase()}`}>{ticket.priority}</span></td><td><div className="table-confidence"><i><b style={{ width: `${ticket.confidence}%` }}/></i><span>{ticket.confidence}%</span></div></td><td>{ticket.team}</td><td><span className={`status-chip ${ticket.status === 'Auto-routed' ? 'success' : ''}`}>{ticket.status}</span></td>{review && <td><button className="review-button">Review <ChevronRight size={14}/></button></td>}</tr>)}</tbody></table>{!rows.length && <div className="empty-state"><Search size={28}/><h3>No matching tickets</h3><p>Try a different search or filter.</p></div>}</div>
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return <div className="chart-tooltip"><b>{label || payload[0].name}</b>{payload.map((item) => <span key={item.dataKey || item.name}>{item.name}: {item.value}{item.dataKey === 'minutes' ? ' min' : ''}</span>)}</div>
}

function AnalyticsPage() {
  const [range, setRange] = useState('7 days')
  return <motion.div className="analytics-v2" initial="hidden" animate="show" transition={{ staggerChildren: .08 }}>
    <GlassCard className="analytics-command">
      <div className="analytics-command-head"><div><span className="eyebrow">Decision intelligence</span><h2>Model performance</h2><p>Operational quality, speed, and confidence across every automated decision.</p></div><div className="range-control" aria-label="Analytics period">{['7 days','30 days','90 days'].map((item) => <button key={item} className={range === item ? 'active' : ''} onClick={() => setRange(item)}>{item}</button>)}</div></div>
      <div className="analytics-kpis"><div><span>Tickets classified</span><strong>1,135</strong><small>+12.8% vs prior period</small></div><div><span>Auto-routing rate</span><strong>92.3%</strong><small>Within target range</small></div><div><span>Median response</span><strong>23m</strong><small>6 minutes faster</small></div><div><span>Review precision</span><strong>98.9%</strong><small>High-confidence decisions</small></div></div>
      <div className="signal-chart"><div className="chart-axis-note"><span><i className="purple-dot"/>Confidence score</span><span><i className="cyan-dot"/>Response time</span></div><ResponsiveContainer width="100%" height={285}><AreaChart data={responseData}><defs><linearGradient id="responseFillV2" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#7C68E8" stopOpacity={.28}/><stop offset="1" stopColor="#7C68E8" stopOpacity={0}/></linearGradient></defs><CartesianGrid stroke="rgba(78,53,211,.07)" vertical={false}/><XAxis dataKey="day" axisLine={false} tickLine={false}/><YAxis axisLine={false} tickLine={false}/><Tooltip content={<ChartTooltip/>}/><Area name="Confidence" type="monotone" dataKey="confidence" stroke="#7561E4" strokeWidth={2.5} fill="url(#responseFillV2)"/><Area name="Minutes" type="monotone" dataKey="minutes" stroke="#69D8E5" strokeWidth={2.5} fill="transparent"/></AreaChart></ResponsiveContainer></div>
    </GlassCard>
    <div className="analytics-split">
      <GlassCard className="chart-card category-intelligence"><div className="card-heading compact"><div><span className="eyebrow">Routing mix</span><h2>Ticket categories</h2><p>Distribution across six operational groups.</p></div><span className="chart-total">1,135 <small>total</small></span></div><ResponsiveContainer width="100%" height={300}><BarChart data={categoryData} layout="vertical" margin={{left:10,right:18}}><defs><linearGradient id="barGlass" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stopColor="#7561E4" stopOpacity={.82}/><stop offset=".72" stopColor="#9C8DEF" stopOpacity={.7}/><stop offset="1" stopColor="#C9C0FA" stopOpacity={.6}/></linearGradient></defs><CartesianGrid stroke="rgba(78,53,211,.065)" horizontal={false}/><XAxis type="number" hide/><YAxis dataKey="name" type="category" axisLine={false} tickLine={false} width={90}/><Tooltip content={<ChartTooltip/>}/><Bar dataKey="value" radius={[0,12,12,0]} fill="url(#barGlass)" barSize={28}/></BarChart></ResponsiveContainer></GlassCard>
      <GlassCard className="chart-card priority-intelligence"><div className="card-heading compact"><div><span className="eyebrow">Impact profile</span><h2>Priority distribution</h2><p>Balanced by operational impact.</p></div></div><div className="donut-wrap"><ResponsiveContainer width="100%" height={235}><PieChart><Pie data={priorityData} dataKey="value" nameKey="name" innerRadius={68} outerRadius={98} paddingAngle={6} stroke="rgba(255,255,255,.72)" strokeWidth={2}>{priorityData.map((entry) => <Cell key={entry.name} fill={entry.color}/>)}</Pie><Tooltip content={<ChartTooltip/>}/></PieChart></ResponsiveContainer><div className="donut-center"><strong>1,135</strong><span>decisions</span></div></div><div className="priority-legend">{priorityData.map(item => <div key={item.name}><i style={{background:item.color}}/><span>{item.name}</span><strong>{Math.round(item.value / 11.35)}%</strong></div>)}</div></GlassCard>
    </div>
    <div className="analytics-lower">
      <GlassCard className="activity-feed"><div className="card-heading compact"><div><span className="eyebrow">Live audit trail</span><h2>Recent model activity</h2><p>Confidence and human-review events.</p></div></div>{tickets.slice(0,5).map((t) => <div className="activity-item" key={t.id}><span className={t.status === 'Auto-routed' ? 'success' : ''}>{t.status === 'Auto-routed' ? <Check size={16}/> : <RefreshCw size={16}/>}</span><div><b>{t.id} · {t.category}</b><p>{t.subject}</p></div><div><strong>{t.confidence}%</strong><small>{t.time}</small></div></div>)}</GlassCard>
      <GlassCard className="model-assurance"><span className="eyebrow">Model assurance</span><div className="assurance-orb"><ShieldCheck size={28}/><span/></div><h2>Healthy and calibrated</h2><p>Confidence remains stable across the current ticket mix. Twelve decisions are waiting for human confirmation.</p><div className="assurance-list"><span><b>0.91</b>Mean confidence</span><span><b>2.4%</b>Override rate</span><span><b>8</b>Routing classes</span></div><button className="secondary-button">Open review queue <ChevronRight size={16}/></button></GlassCard>
    </div>
  </motion.div>
}

function SettingsPage() {
  const [settings, setSettings] = useState({ autoRoute: true, reviewLow: true, notifyHigh: true, digest: false })
  const toggle = (key) => setSettings((old) => ({ ...old, [key]: !old[key] }))
  return <motion.div className="settings-grid" initial="hidden" animate="show" transition={{ staggerChildren: .08 }}><GlassCard className="settings-card"><div className="panel-title"><OrbIcon icon={Gauge}/><div><h2>Decision controls</h2><p>Set how automatic routing and human review work together.</p></div></div><SettingRow icon={TicketCheck} title="Automatic routing" text="Assign high-confidence tickets directly to the recommended team." checked={settings.autoRoute} onChange={() => toggle('autoRoute')}/><SettingRow icon={ShieldCheck} title="Review low confidence" text="Send predictions below the threshold to the review queue." checked={settings.reviewLow} onChange={() => toggle('reviewLow')}/><label className="threshold"><span>Human review threshold <b>82%</b></span><input type="range" min="60" max="98" defaultValue="82"/></label></GlassCard><GlassCard className="settings-card"><div className="panel-title"><OrbIcon icon={Bell} cyan/><div><h2>Notifications</h2><p>Stay informed without adding unnecessary noise.</p></div></div><SettingRow icon={AlertCircle} title="High-priority alerts" text="Notify the operations lead when a high-priority ticket arrives." checked={settings.notifyHigh} onChange={() => toggle('notifyHigh')}/><SettingRow icon={Activity} title="Daily performance digest" text="Receive a concise summary of volume, confidence, and review activity." checked={settings.digest} onChange={() => toggle('digest')}/><button className="primary-button save-button"><Check size={18}/>Save preferences</button></GlassCard></motion.div>
}

function SettingRow({ icon: Icon, title, text, checked, onChange }) { return <div className="setting-row"><span><Icon size={20}/></span><div><b>{title}</b><p>{text}</p></div><Switch label="" checked={checked} onChange={onChange}/></div> }

function HelpPage() {
  const [open, setOpen] = useState(0)
  const faqs = [
    ['How does automatic classification work?', 'SupportFlow compares ticket language and operational context with learned patterns, then returns a category, team, priority, response target, and confidence score.'],
    ['When is a ticket sent for human review?', 'A review is created when confidence falls below your threshold, important signals disagree, or the ticket includes sensitive security or data-loss language.'],
    ['Can I connect my Python model?', 'Yes. Set VITE_API_URL and expose POST /api/predict from Flask or FastAPI. The service layer is isolated in src/services/api.js.'],
    ['Is the interface accessible?', 'The controls use semantic labels, clear focus states, keyboard-friendly interactions, readable contrast, and reduced-motion support.'],
  ]
  return <motion.div className="help-grid" initial="hidden" animate="show" transition={{ staggerChildren: .08 }}><GlassCard className="help-intro"><OrbIcon icon={LifeBuoy} cyan/><span className="eyebrow">SupportFlow guide</span><h2>How can we help?</h2><p>Find answers about ticket decisions, confidence, reviews, and backend integration.</p><div className="help-search"><Search size={19}/><input placeholder="Search help topics" aria-label="Search help topics"/></div><button className="primary-button"><Headphones size={18}/>Contact support</button></GlassCard><GlassCard className="faq-card"><div className="card-heading compact"><div><h2>Frequently asked questions</h2><p>Clear answers for everyday operations.</p></div></div>{faqs.map(([q,a], i) => <div className={`faq ${open === i ? 'open' : ''}`} key={q}><button onClick={() => setOpen(open === i ? -1 : i)} aria-expanded={open === i}><span>{q}</span><Plus size={18}/></button><AnimatePresence initial={false}>{open === i && <motion.p initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }}>{a}</motion.p>}</AnimatePresence></div>)}</GlassCard></motion.div>
}

function AppShell() {
  const [collapsed, setCollapsed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const { pathname } = useLocation()
  useEffect(() => { setMobileOpen(false); window.scrollTo({ top: 0, behavior: 'smooth' }) }, [pathname])
  return <div className={`app ${collapsed ? 'sidebar-collapsed' : ''}`}><LiquidBackground/><LiquidGlassStage/><Sidebar collapsed={collapsed} mobileOpen={mobileOpen} closeMobile={() => setMobileOpen(false)} toggle={() => setCollapsed(!collapsed)}/>{mobileOpen && <button className="mobile-scrim" onClick={() => setMobileOpen(false)} aria-label="Close menu"/>}<div className="app-column"><Topbar openMobile={() => setMobileOpen(true)}/><main><PageHeading/><AnimatePresence mode="wait"><motion.div key={pathname} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}><Routes><Route path="/dashboard" element={<DashboardPage/>}/><Route path="/new-ticket" element={<NewTicketPage/>}/><Route path="/review-queue" element={<ReviewQueuePage/>}/><Route path="/analytics" element={<AnalyticsPage/>}/><Route path="/settings" element={<SettingsPage/>}/><Route path="/help" element={<HelpPage/>}/><Route path="*" element={<DashboardPage/>}/></Routes></motion.div></AnimatePresence></main></div></div>
}

export default function App() {
  return <Routes><Route path="/" element={<LandingPage/>}/><Route path="/*" element={<AppShell/>}/></Routes>
}
