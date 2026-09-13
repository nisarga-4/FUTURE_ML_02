import {
  LayoutDashboard, PlusCircle, ListChecks, BarChart3,
  Settings, CircleHelp, Sparkles, PanelLeftClose, PanelLeftOpen
} from 'lucide-react'
import { NavLink } from 'react-router-dom'
import { motion } from 'framer-motion'

const nav = [
  ['/', 'Dashboard', LayoutDashboard],
  ['/new-ticket', 'New Ticket', PlusCircle],
  ['/review-queue', 'Review Queue', ListChecks, 7],
  ['/analytics', 'Analytics', BarChart3],
  ['/settings', 'Settings', Settings],
  ['/help', 'Help', CircleHelp]
]

export default function Sidebar({ collapsed, setCollapsed, mobileOpen, setMobileOpen }) {
  return (
    <>
      {mobileOpen && <button className="mobile-backdrop" onClick={() => setMobileOpen(false)} aria-label="Close menu" />}
      <motion.aside
        animate={{ width: collapsed ? 94 : 264 }}
        className={`sidebar ${mobileOpen ? 'sidebar-mobile-open' : ''}`}
      >
        <div className="brand-row">
          <div className="brand-logo"><Sparkles size={20}/></div>
          {!collapsed && <div><strong>SupportFlow</strong><span>AI</span></div>}
          <button className="icon-btn collapse-btn" onClick={() => setCollapsed(v => !v)} aria-label="Toggle sidebar">
            {collapsed ? <PanelLeftOpen size={18}/> : <PanelLeftClose size={18}/>}
          </button>
        </div>

        <nav className="nav-list" aria-label="Primary navigation">
          {nav.map(([to, label, Icon, badge]) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              onClick={() => setMobileOpen(false)}
              className={({isActive}) => `nav-item ${isActive ? 'active' : ''}`}
              title={collapsed ? label : undefined}
            >
              <Icon size={20} strokeWidth={1.8}/>
              {!collapsed && <span>{label}</span>}
              {!collapsed && badge && <b>{badge}</b>}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-bottom">
          {!collapsed ? (
            <div className="promo-card">
              <div className="promo-orb" />
              <Sparkles size={18}/>
              <p>Smarter support for brighter experiences.</p>
            </div>
          ) : <div className="mini-orb" />}
        </div>
      </motion.aside>
    </>
  )
}
