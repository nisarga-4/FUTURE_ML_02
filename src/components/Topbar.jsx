import { Bell, Menu, Search, ChevronDown, User, LogOut, Settings } from 'lucide-react'
import { useState } from 'react'

export default function Topbar({ onMenu }) {
  const [open, setOpen] = useState(false)
  return (
    <header className="topbar">
      <button className="icon-btn mobile-menu" onClick={onMenu} aria-label="Open menu"><Menu size={20}/></button>

      <div className="status-pill"><i/><span>Models online</span></div>

      <label className="global-search">
        <Search size={18}/>
        <input aria-label="Global search" placeholder="Search tickets, customers, or anything..." />
        <kbd>⌘ K</kbd>
      </label>

      <div className="top-actions">
        <button className="icon-btn notification" aria-label="Notifications"><Bell size={19}/><i/></button>
        <div className="user-menu">
          <button className="user-trigger" onClick={() => setOpen(v => !v)} aria-expanded={open}>
            <span className="avatar">AR</span>
            <span className="user-copy"><strong>Avery Reed</strong><small>Support Lead</small></span>
            <ChevronDown size={16}/>
          </button>
          {open && (
            <div className="dropdown">
              <button><User size={16}/>Profile</button>
              <button><Settings size={16}/>Preferences</button>
              <hr/>
              <button><LogOut size={16}/>Sign out</button>
            </div>
          )}
        </div>
      </div>
    </header>
  )
}
