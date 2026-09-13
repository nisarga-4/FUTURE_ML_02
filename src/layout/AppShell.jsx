import { useState } from 'react'
import Sidebar from '../components/Sidebar'
import Topbar from '../components/Topbar'
import BackgroundFX from '../components/BackgroundFX'

export default function AppShell({ children }) {
  const [collapsed, setCollapsed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <div className="app-shell">
      <BackgroundFX/>
      <Sidebar collapsed={collapsed} setCollapsed={setCollapsed} mobileOpen={mobileOpen} setMobileOpen={setMobileOpen}/>
      <div className={`main-shell ${collapsed ? 'sidebar-collapsed' : ''}`}>
        <Topbar onMenu={() => setMobileOpen(true)}/>
        <main className="content">{children}</main>
      </div>
    </div>
  )
}
