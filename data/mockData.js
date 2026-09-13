export const tickets = [
  { id: 'SF-2941', subject: 'VPN disconnects every few minutes', category: 'Access', priority: 'High', confidence: 91, team: 'Identity & Access', time: '2 min ago', status: 'Needs review' },
  { id: 'SF-2938', subject: 'Laptop battery no longer charging', category: 'Hardware', priority: 'Medium', confidence: 96, team: 'Hardware Support', time: '11 min ago', status: 'Auto-routed' },
  { id: 'SF-2934', subject: 'Cannot upload quarterly archive', category: 'Storage', priority: 'Medium', confidence: 84, team: 'Storage Operations', time: '18 min ago', status: 'Needs review' },
  { id: 'SF-2929', subject: 'New employee account provisioning', category: 'HR Support', priority: 'Low', confidence: 98, team: 'People Operations', time: '31 min ago', status: 'Auto-routed' },
  { id: 'SF-2922', subject: 'Purchase approval workflow is blocked', category: 'Purchase', priority: 'High', confidence: 88, team: 'Procurement Desk', time: '44 min ago', status: 'Needs review' },
  { id: 'SF-2917', subject: 'Admin rights required for approved tool', category: 'Admin Rights', priority: 'Medium', confidence: 94, team: 'Endpoint Security', time: '1 hr ago', status: 'Auto-routed' },
]

export const categoryData = [
  { name: 'Hardware', value: 312 }, { name: 'Access', value: 248 },
  { name: 'HR Support', value: 205 }, { name: 'Storage', value: 148 },
  { name: 'Purchase', value: 96 }, { name: 'Other', value: 126 },
]

export const priorityData = [
  { name: 'High', value: 186, color: '#FF4F78' },
  { name: 'Medium', value: 492, color: '#6C4DFF' },
  { name: 'Low', value: 457, color: '#5FE7F2' },
]

export const responseData = [
  { day: 'Mon', minutes: 38, confidence: 89 }, { day: 'Tue', minutes: 34, confidence: 91 },
  { day: 'Wed', minutes: 31, confidence: 93 }, { day: 'Thu', minutes: 29, confidence: 92 },
  { day: 'Fri', minutes: 25, confidence: 95 }, { day: 'Sat', minutes: 27, confidence: 94 },
  { day: 'Sun', minutes: 23, confidence: 96 },
]
