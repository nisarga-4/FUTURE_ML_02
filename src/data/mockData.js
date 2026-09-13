export const reviewTickets = [
  { id:'SF-2841', subject:'Laptop display flickering after driver update', category:'Hardware', priority:'High', confidence:96, team:'Hardware Support', time:'5 min ago', status:'Needs review' },
  { id:'SF-2839', subject:'VPN access failing for finance team', category:'Access', priority:'High', confidence:88, team:'Identity & Access', time:'12 min ago', status:'Needs review' },
  { id:'SF-2837', subject:'Mailbox archive quota approaching limit', category:'Storage', priority:'Medium', confidence:91, team:'Messaging Ops', time:'21 min ago', status:'Approved' },
  { id:'SF-2835', subject:'New employee onboarding permissions', category:'HR Support', priority:'Low', confidence:94, team:'People Systems', time:'37 min ago', status:'Approved' },
  { id:'SF-2832', subject:'Procurement request for design workstation', category:'Purchase', priority:'Medium', confidence:83, team:'Procurement', time:'52 min ago', status:'Needs review' },
  { id:'SF-2828', subject:'Admin rights required for approved package', category:'Administrative rights', priority:'Medium', confidence:86, team:'Endpoint Security', time:'1 hr ago', status:'Approved' },
  { id:'SF-2821', subject:'Customer portal intermittent 500 errors', category:'Hardware', priority:'Critical', confidence:79, team:'Hardware Support', time:'2 hrs ago', status:'Needs review' }
]

export const categoryData = [
  { name:'Hardware', value:31 },
  { name:'HR Support', value:22 },
  { name:'Access', value:18 },
  { name:'Misc', value:12 },
  { name:'Storage', value:8 },
  { name:'Purchase', value:9 }
]

export const priorityData = [
  { name:'Low', value:24 },
  { name:'Medium', value:48 },
  { name:'High', value:23 },
  { name:'Critical', value:5 }
]

export const responseData = [
  { day:'Mon', minutes:42 },
  { day:'Tue', minutes:36 },
  { day:'Wed', minutes:31 },
  { day:'Thu', minutes:28 },
  { day:'Fri', minutes:33 },
  { day:'Sat', minutes:24 },
  { day:'Sun', minutes:21 }
]

export const confidenceData = [
  { range:'60–70', tickets:7 },
  { range:'70–80', tickets:14 },
  { range:'80–90', tickets:36 },
  { range:'90–95', tickets:54 },
  { range:'95–100', tickets:71 }
]

export const activityData = [
  { time:'09:00', classified:18, reviewed:4 },
  { time:'10:00', classified:26, reviewed:5 },
  { time:'11:00', classified:31, reviewed:8 },
  { time:'12:00', classified:22, reviewed:3 },
  { time:'13:00', classified:37, reviewed:9 },
  { time:'14:00', classified:43, reviewed:7 },
  { time:'15:00', classified:39, reviewed:6 }
]
