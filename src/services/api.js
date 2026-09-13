const wait = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds))

export async function classifyTicket(payload) {
  const apiUrl = import.meta.env.VITE_API_URL

  if (apiUrl) {
    const response = await fetch(`${apiUrl}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (!response.ok) throw new Error('The prediction service could not process this ticket.')
    return response.json()
  }

  await wait(1450)
  const text = payload.description.toLowerCase()
  const category = text.match(/laptop|screen|battery|keyboard|device|printer/) ? 'Hardware'
    : text.match(/password|login|vpn|access|account/) ? 'Access'
      : text.match(/storage|upload|drive|space/) ? 'Storage'
        : payload.productArea === 'Billing' ? 'Purchase' : 'Hardware'
  const teamMap = {
    Hardware: 'Hardware Support Team', Access: 'Identity & Access Team',
    Storage: 'Storage Operations Team', Purchase: 'Procurement Support Team',
  }
  const severe = Number(payload.customersAffected) >= 100 || Number(payload.downtime) >= 60 || payload.impacts.revenue || payload.impacts.security
  const priority = severe ? 'High' : Number(payload.customersAffected) >= 10 || payload.impacts.disruption ? 'Medium' : 'Low'

  return {
    category,
    team: teamMap[category],
    priority,
    responseTarget: priority === 'High' ? '30 minutes' : priority === 'Medium' ? '2 hours' : '8 hours',
    categoryConfidence: category === 'Hardware' ? 96 : 93,
    priorityConfidence: severe ? 89 : 94,
    automatic: true,
  }
}
