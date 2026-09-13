const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')

function asPercent(value) {
  const number = Number(value)
  if (!Number.isFinite(number)) return 0
  return Math.round(Math.max(0, Math.min(1, number)) * 100)
}

function normalizePrediction(data) {
  const prediction = data?.prediction

  if (!prediction) {
    throw new Error('The backend did not return a prediction.')
  }

  return {
    category: prediction.category,
    team: prediction.assigned_team,
    priority: prediction.priority,
    responseTarget: String(prediction.response_target || '').replace(
      /^Respond within\s+/i,
      '',
    ),
    categoryConfidence: asPercent(data.category_review?.confidence),
    priorityConfidence: asPercent(data.priority_review?.confidence),
    automatic: !data.human_review_required,
    categoryProbabilities: data.category_review?.probabilities || {},
    priorityProbabilities: data.priority_review?.probabilities || {},
    extractedSignals: data.extracted_signals || {},
  }
}

export async function classifyTicket(ticket) {
  const description = String(
    typeof ticket === 'string' ? ticket : ticket?.description || '',
  ).trim()

  if (description.length < 5) {
    throw new Error('Enter a complete support-ticket description.')
  }

  let response

  try {
    response = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        subject: String(ticket?.subject || '').trim(),
        description,
      }),
    })
  } catch {
    throw new Error(
      'Cannot reach the SupportFlow backend. Keep supportflow_app.py running on port 5000.',
    )
  }

  let data
  try {
    data = await response.json()
  } catch {
    throw new Error('The SupportFlow backend returned an invalid response.')
  }

  if (!response.ok) {
    throw new Error(data?.error || 'The trained models could not process this ticket.')
  }

  return normalizePrediction(data)
}
