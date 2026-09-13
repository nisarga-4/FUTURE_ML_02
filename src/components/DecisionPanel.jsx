import { motion, AnimatePresence } from 'framer-motion'
import { BrainCircuit, CheckCircle2, Clock3, ShieldCheck, Sparkles, Users } from 'lucide-react'
import GlassCard from './GlassCard'

function Loader() {
  return (
    <div className="ai-loader">
      <div className="loader-orbit"><i/><i/><i/></div>
      <strong>Analyzing request</strong>
      <span>Routing, priority, impact and confidence signals</span>
    </div>
  )
}

function Confidence({ label, value }) {
  return (
    <div className="confidence-row">
      <div><span>{label}</span><strong>{value}%</strong></div>
      <div className="confidence-track"><i style={{width:`${value}%`}}/></div>
    </div>
  )
}

export default function DecisionPanel({ loading, result }) {
  return (
    <GlassCard className="decision-panel" strong hover={false}>
      <div className="section-heading compact">
        <div><p className="eyebrow">LIVE INFERENCE</p><h2>ML Decision</h2><span>AI analysis based on your input.</span></div>
        <div className="brain-badge"><BrainCircuit size={20}/></div>
      </div>

      <AnimatePresence mode="wait">
        {loading ? (
          <motion.div key="loading" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}}>
            <Loader/>
          </motion.div>
        ) : result ? (
          <motion.div key="result" initial={{opacity:0, y:14}} animate={{opacity:1, y:0}} transition={{duration:.4}} className="decision-content">
            <div className="auto-badge"><CheckCircle2 size={16}/>Automatic decision</div>
            <div className="decision-main">
              <div className="decision-tile classification">
                <span>Predicted category</span><strong>{result.category}</strong>
              </div>
              <div className="decision-tile team">
                <Users size={17}/><span>Assigned team</span><strong>{result.team}</strong>
              </div>
              <div className="decision-tile priority">
                <ShieldCheck size={17}/><span>Priority</span><strong>{result.priority}</strong>
              </div>
              <div className="decision-tile response">
                <Clock3 size={17}/><span>Response target</span><strong>{result.responseTarget}</strong>
              </div>
            </div>
            <div className="confidence-box">
              <Confidence label="Category confidence" value={result.categoryConfidence}/>
              <Confidence label="Priority confidence" value={result.priorityConfidence}/>
            </div>
            <div className="explanation-card"><Sparkles size={18}/><p>{result.explanation}</p></div>
          </motion.div>
        ) : (
          <motion.div key="empty" initial={{opacity:0}} animate={{opacity:1}} className="decision-empty">
            <div className="empty-visual"><BrainCircuit size={36}/></div>
            <strong>Ready for classification</strong>
            <span>Complete the ticket form to generate the category, team, priority and confidence scores.</span>
          </motion.div>
        )}
      </AnimatePresence>
    </GlassCard>
  )
}
