import { motion } from 'framer-motion'

export default function GlassCard({ children, className='', strong=false, hover=true, ...props }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: .45, ease: [0.22, 1, 0.36, 1] }}
      whileHover={hover ? { y: -3 } : undefined}
      className={`${strong ? 'glass-card glass-strong' : 'glass-card'} ${className}`}
      {...props}
    >
      <span className="glass-shine" aria-hidden="true" />
      {children}
    </motion.section>
  )
}
