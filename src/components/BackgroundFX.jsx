import { useEffect, useRef } from 'react'

export default function BackgroundFX() {
  const orb = useRef(null)

  useEffect(() => {
    const onMove = (e) => {
      if (!orb.current) return
      const x = (e.clientX / window.innerWidth - .5) * 16
      const y = (e.clientY / window.innerHeight - .5) * 12
      orb.current.style.transform = `translate3d(${x}px, ${y}px, 0)`
    }
    window.addEventListener('pointermove', onMove, { passive: true })
    return () => window.removeEventListener('pointermove', onMove)
  }, [])

  return (
    <div className="background-fx" aria-hidden="true">
      <div className="ambient ambient-a" />
      <div className="ambient ambient-b" />
      <div ref={orb} className="liquid-shape liquid-shape-a" />
      <div className="liquid-shape liquid-shape-b" />
      <div className="noise-layer" />
    </div>
  )
}
