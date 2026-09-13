import { useEffect, useRef } from 'react'
import * as THREE from 'three'

const vertexShader = /* glsl */ `
  uniform float uTime;
  uniform float uMotion;
  varying vec3 vNormal;
  varying vec3 vWorld;
  varying float vWave;

  float hash(vec3 p) {
    p = fract(p * 0.3183099 + .1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
  }

  float noise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash(i + vec3(0,0,0)), hash(i + vec3(1,0,0)), f.x),
                   mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
               mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
                   mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z);
  }

  void main() {
    float time = uTime * uMotion;
    float broad = sin(position.y * 2.6 + time * .72) * .075;
    float cross = sin(position.x * 3.3 - time * .48) * .052;
    float organic = (noise(normal * 2.25 + time * .16) - .5) * .19;
    float displacement = broad + cross + organic;
    vec3 displaced = position + normal * displacement;
    vWave = displacement;
    vNormal = normalize(normalMatrix * normal);
    vec4 world = modelMatrix * vec4(displaced, 1.0);
    vWorld = world.xyz;
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`

const fragmentShader = /* glsl */ `
  uniform float uTime;
  uniform float uOpacity;
  uniform vec3 uLavender;
  uniform vec3 uCyan;
  varying vec3 vNormal;
  varying vec3 vWorld;
  varying float vWave;

  void main() {
    vec3 viewDirection = normalize(cameraPosition - vWorld);
    float fresnel = pow(1.0 - max(dot(viewDirection, normalize(vNormal)), 0.0), 2.25);
    float ribbon = sin(vWorld.y * 3.4 + vWorld.x * 1.7 + uTime * .55) * .5 + .5;
    float caustic = pow(max(sin((vWorld.x - vWorld.y) * 7.0 + uTime) * .5 + .5, 0.0), 5.0);
    vec3 pearl = vec3(.985, .98, 1.0);
    vec3 liquid = mix(uLavender, uCyan, ribbon * .58 + .08);
    vec3 color = mix(pearl, liquid, .22 + fresnel * .48);
    color += caustic * vec3(.33, .28, .52);
    color += fresnel * vec3(.26, .22, .42);
    float alpha = uOpacity + fresnel * .34 + abs(vWave) * .16;
    gl_FragColor = vec4(color, alpha);
  }
`

const causticVertex = /* glsl */ `
  varying vec2 vUv;
  void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }
`

const causticFragment = /* glsl */ `
  uniform float uTime;
  varying vec2 vUv;
  void main(){
    vec2 p = vUv * 2.0 - 1.0;
    float a = sin((p.x * 5.0 + p.y * 3.2) + uTime * .28);
    float b = sin((p.x * -3.3 + p.y * 5.4) - uTime * .22);
    float bands = smoothstep(.72, .98, (a + b) * .25 + .5);
    float fade = smoothstep(1.1, .12, length(p));
    vec3 color = mix(vec3(.53,.42,1.0), vec3(.35,.88,.95), vUv.x);
    gl_FragColor = vec4(color, bands * fade * .075);
  }
`

export default function LiquidGlassStage() {
  const mountRef = useRef(null)

  useEffect(() => {
    const mount = mountRef.current
    if (!mount || !window.WebGLRenderingContext) {
      mount?.classList.add('webgl-unavailable')
      return undefined
    }

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(38, 1, .1, 100)
    camera.position.set(0, 0, 8.6)
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: 'high-performance' })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.55))
    renderer.setClearColor(0x000000, 0)
    renderer.outputColorSpace = THREE.SRGBColorSpace
    mount.appendChild(renderer.domElement)

    const geometry = new THREE.IcosahedronGeometry(1.48, 6)
    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      uniforms: {
        uTime: { value: 0 },
        uMotion: { value: reduced ? 0 : 1 },
        uOpacity: { value: .14 },
        uLavender: { value: new THREE.Color('#8d72ff') },
        uCyan: { value: new THREE.Color('#69dcea') },
      },
    })

    const leftOrb = new THREE.Mesh(geometry, material)
    leftOrb.position.set(-4.35, -2.05, -.2)
    leftOrb.scale.set(1.2, 1.2, 1.2)
    scene.add(leftOrb)

    const rightMaterial = material.clone()
    rightMaterial.uniforms = THREE.UniformsUtils.clone(material.uniforms)
    rightMaterial.uniforms.uOpacity.value = .085
    const rightOrb = new THREE.Mesh(geometry, rightMaterial)
    rightOrb.position.set(4.9, 2.25, -1.1)
    rightOrb.scale.set(.8, .8, .8)
    scene.add(rightOrb)

    const sheetMaterial = new THREE.ShaderMaterial({
      vertexShader: causticVertex,
      fragmentShader: causticFragment,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: { uTime: { value: 0 } },
    })
    const sheet = new THREE.Mesh(new THREE.PlaneGeometry(12, 7), sheetMaterial)
    sheet.position.z = -2.5
    scene.add(sheet)

    const pointer = new THREE.Vector2()
    const pointerTarget = new THREE.Vector2()
    const onPointer = (event) => {
      pointerTarget.set((event.clientX / window.innerWidth - .5) * .22, (event.clientY / window.innerHeight - .5) * -.16)
    }
    window.addEventListener('pointermove', onPointer, { passive: true })

    const resize = () => {
      const width = window.innerWidth
      const height = window.innerHeight
      renderer.setSize(width, height, false)
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      const desktop = width >= 850
      leftOrb.position.x = desktop ? -4.35 : -3.0
      rightOrb.visible = desktop
    }
    resize()
    window.addEventListener('resize', resize)

    const clock = new THREE.Clock()
    let frame = 0
    const render = () => {
      const time = clock.getElapsedTime()
      pointer.lerp(pointerTarget, .035)
      material.uniforms.uTime.value = time
      rightMaterial.uniforms.uTime.value = time + 4.2
      sheetMaterial.uniforms.uTime.value = reduced ? 0 : time
      if (!reduced) {
        leftOrb.rotation.y = time * .085 + pointer.x
        leftOrb.rotation.x = time * .045 + pointer.y
        rightOrb.rotation.y = -time * .065 - pointer.x
        rightOrb.rotation.z = time * .04
      }
      camera.position.x = pointer.x
      camera.position.y = pointer.y
      camera.lookAt(0, 0, 0)
      renderer.render(scene, camera)
      frame = requestAnimationFrame(render)
    }
    render()

    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('resize', resize)
      window.removeEventListener('pointermove', onPointer)
      geometry.dispose()
      material.dispose()
      rightMaterial.dispose()
      sheet.geometry.dispose()
      sheetMaterial.dispose()
      renderer.dispose()
      renderer.domElement.remove()
    }
  }, [])

  return (
    <>
      <div ref={mountRef} className="webgl-stage" aria-hidden="true" />
      <svg className="glass-filter-defs" aria-hidden="true">
        <filter id="glass-edge-warp" x="-20%" y="-20%" width="140%" height="140%">
          <feTurbulence type="fractalNoise" baseFrequency="0.008 0.016" numOctaves="2" seed="7" result="noise" />
          <feDisplacementMap in="SourceGraphic" in2="noise" scale="5" xChannelSelector="R" yChannelSelector="B" />
        </filter>
      </svg>
    </>
  )
}
