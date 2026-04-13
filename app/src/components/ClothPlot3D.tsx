import React, { useEffect, useRef, useCallback, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import * as d3 from 'd3';

// ── 2-D Canvas fallback (used when WebGL is unavailable) ─────────────────────
// Renders the cloth mesh as a flat XZ projection with per-node coloring.
interface Fallback2DProps {
  worldPositions: [number, number, number][];
  faces: [number, number, number][];
  colorValues?: number[];
  minVal?: number;
  maxVal?: number;
}

const ClothPlot2DFallback: React.FC<Fallback2DProps> = ({
  worldPositions, faces, colorValues, minVal, maxVal,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || worldPositions.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, W, H);

    // Project XZ plane (x→width, z→height flipped)
    const xs = worldPositions.map(p => p[0]);
    const zs = worldPositions.map(p => p[2]);
    const xMin = Math.min(...xs), xMax = Math.max(...xs) || 1;
    const zMin = Math.min(...zs), zMax = Math.max(...zs) || 1;
    const pad = 20;
    const scaleX = (W - pad * 2) / (xMax - xMin || 1);
    const scaleZ = (H - pad * 2) / (zMax - zMin || 1);
    const scale = Math.min(scaleX, scaleZ);
    const offX = pad + ((W - pad * 2) - (xMax - xMin) * scale) / 2;
    const offZ = pad + ((H - pad * 2) - (zMax - zMin) * scale) / 2;

    const px = (i: number) => offX + (worldPositions[i][0] - xMin) * scale;
    const pz = (i: number) => H - (offZ + (worldPositions[i][2] - zMin) * scale);

    const vals = colorValues ?? worldPositions.map(p => Math.sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]));
    const lo = minVal ?? (d3.min(vals) ?? 0);
    const hi = maxVal ?? (d3.max(vals) ?? 1);
    const colorScale = d3.scaleSequential(d3.interpolateTurbo).domain([lo, hi]);

    // Draw triangles — average vertex color for each face
    for (const [a, b, c] of faces) {
      const avgVal = (vals[a] + vals[b] + vals[c]) / 3;
      ctx.fillStyle = colorScale(avgVal);
      ctx.beginPath();
      ctx.moveTo(px(a), pz(a));
      ctx.lineTo(px(b), pz(b));
      ctx.lineTo(px(c), pz(c));
      ctx.closePath();
      ctx.fill();
    }

    // Thin mesh overlay
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 0.4;
    for (const [a, b, c] of faces) {
      ctx.beginPath();
      ctx.moveTo(px(a), pz(a));
      ctx.lineTo(px(b), pz(b));
      ctx.lineTo(px(c), pz(c));
      ctx.closePath();
      ctx.stroke();
    }
  }, [worldPositions, faces, colorValues, minVal, maxVal]);

  return (
    <div className="flex-1 flex flex-col items-stretch relative min-h-[200px]">
      <canvas ref={canvasRef} width={600} height={420}
        className="w-full h-full object-contain" />
      <div className="absolute bottom-2 left-0 right-0 flex justify-center">
        <span className="text-[10px] text-amber-400/70 bg-slate-900/80 px-2 py-0.5 rounded">
          3D view unavailable — WebGL requires GPU acceleration.
          Restart Chrome with <code className="font-mono">--use-gl=swiftshader</code> to enable.
        </span>
      </div>
    </div>
  );
};

export interface ClothPlot3DProps {
  /** [N, 3] world positions for the current frame */
  worldPositions: [number, number, number][];
  /** [F, 3] triangle face indices */
  faces: [number, number, number][];
  title: string;
  /** [N] scalar values for vertex coloring (e.g. position magnitude or error) */
  colorValues?: number[];
  minVal?: number;
  maxVal?: number;
  /** When set, this camera state is applied to the local camera each frame (linked cameras) */
  sharedCameraState?: { position: THREE.Vector3Like; quaternion: THREE.QuaternionLike; target: THREE.Vector3Like } | null;
  /** Called whenever the user interacts with OrbitControls, so the other panel can sync */
  onCameraChange?: (state: { position: THREE.Vector3Like; quaternion: THREE.QuaternionLike; target: THREE.Vector3Like }) => void;
}

export const ClothPlot3D: React.FC<ClothPlot3DProps> = ({
  worldPositions,
  faces,
  title,
  colorValues,
  minVal,
  maxVal,
  sharedCameraState,
  onCameraChange,
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [webglError, setWebglError] = useState<string | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const meshRef = useRef<THREE.Mesh | null>(null);
  const geometryRef = useRef<THREE.BufferGeometry | null>(null);
  const rafRef = useRef<number>(0);
  const isSyncingRef = useRef(false);  // prevent feedback loop when syncing cameras

  // Turbo colormap via d3
  const buildColorScale = useCallback((vals: number[], lo?: number, hi?: number) => {
    const vMin = lo ?? (d3.min(vals) ?? 0);
    const vMax = hi ?? (d3.max(vals) ?? 1);
    return d3.scaleSequential(d3.interpolateTurbo).domain([vMin, vMax]);
  }, []);

  // ── Mount: create renderer, scene, camera, lights, geometry, controls ────────
  useEffect(() => {
    if (!mountRef.current) return;
    const container = mountRef.current;
    const W = container.clientWidth  || 400;
    const H = container.clientHeight || 300;

    // Renderer — may fail if WebGL is not supported (headless / software-only GPU)
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    } catch (err: any) {
      setWebglError(String(err?.message ?? err));
      return;
    }
    if (!renderer.getContext()) {
      setWebglError('WebGL context creation failed (no hardware acceleration available)');
      renderer.dispose();
      return;
    }
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(W, H);
    renderer.setClearColor(0x0f172a);  // slate-950 background
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    // Lights
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(1, 2, 3);
    scene.add(dirLight);
    const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
    backLight.position.set(-1, -1, -2);
    scene.add(backLight);

    // Camera
    const camera = new THREE.PerspectiveCamera(45, W / H, 0.001, 1000);
    camera.position.set(0.5, 0.3, 1.5);
    cameraRef.current = camera;

    // Build initial geometry from current worldPositions and faces
    const geometry = new THREE.BufferGeometry();
    const N = worldPositions.length;
    const F = faces.length;

    const positions = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      positions[i * 3]     = worldPositions[i][0];
      positions[i * 3 + 1] = worldPositions[i][1];
      positions[i * 3 + 2] = worldPositions[i][2];
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const indices = new Uint32Array(F * 3);
    for (let i = 0; i < F; i++) {
      indices[i * 3]     = faces[i][0];
      indices[i * 3 + 1] = faces[i][1];
      indices[i * 3 + 2] = faces[i][2];
    }
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));

    // Vertex colors
    const colors = new Float32Array(N * 3);
    const vals = colorValues ?? worldPositions.map(p => Math.sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]));
    const scale = buildColorScale(vals, minVal, maxVal);
    for (let i = 0; i < N; i++) {
      const c = new THREE.Color(scale(vals[i]));
      colors[i * 3]     = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.computeVertexNormals();
    geometryRef.current = geometry;

    const material = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      shininess: 30,
    });
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);
    meshRef.current = mesh;

    // Center camera on mesh
    geometry.computeBoundingBox();
    const box = geometry.boundingBox!;
    const center = new THREE.Vector3();
    box.getCenter(center);
    const size = new THREE.Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    camera.position.copy(center).addScaledVector(new THREE.Vector3(0.3, 0.5, 1), maxDim * 1.5);

    // OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.copy(center);
    controls.update();
    controlsRef.current = controls;

    // Camera change → notify parent
    controls.addEventListener('change', () => {
      if (isSyncingRef.current) return;
      onCameraChange?.({
        position: camera.position.clone(),
        quaternion: camera.quaternion.clone(),
        target: controls.target.clone(),
      });
    });

    // Render loop
    const animate = () => {
      rafRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Resize observer
    const ro = new ResizeObserver(() => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      if (w > 0 && h > 0) {
        renderer.setSize(w, h);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      }
    });
    ro.observe(container);

    return () => {
      cancelAnimationFrame(rafRef.current);
      ro.disconnect();
      controls.dispose();
      geometry.dispose();
      material.dispose();
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      rendererRef.current = null;
      cameraRef.current   = null;
      controlsRef.current = null;
      sceneRef.current    = null;
      meshRef.current     = null;
      geometryRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);  // Mount once — geometry/color updates handled by the next effect

  // ── Update geometry positions and colors when worldPositions changes ─────────
  useEffect(() => {
    const geometry = geometryRef.current;
    if (!geometry) return;

    const posAttr = geometry.getAttribute('position') as THREE.BufferAttribute;
    const N = worldPositions.length;
    if (posAttr.count !== N) return;  // topology changed — would need full reinit

    for (let i = 0; i < N; i++) {
      posAttr.setXYZ(i, worldPositions[i][0], worldPositions[i][1], worldPositions[i][2]);
    }
    posAttr.needsUpdate = true;
    geometry.computeVertexNormals();

    // Update colors
    const colorAttr = geometry.getAttribute('color') as THREE.BufferAttribute;
    if (colorAttr) {
      const vals = colorValues ?? worldPositions.map(p => Math.sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]));
      const scale = buildColorScale(vals, minVal, maxVal);
      for (let i = 0; i < N; i++) {
        const c = new THREE.Color(scale(vals[i]));
        colorAttr.setXYZ(i, c.r, c.g, c.b);
      }
      colorAttr.needsUpdate = true;
    }
  }, [worldPositions, colorValues, minVal, maxVal, buildColorScale]);

  // ── Sync camera from parent (linked cameras) ──────────────────────────────────
  useEffect(() => {
    if (!sharedCameraState || !cameraRef.current || !controlsRef.current) return;
    isSyncingRef.current = true;
    cameraRef.current.position.copy(sharedCameraState.position as THREE.Vector3);
    cameraRef.current.quaternion.copy(sharedCameraState.quaternion as THREE.Quaternion);
    controlsRef.current.target.copy(sharedCameraState.target as THREE.Vector3);
    controlsRef.current.update();
    isSyncingRef.current = false;
  }, [sharedCameraState]);

  return (
    <div className="flex flex-col h-full w-full bg-slate-900/50 rounded-lg border border-slate-700 overflow-hidden">
      <div className="px-3 py-2 border-b border-slate-700 bg-slate-800/50 flex justify-between items-center">
        <span className="text-xs font-medium text-slate-300 uppercase tracking-wider">{title}</span>
        {!webglError && <span className="text-[10px] text-slate-500">drag: rotate • scroll: zoom • right-drag: pan</span>}
      </div>
      {webglError ? (
        <ClothPlot2DFallback
          worldPositions={worldPositions}
          faces={faces}
          colorValues={colorValues}
          minVal={minVal}
          maxVal={maxVal}
        />
      ) : (
        <div ref={mountRef} className="flex-1 relative min-h-[200px]" />
      )}
    </div>
  );
};
