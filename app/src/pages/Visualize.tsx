import React, { useState, useEffect, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Play, Pause, SkipBack, SkipForward, Info, Download, Trash2 } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, ReferenceArea, Legend } from 'recharts';
import { MeshPlot } from '../components/MeshPlot';
import * as d3 from 'd3';
import { cn } from '@/src/lib/utils';
import { useNavigate } from 'react-router-dom';
import { ClothPlot3D } from '../components/ClothPlot3D';

const LS_LAST_FILE_KEY = 'visualize_last_file';

export const Visualize: React.FC = () => {
  const [searchParams] = useSearchParams();
  // Use ?file= param if present; otherwise restore last viewed file from localStorage
  const paramFile = searchParams.get('file');
  const filename = paramFile ?? localStorage.getItem(LS_LAST_FILE_KEY) ?? null;

  const [metadata, setMetadata] = useState<any>(null);
  const [rmseData, setRmseData] = useState<any>(null);
  const [currentFrame, setCurrentFrame] = useState<any>(null);
  const [t, setT] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const playIntervalRef = useRef<any>(null);

  const [activeTab, setActiveTab] = useState<'viewer' | 'diagnostics' | 'physics'>('viewer');
  const [physicsData, setPhysicsData] = useState<any>(null);
  const [physicsEverLoaded, setPhysicsEverLoaded] = useState(false);
  const [trainEpochs, setTrainEpochs] = useState<any[]>([]);
  const [isLoadingPhysics, setIsLoadingPhysics] = useState(false);

  const navigate = useNavigate();

  // Linked 3D camera state for cloth viewer
  const [sharedCameraState, setSharedCameraState] = useState<{
    position: { x: number; y: number; z: number };
    quaternion: { x: number; y: number; z: number; w: number };
    target: { x: number; y: number; z: number };
  } | null>(null);

  // Cloth physics data
  const [clothPhysicsData, setClothPhysicsData] = useState<any>(null);

  // Persist last viewed file so navigating away and back restores it
  useEffect(() => {
    if (filename) localStorage.setItem(LS_LAST_FILE_KEY, filename);
  }, [filename]);

  useEffect(() => {
    if (!filename) return;
    // Reset state when file changes
    setMetadata(null);
    setRmseData(null);
    setCurrentFrame(null);
    setT(0);
    setIsPlaying(false);
    setPhysicsData(null);
    setPhysicsEverLoaded(false);
    setActiveTab('viewer');
    setColorRange(null);

    fetch(`/api/results/${filename}`).then(r => r.json()).then(setMetadata);
    fetch(`/api/results/${filename}/rmse`).then(r => r.json()).then(setRmseData);
    fetch('/api/train/status').then(r => r.json()).then(d => {
      if (d.epochs && d.epochs.length > 0) setTrainEpochs(d.epochs);
    }).catch(() => {});
  }, [filename]);

  useEffect(() => {
    if (!filename || activeTab !== 'physics') return;
    setIsLoadingPhysics(true);
    fetch(`/api/results/${filename}/physics?t=${t}`)
      .then(r => r.json())
      .then(data => {
        setPhysicsData(data);
        setPhysicsEverLoaded(true);
        setIsLoadingPhysics(false);
      })
      .catch(() => setIsLoadingPhysics(false));
  }, [t, activeTab, filename]);

  useEffect(() => {
    if (!filename || metadata?.domain !== 'flag_simple' || activeTab !== 'physics') return;
    fetch(`/api/results/${filename}/cloth_physics`)
      .then(r => r.json())
      .then(setClothPhysicsData)
      .catch(() => {});
  }, [filename, metadata?.domain, activeTab]);

  const [colorRange, setColorRange] = useState<[number, number] | null>(null);

  const mae = currentFrame ? d3.mean((currentFrame.error as (number|null)[]).filter((x): x is number => x !== null)) ?? 0 : 0;
  const maxErrorIdx = currentFrame ? d3.maxIndex((currentFrame.error as (number|null)[]).filter((x): x is number => x !== null)) : -1;
  const maxErrorVal = currentFrame && maxErrorIdx !== -1 ? (currentFrame.error as number[])[maxErrorIdx] : 0;
  const maxErrorPos = metadata && maxErrorIdx !== -1 ? metadata.crds[maxErrorIdx] : [0, 0];

  // Compute a stable global color range from the first frame so cloth animation
  // doesn't rescale the colormap every step (which looks like "only color change")
  useEffect(() => {
    if (!currentFrame || t !== 0) return;
    const vals = (currentFrame.predicted_magnitude as (number|null)[]).filter((x): x is number => x !== null);
    if (vals.length === 0) return;
    const lo = d3.min(vals) ?? 0;
    const hi = d3.max(vals) ?? 1;
    setColorRange([lo, hi]);
  }, [currentFrame, t]);

  const isGenerate: boolean = metadata?.is_generate === true;

  const fieldLabel = metadata?.domain === "flag_simple"
    ? "World Position (m)"
    : metadata?.target_field === "pressure"
      ? "Pressure (Pa)"
      : "Velocity Magnitude (m/s)";

  const errorLabel = metadata?.domain === "flag_simple"
    ? "Position Error (m)"
    : metadata?.target_field === "pressure"
      ? "Pressure Error (Pa)"
      : "Velocity Error (m/s)";

  const errorUnit = metadata?.domain === "flag_simple"
    ? "m"
    : metadata?.target_field === "pressure"
      ? "Pa"
      : "m/s";

  const overfittingStatus = () => {
    if (trainEpochs.length >= 3) {
      const last3 = trainEpochs.slice(-3);
      const isOverfitting = last3.every((e: any) => e.valid_loss / e.train_loss > 2);
      if (isOverfitting) return { label: '⚠️ Overfitting', color: 'bg-red-500/20 text-red-400', desc: 'Validation loss > 2× training loss for last 3 epochs' };
      const isUnderfitting = trainEpochs.length > 5 && trainEpochs.slice(-5).every((e: any) => e.train_loss > 0.01);
      if (isUnderfitting) return { label: '🟡 Underfitting', color: 'bg-yellow-500/20 text-yellow-400', desc: 'Training loss > 0.01 for last 5 epochs' };
      return { label: '✅ Healthy', color: 'bg-green-500/20 text-green-400', desc: 'Train/valid gap is normal' };
    }
    if (rmseData?.growth_ratio != null) {
      if (rmseData.growth_ratio > 50) return { label: '⚠️ High Error Drift', color: 'bg-red-500/20 text-red-400', desc: `RMSE grew ${rmseData.growth_ratio.toFixed(0)}× — autoregressive errors accumulate rapidly` };
      if (rmseData.growth_ratio > 10) return { label: '🟡 Moderate Error Drift', color: 'bg-yellow-500/20 text-yellow-400', desc: `RMSE grew ${rmseData.growth_ratio.toFixed(1)}× — normal for long rollouts` };
      return { label: '✅ Stable Rollout', color: 'bg-green-500/20 text-green-400', desc: `RMSE growth ratio: ${rmseData.growth_ratio.toFixed(1)}×` };
    }
    return { label: '— No Data', color: 'bg-slate-800 text-slate-500', desc: 'Run training first to see overfitting analysis' };
  };

  useEffect(() => {
    if (!filename || metadata === null) return;
    fetch(`/api/results/${filename}/frame/${t}`)
      .then(r => r.json())
      .then(setCurrentFrame);
  }, [filename, t, metadata]);

  useEffect(() => {
    if (isPlaying) {
      playIntervalRef.current = setInterval(() => {
        setT(prev => (prev + 1) % (metadata?.timesteps ?? 600));
      }, 50);
    } else {
      clearInterval(playIntervalRef.current);
    }
    return () => clearInterval(playIntervalRef.current);
  }, [isPlaying, metadata]);

  if (!filename) {
    return (
      <div className="p-12 text-center space-y-4">
        <Info className="w-12 h-12 text-slate-700 mx-auto" />
        <h2 className="text-xl font-bold text-white">No Rollout Selected</h2>
        <p className="text-slate-400">Please select a rollout from the Predict page to visualize.</p>
      </div>
    );
  }

  if (!metadata || !currentFrame) {
    return <div className="p-8 text-slate-400">Loading visualization data…</div>;
  }

  // For generate rollouts: only viewer tab makes sense (no real GT → no RMSE/diagnostics)
  const availableTabs = isGenerate
    ? (['viewer'] as const)
    : (['viewer', 'diagnostics', 'physics'] as const);

  return (
    <div className="p-8 space-y-6 max-w-full mx-auto">
      <header className="flex justify-between items-center">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-2xl font-bold text-white tracking-tight">{filename}</h2>
            {isGenerate ? (
              <span className="px-2 py-0.5 bg-violet-600/20 text-violet-400 text-[10px] font-bold rounded border border-violet-500/20 uppercase tracking-wider">
                Generated Design
              </span>
            ) : (
              <>
                {metadata.speedup != null && (
                  <span className="px-2 py-0.5 bg-blue-600/20 text-blue-400 text-[10px] font-bold rounded border border-blue-500/20 uppercase tracking-wider">
                    {metadata.speedup}x Real-time
                  </span>
                )}
              </>
            )}
          </div>
          <p className="text-slate-500 text-sm mt-1">
            {isGenerate
              ? `GNN Rollout (no ground truth) • ${metadata.num_nodes} nodes • ${metadata.timesteps} steps`
              : `Trajectory Analysis • ${metadata.num_nodes} nodes • ${metadata.timesteps} steps`}
          </p>
        </div>
        <div className="flex gap-4">
          {availableTabs.length > 1 && (
            <div className="flex bg-slate-900 p-1 rounded-lg border border-slate-800">
              {availableTabs.map(tab => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={cn(
                    "px-4 py-1.5 rounded-md text-xs font-bold uppercase tracking-wider transition-all",
                    activeTab === tab ? "bg-blue-600 text-white shadow-lg" : "text-slate-500 hover:text-slate-300"
                  )}
                >
                  {tab}
                </button>
              ))}
            </div>
          )}
          <div className="flex gap-2">
            <button
              onClick={() => window.open(`/api/results/${filename}/download`, '_blank')}
              className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
              title="Download result file"
            >
              <Download className="w-5 h-5" />
            </button>
            <button
              onClick={async () => {
                if (!window.confirm(`Delete ${filename}?`)) return;
                await fetch(`/api/results/${filename}`, { method: 'DELETE' });
                navigate('/predict');
              }}
              className="p-2 text-slate-400 hover:text-red-400 hover:bg-red-950/30 rounded-lg transition-colors"
              title="Delete this result"
            >
              <Trash2 className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>

      {/* ── VIEWER tab ── */}
      {activeTab === 'viewer' && (
        <>
          {/* Generate rollout: single prediction panel with notice */}
          {isGenerate ? (
            <div className="space-y-3">
              <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-violet-900/20 border border-violet-700/30 text-violet-300 text-xs">
                <Info className="w-3.5 h-3.5 flex-shrink-0" />
                This is a generated candidate — no simulator ground truth exists. The panel shows the GNN forward rollout only.
              </div>
              {metadata.domain === 'flag_simple' ? (
                <div className="h-[520px]">
                  <ClothPlot3D
                    worldPositions={(currentFrame.world_pos_pred ?? []) as [number,number,number][]}
                    faces={metadata.triangles as [number,number,number][]}
                    title={`GNN Prediction — ${fieldLabel}`}
                    colorValues={(currentFrame.predicted_magnitude as (number|null)[]).map(v => v ?? 0)}
                    minVal={colorRange?.[0]}
                    maxVal={colorRange?.[1]}
                  />
                </div>
              ) : (
                <div className="h-[400px]">
                  <MeshPlot
                    crds={metadata.crds}
                    triangles={metadata.triangles}
                    values={(currentFrame.predicted_magnitude as (number|null)[]).map(v => v ?? 0)}
                    title={`GNN Prediction — ${fieldLabel}`}
                    domain={metadata.domain}
                    minVal={colorRange?.[0]}
                    maxVal={colorRange?.[1]}
                  />
                </div>
              )}
            </div>
          ) : (
            /* Full predict rollout: GT + Prediction + Error */
            <>
              {metadata.domain === 'flag_simple' ? (
                /* 3D cloth viewer — two panels side by side */
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 h-[520px]">
                  <ClothPlot3D
                    worldPositions={(currentFrame.world_pos_target ?? []) as [number,number,number][]}
                    faces={metadata.triangles as [number,number,number][]}
                    title={`Ground Truth — ${fieldLabel}`}
                    colorValues={(currentFrame.target_magnitude as (number|null)[]).map(v => v ?? 0)}
                    minVal={colorRange?.[0]}
                    maxVal={colorRange?.[1]}
                    onCameraChange={setSharedCameraState}
                  />
                  <ClothPlot3D
                    worldPositions={(currentFrame.world_pos_pred ?? []) as [number,number,number][]}
                    faces={metadata.triangles as [number,number,number][]}
                    title={`Prediction — ${fieldLabel}`}
                    colorValues={(currentFrame.predicted_magnitude as (number|null)[]).map(v => v ?? 0)}
                    minVal={colorRange?.[0]}
                    maxVal={colorRange?.[1]}
                    sharedCameraState={sharedCameraState}
                  />
                </div>
              ) : (
                /* CFD / pressure: existing 3-panel MeshPlot grid */
                <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 h-[400px]">
                  <MeshPlot
                    crds={metadata.crds}
                    triangles={metadata.triangles}
                    values={(currentFrame.target_magnitude as (number|null)[]).map(v => v ?? 0)}
                    title={`Ground Truth — ${fieldLabel}`}
                    domain={metadata.domain}
                  />
                  <MeshPlot
                    crds={metadata.crds}
                    triangles={metadata.triangles}
                    values={(currentFrame.predicted_magnitude as (number|null)[]).map(v => v ?? 0)}
                    title={`Prediction — ${fieldLabel}`}
                    domain={metadata.domain}
                  />
                  <MeshPlot
                    crds={metadata.crds}
                    triangles={metadata.triangles}
                    values={(currentFrame.error as (number|null)[]).map(v => v ?? 0)}
                    title="Error Magnitude"
                    minVal={0}
                    autoScale={true}
                    colorScale={d3.scaleSequential(d3.interpolateReds).domain([0, 0.1])}
                    domain={metadata.domain}
                  />
                </div>
              )}
            </>
          )}

          {/* Playback Controls */}
          <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-6">
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2">
                <button onClick={() => setT(0)} className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors">
                  <SkipBack className="w-5 h-5" />
                </button>
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  className="w-12 h-12 bg-blue-600 hover:bg-blue-500 text-white rounded-full flex items-center justify-center shadow-lg shadow-blue-900/20 transition-all"
                >
                  {isPlaying ? <Pause className="w-6 h-6 fill-current" /> : <Play className="w-6 h-6 fill-current ml-1" />}
                </button>
                <button onClick={() => setT((metadata?.timesteps ?? 600) - 1)} className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors">
                  <SkipForward className="w-5 h-5" />
                </button>
              </div>

              <div className="flex-1 space-y-2">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-blue-400">Step {t} / {(metadata?.timesteps ?? 600) - 1}</span>
                  <span className="text-slate-500">{(t * (metadata?.dt ?? 0.01)).toFixed(2)}s / {(((metadata?.timesteps ?? 600) - 1) * (metadata?.dt ?? 0.01)).toFixed(2)}s</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max={(metadata?.timesteps ?? 600) - 1}
                  value={t}
                  onChange={(e) => setT(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-full appearance-none cursor-pointer accent-blue-500"
                />
              </div>

              {!isGenerate && (
                <div className="w-32 text-right">
                  <p className="text-[10px] text-slate-500 font-bold uppercase">Current RMSE</p>
                  <p className="text-lg font-bold text-white font-mono">
                    {currentFrame.rmse != null ? currentFrame.rmse.toFixed(6) : '—'}
                  </p>
                </div>
              )}
            </div>
          </section>

          {/* RMSE chart — only for real rollouts */}
          {!isGenerate && (
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
              <section className="lg:col-span-3 bg-slate-900/50 border border-slate-800 rounded-2xl p-6 h-[300px] flex flex-col">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold text-white text-sm">RMSE Over Time</h3>
                  <div className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">Stability Analysis</div>
                </div>
                <div className="flex-1 min-h-0">
                  {rmseData && (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={rmseData.per_step_rmse.map((v: number, i: number) => ({ t: i * 0.01, rmse: v }))}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                        <XAxis dataKey="t" stroke="#64748b" fontSize={10} tickLine={false} axisLine={false} tickFormatter={(v) => `${v.toFixed(1)}s`} />
                        <YAxis stroke="#64748b" fontSize={10} tickLine={false} axisLine={false} tickFormatter={(v) => v.toFixed(4)} />
                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px' }} labelFormatter={(v) => `Time: ${v.toFixed(2)}s`} />
                        <ReferenceLine x={t * 0.01} stroke="#3b82f6" strokeDasharray="3 3" />
                        <Line type="monotone" dataKey="rmse" stroke="#3b82f6" strokeWidth={2} dot={false} isAnimationActive={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </div>
              </section>

              <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-6">
                <h3 className="font-semibold text-white text-sm">Metrics</h3>
                <div className="space-y-4">
                  <MetricItem label="Initial RMSE" value={rmseData?.rmse_at_0 != null ? rmseData.rmse_at_0.toFixed(6) : '—'} />
                  <MetricItem label="Final RMSE" value={rmseData?.rmse_at_599 != null ? rmseData.rmse_at_599.toFixed(6) : '—'} />
                  <MetricItem label="Growth Ratio" value={rmseData?.growth_ratio != null ? `${rmseData.growth_ratio.toFixed(1)}x` : '—'} color="text-orange-400" />
                  <div className="pt-4 border-t border-slate-800">
                    <div className="p-3 bg-blue-600/10 rounded-lg border border-blue-500/20">
                      <p className="text-[10px] text-blue-400 font-bold uppercase mb-1">Stability Rating</p>
                      <p className="text-sm font-bold text-white">
                        {rmseData?.growth_ratio != null
                          ? rmseData.growth_ratio < 5 ? '✅ High (Stable)'
                            : rmseData.growth_ratio < 20 ? '🟡 Moderate'
                            : '🔴 Unstable'
                          : '—'}
                      </p>
                    </div>
                  </div>
                </div>
              </section>
            </div>
          )}
        </>
      )}

      {/* ── DIAGNOSTICS tab (predict rollouts only) ── */}
      {activeTab === 'diagnostics' && !isGenerate && (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-6">
              <div className="flex justify-between items-center">
                <div>
                  <h3 className="font-semibold text-white">Overfitting Detector</h3>
                  <p className="text-[10px] text-slate-500 mt-0.5">{overfittingStatus().desc}</p>
                </div>
                <span className={cn("px-3 py-1 rounded-full text-xs font-bold", overfittingStatus().color)}>{overfittingStatus().label}</span>
              </div>
              <div className="h-64">
                {trainEpochs.length > 0 ? (() => {
                  const areas: { x1: number; x2: number; color: string }[] = [];
                  for (let i = 1; i < trainEpochs.length; i++) {
                    const e = trainEpochs[i];
                    const ratio = e.valid_loss / (e.train_loss + 1e-12);
                    const color = ratio > 2 ? '#ef444420' : e.train_loss > 0.01 ? '#eab30820' : '#22c55e20';
                    if (areas.length > 0 && areas[areas.length - 1].color === color) {
                      areas[areas.length - 1].x2 = e.epoch;
                    } else {
                      areas.push({ x1: trainEpochs[i - 1].epoch, x2: e.epoch, color });
                    }
                  }
                  return (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={trainEpochs}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                        <XAxis dataKey="epoch" stroke="#64748b" fontSize={10} />
                        <YAxis scale="log" domain={['auto', 'auto']} stroke="#64748b" fontSize={10} />
                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }} />
                        {(areas as any[]).map((a, i) => {
                          const raProps: any = { x1: a.x1, x2: a.x2, fill: a.color, ifOverflow: "extendDomain" };
                          return <React.Fragment key={i}><ReferenceArea {...raProps} /></React.Fragment>;
                        })}
                        <Line type="monotone" dataKey="train_loss" stroke="#3b82f6" strokeWidth={2} dot={false} name="Train Loss" />
                        <Line type="monotone" dataKey="valid_loss" stroke="#fb7185" strokeWidth={2} dot={false} name="Valid Loss" />
                      </LineChart>
                    </ResponsiveContainer>
                  );
                })() : rmseData ? (
                  <div className="h-full flex flex-col items-center justify-center gap-3 text-center">
                    <p className="text-slate-500 text-xs">No training log found — showing RMSE curve as proxy</p>
                    <ResponsiveContainer width="100%" height="80%">
                      <LineChart data={rmseData.per_step_rmse.map((v: number, i: number) => ({ t: i, rmse: v }))}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                        <XAxis dataKey="t" stroke="#64748b" fontSize={10} label={{ value: 'timestep', position: 'insideBottom', fill: '#64748b', fontSize: 9 }} />
                        <YAxis stroke="#64748b" fontSize={10} tickFormatter={(v) => v.toFixed(4)} />
                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }} />
                        <Line type="monotone" dataKey="rmse" stroke="#3b82f6" strokeWidth={2} dot={false} name="RMSE" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center text-slate-600 text-sm">No data</div>
                )}
              </div>
            </section>

            <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-6">
              <h3 className="font-semibold text-white">MAE Dashboard</h3>
              <div className="grid grid-cols-3 gap-4">
                <MetricItem label="MAE @ t=0"   value={rmseData?.mae_at_0   != null ? rmseData.mae_at_0.toFixed(6)   : '—'} />
                <MetricItem label="MAE @ t=end" value={rmseData?.mae_at_end != null ? rmseData.mae_at_end.toFixed(6) : '—'} />
                <MetricItem label="MAE/RMSE"    value={
                  rmseData?.mae_at_0 != null && rmseData?.rmse_at_0 != null
                    ? (rmseData.mae_at_0 / (rmseData.rmse_at_0 + 1e-12)).toFixed(2)
                    : '—'
                } />
              </div>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={rmseData?.per_step_rmse?.map((v: number, i: number) => ({
                    t: rmseData.times?.[i] ?? i * 0.01,
                    rmse: v,
                    mae: rmseData.per_step_mae?.[i] ?? null,
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="t" stroke="#64748b" fontSize={9} tickFormatter={(v) => `${parseFloat(v).toFixed(1)}s`} />
                    <YAxis stroke="#64748b" fontSize={9} tickFormatter={(v) => v.toExponential(1)} />
                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }} labelFormatter={(v) => `t=${parseFloat(v).toFixed(2)}s`} />
                    <Legend verticalAlign="top" height={24} iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
                    <Line type="monotone" dataKey="rmse" stroke="#3b82f6" strokeWidth={1.5} dot={false} name="RMSE" connectNulls />
                    <Line type="monotone" dataKey="mae"  stroke="#10b981" strokeWidth={2}   dot={false} name="MAE"  connectNulls />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
          </div>

          <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-6">
            <div className="flex justify-between items-center">
              <h3 className="font-semibold text-white">Error Heatmap & Hotspot Analysis</h3>
              <div className="flex items-center gap-4">
                <div className="text-right">
                  <p className="text-[10px] text-slate-500 font-bold uppercase">{errorLabel}</p>
                  <p className="text-sm font-bold text-red-400">{maxErrorVal.toFixed(6)} {errorUnit}</p>
                </div>
                <div className="text-right">
                  <p className="text-[10px] text-slate-500 font-bold uppercase">Node Index</p>
                  <p className="text-sm font-bold text-white">{maxErrorIdx}</p>
                </div>
              </div>
            </div>
            <div className="h-[400px] relative">
              <MeshPlot
                crds={metadata.crds}
                triangles={metadata.triangles}
                values={currentFrame.error}
                title={`Error Hotspot at t=${t}`}
                colorScale={d3.scaleSequential(d3.interpolateReds).domain([0, 0.1])}
                domain={metadata.domain}
              />
              <div className="absolute top-12 left-4 bg-slate-950/80 p-2 rounded border border-red-500/30 text-[10px]">
                📍 Max Error: {maxErrorVal.toFixed(4)} {errorUnit} at ({maxErrorPos[0].toFixed(2)}, {maxErrorPos[1].toFixed(2)})
              </div>
            </div>
          </section>

          {metadata?.confidence_score != null && (() => {
            const score: number = metadata.confidence_score;
            const isGreen = score >= 0.8;
            const isAmber = score >= 0.5 && score < 0.8;
            const barColor = isGreen ? 'bg-green-500' : isAmber ? 'bg-amber-500' : 'bg-red-500';
            const badge = isGreen
              ? { label: 'High similarity',  cls: 'bg-green-500/20 text-green-400 border-green-500/30' }
              : isAmber
                ? { label: 'Partial match',  cls: 'bg-amber-500/20 text-amber-400 border-amber-500/30' }
                : { label: 'Low similarity', cls: 'bg-red-500/20   text-red-400   border-red-500/30'   };
            const guidance = isGreen
              ? 'This mesh is well within the training distribution. Predictions should be reliable.'
              : isAmber
                ? 'This mesh is somewhat novel. Predictions are estimates — verify critical results independently.'
                : 'This mesh differs significantly from the training set. Consider verifying with a full CFD solver.';
            const scoreColor = isGreen ? 'text-green-400' : isAmber ? 'text-amber-400' : 'text-red-400';
            const sectionCls = isGreen ? 'bg-green-900/10 border-green-500/20' : isAmber ? 'bg-amber-900/10 border-amber-500/20' : 'bg-red-900/10 border-red-500/20';
            return (
              <div className={`border rounded-xl p-4 space-y-3 ${sectionCls}`}>
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Training Similarity</h3>
                  <span className={`px-2.5 py-1 rounded-full text-[10px] font-bold uppercase border ${badge.cls}`}>
                    {badge.label}
                  </span>
                </div>
                <div className="flex items-end gap-4">
                  <span className={`text-4xl font-bold font-mono ${scoreColor}`}>
                    {(score * 100).toFixed(0)}<span className="text-xl text-slate-500">%</span>
                  </span>
                  <div className="flex-1 pb-1.5">
                    <div className="w-full h-2 bg-slate-950 rounded-full overflow-hidden">
                      <div className={`h-full rounded-full transition-all duration-500 ${barColor}`}
                           style={{ width: `${Math.max(0, score * 100)}%` }} />
                    </div>
                    <div className="relative mt-0.5">
                      <span className="absolute text-[8px] text-slate-700" style={{ left: '50%', transform: 'translateX(-50%)' }}>50%</span>
                      <span className="absolute text-[8px] text-slate-700" style={{ left: '80%', transform: 'translateX(-50%)' }}>80%</span>
                    </div>
                  </div>
                </div>
                <p className="text-[10px] text-slate-400 leading-relaxed">{guidance}</p>
                <p className="text-[9px] text-slate-600 italic">
                  Latent-space KDTree score: 1 − (d_min / training_diameter). Higher = more similar to training data.
                </p>
              </div>
            );
          })()}
        </div>
      )}

      {/* ── PHYSICS tab (predict rollouts, CFD only) ── */}
      {activeTab === 'physics' && !isGenerate && (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4">
          {metadata?.domain === 'flag_simple' ? (
            <div className="space-y-6">
              <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-6">
                <div className="flex justify-between items-center">
                  <h3 className="font-semibold text-white">Edge Stretch (Elastic Deformation)</h3>
                  <p className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">
                    stretch = |world_edge_len / rest_len − 1|
                  </p>
                </div>
                {clothPhysicsData ? (
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={clothPhysicsData.per_step_mean_stretch.map((v: number, i: number) => ({
                        t: clothPhysicsData.times?.[i] ?? i * 0.01,
                        mean: v,
                        max: clothPhysicsData.per_step_max_stretch?.[i] ?? null,
                      }))}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                        <XAxis dataKey="t" stroke="#64748b" fontSize={10} tickFormatter={(v) => `${parseFloat(v).toFixed(1)}s`} />
                        <YAxis stroke="#64748b" fontSize={10} tickFormatter={(v) => (v * 100).toFixed(1) + '%'} />
                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px' }}
                          labelFormatter={(v) => `t=${parseFloat(v as string).toFixed(2)}s`}
                          formatter={(v: number) => [(v * 100).toFixed(2) + '%']} />
                        <Legend verticalAlign="top" height={24} iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
                        <Line type="monotone" dataKey="mean" stroke="#3b82f6" strokeWidth={2} dot={false} name="Mean Stretch" />
                        <Line type="monotone" dataKey="max"  stroke="#fb7185" strokeWidth={1.5} dot={false} name="Max Stretch" strokeDasharray="4 2" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="h-64 flex items-center justify-center text-slate-500 text-sm">
                    Loading stretch data…
                  </div>
                )}
                <p className="text-[10px] text-slate-500 italic text-center">
                  Stretch measures how much each mesh edge has deformed from its rest configuration.
                  High stretch indicates large deformation or potential simulation instability.
                </p>
              </section>
            </div>
          ) : (
            <>
              {isLoadingPhysics ? (
                <div className="p-12 text-center text-slate-500">Computing physics metrics...</div>
              ) : physicsData && (
                <>
                  <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-6">
                    <h3 className="font-semibold text-white">Vorticity Comparison (ω = ∂vy/∂x − ∂vx/∂y)</h3>
                    <div className="grid grid-cols-2 gap-4 h-[300px]">
                      <MeshPlot crds={metadata.crds} triangles={metadata.triangles} values={physicsData.vorticity_target} title="Ground Truth ω" minVal={physicsData.omega_min} maxVal={physicsData.omega_max} colorScale={d3.scaleSequential(d3.interpolateRdBu).domain([physicsData.omega_max, physicsData.omega_min])} domain={metadata.domain} />
                      <MeshPlot crds={metadata.crds} triangles={metadata.triangles} values={physicsData.vorticity_pred}   title="Predicted ω"     minVal={physicsData.omega_min} maxVal={physicsData.omega_max} colorScale={d3.scaleSequential(d3.interpolateRdBu).domain([physicsData.omega_max, physicsData.omega_min])} domain={metadata.domain} />
                    </div>
                    <p className="text-[10px] text-slate-500 italic text-center">Positive values (blue) indicate counter-clockwise rotation, negative (red) indicate clockwise rotation.</p>
                  </section>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-6">
                      <div className="flex justify-between items-center">
                        <h3 className="font-semibold text-white">Energy Conservation</h3>
                        <div className="text-right">
                          <p className="text-[10px] text-slate-500 font-bold uppercase">ΔE (Final)</p>
                          <p className={cn("text-sm font-bold", (physicsData.energy_pred_series[physicsData.energy_pred_series.length - 1] - physicsData.energy_target_series[physicsData.energy_target_series.length - 1]) > 0 ? "text-orange-400" : "text-green-400")}>
                            {(physicsData.energy_pred_series[physicsData.energy_pred_series.length - 1] - physicsData.energy_target_series[physicsData.energy_target_series.length - 1]).toFixed(4)}
                          </p>
                        </div>
                      </div>
                      <div className="h-48">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={physicsData.energy_pred_series.map((v: any, i: any) => ({ t: i * 0.01, pred: v, target: physicsData.energy_target_series[i] }))}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                            <XAxis dataKey="t" stroke="#64748b" fontSize={9} tickFormatter={(v) => `${parseFloat(v).toFixed(1)}s`} />
                            <YAxis domain={['auto', 'auto']} stroke="#64748b" fontSize={9} tickFormatter={(v) => v.toExponential(1)} />
                            <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }} labelFormatter={(v) => `t=${parseFloat(v).toFixed(2)}s`} />
                            <Legend verticalAlign="top" height={24} iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
                            <Line type="monotone" dataKey="target" stroke="#64748b" strokeWidth={1.5} dot={false} name="Target E" />
                            <Line type="monotone" dataKey="pred"   stroke="#3b82f6" strokeWidth={2}   dot={false} name="Pred E" />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </section>

                    <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-6">
                      <h3 className="font-semibold text-white">Divergence Proxy (Mass Conservation)</h3>
                      <div className="h-48">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={physicsData.divergence_pred.map((v: any, i: any) => ({ t: i * 0.01, pred: v, target: physicsData.divergence_target[i] }))}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                            <XAxis dataKey="t" hide />
                            <YAxis domain={[0, 0.01]} hide />
                            <Line type="monotone" dataKey="target" stroke="#64748b" strokeWidth={1} dot={false} />
                            <Line type="monotone" dataKey="pred"   stroke="#fb7185" strokeWidth={2} dot={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                      <p className="text-[10px] text-slate-500 text-center">Spikes indicate regions where mass conservation is violated.</p>
                    </section>
                  </div>
                </>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
};

const MetricItem = ({ label, value, color = "text-slate-200" }: any) => (
  <div className="flex justify-between items-center">
    <span className="text-xs text-slate-500">{label}</span>
    <span className={`text-sm font-mono font-bold ${color}`}>{value}</span>
  </div>
);

