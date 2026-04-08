import React, { useState, useEffect, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Play, Pause, SkipBack, SkipForward, Info, Download, Trash2, Maximize2 } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, ReferenceArea, Legend } from 'recharts';
import { MeshPlot } from '../components/MeshPlot';
import * as d3 from 'd3';
import { cn } from '@/src/lib/utils';

export const Visualize: React.FC = () => {
  const [searchParams] = useSearchParams();
  const filename = searchParams.get('file');

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

  useEffect(() => {
    if (!filename) return;

    fetch(`/api/results/${filename}`).then(r => r.json()).then(setMetadata);
    fetch(`/api/results/${filename}/rmse`).then(r => r.json()).then(setRmseData);
    // Load train log for overfitting detector — use /train/status which parses runs/train_ui.log
    fetch('/api/train/status').then(r => r.json()).then(d => {
      if (d.epochs && d.epochs.length > 0) setTrainEpochs(d.epochs);
    }).catch(() => {});
    // NOTE: physics is NOT fetched on mount — only when Physics tab is clicked (see below)
  }, [filename]);

  // Load physics data when Physics tab is first opened, then re-fetch when t changes (on that tab only)
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

  // Compute MAE and hotspot — only after null guard (metadata & currentFrame are guaranteed non-null below)
  const mae = currentFrame ? d3.mean(currentFrame.error as number[]) ?? 0 : 0;
  const maxErrorIdx = currentFrame ? d3.maxIndex(currentFrame.error as number[]) : -1;
  const maxErrorVal = currentFrame && maxErrorIdx !== -1 ? currentFrame.error[maxErrorIdx] : 0;
  const maxErrorPos = metadata && maxErrorIdx !== -1 ? metadata.crds[maxErrorIdx] : [0, 0];

  const fieldLabel = metadata?.domain === "flag_simple"
    ? "Position Magnitude (m)"
    : "Velocity Magnitude (m/s)";

  const errorLabel = metadata?.domain === "flag_simple"
    ? "Position Error (m)"
    : "Velocity Error (m/s)";

  const errorUnit = metadata?.domain === "flag_simple" ? "m" : "m/s";

  const overfittingStatus = () => {
    // If we have actual training epoch data, use it
    if (trainEpochs.length >= 3) {
      const last3 = trainEpochs.slice(-3);
      const isOverfitting = last3.every((e: any) => e.valid_loss / e.train_loss > 2);
      if (isOverfitting) return { label: '⚠️ Overfitting', color: 'bg-red-500/20 text-red-400', desc: 'Validation loss > 2× training loss for last 3 epochs' };
      const isUnderfitting = trainEpochs.length > 5 && trainEpochs.slice(-5).every((e: any) => e.train_loss > 0.01);
      if (isUnderfitting) return { label: '🟡 Underfitting', color: 'bg-yellow-500/20 text-yellow-400', desc: 'Training loss > 0.01 for last 5 epochs' };
      return { label: '✅ Healthy', color: 'bg-green-500/20 text-green-400', desc: 'Train/valid gap is normal' };
    }
    // Fallback: use RMSE growth as a proxy for overfitting
    // A very high growth ratio suggests the model generalized poorly
    if (rmseData?.growth_ratio != null) {
      if (rmseData.growth_ratio > 50) return { label: '⚠️ Poor Generalization', color: 'bg-red-500/20 text-red-400', desc: `RMSE grew ${rmseData.growth_ratio.toFixed(0)}× — model may have overfit` };
      if (rmseData.growth_ratio > 10) return { label: '🟡 Moderate Drift', color: 'bg-yellow-500/20 text-yellow-400', desc: `RMSE grew ${rmseData.growth_ratio.toFixed(1)}× over simulation` };
      return { label: '✅ Stable Rollout', color: 'bg-green-500/20 text-green-400', desc: `RMSE growth ratio: ${rmseData.growth_ratio.toFixed(1)}×` };
    }
    return { label: '— No Data', color: 'bg-slate-800 text-slate-500', desc: 'Run training first to see overfitting analysis' };
  };

  useEffect(() => {
    if (!filename || metadata === null) return;
    
    // Load frame data on demand
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
  }, [isPlaying]);

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
    return <div className="p-8 text-slate-400">Loading visualization data...</div>;
  }

  return (
    <div className="p-8 space-y-6 max-w-full mx-auto">
      <header className="flex justify-between items-center">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-2xl font-bold text-white tracking-tight">{filename}</h2>
            <span className="px-2 py-0.5 bg-blue-600/20 text-blue-400 text-[10px] font-bold rounded border border-blue-500/20 uppercase tracking-wider">
              {metadata.speedup}x Real-time
            </span>
          </div>
          <p className="text-slate-500 text-sm mt-1">
            Trajectory Analysis • {metadata.num_nodes} nodes • {metadata.timesteps} steps
          </p>
        </div>
        <div className="flex gap-4">
          <div className="flex bg-slate-900 p-1 rounded-lg border border-slate-800">
            {(['viewer', 'diagnostics', 'physics'] as const).map(tab => (
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
          <div className="flex gap-2">
            <button className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors">
              <Download className="w-5 h-5" />
            </button>
            <button className="p-2 text-slate-400 hover:text-red-400 hover:bg-red-950/30 rounded-lg transition-colors">
              <Trash2 className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>

      {activeTab === 'viewer' && (
        <>
          {/* Mesh Visualization Grid */}
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 h-[400px]">
        <MeshPlot
          crds={metadata.crds}
          triangles={metadata.triangles}
          values={currentFrame.target_magnitude}
          title={`Ground Truth — ${fieldLabel}`}
        />
        <MeshPlot
          crds={metadata.crds}
          triangles={metadata.triangles}
          values={currentFrame.predicted_magnitude}
          title={`Prediction — ${fieldLabel}`}
        />
        <MeshPlot 
          crds={metadata.crds} 
          triangles={metadata.triangles} 
          values={currentFrame.error} 
          title="Error Magnitude"
          minVal={0}
          maxVal={0.1}
          colorScale={d3.scaleSequential(d3.interpolateReds).domain([0, 0.1])}
        />
      </div>

      {/* Playback Controls */}
      <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-6">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <button 
              onClick={() => setT(0)}
              className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
            >
              <SkipBack className="w-5 h-5" />
            </button>
            <button 
              onClick={() => setIsPlaying(!isPlaying)}
              className="w-12 h-12 bg-blue-600 hover:bg-blue-500 text-white rounded-full flex items-center justify-center shadow-lg shadow-blue-900/20 transition-all"
            >
              {isPlaying ? <Pause className="w-6 h-6 fill-current" /> : <Play className="w-6 h-6 fill-current ml-1" />}
            </button>
            <button
              onClick={() => setT((metadata?.timesteps ?? 600) - 1)}
              className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
            >
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

          <div className="w-32 text-right">
            <p className="text-[10px] text-slate-500 font-bold uppercase">Current RMSE</p>
            <p className="text-lg font-bold text-white font-mono">{currentFrame.rmse.toFixed(6)}</p>
          </div>
        </div>
      </section>

      {/* RMSE Analysis */}
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
                  <XAxis 
                    dataKey="t" 
                    stroke="#64748b" 
                    fontSize={10} 
                    tickLine={false} 
                    axisLine={false}
                    tickFormatter={(v) => `${v.toFixed(1)}s`}
                  />
                  <YAxis 
                    stroke="#64748b" 
                    fontSize={10} 
                    tickLine={false} 
                    axisLine={false}
                    tickFormatter={(v) => v.toFixed(4)}
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px' }}
                    labelFormatter={(v) => `Time: ${v.toFixed(2)}s`}
                  />
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
        </>
      )}

      {activeTab === 'diagnostics' && (
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
                  // Compute vrect zones: classify each epoch range
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
                        {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
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
                <MetricItem label="MAE @ t=0"    value={rmseData?.mae_at_0   != null ? rmseData.mae_at_0.toFixed(6)   : '—'} />
                <MetricItem label="MAE @ t=end"  value={rmseData?.mae_at_end != null ? rmseData.mae_at_end.toFixed(6) : '—'} />
                <MetricItem label="MAE/RMSE"     value={
                  rmseData?.mae_at_0 != null && rmseData?.rmse_at_0 != null
                    ? (rmseData.mae_at_0 / (rmseData.rmse_at_0 + 1e-12)).toFixed(2)
                    : '—'
                } />
              </div>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={rmseData?.per_step_rmse?.map((v: number, i: number) => ({
                    t: i * 0.01,
                    rmse: v,
                    mae: rmseData.per_step_mae?.[i] ?? null,
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="t" stroke="#64748b" fontSize={9} tickFormatter={(v) => `${v.toFixed(1)}s`} />
                    <YAxis stroke="#64748b" fontSize={9} tickFormatter={(v) => v.toExponential(1)} />
                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }} labelFormatter={(v) => `t=${parseFloat(v).toFixed(2)}s`} />
                    <Legend verticalAlign="top" height={24} iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
                    <Line type="monotone" dataKey="rmse" stroke="#3b82f6" strokeWidth={1.5} dot={false} name="RMSE" />
                    <Line type="monotone" dataKey="mae"  stroke="#10b981" strokeWidth={2}   dot={false} name="MAE"  />
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
              />
              <div className="absolute top-12 left-4 bg-slate-950/80 p-2 rounded border border-red-500/30 text-[10px]">
                📍 Max Error: {maxErrorVal.toFixed(4)} {errorUnit} at ({maxErrorPos[0].toFixed(2)}, {maxErrorPos[1].toFixed(2)})
              </div>
            </div>
          </section>
        </div>
      )}

      {activeTab === 'physics' && (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4">
          {metadata?.domain === 'flag_simple' ? (
            <div className="text-center py-12 text-slate-400">
              <p className="text-lg mb-2">Physics analysis not available for cloth simulation</p>
              <p className="text-sm">Vorticity and energy conservation metrics apply to CFD (cylinder flow) only.</p>
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
                  <MeshPlot 
                    crds={metadata.crds} 
                    triangles={metadata.triangles} 
                    values={physicsData.vorticity_target} 
                    title="Ground Truth ω"
                    minVal={physicsData.omega_min}
                    maxVal={physicsData.omega_max}
                    colorScale={d3.scaleSequential(d3.interpolateRdBu).domain([physicsData.omega_max, physicsData.omega_min])}
                  />
                  <MeshPlot 
                    crds={metadata.crds} 
                    triangles={metadata.triangles} 
                    values={physicsData.vorticity_pred} 
                    title="Predicted ω"
                    minVal={physicsData.omega_min}
                    maxVal={physicsData.omega_max}
                    colorScale={d3.scaleSequential(d3.interpolateRdBu).domain([physicsData.omega_max, physicsData.omega_min])}
                  />
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
                        <ReferenceLine
                          y={physicsData.energy_target_series[physicsData.energy_target_series.length - 1]}
                          stroke="#64748b"
                          strokeDasharray="4 2"
                          label={{ value: 'E_target_end', position: 'insideTopRight', fill: '#64748b', fontSize: 9 }}
                        />
                        <ReferenceLine
                          y={physicsData.energy_pred_series[physicsData.energy_pred_series.length - 1]}
                          stroke={physicsData.energy_drift > 0 ? '#f97316' : '#22c55e'}
                          strokeDasharray="4 2"
                          label={{ value: `ΔE=${physicsData.energy_drift?.toFixed(3)}`, position: 'insideBottomRight', fill: physicsData.energy_drift > 0 ? '#f97316' : '#22c55e', fontSize: 9 }}
                        />
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
                        <Line type="monotone" dataKey="pred" stroke="#fb7185" strokeWidth={2} dot={false} />
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
