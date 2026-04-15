import React, { useState, useEffect, useRef } from 'react';
import { Database, Info, AlertTriangle, CheckCircle2, BarChart2, Loader2, Flag, Layers, Clock, Hash, Triangle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { MeshPlot } from '../components/MeshPlot';

const cn = (...classes: any[]) => classes.filter(Boolean).join(' ');

const DOMAIN_OPTIONS = [
  { value: 'cylinder_flow', label: 'Cylinder Flow (CFD)', fieldLabel: 'Velocity', fieldUnit: 'm/s', energyLabel: 'Kinetic Energy (½‖v‖²)', fieldFormula: '‖v‖ = √(vx² + vy²)' },
  { value: 'flag_simple',   label: 'Flag Simple (Cloth)', fieldLabel: 'Position Magnitude', fieldUnit: 'm', energyLabel: 'Position Energy (½‖pos‖²)', fieldFormula: '‖pos‖ = √(x²+y²+z²)' },
];

export const DatasetStudio: React.FC = () => {
  const [domain, setDomain] = useState('cylinder_flow');
  const [domains, setDomains] = useState<Record<string, any>>({});
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [flagging, setFlagging] = useState(false);
  const [flagResult, setFlagResult] = useState<any>(null);
  const [info, setInfo] = useState<any>(null);
  const [preview, setPreview] = useState<any>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const timerRef = useRef<any>(null);

  // Per-domain caches — keyed by domain string, populated on first visit, never evicted
  const dataCache    = useRef<Record<string, any>>({});
  const infoCache    = useRef<Record<string, any>>({});
  const previewCache = useRef<Record<string, any>>({});

  // Load available domains once
  useEffect(() => {
    fetch('/api/status')
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d?.domains) setDomains(d.domains); })
      .catch(() => {});
  }, []);

  // On domain change: restore from cache instantly, only fetch what is missing
  useEffect(() => {
    setError(null);
    setFlagResult(null);

    // Restore cached values immediately (no flash of empty state)
    setData(dataCache.current[domain] ?? null);
    setInfo(infoCache.current[domain] ?? null);
    setPreview(previewCache.current[domain] ?? null);

    const needsSamples = !dataCache.current[domain];
    const needsInfo    = !infoCache.current[domain];
    const needsPreview = !previewCache.current[domain];

    // Start elapsed timer only if we actually need to fetch something
    if (needsSamples) {
      setElapsed(0);
      clearInterval(timerRef.current);
      timerRef.current = setInterval(() => setElapsed(s => s + 1), 1000);
    }

    if (needsSamples) {
      fetch(`/api/dataset/samples?domain=${domain}`)
        .then(r => {
          if (!r.ok) throw new Error(`Server error ${r.status}`);
          return r.json();
        })
        .then(d => {
          clearInterval(timerRef.current);
          dataCache.current[domain] = d;
          setData(d);
        })
        .catch(e => {
          clearInterval(timerRef.current);
          setError(e.message);
        });
    }

    if (needsInfo) {
      fetch(`/api/dataset/info?domain=${domain}&split=train`)
        .then(r => r.ok ? r.json() : null)
        .then(d => { if (d) { infoCache.current[domain] = d; setInfo(d); } })
        .catch(() => {});
    }

    if (needsPreview) {
      setPreviewLoading(true);
      fetch(`/api/dataset/mesh_preview?domain=${domain}&trajectory=0`)
        .then(r => r.ok ? r.json() : null)
        .then(d => {
          if (d) { previewCache.current[domain] = d; setPreview(d); }
          setPreviewLoading(false);
        })
        .catch(() => { setPreviewLoading(false); });
    }

    return () => clearInterval(timerRef.current);
  }, [domain]);

  const domainCfg = DOMAIN_OPTIONS.find(d => d.value === domain) ?? DOMAIN_OPTIONS[0];
  const domainAvailable = domains[domain]?.available !== false;

  const handleFlagOutliers = async () => {
    setFlagging(true);
    setFlagResult(null);
    try {
      const r = await fetch(`/api/dataset/flag_outliers?domain=${domain}&split=test`, { method: 'POST' });
      const d = await r.json();
      setFlagResult(d);
    } catch {
      setFlagResult({ status: 'error', message: 'Request failed' });
    } finally {
      setFlagging(false);
    }
  };

  const totalFieldCount = data?.velocity_bins?.reduce((s: number, b: any) => s + b.count, 0) ?? 0;
  const meanFieldBin = data?.velocity_bins
    ? data.velocity_bins.reduce((sum: number, b: any) => sum + b.bin * b.count, 0) / (totalFieldCount || 1)
    : 0;
  const flaggedCount = data?.outliers?.filter((o: any) => o.flag).length ?? 0;

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      <header className="flex justify-between items-end flex-wrap gap-4">
        <div>
          <h2 className="text-3xl font-bold text-white tracking-tight">Dataset Studio</h2>
          <p className="text-slate-500 mt-1">Statistical analysis and outlier detection for training data.</p>
        </div>

        {/* Domain selector */}
        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-500 font-bold uppercase">Domain</span>
          <div className="flex bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
            {DOMAIN_OPTIONS.map(opt => {
              const available = domains[opt.value]?.available !== false;
              return (
                <button
                  key={opt.value}
                  onClick={() => available && setDomain(opt.value)}
                  disabled={!available}
                  className={cn(
                    "px-4 py-2 text-xs font-bold uppercase tracking-wider transition-all",
                    domain === opt.value
                      ? "bg-blue-600 text-white"
                      : available
                        ? "text-slate-400 hover:text-slate-200"
                        : "text-slate-700 cursor-not-allowed"
                  )}
                  title={!available ? `${opt.label} dataset not available — parse data first` : undefined}
                >
                  {opt.label}
                  {!available && <span className="ml-1 text-[9px] opacity-60">(N/A)</span>}
                </button>
              );
            })}
          </div>

          {data && (
            <div className="flex gap-2 text-[10px] font-bold uppercase text-slate-500">
              <div className="bg-slate-900 border border-slate-800 rounded-lg px-3 py-2">
                {data.outliers.length} Trajectories
              </div>
              <div className="bg-slate-900 border border-slate-800 rounded-lg px-3 py-2">
                Mean = {meanFieldBin.toFixed(3)} {domainCfg.fieldUnit}
              </div>
              <div className={cn("border rounded-lg px-3 py-2", flaggedCount > 0 ? "bg-red-500/10 border-red-500/30 text-red-400" : "bg-green-500/10 border-green-500/30 text-green-400")}>
                {flaggedCount} Flagged
              </div>
            </div>
          )}
        </div>
      </header>

      {/* Not available notice */}
      {!domainAvailable && (
        <div className="flex items-center gap-3 px-4 py-3 bg-amber-900/20 border border-amber-700/30 rounded-xl text-amber-300 text-sm">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          {domain === 'flag_simple'
            ? 'Cloth dataset not yet parsed. Run parse_flag_tfrecord.py to generate data_flag/ index files.'
            : 'Dataset not available for this domain.'}
        </div>
      )}

      {/* Loading */}
      {domainAvailable && !data && !error && (
        <div className="p-8 flex flex-col items-center justify-center gap-4 text-slate-400">
          <Loader2 className="w-8 h-8 animate-spin text-blue-400" />
          <p className="text-sm font-medium">Computing dataset statistics…</p>
          <p className="text-xs text-slate-600">{elapsed}s elapsed — reading field data from disk</p>
          <p className="text-[10px] text-slate-700 italic max-w-xs text-center">
            First load scans all trajectories (~10s). Subsequent visits are instant (cached).
          </p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="p-8 text-center space-y-2">
          <AlertTriangle className="w-8 h-8 text-red-400 mx-auto" />
          <p className="text-red-400 font-bold">Failed to load dataset statistics</p>
          <p className="text-slate-500 text-sm">{error}</p>
        </div>
      )}

      {/* Charts */}
      {data && (
        <>
          {/* Dataset Info Bar */}
          {info && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {[
                { icon: <Hash className="w-3.5 h-3.5 text-blue-400" />, label: 'Trajectories', value: info.num_trajectories.toLocaleString() },
                { icon: <Clock className="w-3.5 h-3.5 text-green-400" />, label: 'Steps / Trajectory', value: info.timesteps_per_trajectory.toLocaleString() },
                { icon: <Layers className="w-3.5 h-3.5 text-purple-400" />, label: 'Total Samples', value: info.total_samples.toLocaleString() },
                { icon: <Clock className="w-3.5 h-3.5 text-amber-400" />, label: 'Time Span', value: `${(info.timesteps_per_trajectory * info.dt).toFixed(1)}s  (dt=${info.dt}s)` },
              ].map(item => (
                <div key={item.label} className="bg-slate-900/50 border border-slate-800 rounded-xl px-4 py-3 flex items-center gap-3">
                  {item.icon}
                  <div>
                    <p className="text-[10px] text-slate-500 font-bold uppercase">{item.label}</p>
                    <p className="text-sm font-bold text-white font-mono">{item.value}</p>
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-white flex items-center gap-2">
                  <BarChart2 className="w-4 h-4 text-blue-400" />
                  {domainCfg.fieldLabel} Distribution
                </h3>
                <span className="text-[10px] text-slate-500 font-bold uppercase">t = 0, sampled trajectories</span>
              </div>
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.velocity_bins} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="bin" stroke="#64748b" fontSize={10} tickFormatter={(v) => v.toFixed(2)} interval={9} />
                    <YAxis stroke="#64748b" fontSize={10} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                      labelFormatter={(v) => `${parseFloat(v).toFixed(3)} ${domainCfg.fieldUnit}`}
                    />
                    <Bar dataKey="count" fill="#3b82f6" radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="text-[10px] text-slate-600 italic">
                {domainCfg.fieldFormula} at t=0. Peak near {meanFieldBin.toFixed(2)} {domainCfg.fieldUnit}.
              </p>
            </section>

            <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-white flex items-center gap-2">
                  <Database className="w-4 h-4 text-green-400" />
                  {domainCfg.energyLabel}
                </h3>
                <span className="text-[10px] text-slate-500 font-bold uppercase">E = ½‖field‖² per node</span>
              </div>
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.energy_bins} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="bin" stroke="#64748b" fontSize={10} tickFormatter={(v) => parseFloat(v).toFixed(4)} interval={9} />
                    <YAxis stroke="#64748b" fontSize={10} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                      labelFormatter={(v) => `${parseFloat(v).toFixed(4)} J`}
                    />
                    <Bar dataKey="count" fill="#10b981" radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="text-[10px] text-slate-600 italic">
                Gaussian-like shape indicates consistent regime across trajectories.
              </p>
            </section>
          </div>

          {/* Node Count Distribution */}
          {data.node_count_bins?.length > 0 && (
            <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-white flex items-center gap-2">
                  <Database className="w-4 h-4 text-purple-400" />
                  Node Count Distribution
                </h3>
                <div className="flex items-center gap-4 text-[10px] text-slate-500 font-bold uppercase">
                  <span>Total nodes: {data.total_nodes?.toLocaleString()}</span>
                  <span>Mean: {data.mean_nodes} nodes/traj</span>
                </div>
              </div>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.node_count_bins} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="bin" stroke="#64748b" fontSize={10} tickFormatter={(v) => v.toLocaleString()} interval={4} label={{ value: 'Nodes per trajectory', position: 'insideBottom', offset: -2, fontSize: 9, fill: '#475569' }} />
                    <YAxis stroke="#64748b" fontSize={10} label={{ value: 'Trajectories', angle: -90, position: 'insideLeft', offset: 10, fontSize: 9, fill: '#475569' }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                      labelFormatter={(v) => `${parseInt(v).toLocaleString()} nodes`}
                    />
                    <Bar dataKey="count" fill="#a855f7" radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </section>
          )}

          {/* Node Type Breakdown + Mesh Preview */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Node Type Breakdown */}
            {data.node_type_counts && Object.keys(data.node_type_counts).length > 0 && (
              <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-4">
                <h3 className="font-semibold text-white flex items-center gap-2 text-sm">
                  <Layers className="w-4 h-4 text-amber-400" />
                  Node Type Breakdown
                </h3>
                <div className="space-y-2">
                  {(() => {
                    const colors: Record<string, string> = {
                      NORMAL: '#3b82f6', OBSTACLE: '#ef4444', AIRFOIL: '#8b5cf6',
                      HANDLE: '#f59e0b', INFLOW: '#10b981', OUTFLOW: '#06b6d4',
                      WALL_BOUNDARY: '#6b7280',
                    };
                    const total = Object.values(data.node_type_counts as Record<string, number>).reduce((a: number, b: number) => a + b, 0);
                    return Object.entries(data.node_type_counts as Record<string, number>).map(([name, count]) => (
                      <div key={name} className="flex items-center gap-3">
                        <div className="w-20 shrink-0 text-[10px] font-bold text-slate-400 uppercase">{name}</div>
                        <div className="flex-1 bg-slate-800 rounded-full h-2 overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all"
                            style={{ width: `${(count / total * 100).toFixed(1)}%`, backgroundColor: colors[name] ?? '#64748b' }}
                          />
                        </div>
                        <div className="w-24 text-right text-[10px] font-mono text-slate-400">
                          {count.toLocaleString()} <span className="text-slate-600">({(count / total * 100).toFixed(1)}%)</span>
                        </div>
                      </div>
                    ));
                  })()}
                </div>
              </section>
            )}

            {/* Mesh Preview */}
            <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-white flex items-center gap-2 text-sm">
                  <Database className="w-4 h-4 text-cyan-400" />
                  Sample Mesh Preview
                </h3>
                {preview && (
                  <span className="text-[10px] text-slate-500 font-bold uppercase">
                    {preview.n_nodes.toLocaleString()} nodes · {preview.n_faces.toLocaleString()} faces
                  </span>
                )}
              </div>
              {previewLoading && (
                <div className="h-48 flex items-center justify-center">
                  <Loader2 className="w-6 h-6 animate-spin text-slate-500" />
                </div>
              )}
              {preview && !previewLoading && (
                <div className="h-48">
                  <MeshPlot
                    crds={preview.positions as [number, number][]}
                    triangles={preview.faces as [number, number, number][]}
                    values={preview.field_values}
                    title=""
                    minVal={Math.min(...preview.field_values)}
                    maxVal={Math.max(...preview.field_values)}
                    domain={domain}
                  />
                </div>
              )}
              {preview && !previewLoading && (
                <p className="text-[10px] text-slate-500 text-center">
                  {domain === 'flag_simple'
                    ? 'Color = Z height (world space at t=0) · X/Y axes = world X/Y'
                    : 'Color = velocity magnitude at t=0'}
                </p>
              )}
              {!preview && !previewLoading && (
                <div className="h-48 flex items-center justify-center text-slate-600 text-sm">
                  No preview available
                </div>
              )}
            </section>
          </div>

          {/* Mesh Quality */}
          {data.mesh_quality && Object.keys(data.mesh_quality).length > 0 && (() => {
            const mq = data.mesh_quality as {
              aspect_ratio_mean: number;
              aspect_ratio_p95: number;
              aspect_ratio_max: number;
              n_degenerate: number;
              n_faces: number;
              quality_ok: boolean;
            };
            const arGood  = mq.aspect_ratio_p95 < 10.0;
            const degGood = mq.n_degenerate === 0;
            const verdictColor = mq.quality_ok ? 'text-green-400' : 'text-amber-400';
            const verdictBg    = mq.quality_ok
              ? 'bg-green-500/10 border-green-500/20'
              : 'bg-amber-500/10 border-amber-500/20';
            return (
              <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-white flex items-center gap-2 text-sm">
                    <Triangle className="w-4 h-4 text-orange-400" />
                    Mesh Quality
                  </h3>
                  <span className={cn('px-2.5 py-1 rounded-full text-[10px] font-bold uppercase border', verdictBg, verdictColor)}>
                    {mq.quality_ok ? '✓ Good' : '⚠ Attention needed'}
                  </span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  {/* Aspect ratio mean */}
                  <div className="bg-slate-800/50 rounded-xl px-4 py-3 space-y-0.5">
                    <p className="text-[10px] text-slate-500 font-bold uppercase">Avg shape ratio</p>
                    <p className="text-lg font-bold font-mono text-white">{mq.aspect_ratio_mean.toFixed(2)}</p>
                    <p className="text-[10px] text-slate-500">longest / shortest edge</p>
                  </div>
                  {/* Aspect ratio p95 */}
                  <div className={cn('rounded-xl px-4 py-3 space-y-0.5', arGood ? 'bg-slate-800/50' : 'bg-amber-900/20 border border-amber-700/30')}>
                    <p className="text-[10px] text-slate-500 font-bold uppercase">95th-pct ratio</p>
                    <p className={cn('text-lg font-bold font-mono', arGood ? 'text-white' : 'text-amber-300')}>{mq.aspect_ratio_p95.toFixed(2)}</p>
                    <p className="text-[10px] text-slate-500">{arGood ? '< 10 — acceptable' : '≥ 10 — very skewed'}</p>
                  </div>
                  {/* Max aspect ratio */}
                  <div className="bg-slate-800/50 rounded-xl px-4 py-3 space-y-0.5">
                    <p className="text-[10px] text-slate-500 font-bold uppercase">Worst ratio</p>
                    <p className="text-lg font-bold font-mono text-white">{mq.aspect_ratio_max.toFixed(1)}</p>
                    <p className="text-[10px] text-slate-500">single worst triangle</p>
                  </div>
                  {/* Degenerate count */}
                  <div className={cn('rounded-xl px-4 py-3 space-y-0.5', degGood ? 'bg-slate-800/50' : 'bg-red-900/20 border border-red-700/30')}>
                    <p className="text-[10px] text-slate-500 font-bold uppercase">Degenerate faces</p>
                    <p className={cn('text-lg font-bold font-mono', degGood ? 'text-green-400' : 'text-red-400')}>{mq.n_degenerate}</p>
                    <p className="text-[10px] text-slate-500">area &lt; 1e-12 m²</p>
                  </div>
                </div>

                <p className="text-[10px] text-slate-600 italic">
                  Sampled from trajectory 0 ({mq.n_faces.toLocaleString()} triangular faces).
                  Aspect ratio = longest ÷ shortest edge per triangle — perfect equilateral = 1.0, ratio ≥ 10 indicates highly skewed elements.
                </p>
              </section>
            );
          })()}

          {/* Outlier table */}
          <section className="bg-slate-900/50 border border-slate-800 rounded-2xl overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/80 flex justify-between items-center">
              <h3 className="font-semibold text-white text-sm uppercase tracking-wider">
                Outlier Detection — Z-Score &gt; 3σ
              </h3>
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2 text-[10px] text-slate-500 font-bold uppercase">
                  <Info className="w-3 h-3" />
                  Based on mean {domainCfg.fieldLabel.toLowerCase()} at t=0 across {data.outliers.length} trajectories
                </div>
                <button
                  onClick={handleFlagOutliers}
                  disabled={flagging || flaggedCount === 0}
                  className="ml-auto flex items-center gap-2 px-3 py-1.5 bg-red-500/10 hover:bg-red-500/20 disabled:opacity-40 text-red-400 border border-red-500/20 rounded-lg text-[10px] font-bold uppercase tracking-wider transition-colors"
                >
                  {flagging ? <Loader2 className="w-3 h-3 animate-spin" /> : <Flag className="w-3 h-3" />}
                  Flag {flaggedCount} Outliers
                </button>
              </div>
            </div>
            {flagResult && (
              <div className={cn(
                "mx-6 mb-2 p-3 rounded-lg text-xs flex items-center gap-2",
                flagResult.status === 'saved'
                  ? "bg-green-500/10 text-green-400 border border-green-500/20"
                  : "bg-red-500/10 text-red-400 border border-red-500/20"
              )}>
                {flagResult.status === 'saved'
                  ? <><CheckCircle2 className="w-3.5 h-3.5 shrink-0" /> Saved {flagResult.n_flagged} outlier flags to <span className="font-mono">{flagResult.path}</span></>
                  : <><AlertTriangle className="w-3.5 h-3.5 shrink-0" /> {flagResult.message}</>
                }
              </div>
            )}
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="bg-slate-950 text-slate-500 uppercase text-[10px] font-bold">
                  <tr>
                    <th className="px-6 py-3">Trajectory ID</th>
                    <th className="px-6 py-3">Mean {domainCfg.fieldLabel} at t=0</th>
                    <th className="px-6 py-3">Z-Score</th>
                    <th className="px-6 py-3">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800">
                  {data.outliers.map((o: any) => (
                    <tr key={o.trajectory} className="hover:bg-slate-800/30 transition-colors">
                      <td className="px-6 py-4 font-mono text-slate-300">traj_{o.trajectory.toString().padStart(4, '0')}</td>
                      <td className="px-6 py-4 text-slate-400">{o.mean_v.toFixed(4)} {domainCfg.fieldUnit}</td>
                      <td className={cn("px-6 py-4 font-bold font-mono", Math.abs(o.z_score) > 3 ? "text-red-400" : "text-green-400")}>
                        {o.z_score > 0 ? '+' : ''}{o.z_score.toFixed(2)}σ
                      </td>
                      <td className="px-6 py-4">
                        {o.flag ? (
                          <span className="flex items-center gap-1.5 text-red-400 text-xs font-bold">
                            <AlertTriangle className="w-3.5 h-3.5" /> OUTLIER — value {o.z_score > 0 ? 'too high' : 'too low'}
                          </span>
                        ) : (
                          <span className="flex items-center gap-1.5 text-green-400 text-xs font-bold">
                            <CheckCircle2 className="w-3.5 h-3.5" /> Normal
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </div>
  );
};
