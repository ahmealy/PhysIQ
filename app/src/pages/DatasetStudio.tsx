import React, { useState, useEffect, useRef } from 'react';
import { Database, Info, AlertTriangle, CheckCircle2, BarChart2, Loader2, Flag } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const cn = (...classes: any[]) => classes.filter(Boolean).join(' ');

export const DatasetStudio: React.FC = () => {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [flagging, setFlagging] = useState(false);
  const [flagResult, setFlagResult] = useState<any>(null);
  const timerRef = useRef<any>(null);
  const fetchedRef = useRef(false);

  useEffect(() => {
    if (fetchedRef.current) return;
    fetchedRef.current = true;

    // Start elapsed timer
    timerRef.current = setInterval(() => setElapsed(s => s + 1), 1000);

    fetch('/api/dataset/samples')
      .then(r => {
        if (!r.ok) throw new Error(`Server error ${r.status}`);
        return r.json();
      })
      .then(d => {
        clearInterval(timerRef.current);
        setData(d);
      })
      .catch(e => {
        clearInterval(timerRef.current);
        setError(e.message);
      });

    return () => clearInterval(timerRef.current);
  }, []);

  if (error) return (
    <div className="p-8 text-center space-y-2">
      <AlertTriangle className="w-8 h-8 text-red-400 mx-auto" />
      <p className="text-red-400 font-bold">Failed to load dataset statistics</p>
      <p className="text-slate-500 text-sm">{error}</p>
    </div>
  );

  if (!data) return (
    <div className="p-8 flex flex-col items-center justify-center gap-4 text-slate-400">
      <Loader2 className="w-8 h-8 animate-spin text-blue-400" />
      <p className="text-sm font-medium">Computing dataset statistics…</p>
      <p className="text-xs text-slate-600">{elapsed}s elapsed — reading velocity data from disk</p>
      <p className="text-[10px] text-slate-700 italic max-w-xs text-center">
        First load scans all trajectories (~10s). Subsequent visits are instant (cached).
      </p>
    </div>
  );

  // Compute stats from data
  const totalVelCount = data.velocity_bins.reduce((s: number, b: any) => s + b.count, 0);
  const meanVelBin = data.velocity_bins.reduce((sum: number, b: any) => sum + b.bin * b.count, 0) / (totalVelCount || 1);
  const flaggedCount = data.outliers.filter((o: any) => o.flag).length;

  const handleFlagOutliers = async () => {
    setFlagging(true);
    setFlagResult(null);
    try {
      const r = await fetch('/api/dataset/flag_outliers?domain=cylinder_flow&split=test', { method: 'POST' });
      const d = await r.json();
      setFlagResult(d);
    } catch {
      setFlagResult({ status: 'error', message: 'Request failed' });
    } finally {
      setFlagging(false);
    }
  };

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      <header className="flex justify-between items-end">
        <div>
          <h2 className="text-3xl font-bold text-white tracking-tight">Dataset Studio</h2>
          <p className="text-slate-500 mt-1">Statistical analysis and outlier detection for training data.</p>
        </div>
        <div className="flex gap-3 text-[10px] font-bold uppercase text-slate-500">
          <div className="bg-slate-900 border border-slate-800 rounded-lg px-3 py-2">
            {data.outliers.length} Trajectories
          </div>
          <div className="bg-slate-900 border border-slate-800 rounded-lg px-3 py-2">
            Mean v̄ = {meanVelBin.toFixed(3)} m/s
          </div>
          <div className={cn("border rounded-lg px-3 py-2", flaggedCount > 0 ? "bg-red-500/10 border-red-500/30 text-red-400" : "bg-green-500/10 border-green-500/30 text-green-400")}>
            {flaggedCount} Flagged
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-white flex items-center gap-2">
              <BarChart2 className="w-4 h-4 text-blue-400" />
              Velocity Magnitude Distribution
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
                  labelFormatter={(v) => `${parseFloat(v).toFixed(3)} m/s`}
                />
                <Bar dataKey="count" fill="#3b82f6" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[10px] text-slate-600 italic">
            ‖v‖ = √(vx² + vy²) at t=0. Peak near {meanVelBin.toFixed(2)} m/s (mean of sampled trajectories).
          </p>
        </section>

        <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-white flex items-center gap-2">
              <Database className="w-4 h-4 text-green-400" />
              Kinetic Energy Distribution
            </h3>
            <span className="text-[10px] text-slate-500 font-bold uppercase">E = ½‖v‖² per node</span>
          </div>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.energy_bins} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                <XAxis dataKey="bin" stroke="#64748b" fontSize={10} tickFormatter={(v) => v.toFixed(3)} interval={9} />
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
            Kinetic energy proxy (unit density). Gaussian-like shape indicates consistent flow regime across trajectories.
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
                <XAxis dataKey="bin" stroke="#64748b" fontSize={10} tickFormatter={(v) => v.toLocaleString()} interval={4} />
                <YAxis stroke="#64748b" fontSize={10} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                  labelFormatter={(v) => `${parseInt(v).toLocaleString()} nodes`}
                />
                <Bar dataKey="count" fill="#a855f7" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[10px] text-slate-600 italic">
            Number of mesh nodes per trajectory. Variation indicates different mesh resolutions or geometries.
          </p>
        </section>
      )}

      <section className="bg-slate-900/50 border border-slate-800 rounded-2xl overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/80 flex justify-between items-center">
          <h3 className="font-semibold text-white text-sm uppercase tracking-wider">
            Outlier Detection — Z-Score &gt; 3σ
          </h3>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 text-[10px] text-slate-500 font-bold uppercase">
              <Info className="w-3 h-3" />
              Based on mean velocity at t=0 across all {data.outliers.length} trajectories
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
            flagResult.status === 'saved' ? "bg-green-500/10 text-green-400 border border-green-500/20" : "bg-red-500/10 text-red-400 border border-red-500/20"
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
                <th className="px-6 py-3">Mean ‖v‖ at t=0</th>
                <th className="px-6 py-3">Z-Score</th>
                <th className="px-6 py-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {data.outliers.map((o: any) => (
                <tr key={o.trajectory} className="hover:bg-slate-800/30 transition-colors">
                  <td className="px-6 py-4 font-mono text-slate-300">traj_{o.trajectory.toString().padStart(4, '0')}</td>
                  <td className="px-6 py-4 text-slate-400">{o.mean_v.toFixed(4)} m/s</td>
                  <td className={cn("px-6 py-4 font-bold font-mono", Math.abs(o.z_score) > 3 ? "text-red-400" : "text-green-400")}>
                    {o.z_score > 0 ? '+' : ''}{o.z_score.toFixed(2)}σ
                  </td>
                  <td className="px-6 py-4">
                    {o.flag ? (
                      <span className="flex items-center gap-1.5 text-red-400 text-xs font-bold">
                        <AlertTriangle className="w-3.5 h-3.5" /> OUTLIER — mean velocity {o.z_score > 0 ? 'too high' : 'too low'}
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
    </div>
  );
};
