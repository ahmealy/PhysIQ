import React, { useState, useEffect } from 'react';
import { CheckCircle2, Circle, RefreshCw, FileText, Settings, Search, Trash2, Edit3, AlertTriangle, Info, Save, History } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { fetchWithRetry } from '../utils/fetch';

const cn = (...classes: any[]) => classes.filter(Boolean).join(' ');

const LINE_COLORS = ['#3b82f6', '#10b981', '#fb7185', '#f59e0b', '#a855f7'];

// DAG layout
const NODE_POSITIONS: Record<string, { x: number; y: number }> = {
  dataset:     { x: 60,  y: 120 },
  preprocess:  { x: 200, y: 120 },
  graph_build: { x: 340, y: 120 },
  train:       { x: 480, y: 120 },
  evaluate:    { x: 620, y: 120 },
  predict:     { x: 760, y: 120 },
  export:      { x: 900, y: 120 },
};

const EDGES = [
  ['dataset', 'preprocess'],
  ['preprocess', 'graph_build'],
  ['graph_build', 'train'],
  ['train', 'evaluate'],
  ['evaluate', 'predict'],
  ['predict', 'export'],
];

// Persist custom experiment names in localStorage
const NAMES_KEY = 'mgn_experiment_names';
function loadNames(): Record<string, string> {
  try { return JSON.parse(localStorage.getItem(NAMES_KEY) || '{}'); } catch { return {}; }
}
function saveName(id: string, name: string) {
  const names = loadNames();
  names[id] = name;
  localStorage.setItem(NAMES_KEY, JSON.stringify(names));
}

export const PipelineView: React.FC = () => {
  // ── Pipeline state ──────────────────────────────────────────────────────────
  const [status, setStatus] = useState<any>(null);
  const [gpuStatus, setGpuStatus] = useState<any>(null);
  const [pipelineNodes, setPipelineNodes] = useState<any[]>([]);
  const [events, setEvents] = useState<{ type: string; message: string; time: string }[]>([]);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  // ── Experiment state ────────────────────────────────────────────────────────
  const [experiments, setExperiments] = useState<any[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [search, setSearch] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [comparisonData, setComparisonData] = useState<any[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingValue, setEditingValue] = useState('');
  const [checkpoint, setCheckpoint] = useState<any>(null);

  // ── Pipeline data — poll every 15s ─────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    const load = async (isInitial = false) => {
      try {
        const r = await fetchWithRetry('/api/status', isInitial ? 6 : 1, 1200);
        if (!cancelled && r.ok) setStatus(await r.json());
      } catch { /* ignore */ }
      try {
        const r = await fetch('/api/status/gpu');
        if (!cancelled && r.ok) setGpuStatus(await r.json());
      } catch { /* ignore */ }
      try {
        const r = await fetch('/api/pipeline');
        if (!cancelled && r.ok) {
          const d = await r.json();
          setPipelineNodes(d.nodes ?? []);
        }
      } catch { /* ignore */ }
      try {
        const r = await fetch('/api/events');
        if (!cancelled && r.ok) {
          const d = await r.json();
          setEvents(Array.isArray(d) ? d : []);
        }
      } catch { /* ignore */ }
    };
    load(true);
    const t = setInterval(() => load(false), 15000);
    return () => { cancelled = true; clearInterval(t); };
  }, []);

  // ── Experiment data — load once on mount ───────────────────────────────────
  const loadExperiments = async () => {
    setIsLoading(true);
    const savedNames = loadNames();
    try {
      const results: any[] = await fetch('/api/results').then(r => r.json());
      const withRmse = await Promise.all(
        results.map(async (res) => {
          try {
            const rmse = await fetch(`/api/results/${res.filename}/rmse`).then(r => r.json());
            return {
              id: res.filename,
              name: savedNames[res.filename] ?? res.filename.replace('.pkl', ''),
              filename: res.filename,
              trajectory_index: res.trajectory_index,
              size_mb: res.size_mb,
              created: res.created,
              rmse_at_0: rmse.rmse_at_0,
              rmse_at_599: rmse.rmse_at_599,
              mae_at_0: rmse.mae_at_0,
              mae_at_end: rmse.mae_at_end,
              per_step_rmse: rmse.per_step_rmse,
              growth_ratio: rmse.growth_ratio,
              status: 'completed',
            };
          } catch {
            return {
              id: res.filename,
              name: savedNames[res.filename] ?? res.filename.replace('.pkl', ''),
              filename: res.filename,
              trajectory_index: res.trajectory_index,
              size_mb: res.size_mb,
              created: res.created,
              rmse_at_0: null, rmse_at_599: null, mae_at_0: null, mae_at_end: null,
              per_step_rmse: [], growth_ratio: null, status: 'failed',
            };
          }
        })
      );
      setExperiments(withRmse);
    } catch {
      setExperiments([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadExperiments();
    fetch('/api/checkpoint').then(r => r.ok ? r.json() : null).then(setCheckpoint).catch(() => {});
  }, []);

  // Build comparison chart data
  useEffect(() => {
    const selected = experiments.filter(e => selectedIds.includes(e.id) && e.per_step_rmse?.length > 0);
    if (selected.length === 0) { setComparisonData([]); return; }
    const maxLen = Math.max(...selected.map(e => e.per_step_rmse.length));
    const data = Array.from({ length: maxLen }, (_, i) => {
      const point: any = { step: i };
      for (const exp of selected) {
        if (i < exp.per_step_rmse.length) point[exp.id] = exp.per_step_rmse[i];
      }
      return point;
    });
    setComparisonData(data);
  }, [selectedIds, experiments]);

  const filtered = experiments.filter(e =>
    e.name.toLowerCase().includes(search.toLowerCase()) ||
    e.filename.toLowerCase().includes(search.toLowerCase())
  );
  const toggleSelect = (id: string) =>
    setSelectedIds(prev => prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]);
  const startEdit = (e: any) => { setEditingId(e.id); setEditingValue(e.name); };
  const commitEdit = (id: string) => {
    const trimmed = editingValue.trim();
    if (trimmed) {
      saveName(id, trimmed);
      setExperiments(prev => prev.map(e => e.id === id ? { ...e, name: trimmed } : e));
    }
    setEditingId(null);
  };
  const selectedExps = experiments.filter(e => selectedIds.includes(e.id));
  const nodeMap = Object.fromEntries(pipelineNodes.map((n: any) => [n.id, n]));
  const selectedPipelineNode = selectedNode ? nodeMap[selectedNode] : null;

  return (
    <div className="p-8 space-y-10 max-w-7xl mx-auto">

      {/* ── Section 1: Pipeline Status ─────────────────────────────────────── */}
      <div className="space-y-6">
        <header>
          <h2 className="text-3xl font-bold text-white tracking-tight">Pipeline & Experiments</h2>
          <p className="text-slate-500 mt-1">End-to-end workflow status and rollout experiment tracking.</p>
        </header>

        {/* DAG */}
        <div className="bg-slate-900/30 border border-slate-800 rounded-3xl p-6">
          {!status ? (
            <div className="flex items-center gap-3 text-slate-400 py-8 justify-center">
              <RefreshCw className="w-5 h-5 animate-spin text-blue-400" />
              <span>Loading pipeline status…</span>
            </div>
          ) : (
            <>
              <svg viewBox="0 0 960 240" className="w-full" style={{ height: 200 }}>
                {EDGES.map(([from, to]) => {
                  const a = NODE_POSITIONS[from];
                  const b = NODE_POSITIONS[to];
                  const fromNode = nodeMap[from];
                  const done = fromNode?.done;
                  return (
                    <line
                      key={`${from}-${to}`}
                      x1={a.x + 28} y1={a.y}
                      x2={b.x - 28} y2={b.y}
                      stroke={done ? '#22c55e40' : '#334155'}
                      strokeWidth={2}
                      strokeDasharray={done ? undefined : '6 3'}
                    />
                  );
                })}
                {pipelineNodes.map((node: any) => {
                  const pos = NODE_POSITIONS[node.id];
                  if (!pos) return null;
                  const isSelected = selectedNode === node.id;
                  const color = node.done ? '#22c55e' : '#475569';
                  const bgColor = node.done ? '#052e16' : '#0f172a';
                  const borderColor = isSelected ? '#3b82f6' : (node.done ? '#22c55e80' : '#334155');
                  return (
                    <g key={node.id} style={{ cursor: 'pointer' }} onClick={() => setSelectedNode(prev => prev === node.id ? null : node.id)}>
                      <rect x={pos.x - 28} y={pos.y - 28} width={56} height={56} rx={12}
                        fill={bgColor} stroke={borderColor} strokeWidth={isSelected ? 2 : 1.5} />
                      {node.done && <circle cx={pos.x} cy={pos.y} r={14} fill={color} opacity={0.15} />}
                      <text x={pos.x} y={pos.y + 5} textAnchor="middle" fill={color} fontSize={20}>
                        {node.done ? '✓' : '○'}
                      </text>
                      <text x={pos.x} y={pos.y + 46} textAnchor="middle" fill={node.done ? '#e2e8f0' : '#64748b'} fontSize={9} fontWeight="600">
                        {node.label}
                      </text>
                    </g>
                  );
                })}
              </svg>
              <div className="flex items-center gap-6 mt-2 px-2">
                <div className="flex items-center gap-2 text-[10px] text-slate-500">
                  <div className="w-3 h-3 rounded bg-green-900 border border-green-500/50" />
                  <span>Complete</span>
                </div>
                <div className="flex items-center gap-2 text-[10px] text-slate-500">
                  <div className="w-3 h-3 rounded bg-slate-900 border border-slate-700" />
                  <span>Pending</span>
                </div>
                <div className="flex items-center gap-2 text-[10px] text-slate-400 ml-auto">Click a node to inspect</div>
              </div>
            </>
          )}
        </div>

        {/* Node detail panel */}
        {selectedPipelineNode && (
          <section className="bg-slate-900/50 border border-blue-500/30 rounded-2xl overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/80 flex items-center gap-3">
              <div className={cn("w-2 h-2 rounded-full", selectedPipelineNode.done ? "bg-green-500" : "bg-slate-600")} />
              <h3 className="font-semibold text-white">{selectedPipelineNode.label}</h3>
              <span className={cn("px-2 py-0.5 rounded text-[9px] font-bold uppercase",
                selectedPipelineNode.done ? "bg-green-500/20 text-green-400" : "bg-slate-800 text-slate-500")}>
                {selectedPipelineNode.done ? 'Complete' : 'Pending'}
              </span>
            </div>
            <div className="p-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-3">
                <h4 className="text-[10px] uppercase text-slate-500 font-bold tracking-wider flex items-center gap-2">
                  <FileText className="w-3 h-3" /> Artifacts
                </h4>
                {selectedPipelineNode.files?.length > 0 ? (
                  <div className="space-y-2">
                    {selectedPipelineNode.files.map((f: any) => (
                      <div key={f.name} className="flex items-center justify-between py-2 border-b border-slate-800">
                        <span className="text-xs text-slate-300 font-mono">{f.name}</span>
                        <div className="text-right">
                          <span className="text-[10px] text-slate-500">{f.size_mb} MB</span>
                          <p className="text-[9px] text-slate-600">{new Date(f.modified).toLocaleString()}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : <p className="text-xs text-slate-600">No artifacts found yet.</p>}
              </div>
              {selectedPipelineNode.config && Object.keys(selectedPipelineNode.config).length > 0 && (
                <div className="space-y-3">
                  <h4 className="text-[10px] uppercase text-slate-500 font-bold tracking-wider flex items-center gap-2">
                    <Settings className="w-3 h-3" /> Config / Metadata
                  </h4>
                  <div className="bg-slate-950 rounded-xl p-4 space-y-2 overflow-auto max-h-48">
                    {Object.entries(selectedPipelineNode.config).map(([k, v]) => (
                      <div key={k} className="flex justify-between text-xs">
                        <span className="text-slate-500 font-mono">{k}</span>
                        <span className="text-slate-300 font-mono ml-4 truncate max-w-[200px]">{String(v)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </section>
        )}

        {/* System Health + Events */}
        {status && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <section className="lg:col-span-2 bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-6">
              <h3 className="font-semibold text-white text-sm uppercase tracking-wider">System Health</h3>
              <div className="grid grid-cols-2 gap-4">
                <HealthCard label="GPU" value={status.gpu_name ?? 'CPU Only'} status={status.gpu_available ? 'healthy' : 'warning'} />
                <HealthCard
                  label="GPU Memory"
                  value={gpuStatus?.mem_alloc_gb != null ? `${gpuStatus.mem_alloc_gb} / ${gpuStatus.mem_reserved_gb} GB` : status.gpu_available ? 'Loading…' : 'N/A'}
                  status="healthy"
                />
                <HealthCard
                  label="GPU Utilization"
                  value={gpuStatus?.utilization != null ? `${gpuStatus.utilization}%` : status.gpu_available ? 'Loading…' : 'N/A'}
                  status="healthy"
                />
                <HealthCard label="Saved Rollouts" value={`${status.saved_rollouts ?? 0}`} status="healthy" />
              </div>
            </section>
            <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-6">
              <h3 className="font-semibold text-white text-sm uppercase tracking-wider">System Events</h3>
              <div className="space-y-4">
                {events.length > 0 ? (
                  events.slice(0, 5).map((ev, i) => {
                    const evType: 'info' | 'warning' | 'error' =
                      ev.type === 'warning' || ev.type === 'error' ? ev.type : 'info';
                    return <React.Fragment key={i}><AlertItem type={evType} message={String(ev.message)} time={String(ev.time)} /></React.Fragment>;
                  })
                ) : <p className="text-xs text-slate-500">No events recorded yet.</p>}
              </div>
            </section>
          </div>
        )}
      </div>

      {/* ── Divider ──────────────────────────────────────────────────────────── */}
      <div className="border-t border-slate-800" />

      {/* ── Section 2: Experiment Tracking ───────────────────────────────────── */}
      <div className="space-y-6">
        <header className="flex justify-between items-end">
          <div>
            <h3 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
              <History className="w-5 h-5 text-slate-400" />
              Experiment Tracking
            </h3>
            <p className="text-slate-500 mt-1">Compare rollout performance across trajectories.</p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={loadExperiments}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg text-sm font-bold flex items-center gap-2 transition-all"
            >
              <RefreshCw className="w-4 h-4" /> Refresh
            </button>
            <button
              disabled={selectedIds.length < 2}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-500 text-white rounded-lg text-sm font-bold flex items-center gap-2 transition-all shadow-lg shadow-blue-900/20"
            >
              Compare Selected ({selectedIds.length})
            </button>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <section className="lg:col-span-2 bg-slate-900/50 border border-slate-800 rounded-2xl overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/80 flex justify-between items-center">
              <div className="relative flex-1 max-w-md">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                <input
                  type="text"
                  placeholder="Search rollouts..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-10 pr-4 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500/50 transition-colors"
                />
              </div>
            </div>
            <div className="overflow-x-auto">
              {isLoading ? (
                <div className="p-12 text-center text-slate-500">Loading rollouts...</div>
              ) : filtered.length === 0 ? (
                <div className="p-12 text-center text-slate-500">
                  <History className="w-8 h-8 text-slate-700 mx-auto mb-2" />
                  <p>No saved rollouts found. Run a prediction first.</p>
                </div>
              ) : (
                <table className="w-full text-sm text-left">
                  <thead className="bg-slate-950 text-slate-500 uppercase text-[10px] font-bold">
                    <tr>
                      <th className="px-6 py-3 w-10">
                        <input type="checkbox" className="rounded border-slate-800 bg-slate-950 text-blue-600"
                          checked={selectedIds.length === filtered.length && filtered.length > 0}
                          onChange={() => selectedIds.length === filtered.length ? setSelectedIds([]) : setSelectedIds(filtered.map(e => e.id))}
                        />
                      </th>
                      <th className="px-6 py-3">Rollout Name</th>
                      <th className="px-6 py-3">Traj</th>
                      <th className="px-6 py-3">Size</th>
                      <th className="px-6 py-3">RMSE@0</th>
                      <th className="px-6 py-3">RMSE@end</th>
                      <th className="px-6 py-3">MAE@end</th>
                      <th className="px-6 py-3">Status</th>
                      <th className="px-6 py-3 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    {filtered.map((e) => (
                      <tr key={e.id} className={cn("hover:bg-slate-800/30 transition-colors", selectedIds.includes(e.id) ? "bg-blue-600/5" : "")}>
                        <td className="px-6 py-4">
                          <input type="checkbox" checked={selectedIds.includes(e.id)} onChange={() => toggleSelect(e.id)}
                            className="rounded border-slate-800 bg-slate-950 text-blue-600" />
                        </td>
                        <td className="px-6 py-4">
                          {editingId === e.id ? (
                            <div className="flex items-center gap-1">
                              <input autoFocus value={editingValue}
                                onChange={ev => setEditingValue(ev.target.value)}
                                onBlur={() => commitEdit(e.id)}
                                onKeyDown={ev => { if (ev.key === 'Enter') commitEdit(e.id); if (ev.key === 'Escape') setEditingId(null); }}
                                className="bg-slate-800 border border-blue-500/50 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none w-32"
                              />
                              <button onClick={() => commitEdit(e.id)} className="text-green-400 hover:text-green-300">
                                <Save className="w-3 h-3" />
                              </button>
                            </div>
                          ) : (
                            <div className="flex items-center gap-2 group">
                              <span className="font-bold text-slate-200">{e.name}</span>
                              <button onClick={() => startEdit(e)} className="opacity-0 group-hover:opacity-100 text-slate-500 hover:text-slate-300 transition-opacity">
                                <Edit3 className="w-3 h-3" />
                              </button>
                            </div>
                          )}
                          <p className="text-[10px] text-slate-500 mt-0.5">{new Date(e.created).toLocaleDateString()} • {e.size_mb} MB</p>
                        </td>
                        <td className="px-6 py-4 text-slate-400 font-mono text-xs">{e.trajectory_index}</td>
                        <td className="px-6 py-4 text-slate-400 font-mono text-xs">{e.size_mb} MB</td>
                        <td className="px-6 py-4 font-bold text-blue-400 font-mono text-xs">{e.rmse_at_0 != null ? e.rmse_at_0.toFixed(4) : '—'}</td>
                        <td className="px-6 py-4 font-bold text-orange-400 font-mono text-xs">{e.rmse_at_599 != null ? e.rmse_at_599.toFixed(4) : '—'}</td>
                        <td className="px-6 py-4 font-bold text-purple-400 font-mono text-xs">{e.mae_at_end != null ? e.mae_at_end.toFixed(4) : '—'}</td>
                        <td className="px-6 py-4">
                          {e.status === 'completed' ? (
                            <span className="flex items-center gap-1.5 text-green-400 text-[10px] font-bold uppercase">
                              <CheckCircle2 className="w-3 h-3" /> OK
                            </span>
                          ) : (
                            <span className="flex items-center gap-1.5 text-red-400 text-[10px] font-bold uppercase">
                              <AlertTriangle className="w-3 h-3" /> Error
                            </span>
                          )}
                        </td>
                        <td className="px-6 py-4 text-right">
                          <button
                            onClick={async () => {
                              if (!confirm(`Delete ${e.filename}?`)) return;
                              await fetch(`/api/results/${e.filename}`, { method: 'DELETE' });
                              loadExperiments();
                            }}
                            className="p-2 text-slate-600 hover:text-red-400 transition-colors"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </section>

          <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-6">
            <div className="flex justify-between items-center">
              <h3 className="font-semibold text-white text-sm uppercase tracking-wider">RMSE Comparison</h3>
              <div className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">{selectedIds.length} selected</div>
            </div>
            {selectedIds.length === 0 ? (
              <div className="h-64 flex items-center justify-center text-slate-600 text-sm text-center">
                Select rollouts from the table to compare their RMSE curves
              </div>
            ) : (
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={comparisonData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="step" stroke="#64748b" fontSize={10} />
                    <YAxis stroke="#64748b" fontSize={10} />
                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }} />
                    <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
                    {selectedExps.map((exp, idx) => (
                      <Line key={exp.id} type="monotone" dataKey={exp.id} name={exp.name}
                        stroke={LINE_COLORS[idx % LINE_COLORS.length]} strokeWidth={2} dot={false} isAnimationActive={false} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
            {selectedExps.length > 0 && (
              <div className="p-4 bg-blue-600/10 rounded-xl border border-blue-500/20 flex gap-3">
                <Info className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" />
                <p className="text-[10px] text-blue-300 leading-relaxed">
                  {selectedExps.length === 1
                    ? `${selectedExps[0].name}: RMSE grows ${selectedExps[0].growth_ratio?.toFixed(1) ?? '—'}× over the rollout.`
                    : `Comparing ${selectedExps.length} rollouts. Lower final RMSE indicates better long-horizon stability.`}
                </p>
              </div>
            )}
          </section>
        </div>

        {checkpoint && (
          <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-4">
            <h3 className="font-semibold text-white text-sm uppercase tracking-wider">Model Hyperparameters</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
              <HparamCard label="Epoch"      value={checkpoint.epoch ?? '—'} />
              <HparamCard label="Valid Loss" value={checkpoint.valid_loss != null ? checkpoint.valid_loss.toExponential(3) : '—'} />
              <HparamCard label="Params"     value={checkpoint.param_count_m != null ? `${checkpoint.param_count_m}M` : '—'} />
              <HparamCard label="Model Size" value={checkpoint.size_mb != null ? `${checkpoint.size_mb} MB` : '—'} />
              <HparamCard label="Trained"    value={checkpoint.last_modified ? new Date(checkpoint.last_modified).toLocaleDateString() : '—'} />
              <HparamCard label="Checkpoint" value={checkpoint.path ? checkpoint.path.split('/').pop() : '—'} />
            </div>
          </section>
        )}
      </div>
    </div>
  );
};

const HealthCard = ({ label, value, status }: any) => (
  <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 flex justify-between items-center">
    <div>
      <p className="text-[10px] text-slate-500 font-bold uppercase">{label}</p>
      <p className="text-sm font-bold text-slate-200 mt-1">{value}</p>
    </div>
    <div className={cn("w-2 h-2 rounded-full shadow-lg", status === 'healthy' ? "bg-green-500 shadow-green-900/50" : "bg-yellow-500 shadow-yellow-900/50")} />
  </div>
);

const AlertItem = ({ type, message, time }: { type: 'info' | 'warning' | 'error'; message: string; time: string }) => (
  <div className="flex gap-3 items-start">
    <div className={cn("p-1.5 rounded-lg mt-0.5",
      type === 'info' ? "bg-blue-500/10 text-blue-400" :
      type === 'warning' ? "bg-yellow-500/10 text-yellow-400" : "bg-red-500/10 text-red-400")}>
      <Circle className="w-3.5 h-3.5" />
    </div>
    <div>
      <p className="text-xs text-slate-300 leading-tight">{message}</p>
      <p className="text-[10px] text-slate-500 mt-1">{time}</p>
    </div>
  </div>
);

const HparamCard = ({ label, value }: { label: string; value: string }) => (
  <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 space-y-1">
    <p className="text-[9px] text-slate-500 font-bold uppercase tracking-wider">{label}</p>
    <p className="text-xs font-bold text-slate-200 font-mono truncate">{value}</p>
  </div>
);
