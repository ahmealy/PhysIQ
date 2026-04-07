import React, { useState, useEffect } from 'react';
import { CheckCircle2, Circle, ChevronDown, ChevronRight, RefreshCw, FileText, Settings } from 'lucide-react';
import { fetchWithRetry } from '../utils/fetch';

const cn = (...classes: any[]) => classes.filter(Boolean).join(' ');

// DAG layout — fixed positions as % of SVG viewport
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

export const PipelineView: React.FC = () => {
  const [status, setStatus] = useState<any>(null);
  const [gpuStatus, setGpuStatus] = useState<any>(null);
  const [pipelineNodes, setPipelineNodes] = useState<any[]>([]);
  const [events, setEvents] = useState<{ type: string; message: string; time: string }[]>([]);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [expandedNode, setExpandedNode] = useState<string | null>(null);

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

  if (!status) return (
    <div className="p-8 flex items-center gap-3 text-slate-400">
      <RefreshCw className="w-5 h-5 animate-spin text-blue-400" />
      <span>Loading pipeline status…</span>
    </div>
  );

  const nodeMap = Object.fromEntries(pipelineNodes.map((n: any) => [n.id, n]));
  const selected = selectedNode ? nodeMap[selectedNode] : null;

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      <header className="flex justify-between items-end">
        <div>
          <h2 className="text-3xl font-bold text-white tracking-tight">Pipeline View</h2>
          <p className="text-slate-500 mt-1">End-to-end simulation workflow status and monitoring.</p>
        </div>
      </header>

      {/* DAG */}
      <div className="bg-slate-900/30 border border-slate-800 rounded-3xl p-6">
        <svg viewBox="0 0 960 240" className="w-full" style={{ height: 200 }}>
          {/* Edges */}
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

          {/* Nodes */}
          {pipelineNodes.map((node: any) => {
            const pos = NODE_POSITIONS[node.id];
            if (!pos) return null;
            const isSelected = selectedNode === node.id;
            const color = node.done ? '#22c55e' : '#475569';
            const bgColor = node.done ? '#052e16' : '#0f172a';
            const borderColor = isSelected ? '#3b82f6' : (node.done ? '#22c55e80' : '#334155');

            return (
              <g key={node.id} style={{ cursor: 'pointer' }} onClick={() => setSelectedNode(prev => prev === node.id ? null : node.id)}>
                <rect
                  x={pos.x - 28} y={pos.y - 28}
                  width={56} height={56}
                  rx={12}
                  fill={bgColor}
                  stroke={borderColor}
                  strokeWidth={isSelected ? 2 : 1.5}
                />
                {node.done ? (
                  <circle cx={pos.x} cy={pos.y} r={14} fill={color} opacity={0.15} />
                ) : null}
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

        {/* Legend */}
        <div className="flex items-center gap-6 mt-2 px-2">
          <div className="flex items-center gap-2 text-[10px] text-slate-500">
            <div className="w-3 h-3 rounded bg-green-900 border border-green-500/50" />
            <span>Complete</span>
          </div>
          <div className="flex items-center gap-2 text-[10px] text-slate-500">
            <div className="w-3 h-3 rounded bg-slate-900 border border-slate-700" />
            <span>Pending</span>
          </div>
          <div className="flex items-center gap-2 text-[10px] text-slate-400 ml-auto">
            Click a node to inspect
          </div>
        </div>
      </div>

      {/* Node detail panel */}
      {selected && (
        <section className="bg-slate-900/50 border border-blue-500/30 rounded-2xl overflow-hidden animate-in fade-in slide-in-from-top-2">
          <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/80 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={cn("w-2 h-2 rounded-full", selected.done ? "bg-green-500" : "bg-slate-600")} />
              <h3 className="font-semibold text-white">{selected.label}</h3>
              <span className={cn("px-2 py-0.5 rounded text-[9px] font-bold uppercase", selected.done ? "bg-green-500/20 text-green-400" : "bg-slate-800 text-slate-500")}>
                {selected.done ? 'Complete' : 'Pending'}
              </span>
            </div>
          </div>
          <div className="p-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Files */}
            <div className="space-y-3">
              <h4 className="text-[10px] uppercase text-slate-500 font-bold tracking-wider flex items-center gap-2">
                <FileText className="w-3 h-3" /> Artifacts
              </h4>
              {selected.files?.length > 0 ? (
                <div className="space-y-2">
                  {selected.files.map((f: any) => (
                    <div key={f.name} className="flex items-center justify-between py-2 border-b border-slate-800">
                      <span className="text-xs text-slate-300 font-mono">{f.name}</span>
                      <div className="text-right">
                        <span className="text-[10px] text-slate-500">{f.size_mb} MB</span>
                        <p className="text-[9px] text-slate-600">{new Date(f.modified).toLocaleString()}</p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-slate-600">No artifacts found yet.</p>
              )}
            </div>

            {/* Config */}
            {selected.config && Object.keys(selected.config).length > 0 && (
              <div className="space-y-3">
                <h4 className="text-[10px] uppercase text-slate-500 font-bold tracking-wider flex items-center gap-2">
                  <Settings className="w-3 h-3" /> Config / Metadata
                </h4>
                <div className="bg-slate-950 rounded-xl p-4 space-y-2 overflow-auto max-h-48">
                  {Object.entries(selected.config).map(([k, v]) => (
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
            ) : (
              <p className="text-xs text-slate-500">No events recorded yet.</p>
            )}
          </div>
        </section>
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
