import React, { useState, useCallback, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Sparkles, Square, AlertCircle, Info, Server } from 'lucide-react';
import { CandidateCard } from '../components/CandidateCard';
import { OptimizationChart } from '../components/OptimizationChart';
import { PipelineSteps } from '../components/PipelineSteps';
import { cn } from '@/src/lib/utils';

// ── Types ────────────────────────────────────────────────────────────────────

interface Candidate {
  id:                   number;
  domain:               string;
  predicted_value:      number;
  target_value:         number;
  ood_confidence:       number;
  is_ood:               boolean;
  mesh_nodes:           number;
  params:               Record<string, number>;
  thumbnail_url?:       string | null;
  session_id?:          string;
}

interface GenerateConfig {
  domain:       string;
  target_value: number;
  n_candidates: number;
  method:       string;
  device:       string;
}

// ── Persistence helpers ───────────────────────────────────────────────────────

const LS_CONFIG     = 'generate_config';
const LS_CANDIDATES = 'generate_candidates';
const LS_BEST_ID    = 'generate_best_id';
const LS_TRAJECTORY = 'generate_trajectory';

function loadLS<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch { return fallback; }
}

function saveLS(key: string, value: unknown) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* quota */ }
}

// ── Domain metadata ──────────────────────────────────────────────────────────

const DOMAIN_CONFIGS = {
  cylinder_flow: {
    label:       'Cylinder Flow (CFD)',
    targetLabel: 'Target Drag Proxy',
    targetMin:   0.001,
    targetMax:   0.15,
    targetStep:  0.001,
    defaultTarget: 0.025,
    description: 'Generate cylinder obstacle geometries with specified aerodynamic drag.',
  },
  flag_simple: {
    label:       'Flag / Cloth',
    targetLabel: 'Target Stress Proxy',
    targetMin:   0.2,
    targetMax:   2.6,
    targetStep:  0.05,
    defaultTarget: 1.0,
    description: 'Generate cloth initial configurations with specified deformation stress.',
  },
};

// ── Main component ───────────────────────────────────────────────────────────

export const Generate: React.FC = () => {
  const [config, setConfig]             = useState<GenerateConfig>(() =>
    loadLS(LS_CONFIG, {
      domain:       'cylinder_flow',
      target_value: 0.025,
      n_candidates: 6,
      method:       'sample',
      device:       'cpu',
    })
  );
  const [gpuAvailable, setGpuAvailable] = useState(false);
  const [remoteGpuHost, setRemoteGpuHost] = useState<string | null>(null);  // set when SSH remote is configured
  const [isGenerating, setIsGenerating] = useState(false);
  const [candidates, setCandidates]     = useState<Candidate[]>(() =>
    loadLS<Candidate[]>(LS_CANDIDATES, [])
  );
  const [selectedId, setSelectedId]     = useState<number | null>(null);
  const [error, setError]               = useState<string | null>(null);
  const [warningMessage, setWarningMessage] = useState<string | null>(null);
  const [bestId, setBestId]             = useState<number | null>(() =>
    loadLS<number | null>(LS_BEST_ID, null)
  );
  const [optTrajectory, setOptTrajectory] = useState<number[]>(() =>
    loadLS<number[]>(LS_TRAJECTORY, [])
  );
  const [genPhase, setGenPhase]           = useState<string | null>(null);
  const [genDone, setGenDone]             = useState(0);
  const [genActiveStep, setGenActiveStep] = useState<number | undefined>(undefined);

  const abortRef = useRef<AbortController | null>(null);
  const navigate = useNavigate();

  const domCfg = DOMAIN_CONFIGS[config.domain as keyof typeof DOMAIN_CONFIGS];

  // ── GPU auto-detection (mirrors Predict.tsx — checks local GPU + remote SSH) ─
  useEffect(() => {
    fetch('/api/status').then(r => r.json()).then(s => {
      if (s.gpu_available) {
        setGpuAvailable(true);
        setConfig(c => c.device === 'cpu' ? { ...c, device: 'cuda:0' } : c);
      }
    }).catch(() => {});
    // Remote GPU overrides local GPU check (same as Predict.tsx)
    fetch('/api/train/remote').then(r => r.ok ? r.json() : null).then(d => {
      if (d && d.enabled && d.host) {
        setGpuAvailable(true);
        setRemoteGpuHost(d.host);
        setConfig(c => c.device === 'cpu' ? { ...c, device: 'cuda:0' } : c);
      }
    }).catch(() => {});
  }, []);

  // ── Analyze handler — run rollout then open Visualize ─────────────────────

  const handleAnalyze = useCallback(async (sessionId: string, candidateId: number) => {
    const res = await fetch(
      `/api/generate/rollout/${sessionId}/${candidateId}?n_steps=50&device=${config.device}`,
      { method: 'POST' }
    );
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail ?? 'Rollout failed');
    }
    const { pkl_filename } = await res.json();
    navigate(`/visualize?file=${encodeURIComponent(pkl_filename)}`);
  }, [config.device, navigate]);

  // ── Persist state to localStorage ─────────────────────────────────────────

  useEffect(() => { saveLS(LS_CONFIG,     config);       }, [config]);
  useEffect(() => { saveLS(LS_CANDIDATES, candidates);   }, [candidates]);
  useEffect(() => { saveLS(LS_BEST_ID,    bestId);       }, [bestId]);
  useEffect(() => { saveLS(LS_TRAJECTORY, optTrajectory);}, [optTrajectory]);

  // ── Start generation ───────────────────────────────────────────────────────

  const handleGenerate = useCallback(async () => {
    setIsGenerating(true);
    setCandidates([]);
    setSelectedId(null);
    setBestId(null);
    setError(null);
    setWarningMessage(null);
    setOptTrajectory([]);
    setGenPhase(null);
    setGenDone(0);
    setGenActiveStep(undefined);

    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      const res = await fetch('/api/generate', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(config),
        signal:  ctrl.signal,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail ?? 'Generation failed');
      }

      const reader  = res.body!.getReader();
      const decoder = new TextDecoder();
      let   buf     = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        // Parse SSE chunks
        const lines = buf.split('\n\n');
        buf = lines.pop() ?? '';   // keep incomplete chunk

        for (const chunk of lines) {
          const eventLine = chunk.match(/^event:\s*(.+)$/m)?.[1]?.trim();
          const dataLine  = chunk.match(/^data:\s*(.+)$/m)?.[1]?.trim();
          if (!dataLine) continue;
          try {
            const payload = JSON.parse(dataLine);
            if (eventLine === 'progress') {
              setGenPhase((payload as any).phase ?? null);
              setGenDone((payload as any).done ?? 0);
              setGenActiveStep((payload as any).step ?? undefined);
            } else if (eventLine === 'candidate') {
              setCandidates(prev => [...prev, payload as Candidate]);
            } else if (eventLine === 'trajectory') {
              setOptTrajectory(payload.values ?? []);
            } else if (eventLine === 'done') {
              setBestId(payload.best_id ?? null);
              setGenPhase(null);
              setGenActiveStep(undefined);
            } else if (eventLine === 'error') {
              setError(payload.detail ?? 'Unknown error');
            } else if (eventLine === 'warning') {
              setWarningMessage((payload as any).detail as string);
            }
          } catch { /* ignore parse errors */ }
        }
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setError(err.message ?? String(err));
      }
    } finally {
      setIsGenerating(false);
      abortRef.current = null;
    }
  }, [config]);

  const handleStop = () => {
    abortRef.current?.abort();
    setIsGenerating(false);
  };

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">

      {/* Header */}
      <div className="flex items-center gap-4">
        <div className="w-10 h-10 rounded-xl bg-violet-600/20 border border-violet-500/30 flex items-center justify-center">
          <Sparkles className="w-5 h-5 text-violet-400" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">
            MeshGraph Generate
          </h1>
          <p className="text-sm text-slate-400 mt-0.5">
            Generate novel mesh designs conditioned on physics targets
          </p>
        </div>
      </div>

      {/* Config panel */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 space-y-6">
        <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
          Configuration
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">

          {/* Domain */}
          <div className="space-y-2">
            <label className="text-xs text-slate-400 font-medium">Domain</label>
            <select
              value={config.domain}
              onChange={e => {
                const d = e.target.value;
                const dc = DOMAIN_CONFIGS[d as keyof typeof DOMAIN_CONFIGS];
                setConfig(c => ({ ...c, domain: d,
                  target_value: dc?.defaultTarget ?? c.target_value }));
              }}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
            >
              {Object.entries(DOMAIN_CONFIGS).map(([k, v]) => (
                <option key={k} value={k}>{v.label}</option>
              ))}
            </select>
          </div>

          {/* Target value */}
          <div className="space-y-2">
            <label className="text-xs text-slate-400 font-medium">
              {domCfg.targetLabel}
              <span className="ml-2 font-mono text-blue-400">
                {config.target_value.toFixed(3)}
              </span>
            </label>
            <input
              type="range"
              min={domCfg.targetMin}
              max={domCfg.targetMax}
              step={domCfg.targetStep}
              value={config.target_value}
              onChange={e => setConfig(c => ({
                ...c, target_value: parseFloat(e.target.value)
              }))}
              className="w-full accent-blue-500"
            />
            <div className="flex justify-between text-[10px] text-slate-500">
              <span>{domCfg.targetMin}</span>
              <span>{domCfg.targetMax}</span>
            </div>
          </div>

          {/* Candidates */}
          <div className="space-y-2">
            <label className="text-xs text-slate-400 font-medium">Candidates</label>
            <input
              type="number"
              min={1}
              max={50}
              value={config.n_candidates}
              onChange={e => setConfig(c => ({
                ...c, n_candidates: Math.max(1, Math.min(50, parseInt(e.target.value) || 1))
              }))}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
            />
          </div>

          {/* Method */}
          <div className="space-y-2">
            <label className="text-xs text-slate-400 font-medium">Method</label>
            <select
              value={config.method}
              onChange={e => setConfig(c => ({ ...c, method: e.target.value }))}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
            >
              <option value="sample">CVAE Sample</option>
              <option value="gradient">Gradient Descent</option>
            </select>
          </div>

          {/* Device */}
          <div className="space-y-2">
            <label className="text-xs text-slate-400 font-medium">Compute Device</label>
            <select
              value={config.device}
              onChange={e => setConfig(c => ({ ...c, device: e.target.value }))}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
            >
              <option value="cpu">{remoteGpuHost ? `CPU (remote: ${remoteGpuHost})` : 'CPU'}</option>
              {gpuAvailable && (
                <option value="cuda:0">
                  {remoteGpuHost ? `cuda:0 (remote: ${remoteGpuHost})` : 'cuda:0 (local GPU)'}
                </option>
              )}
            </select>
          </div>

        </div>

        {/* Remote GPU banner — Generate dispatches via SSH when remote GPU is configured */}
        {remoteGpuHost && (
          <div className="flex items-center gap-2 p-3 bg-emerald-600/10 border border-emerald-500/20 rounded-lg text-xs">
            <Server className="w-4 h-4 text-emerald-400 shrink-0" />
            <div>
              <span className="text-emerald-300 font-bold">Remote GPU active ({remoteGpuHost})</span>
              <span className="text-emerald-400/70 ml-2">— Generate will run on the remote GPU via SSH.</span>
            </div>
          </div>
        )}

        {/* Domain description */}
        <div className="flex items-start gap-2 text-xs text-slate-400 bg-slate-800/40 rounded-lg px-4 py-3">
          <Info className="w-3.5 h-3.5 mt-0.5 text-slate-500 shrink-0" />
          <span>{domCfg.description}</span>
        </div>

        {/* Generate button */}
        <div className="flex items-center gap-3">
          {!isGenerating ? (
            <button
              onClick={handleGenerate}
              className="flex items-center gap-2 px-5 py-2.5 bg-violet-600 hover:bg-violet-500 text-white font-semibold rounded-lg transition-colors text-sm"
            >
              <Sparkles className="w-4 h-4" />
              Generate
            </button>
          ) : (
            <button
              onClick={handleStop}
              className="flex items-center gap-2 px-5 py-2.5 bg-red-600 hover:bg-red-500 text-white font-semibold rounded-lg transition-colors text-sm"
            >
              <Square className="w-4 h-4" />
              Stop
            </button>
          )}

          {isGenerating && (
            <div className="flex items-center gap-2 text-sm text-violet-400">
              <div className="w-2 h-2 bg-violet-400 rounded-full animate-pulse" />
              {genPhase
                ? genPhase.includes('Rendering')
                  ? <>{genPhase} {genDone + 1} / {config.n_candidates}</>
                  : <>{genPhase}</>
                : <>Generating candidates…</>
              }
            </div>
          )}
        </div>
      </div>

      {/* Pipeline steps strip */}
      <PipelineSteps method={config.method} activeStep={genActiveStep} />

      {/* Error */}
      {error && (
        <div className="flex items-start gap-3 rounded-xl border border-red-500/30 bg-red-950/20 px-4 py-3 text-sm text-red-400">
          <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Warning */}
      {warningMessage && (
        <div className="mb-4 flex items-start gap-2 rounded-lg border border-yellow-500/30
                        bg-yellow-500/10 px-4 py-3 text-sm text-yellow-300">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>{warningMessage}</span>
        </div>
      )}

      {/* Results grid */}
      {candidates.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
              Candidates
              {bestId !== null && (
                <span className="ml-3 text-xs text-violet-400 normal-case font-normal">
                  Best: #{bestId + 1}
                </span>
              )}
            </h2>
            <span className="text-xs text-slate-500">
              {candidates.length} generated
              {candidates.some(c => c.ood_confidence >= 0) && (
                <> · {candidates.filter(c => !c.is_ood && c.ood_confidence >= 0).length} in-distribution</>
              )}
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
            {candidates.map(c => (
              <CandidateCard
                key={c.id}
                id={c.id}
                domain={c.domain}
                predictedValue={c.predicted_value}
                targetValue={c.target_value}
                oodConfidence={c.ood_confidence}
                isOod={c.is_ood}
                meshNodes={c.mesh_nodes}
                params={c.params}
                thumbnailUrl={c.thumbnail_url}
                isSelected={selectedId === c.id || bestId === c.id}
                onSelect={() => setSelectedId(c.id)}
                sessionId={c.session_id ?? null}
                onAnalyze={handleAnalyze}
              />
            ))}
          </div>
        </div>
      )}

      {/* Optimisation trajectory (gradient descent method) */}
      {optTrajectory.length > 0 && (
        <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 space-y-4">
          <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
            Optimisation Trajectory
          </h2>
          <OptimizationChart
            trajectory={optTrajectory}
            targetValue={config.target_value}
            label={domCfg.targetLabel}
          />
        </div>
      )}

      {/* Selected candidate detail */}
      {selectedId !== null && (() => {
        const c = candidates.find(x => x.id === selectedId);
        if (!c) return null;
        return (
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 space-y-4">
            <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
              Candidate #{c.id + 1} — Detail
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="space-y-1">
                <div className="text-[11px] text-slate-500 uppercase tracking-wider">
                  {domCfg.targetLabel}
                </div>
                <div className="font-mono text-white text-lg">
                  {c.predicted_value.toFixed(4)}
                </div>
                <div className="text-xs text-slate-400">
                  target: {c.target_value.toFixed(4)}
                </div>
              </div>
              <div className="space-y-1">
                <div className="text-[11px] text-slate-500 uppercase tracking-wider">Confidence</div>
                <div className={cn(
                  "font-mono text-lg",
                  c.ood_confidence < 0 ? "text-slate-500" : c.is_ood ? "text-amber-400" : "text-emerald-400"
                )}>
                  {c.ood_confidence < 0 ? 'N/A' : `${(c.ood_confidence * 100).toFixed(1)}%`}
                </div>
                <div className="text-xs text-slate-400">
                  {c.ood_confidence < 0
                    ? 'Train model to enable OOD scoring'
                    : c.is_ood ? '⚠ Out of distribution' : '✓ In distribution'}
                </div>
              </div>
              <div className="space-y-1">
                <div className="text-[11px] text-slate-500 uppercase tracking-wider">Mesh Nodes</div>
                <div className="font-mono text-white text-lg">
                  {c.mesh_nodes.toLocaleString()}
                </div>
              </div>
              <div className="space-y-1">
                <div className="text-[11px] text-slate-500 uppercase tracking-wider">Parameters</div>
                <div className="space-y-1">
                  {Object.entries(c.params).map(([k, v]) => (
                    <div key={k} className="flex justify-between text-xs">
                      <span className="text-slate-400">{k}</span>
                      <span className="font-mono text-white">
                        {typeof v === 'number' ? v.toFixed(4) : v}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {c.thumbnail_url && (
              <img
                src={c.thumbnail_url}
                alt="Candidate mesh"
                className="rounded-lg max-h-64 object-contain"
              />
            )}
          </div>
        );
      })()}

    </div>
  );
};
