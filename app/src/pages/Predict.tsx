import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Loader2, CheckCircle2, ArrowRight, History, Database, Info, Cpu } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';

export const Predict: React.FC = () => {
  const [datasetInfo, setDatasetInfo] = useState<any>(null);
  const [status, setStatus] = useState<any>(null);
  const [results, setResults] = useState<any[]>([]);
  const [domain, setDomain] = useState('cylinder_flow');
  const [checkpoint, setCheckpoint] = useState<any>(null);
  const [trajectoryIndex, setTrajectoryIndex] = useState(0);
  const [device, setDevice] = useState('cpu');
  const [isRunning, setIsRunning] = useState(false);
  const [statusLoaded, setStatusLoaded] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressStep, setProgressStep] = useState(0);
  const [progressTotal, setProgressTotal] = useState(0);
  const [rolloutResult, setRolloutResult] = useState<any>(null);
  const [gpuStatus, setGpuStatus] = useState<any>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  // Keep isRunning in a ref so the fetch loop sees the latest value after navigation
  const isRunningRef = useRef(false);
  const navigate = useNavigate();
  const cn = (...classes: any[]) => classes.filter(Boolean).join(' ');

  // Restore state on mount — if a rollout is already in progress (tracked by
  // a boolean we store in sessionStorage) reconnect the progress stream.
  useEffect(() => {
    fetch('/api/status').then(r => r.json()).then(s => {
      setStatus(s);
      // Prefer cuda if GPU is available, otherwise default to cpu
      if (s.gpu_available) setDevice('cuda:0');
      setStatusLoaded(true);
    }).catch(() => setStatusLoaded(true));

    fetch('/api/dataset/info').then(r => r.json()).then(setDatasetInfo).catch(() => {});
    fetch('/api/results').then(r => r.json()).then(setResults).catch(() => {});
    fetch('/api/status/gpu').then(r => r.json()).then(setGpuStatus).catch(() => {});
  }, []);

  // Load checkpoint info when domain changes
  useEffect(() => {
    setCheckpoint(null);
    fetch(`/api/checkpoint?domain=${domain}`)
      .then(r => r.ok ? r.json() : null)
      .then(setCheckpoint)
      .catch(() => setCheckpoint(null));
    // Also reload dataset info for the selected domain
    fetch(`/api/dataset/info?domain=${domain}`)
      .then(r => r.ok ? r.json() : null)
      .then(d => d && setDatasetInfo(d))
      .catch(() => {});
  }, [domain]);

  const similarityScore = rolloutResult?.similarity_score ?? null;

  const handleStartRollout = async () => {
    if (isRunningRef.current) return;
    setIsRunning(true);
    isRunningRef.current = true;
    setProgress(0);
    setProgressStep(0);
    setProgressTotal(0);
    setRolloutResult(null);
    setErrorMsg(null);

    try {
      const response = await fetch('/api/rollout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          domain: domain,
          trajectory_index: trajectoryIndex,
          device: device,
        }),
      });

      if (!response.ok || !response.body) {
        const errBody = await response.json().catch(() => ({}));
        throw new Error(errBody.detail || `Server error: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.type === 'progress') {
              setProgressStep(data.step);
              setProgressTotal(data.total);
              setProgress((data.step / data.total) * 100);
            } else if (data.type === 'done') {
              setRolloutResult(data);
              setIsRunning(false);
              isRunningRef.current = false;
              fetch('/api/results').then(r => r.json()).then(setResults).catch(() => {});
            } else if (data.type === 'error') {
              setErrorMsg(data.message);
              setIsRunning(false);
              isRunningRef.current = false;
            }
          } catch {
            // ignore malformed SSE lines
          }
        }
      }
    } catch (err: any) {
      setIsRunning(false);
      isRunningRef.current = false;
      setErrorMsg(`Rollout failed: ${err.message}`);
    }
  };

  // GPU device label
  const gpuLabel = status?.gpu_available && status?.gpu_name ? status.gpu_name : null;

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      <header>
        <h2 className="text-3xl font-bold text-white tracking-tight">Run Inference</h2>
        <p className="text-slate-400 mt-2">Generate autoregressive physics rollouts using the trained GNN</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-1 space-y-6">
          <section className="bg-slate-900/50 border border-slate-800 rounded-2xl overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/80">
              <h3 className="font-semibold text-white">Rollout Configuration</h3>
            </div>
            <div className="p-6 space-y-6">

              {/* Dataset / Domain selector */}
              <div className="space-y-2">
                <label className="text-xs font-bold text-slate-500 uppercase tracking-wider">Dataset</label>
                <select
                  value={domain}
                  onChange={(e) => { setDomain(e.target.value); setTrajectoryIndex(0); }}
                  disabled={isRunning}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-blue-500/50 transition-colors disabled:opacity-50"
                >
                  {status?.domains
                    ? Object.entries(status.domains).map(([key, d]: [string, any]) => (
                        <option key={key} value={key} disabled={!d.available}>
                          {d.label}{!d.available ? ' (coming soon)' : ''}
                        </option>
                      ))
                    : <option value="cylinder_flow">Cylinder Flow (CFD)</option>
                  }
                </select>
                {status?.domains?.[domain]?.description && (
                  <p className="text-[10px] text-slate-500">{status.domains[domain].description}</p>
                )}
              </div>

              {/* Champion model card */}
              {checkpoint ? (
                <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Cpu className="w-4 h-4 text-blue-400" />
                      <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">Champion Model</span>
                    </div>
                    <span className="px-2 py-0.5 bg-yellow-500/20 text-yellow-400 text-[9px] font-bold rounded uppercase tracking-wider">👑 Active</span>
                  </div>
                  <p className="text-[11px] text-slate-400 font-mono truncate">{checkpoint.path.split('/').pop()}</p>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-[11px]">
                    <div className="flex justify-between">
                      <span className="text-slate-500">Epoch</span>
                      <span className="text-slate-200 font-mono">{checkpoint.epoch}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Val Loss</span>
                      <span className="text-green-400 font-mono">{checkpoint.valid_loss?.toFixed(6) ?? '—'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Params</span>
                      <span className="text-slate-200 font-mono">{checkpoint.param_count_m != null ? `${checkpoint.param_count_m} M` : '—'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Size</span>
                      <span className="text-slate-200 font-mono">{checkpoint.size_mb} MB</span>
                    </div>
                  </div>
                  <p className="text-[9px] text-slate-600">
                    Trained {new Date(checkpoint.last_modified).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })}
                  </p>
                </div>
              ) : status?.domains?.[domain]?.available === false ? (
                <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 flex items-center gap-3 text-slate-500 text-xs">
                  <Database className="w-4 h-4 shrink-0" />
                  This domain has no trained model yet
                </div>
              ) : (
                <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 flex items-center gap-3 text-slate-500 text-xs">
                  <div className="w-3 h-3 border-2 border-slate-700 border-t-slate-400 rounded-full animate-spin shrink-0" />
                  Loading model info…
                </div>
              )}

              {/* Trajectory Index with tooltip-style explanation */}
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <label className="text-xs font-bold text-slate-500 uppercase tracking-wider">Test Trajectory</label>
                  <div className="group relative">
                    <Info className="w-3.5 h-3.5 text-slate-600 cursor-help" />
                    <div className="absolute left-5 top-0 w-56 bg-slate-950 border border-slate-700 rounded-lg p-3 text-[10px] text-slate-400 leading-relaxed z-10 hidden group-hover:block shadow-xl">
                      <strong className="text-slate-300">What is a trajectory?</strong><br />
                      The test dataset contains {datasetInfo?.num_trajectories ?? '…'} independent fluid flow simulations,
                      each with {datasetInfo?.timesteps_per_trajectory ?? 600} timesteps (
                      {datasetInfo ? ((datasetInfo.timesteps_per_trajectory * datasetInfo.dt).toFixed(1) + 's') : '…'} of sim time).
                      <br /><br />
                      Each trajectory uses a different cylinder position and inlet velocity.
                      Index 0 is the first test case. Pick any value from 0 to {(datasetInfo?.num_trajectories ?? 1) - 1}.
                    </div>
                  </div>
                </div>
                <input
                  type="number"
                  min="0"
                  max={datasetInfo ? datasetInfo.num_trajectories - 1 : 100}
                  value={trajectoryIndex}
                  onChange={(e) => setTrajectoryIndex(Math.max(0, parseInt(e.target.value) || 0))}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-blue-500/50 transition-colors"
                />
                <p className="text-[10px] text-slate-500">
                  {datasetInfo
                    ? `0 – ${datasetInfo.num_trajectories - 1} available • ${datasetInfo.timesteps_per_trajectory} timesteps each`
                    : 'Loading dataset info…'}
                </p>
              </div>

              {/* Device selector — auto-populated from API */}
              <div className="space-y-2">
                <label className="text-xs font-bold text-slate-500 uppercase tracking-wider">Compute Device</label>
                <select
                  value={device}
                  onChange={(e) => setDevice(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-blue-500/50 transition-colors"
                >
                  {status?.gpu_available && (
                    <option value="cuda:0">cuda:0{gpuLabel ? ` (${gpuLabel})` : ''}</option>
                  )}
                  <option value="cpu">cpu</option>
                </select>
                {!statusLoaded && (
                  <p className="text-[10px] text-slate-500">Detecting GPU…</p>
                )}
                {statusLoaded && !status?.gpu_available && (
                  <p className="text-[10px] text-yellow-600">No GPU detected — will run on CPU (slower)</p>
                )}
              </div>

              {/* Error message */}
              {errorMsg && (
                <div className="p-3 bg-red-600/10 border border-red-500/20 rounded-lg">
                  <p className="text-[11px] text-red-400 font-mono break-all">{errorMsg}</p>
                </div>
              )}

              <button
                onClick={handleStartRollout}
                disabled={isRunning || !statusLoaded || !checkpoint}
                className="w-full py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-500 text-white rounded-xl font-bold flex items-center justify-center gap-2 transition-all shadow-lg shadow-blue-900/20"
              >
                {isRunning ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Running Rollout…
                  </>
                ) : !checkpoint ? (
                  <>
                    <Database className="w-5 h-5" />
                    No Model Available
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5 fill-current" />
                    Launch Simulation
                  </>
                )}
              </button>
            </div>
          </section>

          {/* Progress bar */}
          {isRunning && (
            <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-4">
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Progress</span>
                <span className="text-blue-400 font-mono">
                  {progressTotal > 0 ? `${progressStep} / ${progressTotal}` : `${Math.round(progress)}%`}
                </span>
              </div>
              <div className="w-full h-2 bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                <div
                  className="h-full bg-blue-600 transition-all duration-300 ease-out"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="text-[10px] text-slate-500 text-center uppercase tracking-widest animate-pulse">
                Autoregressive inference in progress
              </p>
            </section>
          )}

          {/* Performance + similarity — shown after rollout */}
          {(rolloutResult || results.length > 0) && !isRunning && (
            <div className="space-y-6">
              <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-4">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Performance Metrics</h3>
                <div className="grid grid-cols-3 gap-4">
                  <MetricCard label="Elapsed" value={rolloutResult ? `${rolloutResult.elapsed_seconds}s` : '—'} />
                  <MetricCard label="Steps/sec" value={rolloutResult ? (progressTotal / rolloutResult.elapsed_seconds).toFixed(1) : '—'} />
                  <MetricCard label="Speedup" value={rolloutResult ? `${rolloutResult.speedup}x` : '—'} />
                  <MetricCard label="GPU Alloc" value={gpuStatus?.mem_alloc_gb != null ? `${gpuStatus.mem_alloc_gb} GB` : '—'} />
                  <MetricCard label="GPU Resrv" value={gpuStatus?.mem_reserved_gb != null ? `${gpuStatus.mem_reserved_gb} GB` : '—'} />
                  <MetricCard label="GPU Util" value={gpuStatus?.utilization != null ? `${gpuStatus.utilization}%` : '—'} />
                </div>
              </section>

              {rolloutResult && similarityScore != null && (
                <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 space-y-4">
                  <div className="flex justify-between items-center">
                    <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Similarity Score</h3>
                    <span className={cn(
                      "px-2 py-0.5 rounded text-[10px] font-bold uppercase",
                      similarityScore > 0.75 ? "bg-green-500/20 text-green-400" :
                      similarityScore > 0.4 ? "bg-yellow-500/20 text-yellow-400" : "bg-red-500/20 text-red-400"
                    )}>
                      {similarityScore > 0.75 ? "🟢 Good" : similarityScore > 0.4 ? "🟡 Moderate" : "🔴 OOD Warning"}
                    </span>
                  </div>
                  <div className="space-y-2">
                    <div className="text-2xl font-bold text-white font-mono">
                      {similarityScore.toFixed(2)}
                    </div>
                    <div className="w-full h-1.5 bg-slate-950 rounded-full overflow-hidden">
                      <div className="h-full bg-blue-600" style={{ width: `${Math.max(0, similarityScore) * 100}%` }} />
                    </div>
                    <p className="text-[10px] text-slate-500 italic">
                      Proxy metric: Euclidean distance in [mean_velocity, num_nodes] space vs existing rollouts.
                    </p>
                  </div>
                </section>
              )}
            </div>
          )}

          {/* Success + View Results button */}
          {rolloutResult && (
            <section className="bg-green-600/10 border border-green-500/20 rounded-2xl p-6 space-y-4">
              <div className="flex items-center gap-2 text-green-400">
                <CheckCircle2 className="w-5 h-5" />
                <span className="font-bold uppercase tracking-wider text-xs">Rollout Complete</span>
                {domain && (
                  <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                    domain === "flag_simple"
                      ? "bg-purple-500/20 text-purple-400"
                      : "bg-blue-500/20 text-blue-400"
                  }`}>
                    {domain === "flag_simple" ? "CLOTH MODEL" : "CFD MODEL"}
                  </span>
                )}
                {domain === 'cylinder_flow' && rolloutResult?.target_field && (
                  <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                    rolloutResult.target_field === "pressure"
                      ? "bg-orange-500/20 text-orange-400"
                      : "bg-cyan-500/20 text-cyan-400"
                  }`}>
                    {rolloutResult.target_field === "pressure" ? "PRESSURE" : "VELOCITY"}
                  </span>
                )}
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-[10px] text-slate-500 font-bold uppercase">Elapsed</p>
                  <p className="text-lg font-bold text-white">{rolloutResult.elapsed_seconds}s</p>
                </div>
                <div>
                  <p className="text-[10px] text-slate-500 font-bold uppercase">Speedup</p>
                  <p className="text-lg font-bold text-green-400">{rolloutResult.speedup}x</p>
                </div>
              </div>
              {rolloutResult?.confidence_score != null && (
                <div className="flex items-center gap-2 mt-2">
                  <span className="text-sm text-gray-400">Confidence:</span>
                  <div className="flex-1 bg-gray-700 rounded-full h-2 max-w-32">
                    <div
                      className={`h-2 rounded-full ${
                        rolloutResult.confidence_score >= 0.7 ? "bg-green-500" :
                        rolloutResult.confidence_score >= 0.4 ? "bg-yellow-500" : "bg-red-500"
                      }`}
                      style={{ width: `${Math.max(0, rolloutResult.confidence_score * 100)}%` }}
                    />
                  </div>
                  <span className="text-sm font-semibold text-gray-200">
                    {Math.round(rolloutResult.confidence_score * 100)}%
                  </span>
                  <span className={`text-xs px-2 py-0.5 rounded font-bold ${
                    rolloutResult.confidence_score >= 0.7 ? "bg-green-500/20 text-green-400" :
                    rolloutResult.confidence_score >= 0.4 ? "bg-yellow-500/20 text-yellow-400" :
                    "bg-red-500/20 text-red-400"
                  }`}>
                    {rolloutResult.confidence_score >= 0.7 ? "HIGH" :
                     rolloutResult.confidence_score >= 0.4 ? "MEDIUM" : "LOW"}
                  </span>
                </div>
              )}
              <button
                onClick={() => navigate(`/visualize?file=${rolloutResult.pkl_path.split('/').pop()}`)}
                className="w-full py-2 bg-green-600 hover:bg-green-500 text-white rounded-lg text-sm font-bold transition-colors flex items-center justify-center gap-2"
              >
                View Results
                <ArrowRight className="w-4 h-4" />
              </button>
            </section>
          )}
        </div>

        {/* Recent rollouts table */}
        <div className="lg:col-span-2 space-y-6">
          <section className="bg-slate-900/50 border border-slate-800 rounded-2xl overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/80 flex items-center gap-2">
              <History className="w-4 h-4 text-slate-400" />
              <h3 className="font-semibold text-white">Saved Rollouts</h3>
            </div>
            <div className="divide-y divide-slate-800">
              {results.length > 0 ? (
                results.map((res) => (
                  <div key={res.filename} className="p-4 hover:bg-slate-900/50 transition-colors flex items-center justify-between group">
                    <div className="flex items-center gap-4">
                      <div className="w-10 h-10 bg-slate-950 rounded-lg flex items-center justify-center text-slate-500">
                        <Database className="w-5 h-5" />
                      </div>
                      <div>
                        <h4 className="text-sm font-medium text-slate-200">{res.filename}</h4>
                        <p className="text-[10px] text-slate-500 uppercase tracking-wider">
                          Trajectory {res.trajectory_index} • {res.size_mb} MB • {new Date(res.created).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                    <Link
                      to={`/visualize?file=${res.filename}`}
                      className="px-4 py-1.5 bg-slate-800 hover:bg-blue-600 text-slate-300 hover:text-white rounded-lg text-xs font-bold transition-all"
                    >
                      Analyze
                    </Link>
                  </div>
                ))
              ) : (
                <div className="p-12 text-center space-y-2">
                  <Database className="w-8 h-8 text-slate-700 mx-auto" />
                  <p className="text-slate-500 text-sm">No saved rollouts yet</p>
                  <p className="text-slate-600 text-xs">Configure a trajectory above and click Launch Simulation</p>
                </div>
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};

const MetricCard = ({ label, value }: any) => (
  <div className="bg-slate-950 p-2 rounded-lg border border-slate-800">
    <p className="text-[8px] text-slate-500 font-bold uppercase">{label}</p>
    <p className="text-xs font-bold text-slate-200">{value}</p>
  </div>
);
