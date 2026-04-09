import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Square, TrendingDown, Target, Clock, Info, Terminal, Database, Server, CheckCircle2, XCircle, Loader2, Copy, Check, Activity, Skull } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export const Train: React.FC = () => {
  const [domains, setDomains] = useState<Record<string, any>>({});
  const [config, setConfig] = useState({
    domain: 'cylinder_flow',
    epochs: 100,
    batch_size: 20,
    lr: 0.0001,
    noise_std: 0.02,
    early_stopping_patience: 10,
    message_passing_steps: 15,
    target_field: 'velocity',
  });

  const [isRunning, setIsRunning] = useState(false);
  const [statusLoaded, setStatusLoaded] = useState(false);
  const [epochs, setEpochs] = useState<any[]>([]);
  const [bestEpoch, setBestEpoch] = useState<any>(null);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [logLines, setLogLines] = useState<string[]>([]);
  const [logPath, setLogPath] = useState<string>('runs/train_ui.log');
  const [copied, setCopied] = useState(false);
  const [arch, setArch] = useState('GNS');
  const [remoteActive, setRemoteActive] = useState(false);  // true when training is running on remote GPU
  const [processes, setProcesses] = useState<any[]>([]);
  const [killingPid, setKillingPid] = useState<number | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  // Remote GPU config
  const [remote, setRemote] = useState({ host: '', port: 22, user: 'ahmealy', venv_python: '/home/ahmealy/.pyenv/versions/venv_gpu/bin/python', enabled: false });
  const [remoteEnabled, setRemoteEnabled] = useState(false);
  const [testStatus, setTestStatus] = useState<{ ok: boolean; message: string } | null>(null);
  const [testing, setTesting] = useState(false);

  // Toggle remote GPU and immediately persist the change
  const handleToggleRemote = async () => {
    const next = !remoteEnabled;
    setRemoteEnabled(next);
    setTestStatus(null);
    const cfg = { ...remote, enabled: next };
    await fetch('/api/train/remote', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg),
    }).catch(() => {});
  };

  // Tick elapsed timer while running
  useEffect(() => {
    if (!isRunning || startTime === null) return;
    const t = setInterval(() => setElapsed(Math.floor((Date.now() - startTime) / 1000)), 1000);
    return () => clearInterval(t);
  }, [isRunning, startTime]);

  // Poll raw log lines every 3s while training is running — shows tqdm progress
  useEffect(() => {
    if (!isRunning) return;
    const poll = async () => {
      try {
        const r = await fetch('/api/train/log?tail=60');
        if (!r.ok) return;
        const d = await r.json();
        if (d.lines && d.lines.length > 0) {
          setLogLines(d.lines.filter((l: string) => l.trim()));
        }
      } catch { /* ignore */ }
    };
    poll();
    const t = setInterval(poll, 3000);
    return () => clearInterval(t);
  }, [isRunning]);

  // Auto-scroll log
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logLines]);

  // Poll process list every 5s (always, not just when running)
  const fetchProcesses = useCallback(async () => {
    try {
      const r = await fetch('/api/train/processes');
      if (!r.ok) return;
      const d = await r.json();
      setProcesses(d.processes || []);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    fetchProcesses();
    const t = setInterval(fetchProcesses, 5000);
    return () => clearInterval(t);
  }, [fetchProcesses]);

  const handleKill = async (pid: number) => {
    setKillingPid(pid);
    try {
      await fetch(`/api/train/kill/${pid}`, { method: 'POST' });
      await fetchProcesses();
      // If this was the managed process, update running state
      setIsRunning(false);
    } catch { /* ignore */ }
    setKillingPid(null);
  };

  const startStreaming = useCallback(() => {
    // No-op if already connected
    if (eventSourceRef.current && eventSourceRef.current.readyState !== EventSource.CLOSED) return;
    if (eventSourceRef.current) eventSourceRef.current.close();
    const es = new EventSource('/api/train/stream');
    eventSourceRef.current = es;
    es.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'epoch') {
        const line = `[Epoch ${data.epoch}] train=${data.train_loss.toExponential(3)}  valid=${data.valid_loss.toExponential(3)}`;
        setEpochs(prev => [...prev, data]);
        setLogLines(prev => [...prev.slice(-200), line]);  // keep last 200 lines
      } else if (data.type === 'best') {
        setLogLines(prev => [...prev, `  ✓ New best checkpoint — epoch ${data.epoch}, valid=${data.valid_loss.toExponential(3)}`]);
        setBestEpoch(data);
      } else if (data.type === 'done') {
        setLogLines(prev => [...prev, '--- Training complete ---']);
        setIsRunning(false);
        es.close();
      } else if (data.type === 'error') {
        setLogLines(prev => [...prev, `❌ Error: ${data.message}`]);
        setIsRunning(false);
        es.close();
      }
    };
    es.onerror = () => {
      setLogLines(prev => [...prev, '[stream disconnected]']);
      setIsRunning(false);
      es.close();
    };
  }, []);

  // On mount: load domains + restore state from API
  useEffect(() => {
    // Use a single state update to avoid the flash where statusLoaded=true
    // but isRunning is still false (shows "Start Training" for one render frame).
    let cancelled = false;

    // Fast path: /api/status already has training_running and is cheap/cached.
    // Use it to set isRunning immediately so the button never flashes.
    fetch('/api/status')
      .then(r => r.ok ? r.json() : null)
      .then(d => {
        if (cancelled || !d) return;
        if (d.domains) setDomains(d.domains);
        // Set isRunning from fast status endpoint before the full train/status loads
        if (d.training_running) {
          setIsRunning(true);
          setStartTime(Date.now());
          startStreaming();
          setStatusLoaded(true);  // already know enough — show Stop button immediately
        }
      })
      .catch(() => {});

    // Load saved remote GPU config
    fetch('/api/train/remote')
      .then(r => r.ok ? r.json() : null)
      .then(d => {
        if (d) { setRemote(d); setRemoteEnabled(!!d.enabled && !!d.host); }
      }).catch(() => {});

    // Full train status for epoch history and best checkpoint
    fetch('/api/train/status')
      .then(r => r.json())
      .then(data => {
        if (cancelled) return;
        setEpochs(data.epochs || []);
        if (data.best_epoch) setBestEpoch({ epoch: data.best_epoch, valid_loss: data.best_valid_loss });
        if (data.log_path) setLogPath(data.log_path);
        // Track whether training is executing on the remote GPU
        if (data.running) setRemoteActive(!!data.remote);
        // Restore accurate elapsed time from log file creation timestamp
        if (data.log_start_ms && data.running) {
          setStartTime(data.log_start_ms);
        }
        // If fast path already set isRunning, don't overwrite — just update epochs.
        // If fast path didn't fire yet, handle running state here.
        if (data.running) {
          setIsRunning(true);
          setStartTime(prev => prev ?? Date.now());
          startStreaming();
          // Sync config dropdowns to the actually-running job
          if (data.active_config) {
            setConfig(prev => ({
              ...prev,
              domain:               data.active_config.domain              ?? prev.domain,
              target_field:         data.active_config.target_field        ?? prev.target_field,
              epochs:               data.active_config.num_epochs          ?? prev.epochs,
              batch_size:           data.active_config.batch_size          ?? prev.batch_size,
              lr:                   data.active_config.lr                  ?? prev.lr,
              noise_std:            data.active_config.noise_std           ?? prev.noise_std,
              early_stopping_patience: data.active_config.early_stopping_patience ?? prev.early_stopping_patience,
              message_passing_steps: data.active_config.message_passing_num ?? prev.message_passing_steps,
            }));
          }
        } else {
          // Training is not running — make sure button shows correctly
          setIsRunning(false);
        }
        setStatusLoaded(true);
      })
      .catch(() => { if (!cancelled) setStatusLoaded(true); });

    return () => {
      cancelled = true;
      eventSourceRef.current?.close();
    };
  }, [startStreaming]);

  const handleStart = async () => {
    const res = await fetch('/api/train/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (res.ok) {
      setIsRunning(true);
      setStartTime(Date.now());
      setElapsed(0);
      setLogLines(['--- Training started ---']);
      setRemoteActive(remoteEnabled && !!remote.host);
      startStreaming();
    } else {
      const err = await res.json().catch(() => ({}));
      alert(err.detail || 'Failed to start training');
    }
  };

  const handleStop = async () => {
    const res = await fetch('/api/train/stop', { method: 'POST' });
    if (res.ok) {
      setIsRunning(false);
      setLogLines(prev => [...prev, '--- Stopped by user ---']);
      eventSourceRef.current?.close();
    }
  };

  const handleSaveRemote = async () => {
    const cfg = { ...remote, enabled: remoteEnabled };
    await fetch('/api/train/remote', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg),
    });
    setTestStatus(null);
  };

  const handleCopyLog = async () => {
    try {
      await navigator.clipboard.writeText(logLines.join('\n'));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch { /* ignore */ }
  };

  const handleTestRemote = async () => {
    setTesting(true);
    setTestStatus(null);
    try {
      const res = await fetch('/api/train/remote/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...remote, enabled: remoteEnabled }),
      });
      const d = await res.json();
      setTestStatus(d);
    } catch {
      setTestStatus({ ok: false, message: 'Request failed — is the API server running?' });
    } finally {
      setTesting(false);
    }
  };

  const lastEpoch = epochs[epochs.length - 1];
  const fmtElapsed = (s: number) => `${String(Math.floor(s / 3600)).padStart(2,'0')}:${String(Math.floor((s % 3600) / 60)).padStart(2,'0')}:${String(s % 60).padStart(2,'0')}`;

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      <header className="flex justify-between items-end">
        <div>
          <h2 className="text-3xl font-bold text-white tracking-tight">Model Training</h2>
          <p className="text-slate-400 mt-2">Configure and monitor MeshGraphNets training</p>
        </div>
        <div className="flex gap-3">
          {!statusLoaded ? (
            <div className="px-6 py-2.5 bg-slate-800 text-slate-500 rounded-lg font-semibold flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-slate-600 border-t-slate-400 rounded-full animate-spin" />
              Checking...
            </div>
          ) : isRunning ? (
            <button
              onClick={handleStop}
              className="px-6 py-2.5 bg-red-600 hover:bg-red-500 text-white rounded-lg font-semibold flex items-center gap-2 transition-all shadow-lg shadow-red-900/20"
            >
              <Square className="w-4 h-4 fill-current" />
              Stop Training
            </button>
          ) : (
            <button
              onClick={handleStart}
              disabled={arch !== 'GNS'}
              className="px-6 py-2.5 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-500 text-white rounded-lg font-semibold flex items-center gap-2 transition-all shadow-lg shadow-blue-900/20"
            >
              <Play className="w-4 h-4 fill-current" />
              Start Training
            </button>
          )}
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        <aside className="space-y-6">
          {/* Dataset / Domain selector */}
          <section className="bg-slate-900/50 border border-slate-800 rounded-2xl overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/80 flex items-center gap-2">
              <Database className="w-4 h-4 text-slate-400" />
              <h3 className="font-semibold text-white text-sm uppercase tracking-wider">Dataset</h3>
            </div>
            <div className="p-6 space-y-3">
              <select
                value={config.domain}
                onChange={(e) => setConfig({ ...config, domain: e.target.value, target_field: 'velocity' })}
                disabled={isRunning}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500/50 transition-colors disabled:opacity-50"
              >
                {Object.keys(domains).length > 0
                  ? Object.entries(domains).map(([key, d]: [string, any]) => (
                    <option key={key} value={key} disabled={!d.available}>
                      {d.label}{!d.available ? ' (coming soon)' : ''}
                    </option>
                  ))
                  : <option value="cylinder_flow">Cylinder Flow (CFD)</option>
                }
              </select>
              {domains[config.domain] && (
                <p className="text-[10px] text-slate-500">{domains[config.domain].description}</p>
              )}
              {Object.values(domains).some((d: any) => !d.available) && (
                <p className="text-[10px] text-slate-600 italic">Other domains are coming soon</p>
              )}
              {/* Target field selector — cylinder_flow only */}
              {config.domain === 'cylinder_flow' && (
                <div className="mt-3 space-y-1.5">
                  <label className="text-[10px] uppercase text-slate-500 font-bold tracking-wider">Target Field</label>
                  <select
                    value={config.target_field}
                    onChange={(e) => setConfig({ ...config, target_field: e.target.value })}
                    disabled={isRunning}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500/50 transition-colors disabled:opacity-50"
                  >
                    <option value="velocity">Velocity — fluid velocity field</option>
                    <option value="pressure">Pressure — fluid pressure field</option>
                  </select>
                  {config.target_field === 'pressure' && (
                    <p className="text-[10px] text-orange-400/80">Requires re-parsed data with pressure.dat files</p>
                  )}
                </div>
              )}
            </div>
          </section>

          <section className="bg-slate-900/50 border border-slate-800 rounded-2xl overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/80">
              <h3 className="font-semibold text-white text-sm uppercase tracking-wider">Architecture</h3>
            </div>
            <div className="p-6 space-y-4">
              <div className="space-y-2">
                <select
                  value={arch}
                  onChange={(e) => setArch(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500/50 transition-colors"
                >
                  <option value="GNS">GNS (Graph Network)</option>
                  <option value="TNS">TNS (Transformer)</option>
                  <option value="SER">SER (Shape Encoding)</option>
                </select>
                {arch !== 'GNS' && (
                  <div className="p-3 bg-blue-600/10 border border-blue-500/20 rounded-lg flex items-start gap-2">
                    <Info className="w-4 h-4 text-blue-400 mt-0.5 shrink-0" />
                    <p className="text-[10px] text-blue-300">Architecture "{arch}" is currently in development. Training is disabled.</p>
                  </div>
                )}
              </div>
              <div className="overflow-hidden border border-slate-800 rounded-lg">
                <table className="w-full text-[10px] text-left">
                  <thead className="bg-slate-950 text-slate-500 uppercase font-bold">
                    <tr>
                      <th className="px-2 py-1.5">Model</th>
                      <th className="px-2 py-1.5">Best For</th>
                      <th className="px-2 py-1.5">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800 text-slate-400">
                    <tr><td className="px-2 py-1.5 font-bold text-slate-200">GNS</td><td className="px-2 py-1.5">Transient CFD</td><td className="px-2 py-1.5 text-green-500">✅ Ready</td></tr>
                    <tr><td className="px-2 py-1.5 font-bold text-slate-200">TNS</td><td className="px-2 py-1.5">Global Physics</td><td className="px-2 py-1.5 text-slate-600">🔒 Soon</td></tr>
                    <tr><td className="px-2 py-1.5 font-bold text-slate-200">SER</td><td className="px-2 py-1.5">Steady-state</td><td className="px-2 py-1.5 text-slate-600">🔒 Soon</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </section>

          <section className="bg-slate-900/50 border border-slate-800 rounded-2xl overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/80">
              <h3 className="font-semibold text-white text-sm uppercase tracking-wider">Hyperparameters</h3>
            </div>
            <div className="p-6 space-y-4">
              <ParamInput label="Learning Rate" value={config.lr} onChange={(v) => setConfig({ ...config, lr: parseFloat(v) })} type="number" step="0.00001" />
              <ParamInput label="Epochs" value={config.epochs} onChange={(v) => setConfig({ ...config, epochs: parseInt(v) })} type="number" />
              <ParamInput label="Batch Size" value={config.batch_size} onChange={(v) => setConfig({ ...config, batch_size: parseInt(v) })} type="number" />
              <ParamInput label="Noise Std" value={config.noise_std} onChange={(v) => setConfig({ ...config, noise_std: parseFloat(v) })} type="number" step="0.01" />
              <ParamInput label="MP Steps" value={config.message_passing_steps} onChange={(v) => setConfig({ ...config, message_passing_steps: parseInt(v) })} type="number" />
            </div>
          </section>

          {bestEpoch && (
            <section className="bg-blue-600/10 border border-blue-500/20 rounded-2xl p-6 space-y-3">
              <div className="flex items-center gap-2 text-blue-400">
                <Target className="w-4 h-4" />
                <span className="text-xs font-bold uppercase tracking-wider">Best Checkpoint</span>
              </div>
              <div>
                <p className="text-2xl font-bold text-white font-mono">{bestEpoch.valid_loss.toFixed(6)}</p>
                <p className="text-xs text-blue-400/70">Epoch {bestEpoch.epoch}</p>
              </div>
            </section>
          )}

          {/* Remote GPU */}
          <section className="bg-slate-900/50 border border-slate-800 rounded-2xl overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/80 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Server className="w-4 h-4 text-slate-400" />
                <h3 className="font-semibold text-white text-sm uppercase tracking-wider">Remote GPU</h3>
              </div>
              <button
                onClick={handleToggleRemote}
                className={`relative w-10 h-5 rounded-full transition-colors ${remoteEnabled ? 'bg-blue-600' : 'bg-slate-700'}`}
              >
                <span className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${remoteEnabled ? 'translate-x-5' : 'translate-x-0'}`} />
              </button>
            </div>
            <div className="p-4 space-y-3">
              <div className="space-y-2">
                <label className="text-[10px] uppercase text-slate-500 font-bold tracking-wider">SSH Host</label>
                <input
                  value={remote.host}
                  onChange={e => setRemote(r => ({ ...r, host: e.target.value }))}
                  placeholder="dvt-gpubig1.wv.mentorg.com"
                  disabled={isRunning}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-blue-500/50 disabled:opacity-50"
                />
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <label className="text-[10px] uppercase text-slate-500 font-bold tracking-wider">Port</label>
                  <input
                    type="number"
                    value={remote.port}
                    onChange={e => setRemote(r => ({ ...r, port: parseInt(e.target.value) || 22 }))}
                    disabled={isRunning}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-blue-500/50 disabled:opacity-50"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-[10px] uppercase text-slate-500 font-bold tracking-wider">User</label>
                  <input
                    value={remote.user}
                    onChange={e => setRemote(r => ({ ...r, user: e.target.value }))}
                    placeholder="ahmealy"
                    disabled={isRunning}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-blue-500/50 disabled:opacity-50"
                  />
                </div>
              </div>
              <div className="space-y-1">
                <label className="text-[10px] uppercase text-slate-500 font-bold tracking-wider">Venv Python path</label>
                <input
                  value={remote.venv_python}
                  onChange={e => setRemote(r => ({ ...r, venv_python: e.target.value }))}
                  placeholder="/home/ahmealy/.pyenv/versions/venv_gpu/bin/python"
                  disabled={isRunning}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-blue-500/50 font-mono disabled:opacity-50"
                />
              </div>

              {testStatus && (
                <div className={`flex items-start gap-2 p-2.5 rounded-lg text-[11px] ${testStatus.ok ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'}`}>
                  {testStatus.ok
                    ? <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                    : <XCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />}
                  <span>{testStatus.message}</span>
                </div>
              )}

              <div className="flex gap-2 pt-1">
                <button
                  onClick={handleTestRemote}
                  disabled={testing || !remote.host || isRunning}
                  className="flex-1 py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-300 rounded-lg text-xs font-medium transition-colors flex items-center justify-center gap-1.5"
                >
                  {testing ? <Loader2 className="w-3 h-3 animate-spin" /> : <Server className="w-3 h-3" />}
                  Test
                </button>
                <button
                  onClick={handleSaveRemote}
                  disabled={isRunning}
                  className="flex-1 py-2 bg-blue-600/20 hover:bg-blue-600/30 disabled:opacity-40 text-blue-400 border border-blue-500/20 rounded-lg text-xs font-medium transition-colors"
                >
                  Save
                </button>
              </div>
              {remoteEnabled && remote.host && (
                <p className="text-[10px] text-blue-400/70 text-center">
                  Training will run on <span className="font-mono">{remote.host}:{remote.port}</span>
                </p>
              )}
            </div>
          </section>
        </aside>

        <main className="lg:col-span-3 space-y-6">
          <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 h-[360px] flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <TrendingDown className="w-5 h-5 text-slate-400" />
                <h3 className="font-semibold text-white">Loss Curves</h3>
              </div>
              <div className="flex gap-4 text-xs">
                <div className="flex items-center gap-2"><div className="w-3 h-3 bg-blue-500 rounded-sm" /><span className="text-slate-400">Train</span></div>
                <div className="flex items-center gap-2"><div className="w-3 h-3 bg-green-500 rounded-sm" /><span className="text-slate-400">Valid</span></div>
              </div>
            </div>
            <div className="flex-1 min-h-0">
              {epochs.length === 0 ? (
                <div className="h-full flex items-center justify-center text-slate-600 text-sm">
                  {isRunning ? 'Waiting for first epoch…' : 'No training data yet — start training to see loss curves'}
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={epochs}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="epoch" stroke="#64748b" fontSize={12} tickLine={false} axisLine={false} label={{ value: 'Epoch', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }} />
                    <YAxis scale="log" domain={['auto', 'auto']} stroke="#64748b" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(v) => v.toExponential(1)} />
                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px' }} itemStyle={{ fontSize: '12px' }} />
                    <Line type="monotone" dataKey="train_loss" stroke="#3b82f6" strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="valid_loss" stroke="#10b981" strokeWidth={2} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </section>

          <div className="grid grid-cols-4 gap-4">
            <div className="bg-slate-900/50 border border-slate-800 p-4 rounded-xl flex items-center gap-4">
              <div className="w-10 h-10 bg-slate-950 rounded-lg flex items-center justify-center text-slate-400">
                <Clock className="w-5 h-5" />
              </div>
              <div>
                <p className="text-[10px] uppercase text-slate-500 font-bold">Time Elapsed</p>
                <p className="text-sm font-semibold text-slate-200 font-mono">
                  {isRunning ? fmtElapsed(elapsed) : elapsed > 0 ? fmtElapsed(elapsed) : '—'}
                </p>
              </div>
            </div>
            <div className="bg-slate-900/50 border border-slate-800 p-4 rounded-xl flex items-center gap-4">
              <div className="w-10 h-10 bg-slate-950 rounded-lg flex items-center justify-center text-slate-400">
                <Info className="w-5 h-5" />
              </div>
              <div>
                <p className="text-[10px] uppercase text-slate-500 font-bold">Epochs Done</p>
                <p className="text-sm font-semibold text-slate-200">{epochs.length > 0 ? `${epochs.length} / ${config.epochs}` : '—'}</p>
              </div>
            </div>
            <div className="bg-slate-900/50 border border-slate-800 p-4 rounded-xl flex items-center gap-4">
              <div className="w-10 h-10 bg-slate-950 rounded-lg flex items-center justify-center text-slate-400">
                <Target className="w-5 h-5" />
              </div>
              <div>
                <p className="text-[10px] uppercase text-slate-500 font-bold">Current Valid Loss</p>
                <p className="text-sm font-semibold text-slate-200 font-mono">
                  {lastEpoch ? lastEpoch.valid_loss.toExponential(3) : '—'}
                </p>
              </div>
            </div>
            <div className={`border p-4 rounded-xl flex items-center gap-4 ${remoteActive || (isRunning && remoteEnabled) ? 'bg-blue-600/10 border-blue-500/20' : 'bg-slate-900/50 border-slate-800'}`}>
              <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${remoteActive || (isRunning && remoteEnabled) ? 'bg-blue-600/20 text-blue-400' : 'bg-slate-950 text-slate-400'}`}>
                <Server className="w-5 h-5" />
              </div>
              <div>
                <p className="text-[10px] uppercase text-slate-500 font-bold">Compute</p>
                <p className={`text-sm font-semibold font-mono ${remoteActive || (isRunning && remoteEnabled) ? 'text-blue-300' : 'text-slate-400'}`}>
                  {remoteActive || (isRunning && remoteEnabled) ? 'Remote GPU' : 'Local CPU'}
                </p>
                {(remoteActive || (isRunning && remoteEnabled)) && remote.host && (
                  <p className="text-[9px] text-blue-500/70 font-mono truncate max-w-[100px]">{remote.host}</p>
                )}
              </div>
            </div>
          </div>

          {/* Process Manager */}
          <section className="bg-slate-900/50 border border-slate-800 rounded-2xl overflow-hidden">
            <div className="px-4 py-2.5 border-b border-slate-800 flex items-center gap-2">
              <Activity className="w-3.5 h-3.5 text-slate-500" />
              <span className="text-[10px] font-bold uppercase text-slate-500 tracking-wider">Running Train Processes</span>
              {processes.length > 0 && (
                <span className="px-1.5 py-0.5 bg-green-500/20 text-green-400 text-[9px] font-bold rounded-full">{processes.length}</span>
              )}
              <button onClick={fetchProcesses} className="ml-auto text-[9px] text-slate-600 hover:text-slate-400 transition-colors">↺ refresh</button>
            </div>
            {processes.length === 0 ? (
              <div className="px-4 py-3 text-[11px] text-slate-600 italic">No train.py processes found</div>
            ) : (
              <div className="divide-y divide-slate-800/60">
                {processes.map((p) => (
                  <div key={p.pid} className={`px-4 py-2.5 flex items-center gap-3 text-[11px] ${p.managed ? 'bg-blue-600/5' : ''}`}>
                    {/* PID + managed badge */}
                    <div className="w-16 shrink-0">
                      <span className="font-mono text-slate-400">{p.pid}</span>
                      {p.managed && <span className="ml-1 px-1 py-0.5 bg-blue-600/20 text-blue-400 text-[8px] font-bold rounded">MGD</span>}
                    </div>
                    {/* Domain */}
                    <div className="w-24 shrink-0">
                      <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold ${p.domain === 'flag_simple' ? 'bg-purple-500/20 text-purple-400' : 'bg-blue-500/20 text-blue-400'}`}>
                        {p.domain === 'flag_simple' ? 'CLOTH' : p.domain === 'cylinder_flow' ? 'CFD' : p.domain}
                      </span>
                    </div>
                    {/* Device */}
                    <div className="w-24 shrink-0">
                      <span className={`text-[9px] font-bold ${p.device === 'remote GPU' ? 'text-blue-400' : 'text-slate-500'}`}>
                        {p.device === 'remote GPU' ? '🖥 Remote GPU' : '💻 Local CPU'}
                      </span>
                    </div>
                    {/* Status */}
                    <div className="w-16 shrink-0">
                      <span className={`text-[9px] font-bold uppercase ${p.status === 'running' ? 'text-green-400' : p.status === 'sleeping' ? 'text-yellow-500' : 'text-slate-500'}`}>
                        {p.status}
                      </span>
                    </div>
                    {/* CPU % */}
                    <div className="w-14 shrink-0 text-slate-400 font-mono">{p.cpu_pct}%</div>
                    {/* Mem */}
                    <div className="w-16 shrink-0 text-slate-400 font-mono">{p.mem_mb != null ? `${p.mem_mb} MB` : '—'}</div>
                    {/* Elapsed */}
                    <div className="w-20 shrink-0 text-slate-400 font-mono">{p.elapsed}</div>
                    {/* Kill button */}
                    <button
                      onClick={() => handleKill(p.pid)}
                      disabled={killingPid === p.pid}
                      className="ml-auto flex items-center gap-1 px-2.5 py-1 bg-red-600/10 hover:bg-red-600/20 disabled:opacity-40 text-red-400 border border-red-500/20 rounded text-[10px] font-bold transition-colors"
                      title={`Kill PID ${p.pid}`}
                    >
                      {killingPid === p.pid
                        ? <Loader2 className="w-3 h-3 animate-spin" />
                        : <Skull className="w-3 h-3" />}
                      Kill
                    </button>
                  </div>
                ))}
              </div>
            )}
          </section>

          {/* Live Log */}
          {(isRunning || logLines.length > 0) && (
            <section className="bg-slate-950 border border-slate-800 rounded-2xl overflow-hidden">
              <div className="px-4 py-2.5 border-b border-slate-800 flex items-center gap-2">
                <Terminal className="w-3.5 h-3.5 text-slate-500" />
                <span className="text-[10px] font-bold uppercase text-slate-500 tracking-wider">Training Log</span>
                {(remoteActive || (isRunning && remoteEnabled)) ? (
                  <span className="px-1.5 py-0.5 bg-blue-600/20 text-blue-400 text-[9px] font-bold rounded uppercase tracking-wider">Remote GPU</span>
                ) : isRunning ? (
                  <span className="px-1.5 py-0.5 bg-slate-800 text-slate-500 text-[9px] font-bold rounded uppercase tracking-wider">Local CPU</span>
                ) : null}
                <span className="text-[10px] font-mono text-slate-700 ml-1 truncate max-w-[200px]">{logPath}</span>
                <button
                  onClick={handleCopyLog}
                  className="ml-1 p-1 hover:bg-slate-800 rounded text-slate-600 hover:text-slate-400 transition-colors"
                  title="Copy log to clipboard"
                >
                  {copied ? <Check className="w-3 h-3 text-green-400" /> : <Copy className="w-3 h-3" />}
                </button>
                {isRunning && <span className="ml-auto text-[10px] text-slate-600 italic">live · updates every 3s</span>}
                {isRunning && <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />}
              </div>
              <div
                ref={logRef}
                className="p-4 h-56 overflow-y-auto font-mono text-[11px] text-slate-400 space-y-0.5"
              >
                {logLines.map((line, i) => (
                  <div key={i} className={`whitespace-pre-wrap break-all ${line.startsWith('  ✓') || line.includes('New best') ? 'text-green-400' : line.startsWith('---') || line.startsWith('Epoch') ? 'text-blue-400' : line.includes('Error') || line.includes('❌') ? 'text-red-400' : ''}`}>
                    {line}
                  </div>
                ))}
                {isRunning && <div className="text-slate-600 animate-pulse">▌</div>}
              </div>
            </section>
          )}
        </main>
      </div>
    </div>
  );
};

const ParamInput = ({ label, value, onChange, type = "text", step }: any) => (
  <div className="space-y-1.5">
    <label className="text-[10px] uppercase text-slate-500 font-bold tracking-wider">{label}</label>
    <input
      type={type}
      step={step}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500/50 transition-colors"
    />
  </div>
);
