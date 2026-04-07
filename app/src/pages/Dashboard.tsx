import React, { useEffect, useState } from 'react';
import { Activity, Database, Cpu, HardDrive, CheckCircle2, AlertCircle, ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import { fetchWithRetry } from '../utils/fetch';

export const Dashboard: React.FC = () => {
  const [status, setStatus] = useState<any>(null);
  const [checkpoint, setCheckpoint] = useState<any>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async (isInitial = false) => {
      try {
        const r = await fetchWithRetry('/api/status', isInitial ? 6 : 1, 1500);
        if (!cancelled && r.ok) {
          const d = await r.json();
          setStatus(d);
          setLoadError(null);
        }
      } catch {
        if (!cancelled) setLoadError('Backend not reachable. Retrying…');
      }
      try {
        const r = await fetchWithRetry('/api/checkpoint', isInitial ? 4 : 1, 1500);
        if (!cancelled) setCheckpoint(r.ok ? await r.json() : null);
      } catch {
        if (!cancelled) setCheckpoint(null);
      }
    };
    load(true);
    // Refresh every 10s so training_running updates without page reload
    const t = setInterval(() => load(false), 10000);
    return () => { cancelled = true; clearInterval(t); };
  }, []);

  if (!status) return (
    <div className="p-8 flex items-center gap-3 text-slate-400">
      <svg className="w-5 h-5 animate-spin text-blue-400" fill="none" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
      </svg>
      <span>{loadError ?? 'Loading system status…'}</span>
    </div>
  );

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      <header className="flex justify-between items-end">
        <div>
          <h2 className="text-3xl font-bold text-white tracking-tight">System Overview</h2>
          <p className="text-slate-400 mt-2">MeshGraphNets Physics AI Engine Status</p>
        </div>
        <div className="flex gap-4">
          <div className="px-4 py-2 bg-slate-900 border border-slate-800 rounded-lg flex items-center gap-2">
            <Cpu className="w-4 h-4 text-blue-400" />
            <span className="text-sm font-medium text-slate-300">{status.gpu_name}</span>
          </div>
          <div className="px-4 py-2 bg-slate-900 border border-slate-800 rounded-lg flex items-center gap-2">
            <div className={status.gpu_available ? "w-2 h-2 bg-green-500 rounded-full" : "w-2 h-2 bg-red-500 rounded-full"} />
            <span className="text-sm font-medium text-slate-300">{status.gpu_available ? 'GPU Ready' : 'CPU Only'}</span>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard 
          icon={Activity} 
          label="Training Status" 
          value={status.training_running ? "Running" : "Idle"} 
          color={status.training_running ? "text-blue-400" : "text-slate-400"}
        />
        <StatCard
          icon={CheckCircle2}
          label="Best Valid Loss"
          value={status.checkpoint_valid_loss != null ? status.checkpoint_valid_loss.toFixed(5) : '—'}
          color="text-green-400"
        />
        <StatCard 
          icon={Database} 
          label="Saved Rollouts" 
          value={status.saved_rollouts} 
          color="text-purple-400"
        />
        <StatCard
          icon={HardDrive}
          label="Model Size"
          value={status.checkpoint_size_mb != null ? `${status.checkpoint_size_mb} MB` : '—'}
          color="text-orange-400"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-6">
          <section className="bg-slate-900/50 border border-slate-800 rounded-2xl overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/80 flex justify-between items-center">
              <h3 className="font-semibold text-white">Available Simulation Domains</h3>
            </div>
            <div className="p-6 space-y-4">
              {Object.entries(status.domains).map(([key, domain]: [string, any]) => (
                <div key={key} className="flex items-center justify-between p-4 bg-slate-950 rounded-xl border border-slate-800 hover:border-blue-500/30 transition-colors group">
                  <div className="flex items-center gap-4">
                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${domain.available ? 'bg-blue-600/10 text-blue-400' : 'bg-slate-800 text-slate-500'}`}>
                      <Database className="w-5 h-5" />
                    </div>
                    <div>
                      <h4 className="font-medium text-slate-200">{domain.label}</h4>
                      <p className="text-xs text-slate-500">{domain.description}</p>
                    </div>
                  </div>
                  {domain.available ? (
                    <Link to="/predict" className="p-2 text-slate-500 hover:text-blue-400 transition-colors">
                      <ArrowRight className="w-5 h-5" />
                    </Link>
                  ) : (
                    <span className="text-[10px] uppercase font-bold text-slate-600 bg-slate-900 px-2 py-1 rounded">Coming Soon</span>
                  )}
                </div>
              ))}
            </div>
          </section>
        </div>

        <div className="space-y-6">
          <section className="bg-slate-900/50 border border-slate-800 rounded-2xl overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/80">
              <h3 className="font-semibold text-white">Model Checkpoint</h3>
            </div>
            <div className="p-6">
              {checkpoint ? (
                <div className="space-y-4">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-500">Epoch</span>
                    <span className="text-slate-200 font-mono">{checkpoint.epoch}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-500">Validation Loss</span>
                    <span className="text-green-400 font-mono">{checkpoint.valid_loss.toFixed(6)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-500">Last Modified</span>
                    <span className="text-slate-400 text-xs">{new Date(checkpoint.last_modified).toLocaleString()}</span>
                  </div>
                  <div className="pt-4">
                    <Link to="/train" className="block w-full py-2 bg-blue-600 hover:bg-blue-500 text-white text-center rounded-lg text-sm font-medium transition-colors">
                      Resume Training
                    </Link>
                  </div>
                </div>
              ) : (
                <div className="flex items-center gap-2 text-slate-500 text-sm">
                  <AlertCircle className="w-4 h-4" />
                  <span>No checkpoint found</span>
                </div>
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};

const StatCard = ({ icon: Icon, label, value, color }: any) => (
  <div className="bg-slate-900/50 border border-slate-800 p-6 rounded-2xl space-y-3">
    <div className={`w-10 h-10 rounded-xl bg-slate-950 flex items-center justify-center border border-slate-800 ${color}`}>
      <Icon className="w-5 h-5" />
    </div>
    <div>
      <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">{label}</p>
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
    </div>
  </div>
);
