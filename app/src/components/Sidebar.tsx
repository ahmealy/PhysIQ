import React, { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Activity, PlayCircle, BarChart3, Database, Cpu, Sparkles, FlaskConical } from 'lucide-react';
import { cn } from '@/src/lib/utils';

const navItems = [
  { icon: LayoutDashboard, label: 'Dashboard',            path: '/' },
  { icon: Database,        label: 'Dataset Studio',       path: '/dataset' },
  { icon: Activity,        label: 'Train',                path: '/train' },
  { icon: PlayCircle,      label: 'Predict',              path: '/predict' },
  { icon: BarChart3,       label: 'Visualize',            path: '/visualize' },
  { icon: Sparkles,        label: 'Generate',             path: '/generate' },
  { icon: FlaskConical,    label: 'Pipeline & Experiments', path: '/pipeline' },
];

export const Sidebar: React.FC = () => {
  const [systemStatus, setSystemStatus] = useState<any>(null);

  useEffect(() => {
    const load = () => {
      fetch('/api/status')
        .then(r => r.ok ? r.json() : null)
        .then(d => d && setSystemStatus(d))
        .catch(() => {});
    };
    load();
    // Refresh every 15s so GPU / training status stays current
    const t = setInterval(load, 15000);
    return () => clearInterval(t);
  }, []);

  const gpuLabel = systemStatus?.gpu_available
    ? (systemStatus.gpu_name ?? 'GPU Ready')
    : systemStatus
      ? 'CPU Only'
      : '…';

  const isTraining = systemStatus?.training_running ?? false;

  return (
    <aside className="w-64 bg-slate-950 border-r border-slate-800 flex flex-col h-screen sticky top-0">
      <div className="p-6">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center shadow-lg shadow-blue-900/20">
            <Database className="text-white w-5 h-5" />
          </div>
          <h1 className="text-lg font-bold text-white tracking-tight">MeshGraphNets</h1>
        </div>

        <nav className="space-y-1">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200",
                  isActive
                    ? "bg-blue-600/10 text-blue-400 border border-blue-500/20"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
                )
              }
            >
              <item.icon className="w-4 h-4" />
              {item.label}
            </NavLink>
          ))}
        </nav>
      </div>

      <div className="mt-auto p-6 border-t border-slate-900 space-y-3">
        {/* GPU status — live from API */}
        <div className="px-4 py-3 bg-slate-900 rounded-xl space-y-2">
          <div className="text-[9px] uppercase text-slate-500 font-bold tracking-wider">System</div>
          <div className="flex items-center gap-2">
            <Cpu className="w-3.5 h-3.5 text-slate-500 shrink-0" />
            <span className="text-xs text-slate-300 truncate">{gpuLabel}</span>
            <div className={cn(
              "ml-auto w-2 h-2 rounded-full shrink-0",
              systemStatus?.gpu_available ? "bg-green-500" : "bg-yellow-500"
            )} />
          </div>
          {isTraining && (
            <div className="flex items-center gap-2 text-[10px] text-blue-400">
              <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse" />
              Training in progress
            </div>
          )}
        </div>
      </div>
    </aside>
  );
};
