import React, { useState } from 'react';
import { cn } from '@/src/lib/utils';
import { AlertTriangle, CheckCircle, PlayCircle, Loader2 } from 'lucide-react';

interface CandidateCardProps {
  id: number;
  domain: string;
  predictedValue: number;
  targetValue: number;
  oodConfidence: number;
  isOod: boolean;
  meshNodes: number;
  params: Record<string, number>;
  thumbnailUrl?: string | null;
  isSelected?: boolean;
  onSelect?: () => void;
  // Analyze button
  sessionId?: string | null;
  onAnalyze?: (sessionId: string, candidateId: number) => Promise<void>;
}

/**
 * CandidateCard — displays a single generated design candidate.
 *
 * Shows:
 *   - Mesh thumbnail (PNG from /api/generate/thumbnail/...)
 *   - Predicted physics value vs target (with % error)
 *   - OOD badge (confidence score)
 *   - Key design parameters
 */
export const CandidateCard: React.FC<CandidateCardProps> = ({
  id,
  domain,
  predictedValue,
  targetValue,
  oodConfidence,
  isOod,
  meshNodes,
  params,
  thumbnailUrl,
  isSelected = false,
  onSelect,
  sessionId,
  onAnalyze,
}) => {
  const [analyzing, setAnalyzing] = useState(false);

  const handleAnalyze = async (e: React.MouseEvent) => {
    e.stopPropagation();  // don't fire onSelect
    if (!sessionId || !onAnalyze || analyzing) return;
    setAnalyzing(true);
    try {
      await onAnalyze(sessionId, id);
    } finally {
      setAnalyzing(false);
    }
  };
  const pctError = targetValue !== 0
    ? ((Math.abs(predictedValue - targetValue) / Math.abs(targetValue)) * 100).toFixed(1)
    : null;  // null when target is 0 — avoids NaN from parseFloat("—")

  const physLabel    = domain === 'cylinder_flow' ? 'Drag proxy' : 'Stress proxy';
  const hasConf      = oodConfidence >= 0;
  const confPct      = hasConf ? (oodConfidence * 100).toFixed(0) : null;

  return (
    <div
      onClick={onSelect}
      className={cn(
        "rounded-xl border transition-all duration-200 overflow-hidden",
        isSelected
          ? "border-blue-500/60 bg-blue-950/30 shadow-lg shadow-blue-900/20"
          : "border-slate-700/50 bg-slate-900/60 hover:border-slate-600",
        onSelect ? "cursor-pointer" : "cursor-default"
      )}
    >
      {/* Thumbnail */}
      <div className="relative h-32 bg-slate-950 flex items-center justify-center border-b border-slate-800">
        {thumbnailUrl ? (
          <img
            src={thumbnailUrl}
            alt={`Candidate ${id} mesh`}
            className="h-full w-full object-cover"
          />
        ) : (
          <div className="text-slate-600 text-xs">No preview</div>
        )}

        {/* OOD badge overlay */}
        <div className={cn(
          "absolute top-2 right-2 flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold",
          !hasConf
            ? "bg-slate-700/60 border border-slate-600/40 text-slate-400"
            : isOod
              ? "bg-amber-500/20 border border-amber-500/40 text-amber-400"
              : "bg-emerald-500/20 border border-emerald-500/40 text-emerald-400"
        )}>
          {!hasConf
            ? <>N/A conf</>
            : isOod
              ? <><AlertTriangle className="w-2.5 h-2.5" /> OOD</>
              : <><CheckCircle className="w-2.5 h-2.5" /> {confPct}% conf</>
          }
        </div>

        {/* Candidate index */}
        <div className="absolute top-2 left-2 w-5 h-5 rounded-full bg-slate-800/80 flex items-center justify-center text-[10px] text-slate-400 font-mono">
          {id + 1}
        </div>
      </div>

      {/* Metrics */}
      <div className="p-3 space-y-2">
        {/* Physics score */}
        <div className="flex justify-between text-xs">
          <span className="text-slate-400">{physLabel}</span>
          <span className="text-slate-200 font-mono">{predictedValue.toFixed(4)}</span>
        </div>

        {/* Mesh nodes */}
        <div className="flex items-center justify-between text-[11px]">
          <span className="text-slate-500">Mesh nodes</span>
          <span className="text-slate-300 font-mono">{meshNodes.toLocaleString()}</span>
        </div>

        {/* Domain params (first 2) */}
        {Object.entries(params).slice(0, 2).map(([key, val]) => (
          <div key={key} className="flex items-center justify-between text-[11px]">
            <span className="text-slate-500">{key}</span>
            <span className="text-slate-300 font-mono">{typeof val === 'number' ? val.toFixed(4) : val}</span>
          </div>
        ))}

        {/* Analyze button — only shown when sessionId available */}
        {sessionId && onAnalyze && (
          <button
            onClick={handleAnalyze}
            disabled={analyzing}
            className={cn(
              "mt-1 w-full flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-lg",
              "text-[11px] font-medium transition-colors",
              analyzing
                ? "bg-violet-900/30 text-violet-400 cursor-wait"
                : "bg-violet-600/20 hover:bg-violet-600/40 text-violet-300 border border-violet-500/30"
            )}
          >
            {analyzing
              ? <><Loader2 className="w-3 h-3 animate-spin" /> Running rollout…</>
              : <><PlayCircle className="w-3 h-3" /> Analyze</>
            }
          </button>
        )}
      </div>
    </div>
  );
};
