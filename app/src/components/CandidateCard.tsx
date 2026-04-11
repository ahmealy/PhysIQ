import React from 'react';
import { cn } from '@/src/lib/utils';
import { Zap, AlertTriangle, CheckCircle } from 'lucide-react';

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
  // Deep mode props
  mode?: 'quick' | 'deep';
  gnnPredictedValue?: number | null;
  scoreGap?: number | null;
  gnnConverged?: boolean | null;
  gnnFailed?: boolean;
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
  mode = 'quick',
  gnnPredictedValue,
  scoreGap,
  gnnConverged,
  gnnFailed,
}) => {
  const pctError = targetValue !== 0
    ? ((Math.abs(predictedValue - targetValue) / Math.abs(targetValue)) * 100).toFixed(1)
    : null;  // null when target is 0 — avoids NaN from parseFloat("—")

  const physLabel    = domain === 'cylinder_flow' ? 'Drag proxy' : 'Stress proxy';
  const hasConf      = oodConfidence >= 0;
  const confPct      = hasConf ? (oodConfidence * 100).toFixed(0) : null;

  // Deep mode — GNN score display helpers
  const isDeep   = mode === 'deep';
  const hasGnn   = isDeep && gnnPredictedValue != null && !gnnFailed;
  const gnnLabel = hasGnn
    ? `${gnnConverged === false ? '~' : ''}${gnnPredictedValue!.toFixed(4)}`
    : null;
  const gapColor = scoreGap == null  ? ''
    : scoreGap < 0.1  ? 'text-emerald-400'
    : scoreGap < 0.2  ? 'text-amber-400'
    : 'text-red-400';
  const gapDot   = scoreGap == null  ? ''
    : scoreGap < 0.1  ? '🟢'
    : scoreGap < 0.2  ? '🟡'
    : '🔴';

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
        {/* Physics score row(s) */}
        {isDeep ? (
          <>
            <div className="flex justify-between text-xs">
              <span className="text-slate-400">Surrogate</span>
              <span className="text-slate-200 font-mono">{predictedValue.toFixed(4)}</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-slate-400">GNN</span>
              {hasGnn ? (
                <span className="text-violet-300 font-mono">{gnnLabel} ✓</span>
              ) : gnnFailed ? (
                <span className="text-slate-500 italic">scoring failed</span>
              ) : (
                <span className="text-slate-500">—</span>
              )}
            </div>
            {hasGnn && scoreGap != null && (
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Gap</span>
                <span className={cn('font-mono', gapColor)}>
                  {scoreGap.toFixed(4)} {gapDot}
                </span>
              </div>
            )}
          </>
        ) : (
          <div className="flex justify-between text-xs">
            <span className="text-slate-400">{physLabel}</span>
            <span className="text-slate-200 font-mono">{predictedValue.toFixed(4)}</span>
          </div>
        )}

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
      </div>
    </div>
  );
};
