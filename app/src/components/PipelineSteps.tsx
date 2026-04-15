import React from 'react';
import { Target, Sparkles, Zap, ChevronRight, FlaskConical } from 'lucide-react';
import { cn } from '@/src/lib/utils';

// ── One unified flow: Set goal → AI designs → Quick estimate → Full simulation
//
// "Full simulation" carries a "via Predict" badge — it's the bridge to Predict page.
// An SVG arc below the icon row shows the feedback loop back to "AI designs".

interface FlowNode {
  icon:       React.ReactNode;
  label:      string;
  sublabel:   string;
  accent:     'violet' | 'amber' | 'blue';
  isAnalyze?: boolean;
}

function getNodes(domain: string, method: string): FlowNode[] {
  const isCFD      = domain !== 'flag_simple';
  const physTarget = isCFD ? 'drag' : 'cloth stress';
  const fastTool   = isCFD ? 'a fast physics formula' : 'a fast trained estimator';

  return [
    {
      icon:     <Target className="w-4 h-4" />,
      label:    'Set your goal',
      sublabel: `Choose the ${physTarget} value you want`,
      accent:   'violet',
    },
    {
      icon:     <Sparkles className="w-4 h-4" />,
      label:    method === 'gradient' ? 'AI optimises' : 'AI proposes',
      sublabel: method === 'gradient'
        ? 'Shapes are iteratively improved toward your target'
        : 'Candidate shapes that should hit your target are generated',
      accent:   'violet',
    },
    {
      icon:     <Zap className="w-4 h-4" />,
      label:    'Quick estimate',
      sublabel: `Each design is scored instantly using ${fastTool}`,
      accent:   'amber',
    },
    {
      icon:     <FlaskConical className="w-4 h-4" />,
      label:    'Full simulation',
      sublabel: 'Analyze any design for a detailed, physics-accurate result with animation',
      accent:   'blue',
      isAnalyze: true,
    },
  ];
}

const ACCENT = {
  violet: {
    idle:   'border-violet-500/50 bg-violet-600/15 text-violet-300',
    active: 'border-violet-400 bg-violet-600/30 text-violet-200 shadow-lg shadow-violet-900/40',
    done:   'border-violet-500/25 bg-violet-900/15 text-violet-400',
    label:  'text-violet-300',
  },
  amber: {
    idle:   'border-amber-500/40 bg-amber-600/10 text-amber-400',
    active: 'border-amber-400 bg-amber-600/25 text-amber-200 shadow-lg shadow-amber-900/30',
    done:   'border-amber-500/25 bg-amber-900/15 text-amber-400',
    label:  'text-amber-300',
  },
  blue: {
    idle:   'border-blue-500/40 bg-blue-600/10 text-blue-400',
    active: 'border-blue-400 bg-blue-600/25 text-blue-200 shadow-lg shadow-blue-900/30',
    done:   'border-blue-500/25 bg-blue-900/15 text-blue-400',
    label:  'text-blue-300',
  },
};

interface PipelineStepsProps {
  domain?:     string;
  method:      string;
  activeStep?: number;
}

export const PipelineSteps: React.FC<PipelineStepsProps> = ({
  domain  = 'cylinder_flow',
  method,
  activeStep,
}) => {
  const nodes = getNodes(domain, method);

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-5">
      <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-5">
        How it works
      </p>

      {/* ── Forward flow ── */}
      <div className="flex flex-col sm:flex-row sm:items-start justify-between">
        {nodes.map((node, idx) => {
          const colors   = ACCENT[node.accent];
          const isActive = activeStep === idx;
          const isDone   = activeStep !== undefined && idx < activeStep;
          const isLast   = idx === nodes.length - 1;
          const ringCls  = isActive ? colors.active : isDone ? colors.done : colors.idle;

          return (
            <React.Fragment key={idx}>
              {/* Node */}
              <div className="flex sm:flex-col items-center gap-3 sm:gap-2 flex-1 min-w-0 px-2">
                <div className={cn(
                  'shrink-0 w-10 h-10 rounded-full border-2 flex items-center justify-center transition-all duration-300',
                  ringCls,
                )}>
                  {node.icon}
                </div>
                <div className="sm:text-center min-w-0">
                  <div className={cn(
                    'text-xs font-semibold leading-tight',
                    isActive ? colors.label : 'text-slate-300',
                  )}>
                    {node.label}
                    {node.isAnalyze && (
                      <span className="ml-1.5 inline-flex items-center px-1.5 py-0.5 rounded
                                       bg-blue-500/15 border border-blue-500/30
                                       text-[9px] font-bold text-blue-400 uppercase tracking-wider align-middle">
                        via Predict
                      </span>
                    )}
                  </div>
                  <div className="text-[10px] text-slate-600 mt-0.5 leading-snug sm:max-w-[105px]">
                    {node.sublabel}
                  </div>
                </div>
              </div>

              {/* Forward connector */}
              {!isLast && (
                <div className="flex sm:flex-none items-center self-center shrink-0">
                  <div className="sm:hidden w-px h-5 bg-slate-700/40 ml-5 my-1" />
                  <ChevronRight className="hidden sm:block w-4 h-4 text-slate-700" />
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>

      {/* ── Loopback arc (desktop only) ──────────────────────────────────────
          4 nodes each take 25% → centres at 12.5%, 37.5%, 62.5%, 87.5%.
          Arc from node 3 (87.5%) back to node 1 (37.5%).
          viewBox="0 0 100 100" preserveAspectRatio="none" scales with container.
          ──────────────────────────────────────────────────────────────────── */}
      <div className="hidden sm:block relative mt-1" style={{ height: '36px' }}>
        <svg
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
          className="absolute inset-0 w-full h-full"
          style={{ overflow: 'visible' }}
        >
          {/* Dashed arc: starts below node 3 (right), curves down, ends below node 1 */}
          <path
            d="M 87.5 5 C 87.5 85, 37.5 85, 37.5 5"
            fill="none"
            stroke="rgba(100,116,139,0.30)"
            strokeWidth="1.5"
            strokeDasharray="4 3"
            vectorEffect="non-scaling-stroke"
          />
          {/* Arrowhead pointing UP at node 1 end (37.5%, top) */}
          <path
            d="M 35.5 12 L 37.5 3 L 39.5 12"
            fill="none"
            stroke="rgba(100,116,139,0.45)"
            strokeWidth="1.5"
            strokeLinejoin="round"
            strokeLinecap="round"
            vectorEffect="non-scaling-stroke"
          />
        </svg>
        {/* "refine" label centred on the arc */}
        <span
          className="absolute text-[9px] text-slate-600 whitespace-nowrap"
          style={{ left: '62.5%', top: '60%', transform: 'translate(-50%, -50%)' }}
        >
          refine with insights
        </span>
      </div>

    </div>
  );
};
