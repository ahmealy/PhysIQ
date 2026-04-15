import React from 'react';
import { Sliders, Wand2, Grid3X3, Zap, ShieldCheck, ChevronRight, Repeat2, Eye } from 'lucide-react';
import { cn } from '@/src/lib/utils';

// ── Step definitions ─────────────────────────────────────────────────────────

interface Step {
  icon:        React.ReactNode;
  label:       string;
  description: string;
}

// CVAE sampling path
// step indices: 0=Set target, 1=CVAE decode, 2=Surrogate predict, 3=OOD check
const STEPS_SAMPLE: Step[] = [
  {
    icon:        <Sliders className="w-4 h-4" />,
    label:       'Set target',
    description: 'Pick a physics goal (e.g. drag = 0.025)',
  },
  {
    icon:        <Wand2 className="w-4 h-4" />,
    label:       'CVAE decoder',
    description: 'Conditioned variational auto-encoder maps target → design params (cx, cy, r, v)',
  },
  {
    icon:        <Zap className="w-4 h-4" />,
    label:       'Surrogate predict',
    description: 'Lightweight MLP predicts drag/stress from params — ~1000× faster than a full GNN rollout',
  },
  {
    icon:        <ShieldCheck className="w-4 h-4" />,
    label:       'OOD check',
    description: 'Each candidate is scored against the training distribution — flagged if the AI is uncertain',
  },
];

// Gradient descent path
// step indices: 0=Set target, 1=Latent opt, 2=Surrogate loss, 3=Render, 4=OOD check
const STEPS_GRADIENT: Step[] = [
  {
    icon:        <Sliders className="w-4 h-4" />,
    label:       'Set target',
    description: 'Pick a physics goal',
  },
  {
    icon:        <Wand2 className="w-4 h-4" />,
    label:       'Latent optimisation',
    description: 'Adam runs 150 steps × 3 restarts in the CVAE latent space, guided by surrogate gradients',
  },
  {
    icon:        <Zap className="w-4 h-4" />,
    label:       'Surrogate gradient',
    description: 'Differentiable MLP loss ∂drag/∂z flows back through the CVAE decoder — no GNN needed',
  },
  {
    icon:        <Repeat2 className="w-4 h-4" />,
    label:       'Diversify',
    description: 'N candidates sampled around z* (optimal point) with small noise to cover the solution space',
  },
  {
    icon:        <ShieldCheck className="w-4 h-4" />,
    label:       'OOD check',
    description: 'Final designs scored for confidence',
  },
];

// Coupling note shown below both pipelines
const COUPLING_NOTE = (
  <div className="mt-4 flex items-start gap-2.5 rounded-lg bg-blue-950/30 border border-blue-500/20 px-3 py-2.5 text-[11px] text-blue-300/80">
    <Eye className="w-3.5 h-3.5 mt-0.5 text-blue-400 shrink-0" />
    <span>
      <span className="font-semibold text-blue-300">Generate → Predict coupling:</span>
      {' '}The surrogate gives a fast physics estimate during design search.
      Click <span className="font-mono bg-blue-900/40 px-1 rounded">Analyze</span> on any
      candidate to run a full <span className="font-semibold">MeshGraphNets GNN rollout</span> in
      Predict — this is the physics-accurate simulation and produces the animated velocity / cloth field.
    </span>
  </div>
);

// ── Props ────────────────────────────────────────────────────────────────────

interface PipelineStepsProps {
  method:      string;   // "sample" | "gradient"
  activeStep?: number;   // 0-based index of the currently running step (optional)
}

// ── Component ────────────────────────────────────────────────────────────────

export const PipelineSteps: React.FC<PipelineStepsProps> = ({
  method,
  activeStep,
}) => {
  const steps = method === 'gradient' ? STEPS_GRADIENT : STEPS_SAMPLE;

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-5">
      <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-4">
        How it works — {method === 'gradient' ? 'Gradient descent (iterative optimisation)' : 'CVAE sampling (instant)'}
      </p>

      {/* Horizontal strip on wide screens, vertical list on narrow */}
      <div className="flex flex-col sm:flex-row sm:items-start gap-0 sm:gap-0">
        {steps.map((step, idx) => {
          const isActive   = activeStep === idx;
          const isDone     = activeStep !== undefined && idx < activeStep;
          const isLast     = idx === steps.length - 1;

          return (
            <React.Fragment key={idx}>
              {/* Step bubble */}
              <div className="flex sm:flex-col items-start sm:items-center gap-3 sm:gap-2 flex-1 min-w-0">
                {/* Icon circle */}
                <div className={cn(
                  "shrink-0 w-9 h-9 rounded-full border flex items-center justify-center transition-all duration-300",
                  isActive
                    ? "border-violet-500 bg-violet-600/30 text-violet-300 shadow-lg shadow-violet-900/30"
                    : isDone
                      ? "border-emerald-500/50 bg-emerald-900/20 text-emerald-400"
                      : "border-slate-700 bg-slate-800/60 text-slate-500"
                )}>
                  {step.icon}
                </div>

                {/* Text */}
                <div className="sm:text-center min-w-0">
                  <div className={cn(
                    "text-xs font-semibold leading-tight",
                    isActive  ? "text-violet-300"
                    : isDone  ? "text-emerald-400"
                    : "text-slate-400"
                  )}>
                    {step.label}
                  </div>
                  <div className="text-[10px] text-slate-600 mt-0.5 leading-snug sm:max-w-[100px]">
                    {step.description}
                  </div>
                </div>
              </div>

              {/* Connector arrow (not after last) */}
              {!isLast && (
                <div className="flex sm:flex-none items-center self-stretch sm:self-center">
                  {/* Vertical line on mobile, horizontal on desktop */}
                  <div className={cn(
                    "sm:hidden w-px h-6 ml-4 my-1",
                    isDone ? "bg-emerald-500/30" : "bg-slate-700/60"
                  )} />
                  <ChevronRight className={cn(
                    "hidden sm:block w-4 h-4 shrink-0 mx-1",
                    isDone ? "text-emerald-500/40" : "text-slate-700"
                  )} />
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>

      {/* Generate → Predict coupling note */}
      {COUPLING_NOTE}
    </div>
  );
};
