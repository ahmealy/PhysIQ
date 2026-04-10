import React from 'react';
import { Sliders, Wand2, Grid3X3, Cpu, ShieldCheck, ChevronRight } from 'lucide-react';
import { cn } from '@/src/lib/utils';

// ── Step definitions ─────────────────────────────────────────────────────────

interface Step {
  icon:        React.ReactNode;
  label:       string;
  description: string;
}

const STEPS_SAMPLE: Step[] = [
  {
    icon:        <Sliders className="w-4 h-4" />,
    label:       'Set target',
    description: 'You pick a physics goal (e.g. drag = 0.025)',
  },
  {
    icon:        <Wand2 className="w-4 h-4" />,
    label:       'Generate design',
    description: 'AI samples candidate shape parameters that should hit the target',
  },
  {
    icon:        <Grid3X3 className="w-4 h-4" />,
    label:       'Build mesh',
    description: 'The shape is turned into a simulation-ready mesh',
  },
  {
    icon:        <Cpu className="w-4 h-4" />,
    label:       'Run simulation',
    description: 'MeshGraphNets predicts the physics field (velocity / cloth motion)',
  },
  {
    icon:        <ShieldCheck className="w-4 h-4" />,
    label:       'Check reliability',
    description: 'Each result is scored — flagged if the AI is uncertain',
  },
];

const STEPS_GRADIENT: Step[] = [
  {
    icon:        <Sliders className="w-4 h-4" />,
    label:       'Set target',
    description: 'You pick a physics goal',
  },
  {
    icon:        <Wand2 className="w-4 h-4" />,
    label:       'Start from random',
    description: 'AI picks a random starting design in latent space',
  },
  {
    icon:        <Cpu className="w-4 h-4" />,
    label:       'Run simulation',
    description: 'Predict physics for this design',
  },
  {
    icon:        <Grid3X3 className="w-4 h-4" />,
    label:       'Adjust design',
    description: 'Tweak the design in the direction that reduces the error (gradient step)',
  },
  {
    icon:        <ChevronRight className="w-4 h-4 text-violet-400" />,
    label:       'Repeat × 100',
    description: 'Keep improving until the design converges on your target',
  },
  {
    icon:        <ShieldCheck className="w-4 h-4" />,
    label:       'Check reliability',
    description: 'Final design scored for confidence',
  },
];

// ── Props ────────────────────────────────────────────────────────────────────

interface PipelineStepsProps {
  method:    string;   // "sample" | "gradient"
  activeStep?: number; // 0-based index of the currently running step (optional)
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
        How it works — {method === 'gradient' ? 'Gradient descent (iterative)' : 'AI sampling (instant)'}
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
    </div>
  );
};
