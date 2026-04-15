import React from 'react';
import { Sliders, Wand2, Zap, ShieldCheck, ChevronRight, Repeat2, Activity, CornerDownLeft } from 'lucide-react';
import { cn } from '@/src/lib/utils';

// ── Step definitions ─────────────────────────────────────────────────────────

interface Step {
  icon:        React.ReactNode;
  label:       string;
  description: string;
}

// ── CFD (cylinder_flow) ──────────────────────────────────────────────────────

const STEPS_CFD_SAMPLE: Step[] = [
  {
    icon:        <Sliders className="w-4 h-4" />,
    label:       'Set target',
    description: 'Pick a drag proxy target (e.g. 0.025)',
  },
  {
    icon:        <Wand2 className="w-4 h-4" />,
    label:       'CVAE decoder',
    description: 'Conditioned VAE maps target → cylinder params (cx, cy, r, v_inlet)',
  },
  {
    icon:        <Zap className="w-4 h-4" />,
    label:       'Drag surrogate',
    description: 'MLP predicts drag proxy r·v²/(1-2r/H) — ~1000× faster than GNN',
  },
  {
    icon:        <ShieldCheck className="w-4 h-4" />,
    label:       'OOD check',
    description: 'Param-space k-NN scores each candidate against training distribution',
  },
];

const STEPS_CFD_GRADIENT: Step[] = [
  {
    icon:        <Sliders className="w-4 h-4" />,
    label:       'Set target',
    description: 'Pick a drag proxy target',
  },
  {
    icon:        <Wand2 className="w-4 h-4" />,
    label:       'Latent search',
    description: 'Adam 150 steps × 3 restarts in CVAE latent space z',
  },
  {
    icon:        <Zap className="w-4 h-4" />,
    label:       'Surrogate grad',
    description: '∂drag/∂z through surrogate MLP + CVAE decoder (no GNN in loop)',
  },
  {
    icon:        <Repeat2 className="w-4 h-4" />,
    label:       'Diversify',
    description: 'N candidates sampled around optimal z* with small noise',
  },
  {
    icon:        <ShieldCheck className="w-4 h-4" />,
    label:       'OOD check',
    description: 'Final designs scored against training distribution',
  },
];

// ── Cloth (flag_simple) ──────────────────────────────────────────────────────

const STEPS_CLOTH_SAMPLE: Step[] = [
  {
    icon:        <Sliders className="w-4 h-4" />,
    label:       'Set target',
    description: 'Pick a stress proxy target (e.g. 1.0)',
  },
  {
    icon:        <Wand2 className="w-4 h-4" />,
    label:       'CVAE decoder',
    description: 'Conditioned VAE maps target → PCA cloth pose coordinates',
  },
  {
    icon:        <Zap className="w-4 h-4" />,
    label:       'Stress surrogate',
    description: 'MLP predicts deformation stress from PCA pose — fast approximation',
  },
  {
    icon:        <ShieldCheck className="w-4 h-4" />,
    label:       'Build mesh',
    description: 'PCA⁻¹ + ClothMeshBuilder reconstruct 3-D cloth world positions',
  },
];

const STEPS_CLOTH_GRADIENT: Step[] = [
  {
    icon:        <Sliders className="w-4 h-4" />,
    label:       'Set target',
    description: 'Pick a stress proxy target',
  },
  {
    icon:        <Wand2 className="w-4 h-4" />,
    label:       'Latent search',
    description: 'Adam optimises z in cloth CVAE latent space',
  },
  {
    icon:        <Activity className="w-4 h-4" />,
    label:       'GNN in loop',
    description: 'FlagSimulator runs K-step rollout — stress loss backpropagates through the full cloth GNN',
  },
  {
    icon:        <Repeat2 className="w-4 h-4" />,
    label:       'Diversify',
    description: 'N candidates sampled around optimal z* with small noise',
  },
  {
    icon:        <ShieldCheck className="w-4 h-4" />,
    label:       'Build mesh',
    description: 'PCA⁻¹ + ClothMeshBuilder reconstruct final cloth positions',
  },
];

// ── Coupling diagram ─────────────────────────────────────────────────────────
//
//  ┌─────────────────────────────────────────────────────────────────┐
//  │  GENERATE                                     PREDICT           │
//  │  ┌──────────┐  fast   ┌──────────┐            ┌──────────────┐  │
//  │  │  CVAE /  │ ──────► │surrogate │            │  MeshGraph   │  │
//  │  │ gradient │  proxy  │   MLP    │            │  Nets GNN    │  │
//  │  └──────────┘         └──────────┘            │  (full sim)  │  │
//  │                            │    ◄──Analyze────►│              │  │
//  │                       candidate                └──────────────┘  │
//  └─────────────────────────────────────────────────────────────────┘
//
// Rendered as two labelled boxes connected by forward + back arrows.

const CouplingDiagram: React.FC<{ domain: string }> = ({ domain }) => {
  const isCloth = domain === 'flag_simple';
  return (
    <div className="mt-4 rounded-lg border border-slate-700/60 bg-slate-950/40 p-3">
      <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-3">
        Generate ↔ Predict coupling
      </p>

      {/* Two-column diagram */}
      <div className="flex items-center gap-2 overflow-x-auto">

        {/* ── GENERATE box ── */}
        <div className="shrink-0 rounded-lg border border-violet-500/40 bg-violet-900/10 px-3 py-2 min-w-[110px]">
          <div className="text-[10px] font-bold text-violet-400 uppercase tracking-wider mb-1">Generate</div>
          <div className="text-[10px] text-slate-400 leading-snug">
            CVAE decoder
          </div>
          <div className="flex items-center gap-1 mt-1">
            <div className="w-1.5 h-1.5 rounded-full bg-violet-500/60" />
            <span className="text-[9px] text-slate-500">
              {isCloth ? 'Stress surrogate' : 'Drag surrogate'}
            </span>
          </div>
          <div className="text-[9px] text-slate-600 mt-0.5 italic">fast proxy</div>
        </div>

        {/* ── Forward arrow: Generate → Predict ── */}
        <div className="flex flex-col items-center shrink-0 gap-0.5">
          {/* top arrow: candidate flows right to Predict */}
          <div className="flex items-center gap-0.5">
            <div className="w-10 h-px bg-blue-500/40" />
            <ChevronRight className="w-3 h-3 text-blue-400 shrink-0" />
          </div>
          <span className="text-[9px] text-blue-400/70">Analyze</span>
          {/* bottom arrow: insight loops back */}
          <div className="flex items-center gap-0.5">
            <CornerDownLeft className="w-3 h-3 text-emerald-400/60 shrink-0 scale-x-[-1]" />
            <div className="w-10 h-px bg-emerald-500/30" />
          </div>
          <span className="text-[9px] text-emerald-400/60">full sim result</span>
        </div>

        {/* ── PREDICT box ── */}
        <div className="shrink-0 rounded-lg border border-blue-500/40 bg-blue-900/10 px-3 py-2 min-w-[110px]">
          <div className="text-[10px] font-bold text-blue-400 uppercase tracking-wider mb-1">Predict</div>
          <div className="text-[10px] text-slate-400 leading-snug">
            MeshGraphNets GNN
          </div>
          <div className="flex items-center gap-1 mt-1">
            <div className="w-1.5 h-1.5 rounded-full bg-blue-500/60" />
            <span className="text-[9px] text-slate-500">
              {isCloth ? 'Cloth rollout' : 'CFD rollout'}
            </span>
          </div>
          <div className="text-[9px] text-slate-600 mt-0.5 italic">physics-accurate</div>
        </div>

        {/* ── Spacer + note ── */}
        <div className="ml-2 text-[10px] text-slate-600 leading-snug hidden lg:block max-w-[160px]">
          Surrogate gives a fast estimate during search.{' '}
          <span className="text-slate-500">Analyze</span> runs the full GNN on
          any candidate you want to inspect in detail.
          {isCloth && (
            <><br/><span className="text-amber-500/70 mt-0.5 inline-block">
              ✦ Gradient mode uses the GNN in-the-loop (fully differentiable)
            </span></>
          )}
        </div>
      </div>

      {/* Mobile note */}
      <p className="lg:hidden text-[10px] text-slate-600 mt-2 leading-snug">
        Surrogate gives fast estimates during search.
        <span className="font-mono text-slate-500"> Analyze</span> runs the full GNN rollout.
        {isCloth && <span className="text-amber-500/70"> ✦ Gradient uses GNN in-loop.</span>}
      </p>
    </div>
  );
};

// ── Props ────────────────────────────────────────────────────────────────────

interface PipelineStepsProps {
  domain?:     string;   // "cylinder_flow" | "flag_simple"
  method:      string;   // "sample" | "gradient"
  activeStep?: number;   // 0-based index of currently running step
}

// ── Component ────────────────────────────────────────────────────────────────

export const PipelineSteps: React.FC<PipelineStepsProps> = ({
  domain = 'cylinder_flow',
  method,
  activeStep,
}) => {
  const isCloth    = domain === 'flag_simple';
  const isGradient = method === 'gradient';

  const steps = isCloth
    ? (isGradient ? STEPS_CLOTH_GRADIENT : STEPS_CLOTH_SAMPLE)
    : (isGradient ? STEPS_CFD_GRADIENT   : STEPS_CFD_SAMPLE);

  const subtitle = isCloth
    ? (isGradient ? 'Cloth — gradient descent (GNN in loop)' : 'Cloth — CVAE sampling (instant)')
    : (isGradient ? 'CFD — gradient descent (surrogate in loop)' : 'CFD — CVAE sampling (instant)');

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-5">
      <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-4">
        How it works — {subtitle}
      </p>

      {/* Horizontal strip on wide screens, vertical list on narrow */}
      <div className="flex flex-col sm:flex-row sm:items-start gap-0 sm:gap-0">
        {steps.map((step, idx) => {
          const isActive = activeStep === idx;
          const isDone   = activeStep !== undefined && idx < activeStep;
          const isLast   = idx === steps.length - 1;

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
                    isActive ? "text-violet-300"
                    : isDone ? "text-emerald-400"
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

      {/* Generate ↔ Predict coupling diagram */}
      <CouplingDiagram domain={domain} />
    </div>
  );
};
