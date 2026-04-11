import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

interface OptimizationChartProps {
  trajectory: number[];    // loss per iteration
  targetValue?: number;    // optional reference line
  label?: string;          // y-axis label
}

/**
 * OptimizationChart — shows the optimisation loss trajectory for
 * gradient-based inverse design (Phase 4).
 *
 * Shows loss vs iteration with a reference line at zero (optimal).
 */
export const OptimizationChart: React.FC<OptimizationChartProps> = ({
  trajectory,
  targetValue,
  label = 'Physics loss',
}) => {
  if (!trajectory || trajectory.length === 0) {
    return (
      <div className="flex items-center justify-center h-40 text-slate-500 text-sm">
        No optimisation trajectory yet
      </div>
    );
  }

  const data = trajectory.map((loss, i) => ({ iter: i + 1, loss }));

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis
          dataKey="iter"
          tick={{ fill: '#64748b', fontSize: 10 }}
          label={{ value: 'Iteration', position: 'insideBottom', offset: -2,
                   fill: '#475569', fontSize: 10 }}
        />
        <YAxis
          tick={{ fill: '#64748b', fontSize: 10 }}
          tickFormatter={(v) => v.toExponential(1)}
        />
        <Tooltip
          contentStyle={{ background: '#0f172a', border: '1px solid #1e293b',
                          borderRadius: 8, fontSize: 11 }}
          labelStyle={{ color: '#94a3b8' }}
          formatter={(v: number) => [v.toExponential(3), label]}
        />
        {targetValue !== undefined && (
          <ReferenceLine
            y={targetValue}
            stroke="#6366f1"
            strokeDasharray="4 4"
            label={{ value: `Target (${targetValue.toExponential(2)})`, fill: '#6366f1', fontSize: 10 }}
          />
        )}
        <Line
          type="monotone"
          dataKey="loss"
          stroke="#3b82f6"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 3 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};
