import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface MeshPlotProps {
  crds: [number, number][];
  triangles: [number, number, number][];
  values: number[];
  title: string;
  minVal?: number;
  maxVal?: number;
  colorScale?: (v: number) => string;
  /** Domain string — used to conditionally render domain-specific overlays. */
  domain?: string;
  /** When true, vMax is computed from data instead of using maxVal prop */
  autoScale?: boolean;
}

export const MeshPlot: React.FC<MeshPlotProps> = ({
  crds, triangles, values, title, minVal, maxVal,
  colorScale: customColorScale, domain,
  autoScale = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !containerRef.current || !crds.length || !values.length) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Set canvas size with high DPI support
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.scale(dpr, dpr);

    // Domain bounds — derive from mesh coordinates so cloth and CFD both render correctly.
    // For cylinder_flow use fixed [0,1.6]×[0,0.41] so aspect ratio is stable across frames.
    // For all other domains (cloth, etc.) compute from actual crds extents.
    let xExtent: [number, number];
    let yExtent: [number, number];

    if (domain === 'cylinder_flow') {
      xExtent = [0, 1.6];
      yExtent = [0, 0.41];
    } else {
      const xs = crds.map(c => c[0]);
      const ys = crds.map(c => c[1]);
      const xMin = d3.min(xs) ?? 0;
      const xMax = d3.max(xs) ?? 1;
      const yMin = d3.min(ys) ?? 0;
      const yMax = d3.max(ys) ?? 1;
      // Add 5% padding so the mesh isn't clipped at the edges
      const xPad = (xMax - xMin) * 0.05 || 0.05;
      const yPad = (yMax - yMin) * 0.05 || 0.05;
      xExtent = [xMin - xPad, xMax + xPad];
      yExtent = [yMin - yPad, yMax + yPad];
    }

    // Padding
    const padding = 20;
    const innerWidth = width - padding * 2;
    const innerHeight = height - padding * 2;

    // Aspect ratio correction
    const targetAspect = (xExtent[1] - xExtent[0]) / (yExtent[1] - yExtent[0]);
    const currentAspect = innerWidth / innerHeight;

    let drawWidth = innerWidth;
    let drawHeight = innerHeight;

    if (currentAspect > targetAspect) {
      drawWidth = innerHeight * targetAspect;
    } else {
      drawHeight = innerWidth / targetAspect;
    }

    const offsetX = (width - drawWidth) / 2;
    const offsetY = (height - drawHeight) / 2;

    const xScale = d3.scaleLinear().domain(xExtent).range([offsetX, offsetX + drawWidth]);
    const yScale = d3.scaleLinear().domain(yExtent).range([offsetY + drawHeight, offsetY]); // Flip Y

    // Color scale
    const vMin = minVal ?? d3.min(values) ?? 0;
    const vMax = autoScale
      ? (d3.max(values) ?? 1.5)
      : (maxVal ?? d3.max(values) ?? 1.5);
    const colorScale = customColorScale ?? d3.scaleSequential(d3.interpolateTurbo).domain([vMin, vMax]);

    // Clear
    ctx.clearRect(0, 0, width, height);

    // Draw triangles
    triangles.forEach(([i1, i2, i3]) => {
      const v1 = values[i1];
      const v2 = values[i2];
      const v3 = values[i3];

      const x1 = xScale(crds[i1][0]), y1 = yScale(crds[i1][1]);
      const x2 = xScale(crds[i2][0]), y2 = yScale(crds[i2][1]);
      const x3 = xScale(crds[i3][0]), y3 = yScale(crds[i3][1]);

      // Centroid and edge midpoints for smooth-shading approximation
      const mx12 = (x1 + x2) / 2, my12 = (y1 + y2) / 2, mv12 = (v1 + v2) / 2;
      const mx23 = (x2 + x3) / 2, my23 = (y2 + y3) / 2, mv23 = (v2 + v3) / 2;
      const mx13 = (x1 + x3) / 2, my13 = (y1 + y3) / 2, mv13 = (v1 + v3) / 2;

      // Simpler: draw the 3 corner sub-triangles + 1 middle
      const subTriangles: [number, number, number, number, number, number, number][] = [
        [x1, y1, mx12, my12, mx13, my13, (v1 + mv12 + mv13) / 3],
        [mx12, my12, x2, y2, mx23, my23, (mv12 + v2 + mv23) / 3],
        [mx13, my13, mx23, my23, x3, y3, (mv13 + mv23 + v3) / 3],
        [mx12, my12, mx23, my23, mx13, my13, (mv12 + mv23 + mv13) / 3],
      ];

      subTriangles.forEach(([ax, ay, bx, by, ccx, ccy, val]) => {
        ctx.fillStyle = colorScale(val);
        ctx.strokeStyle = colorScale(val);
        ctx.lineWidth = 0.3;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.lineTo(ccx, ccy);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      });
    });

    // Draw colorbar on right side (only when there's enough horizontal space)
    const cbWidth = 12;
    const cbHeight = Math.min(drawHeight * 0.7, 150);
    const cbX = offsetX + drawWidth + 8;
    const cbY = offsetY + (drawHeight - cbHeight) / 2;

    if (cbX + cbWidth + 35 <= width) {
      // Gradient bar
      const grad = ctx.createLinearGradient(cbX, cbY + cbHeight, cbX, cbY);
      const nStops = 10;
      for (let s = 0; s <= nStops; s++) {
        const frac = s / nStops;
        const val  = vMin + frac * (vMax - vMin);
        grad.addColorStop(frac, colorScale(val));
      }
      ctx.fillStyle = grad;
      ctx.fillRect(cbX, cbY, cbWidth, cbHeight);
      ctx.strokeStyle = '#475569';
      ctx.lineWidth = 0.5;
      ctx.strokeRect(cbX, cbY, cbWidth, cbHeight);

      // Labels
      ctx.fillStyle = '#94a3b8';
      ctx.font = '9px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(vMax.toFixed(3), cbX + cbWidth + 3, cbY + 9);
      ctx.fillText(vMin.toFixed(3), cbX + cbWidth + 3, cbY + cbHeight);
    }

  }, [crds, triangles, values, minVal, maxVal, customColorScale, domain, autoScale]);

  return (
    <div className="flex flex-col h-full w-full bg-slate-900/50 rounded-lg border border-slate-700 overflow-hidden">
      <div className="px-3 py-2 border-b border-slate-700 bg-slate-800/50 flex justify-between items-center">
        <span className="text-xs font-medium text-slate-300 uppercase tracking-wider">{title}</span>
      </div>
      <div ref={containerRef} className="flex-1 relative min-h-[200px]">
        <canvas ref={canvasRef} className="absolute inset-0" />
      </div>
    </div>
  );
};

