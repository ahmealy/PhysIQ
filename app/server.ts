import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import { fileURLToPath } from "url";
import http from "http";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// When FASTAPI_URL is set (e.g. http://localhost:8000), all /api/* requests
// are forwarded to the real FastAPI backend instead of served by the mock.
const FASTAPI_URL = process.env.FASTAPI_URL;

/** Forward an Express request to the FastAPI backend and pipe the response back. */
function proxyToFastAPI(req: express.Request, res: express.Response): void {
  const target = new URL(FASTAPI_URL!);
  const options: http.RequestOptions = {
    hostname: target.hostname,
    port: parseInt(target.port || "8000"),
    path: req.originalUrl,
    method: req.method,
    headers: { ...req.headers, host: target.host },
  };

  const proxy = http.request(options, (proxyRes) => {
    res.writeHead(proxyRes.statusCode ?? 502, proxyRes.headers);
    proxyRes.pipe(res, { end: true });
  });

  proxy.on("error", (err) => {
    console.error("[proxy] FastAPI unreachable:", err.message);
    res.status(502).json({ error: "FastAPI backend unreachable", detail: err.message });
  });

  req.pipe(proxy, { end: true });
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // In proxy mode all /api/* requests go straight to FastAPI — skip mock routes
  if (FASTAPI_URL) {
    console.log(`[server] Proxying /api/* → ${FASTAPI_URL}`);
    app.all("/api/*", proxyToFastAPI);
  }

  // Mock State
  let trainingRunning = false;
  let trainingEpochs: any[] = [
    { epoch: 1, train_loss: 0.148, valid_loss: 0.00379 },
    { epoch: 2, train_loss: 0.0507, valid_loss: 0.00318 },
    { epoch: 3, train_loss: 0.0321, valid_loss: 0.00295 },
    { epoch: 4, train_loss: 0.0284, valid_loss: 0.00271 },
    { epoch: 5, train_loss: 0.0256, valid_loss: 0.00255 },
    { epoch: 6, train_loss: 0.0238, valid_loss: 0.00248 },
    { epoch: 7, train_loss: 0.0221, valid_loss: 0.00239 },
  ];
  let bestEpoch = 7;
  let bestValidLoss = 0.00239;

  const savedRollouts = [
    {
      filename: "result0.pkl",
      path: "result/result0.pkl",
      trajectory_index: 0,
      created: new Date().toISOString(),
      size_mb: 8.2,
    },
  ];

  // --- Endpoints ---

  app.get("/api/status", (req, res) => {
    res.json({
      checkpoint_exists: true,
      checkpoint_epoch: bestEpoch,
      checkpoint_valid_loss: bestValidLoss,
      checkpoint_size_mb: 34.0,
      gpu_available: true,
      gpu_name: "NVIDIA A100",
      training_running: trainingRunning,
      saved_rollouts: savedRollouts.length,
      domains: {
        cylinder_flow: {
          label: "Cylinder Flow (CFD)",
          description: "2D fluid flow past a cylinder",
          available: true,
        },
        flag_simple: {
          label: "Flag Simple (Cloth)",
          description: "3D cloth simulation — deformable mesh",
          available: false,
        },
      },
    });
  });

  app.get("/api/checkpoint", (req, res) => {
    res.json({
      epoch: bestEpoch,
      valid_loss: bestValidLoss,
      size_mb: 34.0,
      path: "checkpoints/best_model.pth",
      last_modified: new Date().toISOString(),
    });
  });

  app.post("/api/train/start", (req, res) => {
    trainingRunning = true;
    res.json({ pid: 12345, status: "started" });
  });

  app.post("/api/train/stop", (req, res) => {
    trainingRunning = false;
    res.json({ status: "stopped" });
  });

  app.get("/api/train/status", (req, res) => {
    res.json({
      running: trainingRunning,
      pid: 12345,
      epochs: trainingEpochs,
      best_epoch: bestEpoch,
      best_valid_loss: bestValidLoss,
    });
  });

  // SSE for training
  app.get("/api/train/stream", (req, res) => {
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");

    let epochCount = trainingEpochs.length;
    const interval = setInterval(() => {
      if (!trainingRunning) {
        res.write(`data: ${JSON.stringify({ type: "done", reason: "stopped" })}\n\n`);
        clearInterval(interval);
        return;
      }

      epochCount++;
      const newLoss = bestValidLoss * (0.95 + Math.random() * 0.1);
      const event = {
        type: "epoch",
        epoch: epochCount,
        train_loss: newLoss * 1.5,
        valid_loss: newLoss,
      };
      trainingEpochs.push(event);
      res.write(`data: ${JSON.stringify(event)}\n\n`);

      if (newLoss < bestValidLoss) {
        bestValidLoss = newLoss;
        bestEpoch = epochCount;
        res.write(`data: ${JSON.stringify({ type: "best", epoch: bestEpoch, valid_loss: bestValidLoss })}\n\n`);
      }

      if (epochCount >= 100) {
        trainingRunning = false;
        res.write(`data: ${JSON.stringify({ type: "done", reason: "completed" })}\n\n`);
        clearInterval(interval);
      }
    }, 2000);

    req.on("close", () => clearInterval(interval));
  });

  app.get("/api/dataset/info", (req, res) => {
    res.json({
      domain: "cylinder_flow",
      split: "test",
      num_trajectories: 100,
      timesteps_per_trajectory: 600,
      dt: 0.01,
      total_samples: 59900,
    });
  });

  app.post("/api/rollout", (req, res) => {
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");

    let step = 0;
    const total = 600;
    const interval = setInterval(() => {
      step += 50;
      if (step <= total) {
        res.write(`data: ${JSON.stringify({ type: "progress", step, total })}\n\n`);
      } else {
        res.write(
          `data: ${JSON.stringify({
            type: "done",
            elapsed_seconds: 4.2,
            speedup: 1.43,
            pkl_path: "result/result0.pkl",
            rmse_final: 0.0087,
          })}\n\n`
        );
        clearInterval(interval);
      }
    }, 500);

    req.on("close", () => clearInterval(interval));
  });

  app.get("/api/results", (req, res) => {
    res.json(savedRollouts);
  });

  // Mock Mesh Data Generation
  const generateMesh = () => {
    const nodes: [number, number][] = [];
    const nx = 50;
    const ny = 20;
    const dx = 1.6 / (nx - 1);
    const dy = 0.4 / (ny - 1);

    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        nodes.push([i * dx, j * dy]);
      }
    }

    const triangles: [number, number, number][] = [];
    for (let j = 0; j < ny - 1; j++) {
      for (let i = 0; i < nx - 1; i++) {
        const p1 = j * nx + i;
        const p2 = j * nx + i + 1;
        const p3 = (j + 1) * nx + i;
        const p4 = (j + 1) * nx + i + 1;
        triangles.push([p1, p2, p3]);
        triangles.push([p2, p4, p3]);
      }
    }

    return { nodes, triangles };
  };

  const mesh = generateMesh();

  app.get("/api/results/:filename", (req, res) => {
    res.json({
      timesteps: 600,
      num_nodes: mesh.nodes.length,
      dt: 0.01,
      crds: mesh.nodes,
      triangles: mesh.triangles,
      per_step_rmse: Array.from({ length: 600 }, (_, i) => 0.0001 + i * 0.000015),
      elapsed_seconds: 4.2,
      speedup: 1.43,
    });
  });

  app.get("/api/results/:filename/frame/:t", (req, res) => {
    const t = parseInt(req.params.t);
    const N = mesh.nodes.length;

    // Generate some fake vortex-like patterns
    const predicted_magnitude = mesh.nodes.map(([x, y]) => {
      const base = 1.0;
      const vortex = Math.sin(x * 10 - t * 0.1) * Math.cos(y * 10) * 0.2;
      return base + vortex;
    });

    const target_magnitude = predicted_magnitude.map((v) => v + (Math.random() - 0.5) * 0.02);
    const error = predicted_magnitude.map((v, i) => Math.abs(v - target_magnitude[i]));

    res.json({
      t,
      time_seconds: t * 0.01,
      predicted_magnitude,
      target_magnitude,
      error,
      rmse: 0.0034 + (t / 600) * 0.005,
    });
  });

  app.get("/api/results/:filename/rmse", (req, res) => {
    const per_step_rmse = Array.from({ length: 600 }, (_, i) => 0.0001 + i * 0.000015);
    res.json({
      per_step_rmse,
      times: Array.from({ length: 600 }, (_, i) => i * 0.01),
      rmse_at_0: per_step_rmse[0],
      rmse_at_300: per_step_rmse[300],
      rmse_at_599: per_step_rmse[599],
      growth_ratio: 87.0,
    });
  });

  app.delete("/api/results/:filename", (req, res) => {
    res.json({ status: "deleted" });
  });

  app.get("/api/status/gpu", (req, res) => {
    res.json({
      mem_alloc_gb: 4.2,
      mem_reserved_gb: 8.5,
      utilization: 65,
    });
  });

  app.get("/api/results/:filename/physics", (req, res) => {
    const t = parseInt(req.query.t as string || "0");
    const N = mesh.nodes.length;
    const T = 600;

    // Mock physics series (energy, divergence)
    const energy_pred_series = Array.from({ length: T }, (_, i) => 10.5 + Math.sin(i * 0.05) * 0.1 + (i / T) * 0.2);
    const energy_target_series = Array.from({ length: T }, (_, i) => 10.5 + Math.sin(i * 0.05) * 0.1);
    const divergence_pred = Array.from({ length: T }, (_, i) => Math.random() * 0.001);
    const divergence_target = Array.from({ length: T }, () => 0);

    // Mock vorticity for the specific frame
    const vorticity_pred = mesh.nodes.map(([x, y]) => Math.sin(x * 15 - t * 0.1) * Math.cos(y * 15));
    const vorticity_target = vorticity_pred.map(v => v + (Math.random() - 0.5) * 0.05);

    res.json({
      t,
      energy_pred_series,
      energy_target_series,
      divergence_pred,
      divergence_target,
      vorticity_pred,
      vorticity_target,
      omega_min: -1.5,
      omega_max: 1.5,
    });
  });

  app.get("/api/dataset/samples", (req, res) => {
    // Mock histograms
    const velocity_bins = Array.from({ length: 50 }, (_, i) => ({ bin: i * 0.04, count: Math.floor(Math.exp(-Math.pow(i - 15, 2) / 100) * 1000) }));
    const energy_bins = Array.from({ length: 50 }, (_, i) => ({ bin: i * 0.02, count: Math.floor(Math.exp(-Math.pow(i - 10, 2) / 50) * 800) }));
    
    const outliers = [
      { trajectory: 12, mean_v: 1.85, z_score: 3.4, flag: true },
      { trajectory: 45, mean_v: 0.42, z_score: -3.1, flag: true },
      { trajectory: 88, mean_v: 1.25, z_score: 0.2, flag: false },
    ];

    res.json({ velocity_bins, energy_bins, outliers });
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
