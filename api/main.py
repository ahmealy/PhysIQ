"""
MeshGraphNets — FastAPI Backend
Run with: uvicorn api.main:app --reload --port 8000
(from project root with venv activated)
"""

import os
import sys

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import status, train, dataset, rollout, results, physics, generate

app = FastAPI(
    title="MeshGraphNets Physics AI",
    description="Graph neural network physics simulation — REST API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ── CORS — allow all origins for local dev / React/Vue frontend ───────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(status.router,   prefix="/api", tags=["System"])
app.include_router(train.router,    prefix="/api", tags=["Training"])
app.include_router(dataset.router,  prefix="/api", tags=["Dataset"])
app.include_router(rollout.router,  prefix="/api", tags=["Inference"])
app.include_router(results.router,  prefix="/api", tags=["Results"])
app.include_router(physics.router,  prefix="/api", tags=["Physics"])
app.include_router(generate.router, prefix="/api", tags=["Generate"])


@app.get("/api", tags=["System"])
def root():
    return {
        "project": "MeshGraphNets Physics AI",
        "docs":    "/api/docs",
        "redoc":   "/api/redoc",
    }
