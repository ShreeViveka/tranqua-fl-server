"""
fl_server.py — Real Federated Learning Aggregation Server
===========================================================
Deployed on Railway (free tier).

What this does:
  1. Receives weight updates from all users' laptops
  2. Runs FedAvg to combine them into a better global model
  3. Serves the improved global model back to users
  4. Tracks contribution history

Deploy to Railway:
  1. Push this file + requirements_server.txt to GitHub
  2. Connect Railway to your GitHub repo
  3. Set ROOT_DIR = /app in Railway environment

Local test:
  python fl_server/fl_server.py
  → runs on http://localhost:9000
"""

import os
import io
import json
import time
import copy
import logging
import hashlib
import threading
from datetime import datetime, date
from typing import Optional
from collections import defaultdict

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import uvicorn

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [FL-SERVER] %(message)s'
)
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Serenity FL Server",
    description = "Federated Learning aggregation server for Serenity",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── In-memory state (Railway free tier has ephemeral filesystem) ──────────────
# In production you'd use Railway's PostgreSQL addon for persistence
_state = {
    "global_weights"      : {},       # current global model weights
    "round_number"        : 0,        # current FL round
    "pending_updates"     : [],       # weight updates waiting to be aggregated
    "client_contributions": defaultdict(int),  # client_id -> contribution count
    "model_version"       : "1.0.0",
    "last_aggregation"    : None,
    "min_clients_per_round": 2,       # aggregate after this many updates
    "total_rounds"        : 0,
    "lock"                : threading.Lock(),
}

# ── Model storage path ────────────────────────────────────────────────────────
MODEL_DIR  = os.environ.get('MODEL_DIR', '/tmp/serenity_models')
MODEL_FILE = os.path.join(MODEL_DIR, 'global_model.npz')
META_FILE  = os.path.join(MODEL_DIR, 'meta.json')

os.makedirs(MODEL_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# FEDAVG IMPLEMENTATION
# ════════════════════════════════════════════════════════════════════════════

def fedavg(global_weights: dict, updates: list[dict],
           learning_rate: float = 0.1) -> dict:
    """
    Federated Averaging (McMahan et al., 2017).

    Instead of simple averaging, we apply the weight updates
    scaled by learning_rate to the global model.
    This is more stable than full replacement.

    global_weights + lr * mean(all_deltas) = new_global_weights
    """
    if not updates:
        return global_weights

    if not global_weights:
        # First round — use first update as the base
        log.info("[FedAvg] Initialising global model from first contribution.")
        return copy.deepcopy(updates[0])

    new_weights = copy.deepcopy(global_weights)

    for key in new_weights:
        if key not in updates[0]:
            continue

        # Stack all updates for this key
        all_deltas = []
        for update in updates:
            if key in update:
                delta = np.array(update[key])
                if delta.shape == np.array(new_weights[key]).shape:
                    all_deltas.append(delta)

        if not all_deltas:
            continue

        # Average the deltas
        mean_delta = np.mean(all_deltas, axis=0)

        # Apply to global weights
        current = np.array(new_weights[key])
        new_weights[key] = (current + learning_rate * mean_delta).tolist()

    log.info(f"[FedAvg] Aggregated {len(updates)} updates. "
             f"Keys updated: {len(new_weights)}")
    return new_weights


def save_global_model(weights: dict, round_num: int):
    """Save the global model weights to disk."""
    try:
        # Save weights as npz
        np_weights = {k: np.array(v) for k, v in weights.items()}
        np.savez(MODEL_FILE, **np_weights)

        # Save metadata
        meta = {
            "round_number" : round_num,
            "model_version": _state["model_version"],
            "saved_at"     : datetime.now().isoformat(),
            "total_clients": len(_state["client_contributions"]),
            "total_rounds" : _state["total_rounds"],
        }
        with open(META_FILE, 'w') as f:
            json.dump(meta, f, indent=2)

        log.info(f"[Server] Global model saved. Round {round_num}.")
    except Exception as e:
        log.error(f"[Server] Failed to save model: {e}")


def load_global_model() -> dict:
    """Load the global model from disk if it exists."""
    try:
        if os.path.exists(MODEL_FILE):
            data = np.load(MODEL_FILE, allow_pickle=True)
            weights = {k: data[k].tolist() for k in data.files}
            log.info(f"[Server] Loaded global model ({len(weights)} weight tensors)")
            return weights
    except Exception as e:
        log.error(f"[Server] Failed to load model: {e}")
    return {}


def maybe_aggregate():
    """
    Check if we have enough updates to run FedAvg.
    Called after each new update is received.
    """
    with _state["lock"]:
        pending = _state["pending_updates"]
        if len(pending) < _state["min_clients_per_round"]:
            log.info(f"[Server] {len(pending)}/{_state['min_clients_per_round']} "
                     f"updates needed. Waiting...")
            return

        log.info(f"[Server] Aggregating {len(pending)} updates...")

        # Run FedAvg
        new_weights = fedavg(
            _state["global_weights"],
            [u["weight_delta"] for u in pending]
        )

        _state["global_weights"]    = new_weights
        _state["round_number"]     += 1
        _state["total_rounds"]     += 1
        _state["last_aggregation"]  = datetime.now().isoformat()
        _state["pending_updates"]   = []   # clear queue

        save_global_model(new_weights, _state["round_number"])

        log.info(f"[Server] Round {_state['round_number']} complete. "
                 f"Total rounds: {_state['total_rounds']}")


# ════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def root():
    return {
        "status"        : "running",
        "service"       : "Serenity FL Server",
        "round"         : _state["round_number"],
        "total_clients" : len(_state["client_contributions"]),
        "pending"       : len(_state["pending_updates"]),
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status"           : "ok",
        "round_number"     : _state["round_number"],
        "total_rounds"     : _state["total_rounds"],
        "registered_clients": len(_state["client_contributions"]),
        "pending_updates"  : len(_state["pending_updates"]),
        "last_aggregation" : _state["last_aggregation"],
        "model_version"    : _state["model_version"],
        "min_clients"      : _state["min_clients_per_round"],
    }


@app.post("/fl/upload", tags=["FL"])
async def receive_update(request: Request):
    """
    Receive a weight update from a client laptop.

    Expected payload:
    {
        "client_id"   : "uuid-string",
        "round_number": 0,
        "weight_delta": {"layer_name": [values...]},
        "loss_before" : 0.85,
        "timestamp"   : "2026-04-18T...",
        "data_hash"   : "abc123"
    }
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    client_id    = payload.get("client_id", "unknown")
    round_number = payload.get("round_number", 0)
    weight_delta = payload.get("weight_delta", {})
    loss_before  = payload.get("loss_before", 0)

    # Validate
    if not client_id or len(client_id) < 8:
        raise HTTPException(status_code=400, detail="Invalid client_id")

    if not weight_delta:
        raise HTTPException(status_code=400, detail="Empty weight_delta")

    # Rate limiting — one update per client per day
    today_key = f"{client_id[:8]}_{date.today().isoformat()}"
    if today_key in [u.get("today_key") for u in _state["pending_updates"]]:
        return JSONResponse({
            "accepted"     : False,
            "reason"       : "Already received update from this client today",
            "round"        : _state["round_number"],
            "model_version": _state["model_version"],
        })

    log.info(f"[Server] Received update from {client_id[:8]}... "
             f"round={round_number} loss={loss_before:.4f} "
             f"keys={len(weight_delta)}")

    # Store the update
    with _state["lock"]:
        _state["pending_updates"].append({
            "client_id"  : client_id,
            "today_key"  : today_key,
            "weight_delta": weight_delta,
            "loss_before": loss_before,
            "received_at": datetime.now().isoformat(),
        })
        _state["client_contributions"][client_id[:8]] += 1

    # Try to aggregate
    threading.Thread(target=maybe_aggregate, daemon=True).start()

    return {
        "accepted"     : True,
        "client_id"    : client_id[:8] + "...",
        "round"        : _state["round_number"] + 1,
        "pending"      : len(_state["pending_updates"]),
        "model_version": _state["model_version"],
        "message"      : "Update received. Thank you for contributing!"
    }


@app.get("/fl/model", tags=["FL"])
def get_global_model_info():
    """Get metadata about the current global model."""
    has_model = bool(_state["global_weights"])
    return {
        "available"     : has_model,
        "round_number"  : _state["round_number"],
        "model_version" : _state["model_version"],
        "total_clients" : len(_state["client_contributions"]),
        "last_updated"  : _state["last_aggregation"],
        "weight_keys"   : len(_state["global_weights"]),
        "download_url"  : "/fl/model/weights" if has_model else None,
    }


@app.get("/fl/model/weights", tags=["FL"])
def download_global_weights():
    """
    Download the global model weights as JSON.
    Clients call this to update their local model after FL aggregation.
    """
    if not _state["global_weights"]:
        raise HTTPException(
            status_code=404,
            detail="No global model available yet. "
                   "Need at least 2 clients to contribute first."
        )

    return {
        "round_number"  : _state["round_number"],
        "model_version" : _state["model_version"],
        "weights"       : _state["global_weights"],
        "downloaded_at" : datetime.now().isoformat(),
    }


@app.get("/fl/stats", tags=["FL"])
def get_stats():
    """Get FL server statistics."""
    return {
        "round_number"       : _state["round_number"],
        "total_rounds"       : _state["total_rounds"],
        "registered_clients" : len(_state["client_contributions"]),
        "pending_updates"    : len(_state["pending_updates"]),
        "last_aggregation"   : _state["last_aggregation"],
        "client_contributions": dict(list(_state["client_contributions"].items())[:10]),
        "min_clients_per_round": _state["min_clients_per_round"],
    }


# ════════════════════════════════════════════════════════════════════════════
# STARTUP
# ════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    """Load existing model on server start."""
    weights = load_global_model()
    if weights:
        _state["global_weights"] = weights

        # Load metadata
        if os.path.exists(META_FILE):
            with open(META_FILE) as f:
                meta = json.load(f)
            _state["round_number"] = meta.get("round_number", 0)
            _state["total_rounds"] = meta.get("total_rounds", 0)

        log.info(f"[Server] Resumed from round {_state['round_number']}")
    else:
        log.info("[Server] Starting fresh — no existing model found.")


if __name__ == "__main__":
    log.info("Starting Serenity FL Server on port 9000...")
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")
