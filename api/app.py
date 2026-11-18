"""PathSnap FastAPI server (runs on Laptop 1).

    PATHSNAP_DATA=/path/to/dataset PATHSNAP_PARTITION=evaluator_test \
        uvicorn api.app:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /health            liveness + loaded model/config
    GET  /trips             list of trip ids available in the evaluation partition
    POST /predict           {"trip_id": "..."} -> predicted route + server-side accuracy

Reference routes are read ONLY inside score_trip(), after prediction, and are never returned to the UI.
"""
import json, logging, os, sys, time

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.io import load_trip, list_trips
from pathsnap.pipeline import predict_trip, score_trip

DATA = os.environ.get("PATHSNAP_DATA", os.path.join(ROOT, "data"))
PART = os.environ.get("PATHSNAP_PARTITION", "evaluator_test")
MODEL_FILE = os.environ.get("PATHSNAP_MODEL", os.path.join(ROOT, "models", "final_model.joblib"))
HMM_FILE = os.path.join(ROOT, "configs", "hmm_params.json")

os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(os.path.join(ROOT, "logs", "api.log"))])
log = logging.getLogger("pathsnap")

MODEL = joblib.load(MODEL_FILE)
HMM = json.load(open(HMM_FILE))["params"] if os.path.exists(HMM_FILE) else None
log.info("model loaded: %s | hmm params: %s | data: %s/%s", MODEL_FILE, HMM, DATA, PART)

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


def route_lines(nodes, arcs, route):
    """Arc route -> list of lon/lat polylines (split where consecutive arcs are not connected)."""
    lines, cur = [], []
    for k, aid in enumerate(route):
        f, t = arcs[aid]
        if not cur or arcs[route[k - 1], 1] != f:
            if cur:
                lines.append(cur)
            cur = [nodes[f].tolist()]
        cur.append(nodes[t].tolist())
    if cur:
        lines.append(cur)
    return [[[p[1], p[0]] for p in ln] for ln in lines]   # -> [lat, lon] for Leaflet


@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_FILE), "hmm": HMM, "partition": PART}


@app.get("/trips")
def trips(request: Request):
    ids = list_trips(DATA, PART)
    log.info("GET /trips from %s -> %d trips", request.client.host, len(ids))
    return {"partition": PART, "trips": ids}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    if req.trip_id not in list_trips(DATA, PART):
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    t0 = time.time()
    log.info("POST /predict trip=%s from %s", req.trip_id, request.client.host)
    trip = load_trip(DATA, PART, req.trip_id, with_route=False)        # prediction path: no labels
    pred = predict_trip(trip, MODEL, HMM)
    trip.route = load_trip(DATA, PART, req.trip_id).route              # server-side scoring only
    metrics = score_trip(trip, pred) if trip.route is not None else None
    gps = trip.track[:, [1, 0]].round(6).tolist()
    resp = dict(trip_id=req.trip_id, n_gps=len(gps), gps=gps,
                predicted_route=route_lines(trip.nodes, trip.arcs, pred["route"]),
                n_route_arcs=int(len(pred["route"])), metrics=metrics,
                timing={k: round(v, 3) for k, v in pred["timing"].items()},
                server_total_s=round(time.time() - t0, 3))
    if metrics:
        log.info("trip=%s | obs=%d | top1 clf=%.4f hmm=%.4f | route F1=%.4f | continuity=%.3f | %.2fs",
                 req.trip_id, metrics["n_obs"], metrics["clf"]["top1_accuracy"], metrics["hmm"]["top1_accuracy"],
                 metrics["hmm_filled_route"]["route_f1"], metrics["hmm_filled_route"]["route_continuity"],
                 resp["server_total_s"])
    return resp
