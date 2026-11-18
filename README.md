# PathSnap — Map-Matching & Route Reconstruction (UE24CS352A Hackathon)

PathSnap snaps noisy GPS observations onto a road network. It scores every candidate road
segment with a learned classifier, then decodes a connected route with an HMM and Viterbi.

```
raw .track/.nodes/.arcs ──► AEQD projection ──► candidate arcs (≤60 m, ≤10/pt) ──► 21 features
        │                                                                            │
   .route ─► monotone DP alignment ─► per-point label (training / server-side scoring only)
                                                                                     ▼
                         XGBoost (Optuna-tuned) P(candidate = true arc) ──► HMM + Viterbi
                                                                                     ▼
                                     shortest-path gap filling ──► connected predicted route
```

## Repository layout

| Folder | Contents |
|---|---|
| `pathsnap/` | Library: `io.py` (readers, cleaning), `geo.py` (projection, geometry), `prep.py` (network, labels, candidates, features), `models.py` (model zoo), `metrics.py`, `hmm.py` (HMM, Viterbi, route filling), `pipeline.py` (end-to-end inference + scoring) |
| `scripts/` | `eda.py`, `build_dataset.py`, `train_models.py`, `bayes_opt.py`, `hmm_eval.py`, `make_figures.py` |
| `api/` | FastAPI server (Laptop 1) |
| `ui/` | Evaluator UI, a single static HTML page using Leaflet (Laptop 2) |
| `configs/` | `best_params.json` (Optuna), `hmm_params.json` (decoder), `models.json` |
| `models/` | Saved models (`final_model.joblib` is the deployed XGBoost). The RF, ExtraTrees and KNN weights (19–92 MB) are git-ignored because of their size; regenerate them with `scripts/train_models.py` |
| `results/` | CSV logs: EDA inventory, CV folds, model comparison, Optuna trials, tuned vs untuned, HMM results |
| `figures/` | Plots used in the report |
| `maps/` | Interactive Folium maps (EDA and predictions) |
| `logs/` | API request log (`api.log`) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the dataset next to the code, or point `--data` / `PATHSNAP_DATA` at it:

```
data/
  train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
  student_test/...
  evaluator_test/...        # only on Laptop 1, only for the final evaluation
```

## Reproduce the experiments (only `train` and `student_test` are used)

```bash
python scripts/build_dataset.py --data data          # labels, candidates, features      (~1 min)
python scripts/eda.py --data data                    # inventory, EDA plots, Folium maps  (~2 min)
python scripts/train_models.py                       # 9 models, GroupKFold CV + student test
python scripts/bayes_opt.py --model xgboost --trials 40   # Optuna TPE + tuned vs untuned
cp models/xgboost_tuned.joblib models/final_model.joblib
python scripts/hmm_eval.py                           # decoder tuning + classifier vs HMM
python scripts/make_figures.py                       # all plots + prediction maps
```

Seeds are fixed (`SEED=42`). The evaluator-test partition is never read by any training,
tuning or EDA script.

## Two-laptop deployment

**Laptop 1 (model + API).** Both machines must be on the same network.

```bash
# Linux / macOS
PATHSNAP_DATA=./data PATHSNAP_PARTITION=evaluator_test uvicorn api.app:app --host 0.0.0.0 --port 8000
# Windows (PowerShell)
$env:PATHSNAP_DATA="./data"; $env:PATHSNAP_PARTITION="evaluator_test"; uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Find Laptop 1's IP address (`ipconfig` or `ifconfig`) and allow port 8000 through the firewall.
Every request is logged to the console and to `logs/api.log`, with the trip, accuracy and latency.

**Laptop 2 (UI).**

```bash
cd ui && python -m http.server 5500
# open http://localhost:5500/?api=http://<LAPTOP1_IP>:8000
```

Click **Load evaluation trips**, then pick any trip. The UI sends only the `trip_id` to `POST /predict`
and draws the raw GPS points and the predicted route. It also shows the accuracy metrics computed
on the server.

### API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | – | status, model, HMM params |
| GET | `/trips` | – | trip ids in the evaluation partition |
| POST | `/predict` | `{"trip_id": "00000052"}` | GPS points, predicted route polylines, metrics, timing |

```bash
curl http://<LAPTOP1_IP>:8000/trips
curl -X POST http://<LAPTOP1_IP>:8000/predict -H "Content-Type: application/json" -d "{\"trip_id\":\"00000052\"}"
```

## Leakage safeguards

- Folds are split by trip (`GroupKFold`), so no trip appears in both training and validation.
- On the prediction path, the trip is loaded with `with_route=False`. The reference route is read
  only afterwards, inside `score_trip()`, to compute accuracy on the server. It is never returned to the UI.
- Optuna and the decoder grid use only training trips. Student-test trips are used only for
  development comparison.
- No predictions are hardcoded. Every request runs the full pipeline.
