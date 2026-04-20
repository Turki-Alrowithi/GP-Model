# GP-Model

Real-time security drone detection engine — **YOLO11 + ByteTrack**, polygon geofencing, crowd monitoring, face-based staff/intruder identification, all optimised for Apple Silicon (MPS / CoreML).

[![CI](https://github.com/Turki-Alrowithi/GP-Model/actions/workflows/ci.yml/badge.svg)](https://github.com/Turki-Alrowithi/GP-Model/actions/workflows/ci.yml)

This repo contains the ML/inference side only. The FastAPI backend and React dashboard live in separate repositories.

---

## Status

- **Phase 0** — baseline pipeline ✅
- **Phase 1** — virtual RTSP stream + decision engine (geofence, crowd, weapon, intruder) ✅
- **Phase 2** — training + evaluation CLIs, CI ✅
- **Phase 3** — CoreML export (blocked by upstream), weapon specialist fine-tune
- **Phase 4** — backend integration (WebSocket / REST publishers)

**Measured on MacBook Pro M4 Pro:**

| Config | Throughput | p50 latency | p95 latency |
|---|---|---|---|
| YOLO11s, PyTorch, MPS, 640² | **77.9 FPS** | 11.9 ms | 24.4 ms |
| YOLO11n, PyTorch, MPS, 640² | ~100 FPS | 9 ms | — |

Test suite: **140 tests, ruff clean, mypy-strict clean** across 47 source files.

---

## Architecture

```
 VideoSource  ──►  Detector  ──►  Tracker  ──►  RulesEngine  ──►  AlertDispatcher  ──►  Subscribers
  (webcam,         (YOLO11 on      (ByteTrack)   (geofence,        (Observer)            (Console,
   file, RTSP)      MPS/CoreML/                   crowd, weapon,                          JSONL,
                    ONNX)                         intruder)                               Metrics,
                                                                                          WS/REST
                                                                                          later)
```

Everything between arrows is a Python `Protocol`; concrete classes are wired from YAML by `src/gpmodel/config/factory.py`. Adding a new source type, detector backend, or rule is a one-case addition there — nothing else moves.

### Design patterns applied

| Pattern | Where | Why |
|---|---|---|
| **Strategy** | `VideoSource`, `Detector`, `Tracker` | Swap hardware / runtime without touching the pipeline. |
| **Observer** | `AlertDispatcher` → `Subscriber`s | Decouple inference from delivery. Add a WebSocket publisher later without editing a single byte of rule code. |
| **Chain of Responsibility** | `InferenceEngine` stages | Each stage independently testable and replaceable. |
| **Factory** | `config/factory.py` | Config-driven wiring; no `if source_type == …` anywhere else. |
| **Template method** | `BaseVideoSource` | Subclasses supply only `_open_capture()` + loop-on-EOF policy. |
| **State machine** | `IntruderRule` | Per-track `UNKNOWN → {STAFF, INTRUDER, INDETERMINATE}` avoids redundant face inference. |

---

## Repository layout

```
GP-Model/
├── .github/workflows/    # CI pipeline
├── src/gpmodel/
│   ├── core/             # Domain types, events, interfaces, dispatcher
│   ├── sources/          # Webcam, file, RTSP, threaded reader wrapper
│   ├── detectors/        # YOLO (PyTorch / CoreML / ONNX)
│   ├── trackers/         # ByteTrack (via supervision)
│   ├── rules/            # Geofence, crowd, weapon, intruder + RulesEngine
│   ├── reid/             # InsightFace encoder, StaffFaceDB
│   ├── pipeline/         # InferenceEngine orchestrator
│   ├── publishers/       # Console, JSONL, metrics subscribers
│   ├── streaming/        # (reserved for virtual-stream helpers)
│   ├── telemetry/        # Rolling perf meter, structured logging
│   ├── export/           # PT → CoreML/ONNX + detector benchmarker
│   ├── training/         # train(), evaluate() around Ultralytics
│   ├── config/           # Pydantic schema, YAML loader, factory
│   └── cli.py            # `gpmodel` CLI
├── apps/                 # Entry scripts (run_inference, export, train, eval)
├── configs/              # laptop.yaml, file_demo.yaml, rtsp_demo.yaml
├── docker/               # MediaMTX virtual-stream compose
├── datasets/             # YAML definitions + README (images/labels gitignored)
├── assets/samples/       # Sample clip
├── tests/                # 140 tests mirroring src/
├── models/               # Trained/exported weights (gitignored)
└── pyproject.toml
```

---

## Setup

Requires **Python 3.11** and **uv** (Astral's fast Python package manager).

```bash
brew install uv                         # once
uv sync                                 # core deps
uv sync --extra face --extra export     # add InsightFace + CoreML/ONNX helpers
```

---

## Running the engine

### Webcam (1080p, MPS)

```bash
uv run gpmodel --config configs/laptop.yaml
```

### File playback (deterministic demo)

```bash
uv run gpmodel --config configs/file_demo.yaml
```

### Virtual RTSP (as if a real drone were streaming)

```bash
docker compose -f docker/compose.yaml up -d        # starts MediaMTX + ffmpeg publisher
uv run gpmodel --config configs/rtsp_demo.yaml
```

You'll see per-frame detection summaries with track ids, periodic `PERF` lines, and colour-coded alerts from each enabled rule. Ctrl-C drains cleanly via the threaded frame reader.

---

## Rules enabled by the file demo

- **Geofence** — a "right-half" restricted zone; any `person` whose feet are inside fires HIGH.
- **Crowd** — 3+ persons sustained for 2 s fires MEDIUM.
- **Weapon** — disabled by default (stock YOLO11 knife is weak — enable once a specialist model is trained).
- **Intruder** — disabled by default (requires staff photos under `data/authorized/<name>/`).

---

## Model workflow

```bash
# 1. Create datasets/<your>.yaml from the template
cp datasets/example_data.yaml datasets/weapons.yaml

# 2. Train
uv run python apps/train.py --weights yolo11s.pt --data datasets/weapons.yaml --epochs 50

# 3. Evaluate
uv run python apps/eval.py --weights runs/train/<run>/weights/best.pt --data datasets/weapons.yaml

# 4. Export for deployment
uv run python apps/export.py export --weights runs/train/<run>/weights/best.pt --format onnx

# 5. Benchmark
uv run python apps/export.py bench --weights models/best.onnx \
    --source assets/samples/drone_01.mp4 --frames 200
```

Training runs both locally (`--device mps`) and on Colab (`--device 0`). See [`datasets/README.md`](datasets/README.md) for the expected on-disk layout and candidate public datasets.

---

## Development

```bash
uv run pytest                           # full suite
uv run ruff check --fix src tests apps  # lint (auto-fix)
uv run ruff format src tests apps       # format
uv run mypy src/gpmodel                 # strict type check
uv run pre-commit install               # enable hooks
```

CI enforces all four on every push and PR.

---

## Roadmap

### Phase 3 — model + export

- CoreML export (blocked on upstream `coremltools` ↔ `torch ≥ 2.10` fix; ONNX path works today).
- Weapon-specialist fine-tune on merged public datasets (Sohas / Olmos / Objects365 weapons), SAHI tiled inference at imgsz=1280, two-stage verifier on hard negatives.
- TensorRT export track for non-Apple edge devices.

### Phase 4 — backend integration

- `WebSocketSubscriber` and `RestSubscriber` wired to the sibling FastAPI backend (JWT auth, auto-reconnect).
- Schema contract tests against the backend's OpenAPI spec.

### Phase 5 — productionisation

- Prometheus-compatible metrics endpoint.
- Dockerfile for edge deployment (Jetson Orin / Raspberry Pi + Hailo).
- Multi-stream orchestration (one process, N engines).

---

## License

MIT. See `pyproject.toml`.
