# GP-Model

Real-time security drone detection engine — **YOLO11 + ByteTrack**, optimized for Apple Silicon via **MPS** and **CoreML**.

This repo contains the ML/inference side only. The FastAPI backend and React dashboard live in separate repositories.

---

## Status — Phase 0 complete

Verified end-to-end on a MacBook Pro M4 Pro:

- Stock YOLO11n + ByteTrack on the bundled sample clip: **~100 FPS, ~9 ms/frame** on MPS.
- Target with YOLO11s @ 1080p on CoreML: **25–30 FPS, < 200 ms end-to-end** latency.
- 69 unit + integration tests, ruff-clean, mypy-strict (no errors).

---

## Architecture at a glance

The pipeline is a chain of interchangeable components, wired by config:

```
 VideoSource  ──►  Detector  ──►  Tracker  ──►  AlertDispatcher  ──►  Subscribers
  (webcam,         (YOLO11 on      (ByteTrack)     (Observer)           (Console,
   file, RTSP…)     MPS/CoreML)                                         JSONL,
                                                                        Metrics,
                                                                        … future:
                                                                        WebSocket,
                                                                        REST)
```

Every arrow is a Python `Protocol`; concrete classes are built from YAML by the Factory in `src/gpmodel/config/factory.py`. Swapping the webcam for an RTSP stream, or PyTorch for CoreML, is a YAML change — no code edits.

### Design patterns applied

| Pattern | Where | Why |
|---|---|---|
| Strategy | `VideoSource`, `Detector`, `Tracker` | Swap hardware/runtime without touching the pipeline. |
| Observer | `AlertDispatcher` + subscribers | Decouple inference from delivery (console today, WebSocket later). |
| Chain of Responsibility | `InferenceEngine` stages | Each stage is independently testable and replaceable. |
| Factory | `config/factory.py` | Config-driven wiring; no `if source_type == …` elsewhere. |

---

## Repository layout

```
GP-Model/
├── src/gpmodel/
│   ├── core/         # Domain types, events, interfaces, dispatcher
│   ├── sources/      # Webcam, file (RTSP coming in Phase 1)
│   ├── detectors/    # YOLO wrapper (PyTorch / CoreML / ONNX)
│   ├── trackers/     # ByteTrack
│   ├── pipeline/     # InferenceEngine orchestrator
│   ├── publishers/   # Console, JSONL file, metrics subscribers
│   ├── rules/        # (Phase 1) intruder / weapon / crowd / geofence
│   ├── reid/         # (Phase 1) face recognition for staff vs intruder
│   ├── streaming/    # (Phase 1) virtual RTSP simulator
│   ├── telemetry/    # Rolling perf meter, structured logging
│   ├── config/       # Pydantic schema, YAML loader, factory
│   ├── cli.py        # Entrypoint (gpmodel / python -m gpmodel)
│   └── __main__.py
├── apps/             # Thin scripts (run_inference.py)
├── configs/          # laptop.yaml, file_demo.yaml
├── assets/samples/   # Sample drone clip
├── tests/            # 69 tests, mirrors src/ layout
├── models/           # (gitignored) weights
├── datasets/         # (gitignored) training data
├── notebooks/        # Exploratory only
└── pyproject.toml
```

---

## Setup

Requires **Python 3.11** and **uv** (Astral's fast Python package manager).

```bash
# install uv once
brew install uv

# install project deps into a local .venv
uv sync
```

---

## Run the engine

Either of these runs the pipeline end-to-end:

```bash
# webcam profile — 1080p, YOLO11s on MPS, rich console output
uv run gpmodel --config configs/laptop.yaml

# deterministic demo — plays assets/samples/drone_01.mp4 on loop
uv run gpmodel --config configs/file_demo.yaml
```

You'll see per-frame detection summaries, periodic `PERF` lines with FPS + latency, and a run summary with alert counts on exit. Ctrl-C drains cleanly.

### Switching to CoreML for max Apple Silicon performance

Once you've exported YOLO11s to CoreML (Phase 1 deliverable), change one line in your config:

```yaml
detector:
  weights: models/yolo11s.mlpackage
  # `device` is ignored for CoreML — it runs on ANE + GPU automatically
```

---

## Development

```bash
# run the full suite
uv run pytest

# lint + format
uv run ruff check --fix src tests
uv run ruff format src tests

# strict type check
uv run mypy src/gpmodel

# install the pre-commit hooks
uv run pre-commit install
```

---

## Roadmap

### Phase 0 — baseline pipeline *(complete)*

- Domain model, dispatcher, Strategy-based interfaces.
- Webcam / file sources, YOLO11 on MPS, ByteTrack.
- Console / JSONL / metrics subscribers.
- Config-driven CLI.

### Phase 1 — virtual stream + decision engine

- MediaMTX-based RTSP simulator (stream the sample clip as if it were a real drone feed).
- RTSP source.
- Decision engine + rules: intruder, weapon, crowd, geofence.
- Staff-vs-intruder via InsightFace against a local `data/authorized/` folder.
- Hardcoded geofence polygons on the frame.

### Phase 2 — Apple Silicon export

- CoreML export workflow + benchmarks on M4 Pro.
- `export.py` CLI.

### Phase 3 — specialist weapon detector

- Public weapons dataset curation (Sohas, Olmos, Objects365 subset).
- Fine-tune on Colab T4/A100.
- SAHI tiled inference for small targets.
- Two-stage verifier to knock down false positives.

### Phase 4 — backend integration

- WebSocket and REST publishers.
- Wire to FastAPI backend in the sibling repo.

---

## License

MIT. See `pyproject.toml`.
