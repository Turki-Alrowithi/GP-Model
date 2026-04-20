# Datasets

This directory is `.gitignored` for **images and labels** (they can be huge).
Only dataset **YAML definitions** and this README are tracked.

## Layout

For each dataset, place the raw data under `datasets/<name>/` and the YAML that describes it alongside:

```
datasets/
├── README.md
├── example_data.yaml          ← tracked: template to copy
├── weapons.yaml               ← tracked: your real definitions
└── weapons/                   ← ignored: the actual data
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

Each label file is YOLO-format: one line per object with normalized `cx cy w h`:

```
0 0.524 0.611 0.082 0.154
2 0.211 0.883 0.031 0.065
```

## Workflow

1. Copy `example_data.yaml` to `datasets/<your-name>.yaml` and edit paths + class names.
2. Place images + labels under `datasets/<your-name>/`.
3. Train: `uv run python apps/train.py --weights yolo11s.pt --data datasets/<your-name>.yaml --epochs 50`.
4. Evaluate: `uv run python apps/eval.py --weights runs/train/<name>/weights/best.pt --data datasets/<your-name>.yaml`.

## Public datasets for drone security

Candidates to combine for weapon / intruder / crowd fine-tuning:

- **VisDrone-DET** — 10 classes from drone altitude (person/vehicle/bike).
- **Sohas Weapons** — handgun / knife / phone / card / key (good negative set).
- **Weapons Detection (Olmos et al.)** — handgun focus.
- **Objects365** — weapon subset (~600 classes total; filter to `knife`, `gun`, `baseball bat`).
- **ShanghaiTech Part B** — outdoor crowd counting.
- **UCF-QNRF**, **JHU-CROWD++** — dense crowd density maps.
