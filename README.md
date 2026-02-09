# AdapterTune ViT (ECCV 2026)

AdapterTune ViT adapts pretrained ViTs by freezing the backbone and inserting small residual adapters into transformer blocks. Only adapters and the classification head are trained.

## Quickstart

```bash
pip install -r requirements.txt
python -m scripts.reproduce --devices 0,1,2
```

## Training

```bash
python -m scripts.train \
  --config configs/base.yaml \
  --config configs/datasets/cifar10.yaml \
  --config configs/models/vit_small.yaml \
  --config configs/experiments/quick_sanity.yaml \
  --override method.name=adapter_tune \
  --override method.adapter_rank=16 \
  --device 0
```

## Structure

- `configs/`: YAML configs (base, datasets, models, experiments)
- `src/`: datasets, models, training, evaluation, utils
- `scripts/`: runnable entry points
- `results/`: JSONL logs, summaries, checkpoints
- `paper/`: LaTeX paper skeleton and generated tables/figures

## Notes

- Datasets are torchvision-only and cached under `data/`.
- Deterministic splits are saved under `data/splits/`.
- Results are aggregated with `python -m scripts.collect_results`.
