# Example: image classifier

A ResNet-18 trained from scratch (`weights=None`, nothing is downloaded) with PyTorch DDP on Ray Train. One class, two datasets:

| Spec file | Model name | Dataset | Used for |
| --- | --- | --- | --- |
| `model.yaml` | `image-classifier` | EuroSAT land use, 64×64 RGB, 10 classes | full profile, `make demo` |
| `model.fashion.yaml` | `fashion-classifier` | Fashion-MNIST, 28×28 grayscale, 10 classes | CI (smoke profile) |

```bash
modelport validate examples/models/image-classifier --spec model.fashion.yaml
modelport submit examples/models/image-classifier --spec model.fashion.yaml --wait
```

## How it meets the contract

- `preprocess` decodes PNG bytes with Pillow, converts to RGB (grayscale is repeated to three channels) and resizes to `train.params.image_size`.
- `predict` returns the softmax maximum as the confidence.
- `train_loop` resumes from `modelport.latest_artifacts()` and calls `modelport.report()` every epoch; rank 0 saves `model.pt` and `meta.json` as the checkpoint.
- `input_stats` returns per-image `brightness` and `contrast`, the drift features for this model.
- `train.params.shuffle_labels: true` makes the labels meaningless. That variant exists to prove the evaluation gate rejects a bad model.

## CI tolerances (`model.fashion.yaml`)

CI trains on a 6,000-image sample for 2 epochs on CPU, so the thresholds are looser than the full-profile defaults: `min_absolute: 0.6`, `min_improvement: -0.02`, `max_drop_per_class: 0.10`, `max_latency_increase: 0.5`, a 2-minute canary window and 30 canary requests. They keep a good retrain passing despite training noise, while the shuffled-label variant still fails.

Datasets: Fashion-MNIST (Xiao, Rasul and Vollgraf, 2017, MIT) and EuroSAT (Helber et al., 2019, MIT). See `NOTICE`.
