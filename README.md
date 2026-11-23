# MultiAttCalib

**Validity-Guided Reciprocal Matching for LiDAR–Camera Extrinsic Calibration**

This repository contains the authors' **experimental implementation** of MultiAttCalib, a targetless residual LiDAR–camera calibration framework. It includes the model architecture, training pipeline, validity-guided reciprocal matching, chance-centered consistency score, confidence-gated SE(3) refinement, fixed-reference geometric loss, evaluation metrics, and ablation configurations used for the accompanying manuscript.

> **Version 0.1.1:** Manuscript submission package. Documentation, data-access information, and package metadata have been updated; the model, training and evaluation logic, and experimental configurations are unchanged from v0.1.0. See the [implementation details](docs/IMPLEMENTATION_NOTES.md) and [software validation notes](docs/VALIDATION.md). Raw data, original experiment manifests, trained checkpoints, and prediction outputs are not bundled.

## Overview

1. Project LiDAR points with the current extrinsic estimate and retain the nearest positive depth per pixel.
2. Extract RGB and sparse-depth features using separate ResNet-18 branches. Area-pool a binary projection mask to obtain token occupancy.
3. Compute occupancy-weighted image-to-depth probabilities and valid depth-to-image probabilities. Multiply reciprocal associations and derive their chance-centered consistency.
4. Predict a bounded twist, modulate it with a monotonic consistency gate, left-compose the SE(3) update, and reproject for the next iteration.

See [equation-to-code mapping](docs/METHOD.md) and [data specification](docs/DATA.md).

## Installation

Python 3.10–3.12 is recommended. Install a compatible PyTorch build using the [official installation instructions](https://pytorch.org/get-started/locally/); choose CUDA for full KITTI training. Then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The implementation does not require pretrained weights, torchvision, a proprietary CUDA extension, or any external service.

## Quick smoke test (no KITTI or private data required)

Generate artificial examples only for checking the software pipeline:

```bash
python scripts/make_synthetic.py --out data/synthetic
python -m multiattcalib train --config configs/smoke.json --root data/synthetic --train data/synthetic/train.jsonl --val data/synthetic/val.jsonl --out runs/smoke --device cpu
python -m multiattcalib evaluate --checkpoint runs/smoke/best.pt --root data/synthetic --manifest data/synthetic/test.jsonl --out runs/smoke/test --device cpu
```

The smoke outputs are not scientific evidence and should never be used as paper results.

## KITTI Raw

Download synced+rectified data and the calibration archive from the [KITTI Raw dataset website](https://www.cvlibs.net/datasets/kitti/raw_data.php), following its terms. Use the 2011_09_26 drives and camera image_02. Unpack under a local data root; no dataset is distributed here.

```bash
python -m multiattcalib prepare-kitti --root data/kitti --out data/splits --config configs/paper.json
```

For exact reproduction of the manuscript's 7,000/1,500/1,500 split, use the original train, validation, and test manifests, including the fixed evaluation perturbations. Contact the corresponding author for the original split and perturbation metadata. The `--drive-lists data/original_drives.json` option accepts explicit drive assignments; without it, the preparation utility generates a deterministic, drive-disjoint split for independent experiments. A newly generated split is not a substitute for the original manifests unless the selected frames and perturbations are confirmed to match. All methods must use the same saved manifests and fixed test perturbations.

### Train and evaluate

```bash
python -m multiattcalib train --config configs/paper.json --root data/kitti --train data/splits/train.jsonl --val data/splits/val.jsonl --variant full --out runs/full
python -m multiattcalib evaluate --checkpoint runs/full/best.pt --root data/kitti --manifest data/splits/test.jsonl --out runs/full/test
```

Evaluation produces predictions.jsonl, samples.csv, summary.json and evaluation.json. Report aggregate rotation and translation MAE, marginal success percentages and the joint success percentage. Each CSV row contains per-axis errors and each iteration's gate and reciprocal consistency diagnostics. Each checkpoint includes its actual configuration.

### Ablations

```bash
bash scripts/run_ablations.sh data/kitti data/splits runs/ablations
```

| Variant | Intervention |
| --- | --- |
| full | All proposed mechanisms |
| no_validity | Set occupancy to one for all depth tokens |
| no_reciprocal_weighting | Use directional feature aggregation while preserving the reciprocal consistency score |
| no_gate | Set alpha to one |
| constant_gate | Learn sigmoid(b), retaining the valid-count indicator but removing kappa dependence |

Every variant is independently retrained using the same configuration and split files. The plotting script reproduces the *structure* of Fig. 5 from actual prediction errors; the constant-gate comparison is reported separately in its own summary.

### External baselines

MSANet and CalibFormer are third-party methods; their source code and weights are not bundled. To reproduce the comparison, retrain their implementations under the same train/validation manifests and perturbation protocol, preserving their inference schemes. Export fixed inputs:

```bash
python scripts/export_protocol.py --config configs/paper.json --root data/kitti --manifest data/splits/test.jsonl --out data/splits/baseline_inputs.jsonl
```

For each baseline, export JSONL records containing `id`, `T0` and `T_pred` (each transform is 4x4, LiDAR-to-rectified-camera-02). The common scorer rejects missing, extra, or mismatched initial poses.

```bash
python -m multiattcalib score-external --config configs/paper.json --root data/kitti --manifest data/splits/test.jsonl --predictions runs/baseline_predictions.jsonl --out runs/baseline_metrics
```

### Private-platform / zero-shot evaluation

Prepare synchronized rectified RGB and point-cloud data with the corresponding camera intrinsics and independently estimated reference extrinsics. The manuscript's private VLP-16–FLIR dataset and associated calibration information are available from the corresponding author upon reasonable request. No private sensor recordings or calibration files are bundled. See [DATA.md](docs/DATA.md).

```bash
python -m multiattcalib evaluate --root private/platform --manifest private/platform/pairs.jsonl --checkpoint runs/full/best.pt --trials 3 --out runs/platform_zero_shot
```

No training or fine-tuning is performed by this command. For deployed inference without a reference pose, use explicit T0 entries and one trial; the evaluator exports predicted extrinsics without calculating ground-truth-dependent metrics.

### Qualitative projection

```bash
python -m multiattcalib visualize --root data/kitti --manifest data/splits/test.jsonl --checkpoint runs/full/best.pt --index 0 --out runs/qualitative.png
```

## Citation

If you use this implementation, please cite the accompanying MultiAttCalib manuscript. Software citation metadata is provided in [CITATION.cff](CITATION.cff).

## Data availability

KITTI Raw is publicly available from the [official dataset website](https://www.cvlibs.net/datasets/kitti/raw_data.php). The privately collected VLP-16–FLIR dataset and associated calibration information are not publicly distributed but are available from the corresponding author upon reasonable request.

Corresponding author: Tao Cao (caotao@sqxy.edu.cn).

## License

The MultiAttCalib source code in this package is provided under the MIT license; see [LICENSE](LICENSE). This software license does not extend to the KITTI dataset, privately collected sensor data, third-party methods, or their weights. Dataset access and redistribution remain subject to the applicable terms. See [DATA.md](docs/DATA.md) for data preparation and access instructions.
