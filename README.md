# GaugeDepth

Implementation of **Gauge-Consistent Scale Calibration for Monocular Depth Measurement**.

## Installation

Requires Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3 and scikit-learn 1.6.1; neural entries use the installed PyTorch 2.7.1+cu118. Checks cover Python syntax, TOML/JSON and CLI options against parser definitions. Training and evaluation were not run.

## Data and preprocessing

Generate relative inverse-depth fields, affine gauges and smooth representation mismatch; depth is in meters. Custom numeric NPZ input uses `r[N,H,W]`, registered camera-axis `depth[N,H,W]`, optional `valid[N,H,W]` and integer scene/drive `group[N]`. Complete groups remain together in tuning/calibration/test.

```bash
gauge-depth generate --output data/frames.npz --frames 300 --side 32 --seed 13
gauge-depth prepare --input data/frames.npz --output data/depth
```

## Training and inference

```bash
gauge-depth fit --input data/depth/tuning.npz --output data/tuning.json --budget 8 --bounds 0.001 4 -2 2 --sigma 0.01 --anchor-rho 0.01 --pixels 128 --seed 13
gauge-depth calibrate --input data/depth/calibration.npz --config data/tuning.json --output data/calibrated.json --alpha 0.05 --seed 10013
```

## Evaluation

```bash
gauge-depth evaluate --input data/depth/test.npz --config data/calibrated.json --output runs/full --eta 0.02 --seed 20013
```

## Ablations

Variants: `full`, `wls`, `scalar`, `no_residual`, `no_reference`. Acquisition policies are `disagreement`, `random`, `uniform`, `extremes`; rerun fit and calibration when changing the policy or budget.

```bash
gauge-depth evaluate --input data/depth/test.npz --config data/calibrated.json --output runs/no_residual --variant no_residual --eta 0.02 --seed 20013
```

## Outputs

`data/tuning.json` and `calibrated.json` retain the frozen query/calibration settings. Each evaluation directory contains `predictions.npz` (points and bounds), `frames.json`, `ranges.json` and `metrics.json`. Incompatible/unbounded sets retain their explicit status.

```bash
gauge-depth plot --input runs --output runs/figures
```
