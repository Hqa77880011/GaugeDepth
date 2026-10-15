# GaugeDepth

Implementation of **Gauge-Consistent Scale Calibration for Sparse-Reference Depth Measurement**. A positive affine inverse-depth gauge is constrained by uncertain sparse references. The surviving gauge polygon produces depth intervals, a constrained point estimate, and explicit incompatible or unbounded outputs.

## Install

Use Python 3.10 or later:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then:

```bash
python -m pip install -e .
```

## Data preparation

```bash
gauge-depth generate --output data/frames.npz --frames 300 --side 32 --seed 13
gauge-depth prepare --input data/frames.npz --output data/depth
```

The supplied generator defines independent relative-inverse-depth fields, positive affine gauges, and a smooth representation mismatch. Its depth units are meters. It provides a fully specified calibration example without requiring a backbone download.

For image datasets, run a frozen relative-depth backbone first and export numeric arrays to the same NPZ schema:

| Key | Shape | Meaning |
| --- | --- | --- |
| `r` | N × H × W | Relative inverse depth after the frozen training-defined coordinate transformation |
| `depth` | N × H × W | Registered camera-axis reference depth in meters |
| `valid` | N × H × W | Optional validity mask available to the query policy |
| `group` | N | Optional integer scene/drive IDs; all related frames remain together |

Keep camera resizing, registration, intrinsic conventions, backbone checkpoint, and depth units consistent when exporting. Radial range must be converted to camera-axis depth before use. `prepare` allocates complete ordered groups to 40% tuning, 30% calibration, and 30% test and excludes nonfinite or nonpositive references. The valid mask is part of the allowed information and must not encode test prediction errors.

## Freeze the gauge and acquisition rule

```bash
gauge-depth fit --input data/depth/tuning.npz --output data/tuning.json --budget 8 --bounds 0.001 4 -2 2 --sigma 0.01 --anchor-rho 0.01 --pixels 128
```

`--bounds` is `[a_min, a_max, b_min, b_max]` for inverse depth `a*r+b`, with positive slope. These are development-defined bounds, fixed before calibration. Polygon clipping also imposes nonnegative inverse depth over valid pixels. A boundary at zero is retained and can produce an unbounded depth interval.

The default query policy first spans predicted inverse-depth extremes and then queries the largest disagreement between surviving gauges. Only `AnchorService.query` reveals a reference value. Every query is charged, including a rejected reference. Tuning fits a positive global residual scale from discrepancy outside gauge projections; it does not alter the backbone. Development coordinate moments are stored for the anchor conditioning diagnostic.

The reference simulator uses bounded uniform reading error with half-width `--sigma` meters. Each reported anchor interval contains its nominal shared-scale target under that model. `--anchor-rho` is a separate frozen representation tolerance in inverse meters. Noise and representation mismatch therefore enter different constraints.

## Calibrate and execute

```bash
gauge-depth calibrate --input data/depth/calibration.npz --config data/tuning.json --output data/calibrated.json --alpha 0.05
gauge-depth evaluate --input data/depth/test.npz --config data/calibrated.json --output runs/full --eta 0.02
gauge-depth plot --input runs --output runs/figures
```

Calibration replays the frozen query episode and samples a fixed-size non-anchor pixel subset using only locations and declared validity. Anchor neighborhoods are excluded. The frame score is maximum inverse-depth distance outside the gauge projection divided by the frozen scale. Frames in the same group are combined by a maximum, so the conformal sampling unit is the group. The rank is `ceil((n+1)*(1-alpha))`. A rank beyond the available units, or an infinite selected score, gives an infinite residual tolerance, stored as JSON `q: null` and read back as infinity.

An incompatible gauge returns no point prediction and the uninformative positive-depth fallback. It remains in set-inclusion accounting and counts as failure for operational coverage. A nonempty gauge whose inverse-depth interval reaches zero keeps its upper depth endpoint infinite. Neither case is replaced by an arbitrary depth cap.

Point estimates minimize interval-weighted anchor error subject to the original gauge constraints. A tiny development-centered tie penalty selects among underdetermined solutions. Shared reference-scale uncertainty expands intervals by division by `1+eta` and `1-eta`; it leaves the point estimate unchanged.

## Baselines, acquisition comparisons, and ablations

```bash
gauge-depth evaluate --input data/depth/test.npz --config data/calibrated.json --output runs/wls --variant wls
gauge-depth evaluate --input data/depth/test.npz --config data/calibrated.json --output runs/scalar --variant scalar
gauge-depth evaluate --input data/depth/test.npz --config data/calibrated.json --output runs/no_residual --variant no_residual
gauge-depth evaluate --input data/depth/test.npz --config data/calibrated.json --output runs/no_reference --variant no_reference
```

`wls` replaces the constrained point gauge with an unconstrained affine WLS fit. `scalar` fits only inverse-depth scale with zero offset. Their reported intervals retain the common gauge-set procedure, so compare their point metrics rather than interpreting them as separate interval methods. `no_residual` removes calibrated residual expansion; `no_reference` removes only the shared-scale envelope.

For acquisition policies, rerun `fit` and `calibrate` with `--policy random`, `uniform`, or `extremes`. Calibration is specific to the complete policy and budget. Budget sweeps use 1, 2, 4, 8, 16, and 32 references, with the one-anchor case exposing nonidentifiability. Seeds 13, 37, and 73 can be used for repeated query experiments.

`fit --resolution <meters>` enables provisional width stopping using the fixed `--stop-rho` inverse-depth tolerance. `--stop-fraction` enables residual-limited stopping. Calibration never feeds its fitted quantile back into acquisition. `resolution_met` checks the final calibrated set, separately from the provisional stop status.

For shared-reference drift and repeatability:

```bash
gauge-depth evaluate --input data/depth/test.npz --config data/calibrated.json --output runs/drift --shared-scale 1.02 --eta 0.02 --repeats 10
```

Repeatability is the square root of mean within-pixel sample variance of repeated point predictions normalized by fixed reference depth, computed on pixels finite in every repetition. Shared scale changes the reference revelation; it does not change the physical ground truth used to score the output.

## Results

| File | Contents |
| --- | --- |
| `predictions.npz` | Point estimates, lower/upper depth bounds, and evaluation reference |
| `frames.json` | Per-frame status, queries, point metrics, inclusion, width, availability, conditioning, and latency |
| `ranges.json` | Descriptive range-stratified coverage, RMSE, width, and denominators |
| `metrics.json` | Frame-averaged metrics, group-simultaneous inclusion, repeatability, and group-bootstrap AbsRel interval |
| `*.png` | Metric comparisons from existing evaluation outputs |

Point metrics are RMSE in meters, signed bias in meters, and dimensionless AbsRel on finite point predictions. Intervals report set coverage, operational coverage, simultaneous subset/group inclusion, finite-output fraction, unbounded fraction, incompatibility, and finite-width means. Conditioning uses the two-column anchor design standardized with tuning moments. Bootstrap units are scene/drive groups, never individual pixels. Range edges default to 0, 2, 5, and infinity meters and can be changed with `--range-edges`.

## Implementation conventions

This implementation uses a bounded, declared initial gauge domain and double-precision polygon clipping. It does not replace an unbounded coefficient domain with an undocumented numerical box. Outward projection tolerances retain boundary cases. The supplied residual-scale model is global, and the frozen backbone is an external data-preparation stage.

The interval statement concerns the prespecified sampled pixels and grouping rule, not every image pixel or arbitrary range-conditional events. Data with temporal or source dependence must use groups matching the intended deployment unit. Calibration-layer runtime excludes backbone execution; query counts describe logical reference access. CI checks syntax and package construction without running calibration or experiments.
