import argparse
import json
import time
from pathlib import Path

import numpy as np

from .core import AnchorService, acquire, conformal_quantile, extrema, point_fit, project


def read(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def generate(args):
    rng = np.random.default_rng(args.seed)
    relative = rng.uniform(.1, 1, (args.frames, args.side, args.side))
    slope = rng.uniform(.5, .8, (args.frames, 1, 1))
    offset = rng.uniform(.03, .08, (args.frames, 1, 1))
    inverse = slope * relative + offset + .004 * np.sin(5 * relative)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, r=relative, depth=1 / inverse,
                        valid=np.ones(relative.shape, bool), group=np.arange(args.frames))


def prepare(args):
    data = read(args.input)
    if data["r"].shape != data["depth"].shape or data["r"].ndim != 3:
        raise ValueError("r and depth must have matching frame-height-width shape")
    data.setdefault("valid", np.isfinite(data["depth"]) & (data["depth"] > 0) & np.isfinite(data["r"]))
    data["valid"] = data["valid"].astype(bool) & np.isfinite(data["depth"]) & (data["depth"] > 0) & np.isfinite(data["r"])
    data.setdefault("group", np.arange(len(data["r"])))
    groups = np.unique(data["group"])
    if len(groups) < 10:
        raise ValueError("At least ten independent frame groups required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    n1, n2 = int(.4 * len(groups)), int(.7 * len(groups))
    for name, selected in zip(["tuning", "calibration", "test"],
                              [groups[:n1], groups[n1:n2], groups[n2:]]):
        mask = np.isin(data["group"], selected)
        np.savez_compressed(output / f"{name}.npz", **{key: value[mask] for key, value in data.items()})


def episode(data, index, config, seed):
    r, depth, valid = data["r"][index], data["depth"][index], data["valid"][index].astype(bool)
    service = AnchorService(depth, config["sigma"], np.random.default_rng(seed), valid, config.get("shared_scale", 1.))
    result = acquire(r, valid, service, config["budget"], config["bounds"], config["anchor_rho"],
                     config["policy"], seed, config["stop_rho"], config["resolution"], config["stop_fraction"])
    available = valid.flatten().copy()
    height, width = r.shape
    for query in result["queries"]:
        row, column = divmod(query, width)
        radius = config["exclude_radius"]
        for rr in range(max(0, row - radius), min(height, row + radius + 1)):
            available[rr * width + max(0, column - radius):rr * width + min(width, column + radius + 1)] = False
    candidates = np.flatnonzero(available)
    if len(candidates) < config["pixels"]:
        raise ValueError("Insufficient non-anchor pixels for the fixed-size frame score")
    sample = np.random.default_rng(seed + 1000000).choice(candidates, config["pixels"], replace=False)
    result["evaluation_indices"] = sample
    result["valid"] = valid
    return result


def configuration(args):
    return {key: getattr(args, key) for key in ["budget", "bounds", "sigma", "anchor_rho", "policy", "pixels",
            "exclude_radius", "stop_rho", "resolution", "stop_fraction"]}


def fit(args):
    data = read(args.input)
    config = configuration(args)
    coordinates = data["r"][data["valid"].astype(bool)]
    config["r_mean"], config["r_std"] = float(coordinates.mean()), float(max(coordinates.std(), 1e-8))
    deviations = []
    for index in range(len(data["r"])):
        result = episode(data, index, config, args.seed + index)
        if len(result["vertices"]):
            sample = result["evaluation_indices"]
            low, high = extrema(data["r"][index].flatten()[sample], result["vertices"])
            inverse = 1 / data["depth"][index].flatten()[sample]
            deviations.extend(np.maximum(np.maximum(low - inverse, inverse - high), 0).tolist())
    config["scale"] = max(float(np.median(deviations)) if deviations else 1., 1e-6)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, config)


def calibrate(args):
    data = read(args.input)
    config = json.loads(Path(args.config).read_text())
    scores, groups = [], []
    for index in range(len(data["r"])):
        result = episode(data, index, config, args.seed + index)
        score = np.inf
        if len(result["vertices"]):
            sample = result["evaluation_indices"]
            low, high = extrema(data["r"][index].flatten()[sample], result["vertices"])
            target = 1 / data["depth"][index].flatten()[sample]
            score = np.max(np.maximum(np.maximum(low - target, target - high), 0)) / config["scale"]
        scores.append(score)
        groups.append(data["group"][index])
    unit_scores = [max(s for s, g in zip(scores, groups) if g == group) for group in np.unique(groups)]
    q = conformal_quantile(unit_scores, args.alpha)
    config.update(q=None if not np.isfinite(q) else q, alpha=args.alpha,
                  calibration_units=len(unit_scores))
    write(args.output, config)


def evaluate(args):
    data = read(args.input)
    config = json.loads(Path(args.config).read_text())
    config["shared_scale"] = args.shared_scale
    q = np.inf if config["q"] is None else config["q"]
    rho = 0. if args.variant == "no_residual" else q * config["scale"]
    eta = 0. if args.variant == "no_reference" else args.eta
    records, predictions, lowers, uppers = [], [], [], []
    repeats, range_records = [], []
    for index, r in enumerate(data["r"]):
        repeated = []
        for repeat in range(args.repeats):
            start = time.perf_counter()
            result = episode(data, index, config, args.seed + index + 100000 * repeat)
            sample = result["evaluation_indices"]
            depth = data["depth"][index].flatten()
            if not len(result["vertices"]):
                lower, upper = np.zeros(r.shape), np.full(r.shape, np.inf)
                prediction = np.full(r.shape, np.nan)
            else:
                lo, hi = extrema(r, result["vertices"])
                lower, upper = project(lo, hi, rho, eta)
                theta = point_fit(r, result["anchors"], result["vertices"], config["bounds"], np.min(r[result["valid"]]))
                if args.variant in {"wls", "scalar"}:
                    indices = [item[0] for item in result["anchors"]]
                    design = np.c_[r.flatten()[indices], np.ones(len(indices))]
                    midpoint = np.array([.5 * (item[1] + item[2]) for item in result["anchors"]])
                    weight = 1 / np.maximum(np.array([.5 * (item[2] - item[1]) for item in result["anchors"]]), 1e-6)
                    if args.variant == "scalar":
                        slope = np.linalg.lstsq(design[:, :1] * weight[:, None], midpoint * weight, rcond=None)[0][0]
                        theta = np.array([slope, 0.])
                    else:
                        theta = np.linalg.lstsq(design * weight[:, None], midpoint * weight, rcond=None)[0]
                inverse = theta[0] * r + theta[1]
                prediction = np.full(r.shape, np.nan)
                np.divide(1., inverse, out=prediction, where=inverse > 0)
            elapsed = time.perf_counter() - start
            repeated.append(prediction)
            if repeat:
                continue
            true, point = depth[sample], prediction.flatten()[sample]
            low, high = lower.flatten()[sample], upper.flatten()[sample]
            finite_point = np.isfinite(point)
            finite_set = np.isfinite(low) & np.isfinite(high)
            inclusion = (true >= low) & (true <= high)
            compatible = bool(len(result["vertices"]))
            metrics = {"frame": index, "group": int(data["group"][index]), "queries": len(result["queries"]),
                "status": result["status"], "incompatible": not compatible,
                "set_coverage": float(inclusion.mean()), "subset_inclusion": bool(inclusion.all()),
                "operational_coverage": float(inclusion.mean()) if compatible else 0.,
                "finite_fraction": float(finite_set.mean()), "unbounded_fraction": float(np.isinf(high).mean()),
                "mean_width": float((high - low)[finite_set].mean()) if finite_set.any() else None,
                "calibration_ms": 1000 * elapsed,
                "point_rmse": float(np.sqrt(np.mean((point[finite_point] - true[finite_point]) ** 2))) if finite_point.any() else None,
                "absrel": float(np.mean(abs(point[finite_point] - true[finite_point]) / true[finite_point])) if finite_point.any() else None,
                "bias_m": float(np.mean(point[finite_point] - true[finite_point])) if finite_point.any() else None}
            metrics["resolution_met"] = bool(compatible and finite_set.all() and np.median(high - low) <= config["resolution"]) if config["resolution"] > 0 else None
            anchors = [item[0] for item in result["anchors"]]
            if len(anchors) > 1:
                design = np.c_[(r.flatten()[anchors] - config["r_mean"]) / config["r_std"], np.ones(len(anchors))]
                singular = np.linalg.svd(design, compute_uv=False)
                metrics["anchor_condition_number"] = float(singular[0] / singular[-1]) if singular[-1] > 1e-12 else None
            else:
                metrics["anchor_condition_number"] = None
            for band_low, band_high in zip(args.range_edges[:-1], args.range_edges[1:]):
                mask = (true >= band_low) & (true < band_high)
                if mask.any():
                    valid_point = mask & finite_point
                    valid_width = mask & finite_set
                    range_records.append({"range": f"{band_low}-{band_high}", "frame": index,
                        "coverage": float(inclusion[mask].mean()) if compatible else 0.,
                        "rmse": float(np.sqrt(np.mean((point[valid_point] - true[valid_point]) ** 2))) if valid_point.any() else None,
                        "width": float((high - low)[valid_width].mean()) if valid_width.any() else None,
                        "pixels": int(mask.sum())})
            records.append(metrics)
            predictions.append(prediction)
            lowers.append(lower)
            uppers.append(upper)
        if args.repeats > 1:
            stack = np.stack(repeated)
            fixed = result["valid"] & np.isfinite(stack).all(0)
            repeats.append(float(np.sqrt(np.mean(np.var(stack[:, fixed] / data["depth"][index][fixed], axis=0, ddof=1)))) if fixed.any() else None)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "predictions.npz", point=np.stack(predictions),
                        lower=np.stack(lowers), upper=np.stack(uppers), truth=data["depth"])
    write(output / "frames.json", records)
    write(output / "ranges.json", range_records)
    summary = {"variant": args.variant, "frames": len(records), "repeatability": repeats}
    for key in records[0]:
        if key in {"frame", "group", "status"}:
            continue
        values = [r[key] for r in records if r[key] is not None]
        summary[key] = float(np.mean(values)) if values else None
    group_success = [all(r["subset_inclusion"] for r in records if r["group"] == group) for group in np.unique(data["group"])]
    summary["group_simultaneous_inclusion"] = float(np.mean(group_success))
    rng = np.random.default_rng(args.seed)
    group_errors = []
    for group in np.unique(data["group"]):
        values = [r["absrel"] for r in records if r["group"] == group and r["absrel"] is not None]
        if values:
            group_errors.append(float(np.mean(values)))
    if len(group_errors) > 1:
        means = [np.mean(rng.choice(group_errors, len(group_errors), replace=True)) for _ in range(args.resamples)]
        summary["absrel_group_bootstrap_ci95"] = np.quantile(means, [.025, .975]).tolist()
    write(output / "metrics.json", summary)


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = list(Path(args.input).rglob("metrics.json"))
    if not paths:
        raise ValueError("No evaluated metrics found")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for metric in ["point_rmse", "absrel", "bias_m", "operational_coverage", "finite_fraction", "mean_width", "queries", "calibration_ms"]:
        fig, ax = plt.subplots(figsize=(9, 4))
        labels = [str(p.parent.relative_to(args.input)) for p in paths]
        values = [json.loads(p.read_text()).get(metric) for p in paths]
        ax.bar(labels, [np.nan if v is None else v for v in values], color="#4e819a")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", labelrotation=35)
        fig.tight_layout()
        fig.savefig(out / f"{metric}.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(prog="gauge-depth")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("generate")
    p.add_argument("--output", required=True)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--side", type=int, default=32)
    p.add_argument("--seed", type=int, default=13)
    p = commands.add_parser("prepare")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("fit")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bounds", nargs=4, type=float, default=[.001, 4., -2., 2.])
    p.add_argument("--budget", type=int, default=8)
    p.add_argument("--pixels", type=int, default=128)
    p.add_argument("--exclude-radius", type=int, default=1)
    p.add_argument("--sigma", type=float, default=.01)
    p.add_argument("--anchor-rho", type=float, default=.01)
    p.add_argument("--policy", choices=["disagreement", "random", "uniform", "extremes"], default="disagreement")
    p.add_argument("--stop-rho", type=float, default=.01)
    p.add_argument("--resolution", type=float, default=0.)
    p.add_argument("--stop-fraction", type=float, default=0.)
    p.add_argument("--seed", type=int, default=13)
    p = commands.add_parser("calibrate")
    p.add_argument("--input", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--alpha", type=float, default=.05)
    p.add_argument("--seed", type=int, default=10013)
    p = commands.add_parser("evaluate")
    p.add_argument("--input", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--variant", choices=["full", "no_residual", "no_reference", "wls", "scalar"], default="full")
    p.add_argument("--eta", type=float, default=.02)
    p.add_argument("--shared-scale", type=float, default=1.)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=20013)
    p.add_argument("--range-edges", nargs="+", type=float, default=[0., 2., 5., float("inf")])
    p.add_argument("--resamples", type=int, default=2000)
    p = commands.add_parser("plot")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
