import numpy as np
from scipy.optimize import LinearConstraint, minimize


def clip_polygon(vertices, normal, upper):
    if not len(vertices):
        return vertices
    output = []
    for first, second in zip(vertices, np.roll(vertices, -1, axis=0)):
        fa, fb = first @ normal - upper, second @ normal - upper
        inside_a, inside_b = fa <= 1e-12, fb <= 1e-12
        if inside_a:
            output.append(first)
        if inside_a != inside_b:
            output.append(first + fa / (fa - fb) * (second - first))
    return np.asarray(output, dtype=float).reshape(-1, 2)


def domain(r, bounds, minimum_inverse=0):
    amin, amax, bmin, bmax = bounds
    if amin <= 0 or amax < amin or bmax < bmin:
        raise ValueError("Require 0 < amin <= amax and bmin <= bmax")
    vertices = np.array([[amin, bmin], [amax, bmin], [amax, bmax], [amin, bmax]], float)
    return clip_polygon(vertices, np.array([-np.min(r), -1.]), -minimum_inverse)


def extrema(r, vertices):
    if not len(vertices):
        return np.full(r.shape, -np.inf), np.full(r.shape, np.inf)
    values = r[..., None] * vertices[:, 0] + vertices[:, 1]
    error = 1e-10 * np.maximum(1., np.abs(values).max(-1))
    return values.min(-1) - error, values.max(-1) + error


def project(lower_inverse, upper_inverse, rho, eta=0):
    if not 0 <= eta < 1:
        raise ValueError("Reference-scale bound eta must be in [0,1)")
    low, high = lower_inverse - rho, upper_inverse + rho
    lower = np.zeros_like(low)
    np.divide(1., high, out=lower, where=high > 0)
    upper = np.full_like(low, np.inf)
    np.divide(1., low, out=upper, where=low > 0)
    incompatible = high <= 0
    lower[incompatible], upper[incompatible] = np.nan, np.nan
    return lower / (1 + eta), upper / (1 - eta)


class AnchorService:
    def __init__(self, depth, sigma, rng, valid, shared_scale=1.):
        self._depth = depth.flatten()
        self.sigma, self.rng = sigma, rng
        self.valid = valid.flatten()
        self.shared_scale = shared_scale
        self.log = []

    def query(self, index):
        self.log.append(int(index))
        if not self.valid[index] or not np.isfinite(self._depth[index]) or self._depth[index] <= 0:
            return None
        observed = self._depth[index] * self.shared_scale + self.rng.uniform(-self.sigma, self.sigma)
        lower = observed - self.sigma
        if lower <= 0:
            return None
        return lower, observed + self.sigma


def point_fit(r, anchors, vertices, bounds, minimum_r=None):
    if not anchors or not len(vertices):
        return np.full(2, np.nan)
    indices = [item[0] for item in anchors]
    design = np.c_[r.flatten()[indices], np.ones(len(indices))]
    low = np.array([item[1] for item in anchors])
    high = np.array([item[2] for item in anchors])
    target = .5 * (low + high)
    weight = 1 / np.maximum(.5 * (high - low), 1e-6) ** 2
    scales = np.array([bounds[1] - bounds[0], bounds[3] - bounds[2]]).clip(1e-8)
    development = np.array([.5 * (bounds[0] + bounds[1]), .5 * (bounds[2] + bounds[3])])
    weight = weight / weight.mean()

    def loss(theta):
        residual = design @ theta - target
        tie = (theta - development) / scales
        return (float(np.mean(weight * residual ** 2) + 1e-12 * np.sum(tie ** 2)),
                2 * design.T @ (weight * residual) / len(residual) + 2e-12 * tie / scales)

    minimum_r = np.nanmin(r) if minimum_r is None else minimum_r
    constraint_matrix = np.vstack([design, [minimum_r, 1.]])
    constraints = [LinearConstraint(constraint_matrix, np.r_[low, 0.], np.r_[high, np.inf])]
    result = minimize(loss, vertices.mean(0), jac=True, method="SLSQP",
                      bounds=[bounds[:2], bounds[2:]], constraints=constraints,
                      options={"ftol": 1e-12, "maxiter": 200})
    if not result.success:
        raise RuntimeError(f"Constrained gauge fit failed: {result.message}")
    return result.x


def acquire(r, valid, service, budget, bounds, anchor_rho=.01,
            policy="disagreement", seed=0, stop_rho=0., resolution=0., stop_fraction=0.):
    candidates = np.flatnonzero(valid.flatten() & np.isfinite(r.flatten()))
    if not len(candidates):
        return {"vertices": np.empty((0, 2)), "anchors": [], "queries": [], "status": "incompatible"}
    vertices = domain(r[valid], bounds)
    rng = np.random.default_rng(seed)
    anchors, status = [], "budget"
    ordered = candidates[np.argsort(r.flatten()[candidates])]
    uniform = candidates[np.linspace(0, len(candidates) - 1, min(budget, len(candidates))).astype(int)]
    for step in range(min(budget, len(candidates))):
        remaining = np.setdiff1d(candidates, service.log)
        if step < 2 and policy == "disagreement":
            index = ordered[0] if step == 0 else ordered[-1]
            if index in service.log:
                index = remaining[0]
        elif policy == "random":
            index = rng.choice(remaining)
        elif policy == "uniform":
            index = uniform[step]
        elif policy == "extremes":
            distance = np.abs(r.flatten()[remaining] - np.median(r.flatten()[candidates]))
            index = remaining[np.argmax(distance)]
        else:
            low, high = extrema(r.flatten()[remaining], vertices)
            index = remaining[np.argmax(high - low)]
        observation = service.query(index)
        if observation is None:
            continue
        low, high = 1 / observation[1] - anchor_rho, 1 / observation[0] + anchor_rho
        row = np.array([r.flatten()[index], 1.])
        vertices = clip_polygon(clip_polygon(vertices, row, high), -row, -low)
        anchors.append((int(index), float(low), float(high)))
        if not len(vertices):
            status = "incompatible"
            break
        lower, upper = extrema(r[valid], vertices)
        dl, dh = project(lower, upper, stop_rho)
        half = (upper - lower) / 2
        if resolution > 0 and np.isfinite(dh).all() and np.median(dh - dl) <= resolution:
            status = "provisional_resolution"
            break
        if stop_fraction > 0 and np.median(half / (half + stop_rho + 1e-12)) < stop_fraction:
            status = "residual_limited"
            break
    return {"vertices": vertices, "anchors": anchors, "queries": list(service.log), "status": status}


def conformal_quantile(scores, alpha):
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return float(np.sort(scores)[k - 1]) if k <= len(scores) else np.inf
