"""Compact arrays behind the run page's visual diagnostics: expected-vs-observed, odd/even, secondary, per-transit views."""
from __future__ import annotations

import numpy as np

from .bls import trapezoid
from .physics import binned_median, local_view
from .synthetic import transit_flux

STACK_BINS = 41
MAX_STACK_ROWS = 90


def _window_view(x_days, flux, half_days, nbins=61):
    """Median-binned flux in a window of +/- half_days around x = 0 (NaN where no data)."""
    idx = np.floor((x_days + half_days) / (2 * half_days) * nbins).astype(np.intp)
    med, counts = binned_median(idx, flux, nbins)
    return med, counts


def build_visuals(t, flat, period, t0, t14, depth, q_local, fit, params, impact_param, shape=None, refined=None) -> dict:
    half = 2.5 * q_local * period  # days: same span as the CNN local view
    n = 61
    x_grid = ((np.arange(n) + 0.5) / n - 0.5) * 2 * half
    x = (t - t0 + 0.5 * period) % period - 0.5 * period
    epoch = np.round((t - t0) / period).astype(int)
    phase = x / period

    # expected behaviour: the fitted trapezoid, and the transit predicted by the derived physical parameters
    model_fit = trapezoid(x_grid, fit["depth_ppm"] * 1e-6 * fit.get("baseline", 1.0), fit["t14_hours"] / 24.0,
                          fit["t12_hours"] / 24.0, 0.0, fit.get("baseline", 1.0))
    k, a_rs = params["radius_ratio"], max(params["a_over_rstar"], 1.0 + params["radius_ratio"] + 1e-3)
    fine = np.linspace(-half, half, 241)
    model_phys = transit_flux(fine, period, 0.0, k, a_rs, impact_param)
    # the geometry the light curve itself implies (falls back to the catalogue geometry when the shape is unconstrained)
    seen = {"a_rs": a_rs, "b": impact_param, "from_shape": False}
    if shape and shape.get("valid") and shape["a_over_rstar"] > 1.0 + k:
        seen = {"a_rs": shape["a_over_rstar"], "b": shape["impact_param"], "from_shape": True}
    k_seen = k
    if refined and refined.get("converged"):  # the limb-darkened fit is the best description of what was observed
        seen = {"a_rs": refined["a_over_rstar"], "b": refined["impact_param"], "from_shape": True}
        k_seen = refined["radius_ratio"]
        model_shape = np.interp(fine * 24.0, refined["curve"]["hours"], refined["curve"]["flux"])
    else:
        model_shape = transit_flux(fine, period, 0.0, k, seen["a_rs"], seen["b"])
    observed = local_view(phase, flat, q_local, n)
    resid = observed - model_fit

    def parity(mask):
        if mask.sum() < 20:
            return None
        med, counts = _window_view(x[mask], flat[mask], half, n)
        return None if (counts > 0).sum() < 0.5 * n else med

    odd, even = parity(epoch % 2 != 0), parity(epoch % 2 == 0)
    xs = (t - t0) % period - 0.5 * period  # centred on phase 0.5
    sec, sec_counts = _window_view(xs, flat, half, n)

    # one row per observed transit
    rows, depths = [], []
    near = np.abs(x) < half
    sigma = float(1.4826 * np.median(np.abs(flat[np.abs(x) > t14] - 1.0))) if (np.abs(x) > t14).any() else 0.0
    for e in np.unique(epoch[near]):
        m = near & (epoch == e)
        idx = np.floor((x[m] + half) / (2 * half) * STACK_BINS).astype(np.intp)
        med, counts = binned_median(idx, flat[m], STACK_BINS)
        if (counts > 0).sum() < 0.6 * STACK_BINS:
            continue  # transit fell in a data gap
        core = m & (np.abs(x) < 0.25 * t14)
        if core.sum() >= 2:
            depths.append({"epoch": int(e), "time": float(t0 + e * period), "depth_ppm": float((1.0 - flat[core].mean()) * 1e6),
                           "err_ppm": float(sigma / np.sqrt(core.sum()) * 1e6)})
        if len(rows) < MAX_STACK_ROWS:
            rows.append({"epoch": int(e), "values": np.round((med - 1.0) / max(depth, 1e-9), 3)})

    return {
        "x_hours": np.round(x_grid * 24.0, 4), "observed": np.round(observed, 6), "model_fit": np.round(model_fit, 6),
        "residual_ppm": np.round(resid * 1e6, 1), "residual_rms_ppm": float(np.nanstd(resid) * 1e6),
        "phys_hours": np.round(fine * 24.0, 4), "model_physical": np.round(model_phys, 7),
        "odd": None if odd is None else np.round(odd, 6), "even": None if even is None else np.round(even, 6),
        "secondary": np.round(sec, 6) if (sec_counts > 0).sum() >= 0.5 * n else None,
        "stack": rows, "transit_depths": depths,
        "model_shape": np.round(model_shape, 7),
        "geometry": {"k": k_seen, "a_rs": seen["a_rs"], "b": seen["b"], "from_shape": seen["from_shape"], "period_days": period,
                     "t14_hours": t14 * 24.0, "depth": depth, "a_rs_catalogue": a_rs, "b_assumed": impact_param},
    }
