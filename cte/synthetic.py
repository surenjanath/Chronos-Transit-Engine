"""Synthetic photometry: transit / eclipse injection for demos, training and validation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import AU_M, DAYS_PER_YEAR, R_SUN_M

KINDS = ("planet", "eclipsing_binary", "blend", "noise")


def overlap_fraction(z: np.ndarray, k: float) -> np.ndarray:
    """Area of intersection of a unit disc and a disc of radius k at separation z, over pi."""
    z = np.abs(z)
    out = np.zeros_like(z)
    inside = z <= abs(1.0 - k)
    out[inside] = min(1.0, k) ** 2
    part = (z > abs(1.0 - k)) & (z < 1.0 + k)
    zp = z[part]
    k0 = np.arccos(np.clip((k * k + zp * zp - 1.0) / (2.0 * k * zp), -1, 1))
    k1 = np.arccos(np.clip((1.0 - k * k + zp * zp) / (2.0 * zp), -1, 1))
    lens = k * k * k0 + k1 - 0.5 * np.sqrt(np.clip(4.0 * zp * zp - (1.0 + zp * zp - k * k) ** 2, 0, None))
    out[part] = lens / np.pi
    return out


def a_over_rstar(period_days: float, m_star: float, r_star: float) -> float:
    a_au = (m_star * (period_days / DAYS_PER_YEAR) ** 2) ** (1.0 / 3.0)
    return a_au * AU_M / (r_star * R_SUN_M)


def sky_separation(time, period, t0, a_rs, b):
    ang = 2.0 * np.pi * (time - t0) / period
    z = np.hypot(a_rs * np.sin(ang), b * np.cos(ang))
    return z, np.cos(ang) > 0


def transit_flux(time, period, t0, k, a_rs, b=0.0, u=(0.0, 0.0)) -> np.ndarray:
    """Planet transit with small-planet quadratic limb darkening (u=(0,0): uniform disc, depth k^2)."""
    z, front = sky_separation(time, period, t0, a_rs, b)
    drop = overlap_fraction(z, k)
    if u[0] or u[1]:
        mu = np.sqrt(np.clip(1.0 - np.minimum(z, 1.0) ** 2, 0.0, 1.0))
        intensity = 1.0 - u[0] * (1 - mu) - u[1] * (1 - mu) ** 2
        drop = drop * intensity / (1.0 - u[0] / 3.0 - u[1] / 6.0)
    return 1.0 - np.where(front, drop, 0.0)


def box_flux(time, period, t0, depth, duration) -> np.ndarray:
    x = np.abs((time - t0 + 0.5 * period) % period - 0.5 * period)
    return 1.0 - depth * (x < 0.5 * duration)


def eclipsing_binary_flux(time, period, t0, k, a_rs, b, sb_ratio, dilution=1.0) -> np.ndarray:
    """Two uniform discs (radius ratio k, surface-brightness ratio J); secondary eclipse at phase 0.5."""
    z, front = sky_separation(time, period, t0, a_rs, b)
    area = overlap_fraction(z, k)
    total = 1.0 + k * k * sb_ratio
    drop = np.where(front, area, area * sb_ratio) / total
    return 1.0 - dilution * drop


def planet_duration_days(period, k, a_rs, b) -> float:
    arg = np.sqrt(max((1 + k) ** 2 - b * b, 0.0)) / a_rs
    return float(period / np.pi * np.arcsin(min(arg, 1.0)))


@dataclass
class SyntheticSpec:
    kind: str = "planet"
    period_days: float = 3.524748
    depth: float = 0.0146
    t0: float | None = None
    impact_param: float = 0.1
    noise_ppm: float = 1000.0
    baseline_days: float = 27.4
    cadence_min: float = 2.0
    start_time: float = 1325.0
    variability_amp: float = 0.002
    limb_darkening: bool = True
    box: bool = False
    duration_hours: float | None = None
    sb_ratio: float = 0.5
    dilution: float = 1.0
    gap: bool = True
    outlier_fraction: float = 0.001
    seed: int | None = None


def synthesize(spec: SyntheticSpec, r_star: float = 1.0, m_star: float = 1.0) -> dict:
    """Generate a raw light curve (arbitrary e-/s units) with systematics, noise and outliers."""
    if spec.kind not in KINDS:
        raise ValueError(f"unknown synthetic kind '{spec.kind}'")
    rng = np.random.default_rng(spec.seed)
    dt = spec.cadence_min / 1440.0
    time = spec.start_time + np.arange(0.0, spec.baseline_days, dt)
    if spec.gap and spec.baseline_days > 20:  # TESS-like mid-sector data downlink gap
        mid = spec.start_time + spec.baseline_days / 2.0
        time = time[np.abs(time - mid) > 0.5]
    n = len(time)
    period = spec.period_days
    t0 = spec.t0 if spec.t0 is not None else spec.start_time + rng.uniform(0.15, 0.85) * min(period, spec.baseline_days)
    k = float(np.sqrt(max(spec.depth, 0.0)))
    a_rs = a_over_rstar(period, m_star, r_star)
    centroid_shift = np.zeros(n)
    truth: dict = {"kind": spec.kind}

    if spec.kind == "planet":
        if spec.box:
            dur = (spec.duration_hours / 24.0) if spec.duration_hours else planet_duration_days(period, k, a_rs, 0.0)
            signal = box_flux(time, period, t0, spec.depth, dur)
        else:
            u = (0.4, 0.25) if spec.limb_darkening else (0.0, 0.0)
            signal = transit_flux(time, period, t0, k, a_rs, spec.impact_param, u)
            dur = planet_duration_days(period, k, a_rs, spec.impact_param)
        truth.update(period_days=period, t0=t0, depth=float(1.0 - signal.min()), radius_ratio=k, duration_hours=dur * 24.0,
                     impact_param=spec.impact_param)
    elif spec.kind in ("eclipsing_binary", "blend"):
        # `depth` sets the companion size: k = clip(sqrt(depth) * 3, 0.25, 0.95) gives stellar-sized eclipses
        k_eb = float(np.clip(3.0 * k, 0.25, 0.95))
        b = max(spec.impact_param, 0.6)
        dilution = spec.dilution if spec.kind == "eclipsing_binary" else min(spec.dilution, 0.08)
        signal = eclipsing_binary_flux(time, period, t0, k_eb, a_rs * 1.26, b, spec.sb_ratio, dilution)
        truth.update(period_days=period, t0=t0, depth=float(1.0 - signal.min()), radius_ratio=k_eb,
                     sb_ratio=spec.sb_ratio, dilution=dilution, impact_param=b)
        if spec.kind == "blend":  # light from a background binary pulls the photocentre during eclipse
            centroid_shift = (1.0 - signal) * 1.5  # the binary sits 1.5 px from the target
    else:
        signal = np.ones(n)

    # low-frequency stellar variability + spacecraft thermal drift
    phase = rng.uniform(0, 2 * np.pi, 3)
    p_rot = rng.uniform(6.0, 14.0)
    trend = 1.0 + spec.variability_amp * (
        np.sin(2 * np.pi * time / p_rot + phase[0])
        + 0.4 * np.sin(2 * np.pi * time / (p_rot / 2.1) + phase[1])
    ) + 0.5 * spec.variability_amp * ((time - time[0]) / spec.baseline_days - 0.5)
    sigma = spec.noise_ppm * 1e-6
    flux = signal * trend + rng.normal(0.0, sigma, n)
    n_out = int(spec.outlier_fraction * n)
    if n_out:
        where = rng.choice(n, n_out, replace=False)
        flux[where] += rng.choice([-1, 1], n_out) * rng.uniform(8, 20, n_out) * sigma
    scale = 15000.0  # e-/s, so that median normalisation is exercised
    return {
        "time": time,
        "flux": flux * scale,
        "flux_err": np.full(n, sigma * scale),
        "centroid_col": 1000.0 + rng.normal(0, 0.0008, n) + centroid_shift,
        "centroid_row": 500.0 + rng.normal(0, 0.0008, n) + 0.6 * centroid_shift,
        "truth": truth,
    }
