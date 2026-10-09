"""Physics & feature engine: analytical parameters and ML tensors (PRD §2.4, §3.2, §4)."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .config import AU_M, DAYS_PER_YEAR, R_EARTH_M, R_SUN_M


@dataclass
class StellarParameters:
    t_eff_k: float
    r_star_solar: float
    m_star_solar: float


@dataclass
class DerivedExoplanetParameters:
    period_days: float
    transit_depth_ppm: float
    radius_ratio: float
    radius_earth: float
    semi_major_axis_au: float
    transit_duration_hours: float
    equilibrium_temp_k: float


class TransitAnalyticsEngine:
    def __init__(self, stellar_params: StellarParameters, bond_albedo: float = 0.3):
        self.star = stellar_params
        self.ab = bond_albedo
        self.r_sun_meters = R_SUN_M
        self.au_meters = AU_M
        self.r_earth_meters = R_EARTH_M

    def compute_all_parameters(
        self, period_days: float, raw_depth: float, impact_param: float = 0.0
    ) -> DerivedExoplanetParameters:
        # 1. Depth & radius ratio: delta = (Rp / Rstar)^2
        k = math.sqrt(max(0.0, raw_depth))
        r_star_m = self.star.r_star_solar * self.r_sun_meters
        r_p_earth = k * r_star_m / self.r_earth_meters

        # 2. Kepler's third law: a [AU] = (M*/Msun)^(1/3) (P / 365.256 d)^(2/3)
        p_years = period_days / DAYS_PER_YEAR
        a_au = (self.star.m_star_solar * p_years ** 2) ** (1.0 / 3.0)
        a_meters = a_au * self.au_meters

        # 3. Circular-orbit duration: T_dur = (P/pi) asin( (R*/a) sqrt((1+k)^2 - b^2) / sin i )
        cos_i = min(1.0, impact_param * r_star_m / a_meters)
        sin_i = math.sqrt(max(1e-12, 1.0 - cos_i ** 2))
        term_sq = (1.0 + k) ** 2 - impact_param ** 2
        arg = (r_star_m / a_meters) * math.sqrt(term_sq) / sin_i if term_sq > 0 else 2.0
        if arg <= 1.0:
            duration_hours = (period_days / math.pi) * math.asin(arg) * 24.0
        else:  # fallback for grazing geometries
            duration_hours = (period_days * 24.0 / math.pi) * (r_star_m / a_meters)

        # 4. Equilibrium temperature: Teq = T* sqrt(R* / 2a) (1 - A_B)^(1/4)
        t_eq_k = self.star.t_eff_k * math.sqrt(r_star_m / (2.0 * a_meters)) * (1.0 - self.ab) ** 0.25

        return DerivedExoplanetParameters(
            period_days=round(period_days, 6),
            transit_depth_ppm=round(raw_depth * 1e6, 2),
            radius_ratio=round(k, 6),
            radius_earth=round(r_p_earth, 3),
            semi_major_axis_au=round(a_au, 5),
            transit_duration_hours=round(duration_hours, 3),
            equilibrium_temp_k=round(t_eq_k, 2),
        )

    @staticmethod
    def phase_fold(timestamps: np.ndarray, flux: np.ndarray, period: float, t0: float):
        """phi = ((t - t0 + P/2) mod P - P/2) / P, sorted, on [-0.5, 0.5)."""
        half_p = period / 2.0
        phase = ((timestamps - t0 + half_p) % period - half_p) / period
        order = np.argsort(phase, kind="stable")
        return phase[order], flux[order], order


def derive(star: StellarParameters, period_days: float, depth: float, impact_param: float = 0.0,
           bond_albedo: float = 0.3) -> dict:
    out = asdict(TransitAnalyticsEngine(star, bond_albedo).compute_all_parameters(period_days, depth, impact_param))
    a_over_rstar = out["semi_major_axis_au"] * AU_M / (star.r_star_solar * R_SUN_M)
    out["a_over_rstar"] = round(a_over_rstar, 3)
    out["radius_jupiter"] = round(out["radius_earth"] / 11.209, 3)
    out["insolation_earth"] = round(
        (star.r_star_solar ** 2) * (star.t_eff_k / 5772.0) ** 4 / out["semi_major_axis_au"] ** 2, 2
    )
    return out


def binned_median(index: np.ndarray, values: np.ndarray, nbins: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised per-bin median. Returns (medians with NaN for empty bins, counts)."""
    ok = (index >= 0) & (index < nbins)
    index, values = index[ok], values[ok]
    order = np.lexsort((values, index))
    index, values = index[order], values[order]
    counts = np.bincount(index, minlength=nbins)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    med = np.full(nbins, np.nan)
    has = counts > 0
    lo = starts[has] + (counts[has] - 1) // 2
    hi = starts[has] + counts[has] // 2
    med[has] = 0.5 * (values[lo] + values[hi])
    return med, counts


def global_view(phase: np.ndarray, flux: np.ndarray, num_bins: int = 201) -> np.ndarray:
    """Length-201 vector over phase [-0.5, 0.5], median aggregation; empty bins = 1.0 baseline."""
    idx = np.floor((phase + 0.5) * num_bins).astype(np.intp)
    idx[idx == num_bins] = num_bins - 1
    med, _ = binned_median(idx, flux, num_bins)
    med[np.isnan(med)] = 1.0
    return med


def local_view(phase: np.ndarray, flux: np.ndarray, q: float, num_bins: int = 61) -> np.ndarray:
    """Length-61 vector over phase [-2.5 q, 2.5 q]; empty bins filled by linear interpolation."""
    half = 2.5 * q
    idx = np.floor((phase + half) / (2 * half) * num_bins).astype(np.intp)
    med, _ = binned_median(idx, flux, num_bins)
    good = ~np.isnan(med)
    if not good.any():
        return np.ones(num_bins)
    centers = np.arange(num_bins)
    med[~good] = np.interp(centers[~good], centers[good], med[good])
    return med


def normalize_view(view: np.ndarray) -> np.ndarray:
    """Median-subtract and scale so the deepest bin is -1 (AstroNet convention)."""
    v = view - np.median(view)
    scale = abs(float(v.min()))
    return (v / scale if scale > 0 else v).astype(np.float32)


def shape_geometry(period_days: float, depth: float, t14_days: float, t12_days: float) -> dict:
    """Impact parameter, a/R* and mean stellar density from the transit shape alone (Seager & Mallen-Ornelas 2003).

    Uses only the light curve (depth, total duration T14, ingress time T12), so comparing the result with the
    catalogue star is an independent check: a planet on the catalogued star must give a consistent density.
    """
    k = math.sqrt(max(depth, 0.0))
    t23 = t14_days - 2.0 * t12_days
    out = {"valid": False, "impact_param": None, "a_over_rstar": None, "density_solar": None, "t23_hours": t23 * 24.0}
    if not (k > 0 and t14_days > 0 and t23 > 0 and t14_days < 0.5 * period_days):
        return out
    s14 = math.sin(math.pi * t14_days / period_days) ** 2
    ratio = math.sin(math.pi * t23 / period_days) ** 2 / s14
    if ratio >= 1.0:
        return out
    b2 = ((1 - k) ** 2 - ratio * (1 + k) ** 2) / (1 - ratio)
    b = math.sqrt(min(max(b2, 0.0), (1 + k) ** 2 * 0.98))
    a_rs = math.sqrt(((1 + k) ** 2 - b * b * (1 - s14)) / s14)
    # Kepler's third law: rho* = 3 pi (a/R*)^3 / (G P^2); in solar units with P in days
    density = 0.01342 * a_rs ** 3 / period_days ** 2  # rho_sun: (a/R*)^3 / P[d]^2 = 74.5 for the Sun
    out.update(valid=True, impact_param=round(b, 3), a_over_rstar=round(a_rs, 3), density_solar=round(density, 4))
    return out


def validate_transit(period_days: float, duration_days: float) -> None:
    """Reject non-physical geometry: a transit must have a finite, positive duration shorter than the orbit."""
    if not (math.isfinite(period_days) and math.isfinite(duration_days)) or period_days <= 0 or duration_days <= 0:
        raise ValueError(f"non-physical transit: period {period_days} d, duration {duration_days} d")
    if duration_days >= period_days:
        raise ValueError(f"non-physical transit: duration {duration_days:.4f} d is not shorter than the period {period_days:.4f} d")


LOCAL_CHANNELS = ("all transits", "odd transits", "even transits", "phase 0.5")


def local_channels(time: np.ndarray, flat: np.ndarray, period: float, t0: float, q: float, num_bins: int = 61) -> np.ndarray:
    """The classifier's local input, shape (4, num_bins): every transit, odd transits only, even transits only, and
    the same window half an orbit later. All four share one scale (the depth of the first), so a missing odd or even
    dip, or a secondary eclipse, appears at its true relative size."""
    half = 2.5 * q
    phase = ((time - t0 + 0.5 * period) % period - 0.5 * period) / period
    epoch = np.round((time - t0) / period).astype(int)

    def view(mask, centre_shift=0.0):
        ph = phase[mask] if centre_shift == 0.0 else (((time[mask] - t0) / period) % 1.0 - 0.5)
        idx = np.floor((ph + half) / (2 * half) * num_bins).astype(np.intp)
        med, _ = binned_median(idx, flat[mask], num_bins)
        good = ~np.isnan(med)
        if good.sum() < num_bins // 2:
            return None
        centres = np.arange(num_bins)
        med[~good] = np.interp(centres[~good], centres[good], med[good])
        return med

    every = np.ones(len(time), bool)
    main = view(every)
    if main is None:
        main = np.ones(num_bins)
    base = float(np.median(main))
    scale = abs(float((main - base).min())) or 1.0
    out = np.empty((4, num_bins), np.float32)
    out[0] = (main - base) / scale
    for row, v in ((1, view(epoch % 2 != 0)), (2, view(epoch % 2 == 0))):
        out[row] = out[0] if v is None else (v - np.median(v)) / scale
    sec = view(every, centre_shift=0.5)
    out[3] = 0.0 if sec is None else (sec - np.median(sec)) / scale
    return out
