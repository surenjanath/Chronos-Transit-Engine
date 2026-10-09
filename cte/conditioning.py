"""Signal conditioning: normalization, outlier rejection, detrending (PRD §2.1, §2.2)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter


def median_normalize(flux: np.ndarray) -> tuple[np.ndarray, float]:
    """F~(t_i) = F_raw(t_i) / median(F_raw)."""
    med = float(np.nanmedian(flux))
    if not np.isfinite(med) or med == 0:
        raise ValueError("flux median is zero or undefined; cannot normalize")
    return flux / med, med


def rolling_mean_std(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    window = max(3, min(window, len(x)))
    d = x - 1.0  # keep precision when subtracting squares
    mu = uniform_filter1d(d, window, mode="nearest")
    var = uniform_filter1d(d * d, window, mode="nearest") - mu * mu
    return mu + 1.0, np.sqrt(np.clip(var, 0.0, None))


def sigma_clip(flux: np.ndarray, k: float = 5.0, window: int = 101, max_iter: int = 10) -> tuple[np.ndarray, int]:
    """Iteratively reject |F~ - mu_roll| > k * sigma_roll. Returns (keep mask, iterations)."""
    keep = np.isfinite(flux)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        idx = np.flatnonzero(keep)
        if len(idx) < window:
            break
        x = flux[idx]
        mu, sd = rolling_mean_std(x, window)
        bad = np.abs(x - mu) > k * sd
        if not bad.any():
            break
        keep[idx[bad]] = False
    return keep, iterations


def window_cadences(time: np.ndarray, window_days: float) -> int:
    dt = float(np.median(np.diff(time)))
    w = int(round(window_days / dt))
    return max(5, w + 1 if w % 2 == 0 else w)


def savgol_detrend(
    time: np.ndarray,
    flux: np.ndarray,
    window_days: float = 1.5,
    polyorder: int = 3,
    transit_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Savitzky-Golay trend F_trend (degree `polyorder`, window 2m+1 cadences).

    The filter runs independently on each contiguous segment (data gaps would
    otherwise be treated as adjacent cadences). When `transit_mask` is given,
    in-transit samples are replaced by a linear bridge so the trend does not
    absorb the transit signal.
    """
    window = window_cadences(time, window_days)
    dt = float(np.median(np.diff(time)))
    breaks = np.flatnonzero(np.diff(time) > max(0.5, 20 * dt)) + 1
    trend = np.empty_like(flux)
    for seg in np.split(np.arange(len(time)), breaks):
        y = flux[seg].copy()
        if transit_mask is not None:
            m = transit_mask[seg]
            if m.any() and (~m).sum() >= 2:
                # bridge from locally averaged out-of-transit flux, not from two noisy edge samples
                smooth = uniform_filter1d(y[~m], max(3, window // 4), mode="nearest")
                y[m] = np.interp(time[seg][m], time[seg][~m], smooth)
        w = min(window, len(seg) if len(seg) % 2 == 1 else len(seg) - 1)
        if w <= polyorder:
            trend[seg] = np.median(y) if len(y) else 1.0
        else:
            trend[seg] = savgol_filter(y, w, polyorder, mode="interp")
    return trend, window


def red_noise_ratio(time: np.ndarray, resid: np.ndarray, bin_days: float = 0.125) -> float:
    """How far the scatter of 3-hour averages exceeds what white noise would give (1.0 = purely white).

    Robust estimators are used throughout, so the handful of bins that fall in a transit do not drive the result.
    """
    idx = np.floor((time - time[0]) / bin_days).astype(np.intp)
    counts = np.bincount(idx)
    sums = np.bincount(idx, resid)
    ok = counts >= max(2, int(0.5 * np.median(counts[counts > 0])))
    if ok.sum() < 20:
        return 1.0
    means = sums[ok] / counts[ok]
    mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))  # noqa: E731
    white = mad(resid) / np.sqrt(np.median(counts[ok]))
    return float(mad(means) / white) if white > 0 else 1.0


def choose_detrend_window(time: np.ndarray, flux: np.ndarray, base_days: float, polyorder: int = 3,
                          floor_days: float = 0.5, tolerance: float = 1.6) -> tuple[float, list[dict]]:
    """Pick the longest Savitzky-Golay window that actually removes the star's variability.

    A long window protects transit depth, but a star that rotates or pulsates in about a day is left with
    correlated residuals that swamp any transit. Windows are tried from the configured length downward and the
    first whose residuals are close to white is used.
    """
    tried = []
    window = base_days
    candidates = []
    while window >= floor_days - 1e-9:
        candidates.append(round(window, 4))
        window /= 1.5
    for w in candidates:
        trend, _ = savgol_detrend(time, flux, w, polyorder)
        ratio = red_noise_ratio(time, flux / trend - 1.0)
        tried.append({"window_days": w, "red_noise_ratio": round(ratio, 3)})
        if ratio <= tolerance:
            return w, tried
    best = min(tried, key=lambda x: x["red_noise_ratio"])
    return best["window_days"], tried


def periodic_variability_model(time: np.ndarray, flux: np.ndarray, period: float, nbins: int = 60) -> np.ndarray:
    """The star's repeating waveform at `period`: phase-binned medians, interpolated around the cycle.

    Used to divide out strictly periodic stellar variability (pulsation, rotation, ellipsoidal modulation) that
    is not a transit, in the manner of the harmonic-removal step of the Kepler pipeline.
    """
    phase = ((time - time[0]) / period) % 1.0
    idx = np.minimum((phase * nbins).astype(np.intp), nbins - 1)
    order = np.lexsort((flux, idx))
    idx_s, flux_s = idx[order], flux[order]
    counts = np.bincount(idx_s, minlength=nbins)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    good = counts >= 3
    if good.sum() < nbins // 2:
        return np.ones_like(flux)
    med = np.empty(nbins)
    med[good] = 0.5 * (flux_s[starts[good] + (counts[good] - 1) // 2] + flux_s[starts[good] + counts[good] // 2])
    centres = (np.arange(nbins) + 0.5) / nbins
    xp = np.concatenate((centres[good] - 1.0, centres[good], centres[good] + 1.0))  # periodic interpolation
    fp = np.tile(med[good], 3)
    model = np.interp(phase, xp, fp)
    return model / np.median(model)
