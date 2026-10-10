"""Box Least Squares periodogram (PRD §2.3).

SR(P, q, t0) = sqrt( s^2 / (r (1 - r)) ),  r = sum_in w_i,  s = sum_in w_i (F_flat - 1),
with inverse-variance weights w_i normalised to unit sum.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from scipy.optimize import least_squares

MAX_PERIODS = 60000
MAX_SEARCH_POINTS = 20000


@dataclass
class BLSResult:
    period: float
    t0: float
    q: float
    duration_days: float
    depth: float
    sr: float
    sde: float
    n_periods: int
    period_min: float
    period_max: float
    freq_step: float

    def to_dict(self) -> dict:
        return asdict(self)


def normalized_weights(flux_err: np.ndarray | None, n: int) -> np.ndarray:
    if flux_err is None or not np.all(np.isfinite(flux_err)) or np.any(flux_err <= 0):
        return np.full(n, 1.0 / n)
    w = flux_err ** -2.0
    return w / w.sum()


def period_grid(baseline: float, pmin: float, pmax: float, qmin: float, oversample: int) -> tuple[np.ndarray, float]:
    """Uniform-in-frequency grid; the step keeps phase smear below qmin/oversample."""
    pmax = min(pmax, baseline / 2.0)  # require at least two transits
    if pmax <= pmin:
        raise ValueError(f"baseline of {baseline:.2f} d is too short to search periods >= {pmin} d")
    fmin, fmax = 1.0 / pmax, 1.0 / pmin
    df = qmin / (oversample * baseline)
    n = int(np.ceil((fmax - fmin) / df)) + 1
    if n > MAX_PERIODS:
        n = MAX_PERIODS
        df = (fmax - fmin) / (n - 1)
    return 1.0 / (fmin + df * np.arange(n))[::-1], df


def bls_periodogram(
    time: np.ndarray,
    flux: np.ndarray,
    weights: np.ndarray,
    periods: np.ndarray,
    qmin: float = 0.01,
    qmax: float = 0.1,
    nbins: int = 200,
    progress: Callable[[float], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return SR(P) maximised over (q, t0), with the best phase-bin start and width per period."""
    tt = time - time[0]
    y = flux - 1.0
    wy = weights * y
    widths = np.unique(np.clip(np.round(np.geomspace(qmin, qmax, 12) * nbins).astype(int), 1, nbins // 2))
    kmax = int(widths.max())
    lo = np.arange(nbins)[None, :].repeat(len(widths), 0)
    hi = lo + widths[:, None]

    sr = np.zeros(len(periods))
    start = np.zeros(len(periods), dtype=np.int32)
    width = np.zeros(len(periods), dtype=np.int32)
    cr = np.zeros(nbins + kmax + 1)
    cs = np.zeros(nbins + kmax + 1)
    step = max(1, len(periods) // 50)
    # The loop below runs tens of thousands of times over every sample, so it reuses its buffers.
    phase = np.empty_like(tt)
    whole = np.empty_like(tt)
    idx = np.empty(len(tt), dtype=np.intp)
    for j, period in enumerate(periods):
        np.divide(tt, period, out=phase)  # orbits elapsed; dropping the whole turns leaves the phase
        np.floor(phase, out=whole)        # (x - floor(x) is exactly x % 1 for these positive values, and many times faster)
        phase -= whole
        phase *= nbins
        idx[:] = phase
        np.minimum(idx, nbins - 1, out=idx)
        rb = np.bincount(idx, weights, nbins)
        sb = np.bincount(idx, wy, nbins)
        np.cumsum(np.concatenate((rb, rb[:kmax])), out=cr[1:])
        np.cumsum(np.concatenate((sb, sb[:kmax])), out=cs[1:])
        r = cr[hi] - cr[lo]
        s = cs[hi] - cs[lo]
        ok = (s < 0) & (r > 1e-9) & (r < 1 - 1e-9)
        val = np.zeros_like(r)
        val[ok] = s[ok] ** 2 / (r[ok] * (1.0 - r[ok]))
        flat = int(val.argmax())
        wi, bi = divmod(flat, nbins)
        sr[j] = np.sqrt(val[wi, bi])
        start[j] = bi
        width[j] = widths[wi]
        if progress and j % step == 0:
            progress(j / len(periods))
    return sr, start, width


_pool = None
_pool_lock = None


def _chunk(args):
    time, flux, weights, periods, qmin, qmax, nbins = args
    return bls_periodogram(time, flux, weights, periods, qmin, qmax, nbins)


def _get_pool(workers: int):
    """One long-lived pool of worker processes, created on first use and shared by every search in this process."""
    global _pool, _pool_lock
    import multiprocessing
    import threading
    from concurrent.futures import ProcessPoolExecutor

    if _pool_lock is None:
        _pool_lock = threading.Lock()
    with _pool_lock:
        if _pool is None:
            _pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
        return _pool


def bls_periodogram_parallel(time, flux, weights, periods, qmin, qmax, nbins=200, progress=None):
    """The same periodogram with the trial periods split across CPU cores. Falls back to the serial loop when
    parallelism is disabled (CTE_BLS_WORKERS=1), unavailable, or not worth the overhead."""
    import os

    workers = int(os.environ.get("CTE_BLS_WORKERS", min(8, max(1, (os.cpu_count() or 2) - 2))))
    if workers <= 1 or len(periods) * len(time) < 5e7:
        return bls_periodogram(time, flux, weights, periods, qmin, qmax, nbins, progress)
    try:
        pool = _get_pool(workers)
        chunks = np.array_split(np.arange(len(periods)), workers * 3)
        futures = [pool.submit(_chunk, (time, flux, weights, periods[idx], qmin, qmax, nbins)) for idx in chunks]
        parts = []
        for i, fut in enumerate(futures):
            parts.append(fut.result())
            if progress:
                progress((i + 1) / len(futures))
        return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))
    except Exception:  # a broken pool must never fail a run
        global _pool
        _pool = None
        return bls_periodogram(time, flux, weights, periods, qmin, qmax, nbins, progress)


def box_stat(time, y, weights, period, t0, duration):
    x = np.abs((time - t0 + 0.5 * period) % period - 0.5 * period)
    inside = x < 0.5 * duration
    r = weights[inside].sum()
    s = (weights[inside] * y[inside]).sum()
    if r <= 0 or r >= 1 or s >= 0:
        return 0.0, 0.0
    return float(np.sqrt(s * s / (r * (1 - r)))), float(-s / (r * (1 - r)))


def refine_ephemeris(
    time: np.ndarray,
    flux: np.ndarray,
    weights: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    dperiod: float,
    n_period: int = 161,
    n_t0: int = 41,
) -> tuple[float, float]:
    """Maximise the un-binned box statistic over a fine (P, t0) grid around the BLS peak."""
    baseline = time[-1] - time[0]
    y = flux - 1.0
    for _ in range(3):
        margin = duration + dperiod * baseline / period
        near = np.abs((time - t0 + 0.5 * period) % period - 0.5 * period) < margin
        t_s, y_s, w_s = time[near], y[near], weights[near]
        if len(t_s) < 5:
            return period, t0
        w_total = 1.0  # weights are normalised over the full series
        p_grid = period + np.linspace(-dperiod, dperiod, n_period)
        t_grid = t0 + np.linspace(-0.5, 0.5, n_t0) * duration
        stat = np.zeros((n_period, n_t0))
        for i, p in enumerate(p_grid):
            x = np.abs((t_s[None, :] - t_grid[:, None] + 0.5 * p) % p - 0.5 * p)
            inside = x < 0.5 * duration
            r = inside @ w_s
            s = inside @ (w_s * y_s)
            ok = (s < 0) & (r > 0) & (r < w_total)
            stat[i, ok] = s[ok] ** 2 / (r[ok] * (1 - r[ok]))
        by_p = stat.max(axis=1)
        peak = by_p.max()
        if peak <= 0:
            return period, t0
        # centroid of the peak is steadier than the arg-max of a piecewise-constant surface
        wgt = np.clip(by_p - 0.9 * peak, 0, None)
        period = float((p_grid * wgt).sum() / wgt.sum())
        t0 = float(t_grid[stat[int(np.abs(p_grid - period).argmin())].argmax()])
        dperiod /= 8.0
    return period, t0


def bls_search(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray | None,
    pmin: float = 0.5,
    pmax: float = 30.0,
    qmin: float = 0.01,
    qmax: float = 0.1,
    oversample: int = 3,
    progress: Callable[[float], None] | None = None,
) -> tuple[BLSResult, np.ndarray, np.ndarray]:
    """Full BLS search. Returns (best-fit result, trial periods, SR spectrum)."""
    weights = normalized_weights(flux_err, len(time))
    baseline = float(time[-1] - time[0])
    periods, df = period_grid(baseline, pmin, pmax, qmin, oversample)
    nbins = 200
    # Very long series (short-cadence data) are averaged in time for the grid search only; the box statistic is
    # insensitive to this as long as a bin is shorter than the shortest transit searched. Refinement uses every cadence.
    step = int(np.ceil(len(time) / MAX_SEARCH_POINTS))
    if step > 1 and step * float(np.median(np.diff(time))) < qmin * pmin:
        n = len(time) // step * step
        tb, fb = time[:n].reshape(-1, step).mean(1), flux[:n].reshape(-1, step).mean(1)
        wb = weights[:n].reshape(-1, step).sum(1)
        sr, start, width = bls_periodogram_parallel(tb, fb, wb / wb.sum(), periods, qmin, qmax, nbins, progress)
        start = np.round(start + (tb[0] - time[0]) / periods * nbins).astype(np.int32) % nbins  # same phase origin
    else:
        sr, start, width = bls_periodogram_parallel(time, flux, weights, periods, qmin, qmax, nbins, progress)
    j = int(sr.argmax())
    period = float(periods[j])
    q = float(width[j]) / nbins
    t0 = float(time[0] + (start[j] + 0.5 * width[j]) / nbins * period)
    best_sr = float(sr[j])

    # Alias check. On a long baseline the grid above is capped and coarser than a long-period transit needs, which
    # smears the true peak and lets P/2 or P/4 (whose duty cycle is larger, so they tolerate the smear) win.
    # Re-examine the peak and its simple multiples on a locally fine grid and keep whichever is truly strongest.
    df_fine = qmin / (oversample * baseline)
    if df > 1.5 * df_fine:
        p_hi = min(pmax, baseline / 2.0)
        for _ in range(3):  # a quarter-period alias needs two steps to climb back to the true period
            improved = False
            base_period = period
            for mult in (1.0, 2.0, 3.0, 0.5, 1.0 / 3.0):
                pc = base_period * mult
                if not (pmin <= pc <= p_hi):
                    continue
                local_f = 1.0 / pc + np.arange(-1.5 * df, 1.5 * df + df_fine, df_fine) / max(mult, 1.0)
                local_p = 1.0 / local_f[(local_f > 1.0 / p_hi) & (local_f < 1.0 / pmin)]
                if len(local_p) == 0:
                    continue
                sr_l, st_l, wd_l = bls_periodogram(time, flux, weights, local_p, qmin, qmax, nbins)
                k = int(sr_l.argmax())
                # a multiple must be clearly better to replace the current peak
                if sr_l[k] > best_sr * (1.0 if mult == 1.0 else 1.05):
                    best_sr = float(sr_l[k])
                    period, q = float(local_p[k]), float(wd_l[k]) / nbins
                    t0 = float(time[0] + (st_l[k] + 0.5 * wd_l[k]) / nbins * period)
                    improved = improved or mult != 1.0
            if not improved:
                break
    duration = q * period

    period, t0 = refine_ephemeris(time, flux, weights, period, t0, duration, 1.5 * period * period * df)
    # centre t0 on the first transit inside the observing window
    t0 = float(t0 + np.ceil((time[0] - t0) / period) * period)
    stat, depth = box_stat(time, flux - 1.0, weights, period, t0, duration)
    spread = float(np.std(sr))
    sde = float((sr[j] - np.mean(sr)) / spread) if spread > 0 else 0.0
    result = BLSResult(
        period=period, t0=t0, q=q, duration_days=duration, depth=depth, sr=float(max(stat, best_sr)),
        sde=sde, n_periods=len(periods), period_min=float(periods[0]), period_max=float(periods[-1]),
        freq_step=float(df),
    )
    return result, periods, sr


def trapezoid(x: np.ndarray, depth: float, t14: float, t12: float, dt0: float, base: float) -> np.ndarray:
    t12 = max(min(t12, 0.5 * t14), 1e-6)
    return base - depth * np.clip((0.5 * t14 - np.abs(x - dt0)) / t12, 0.0, 1.0)


def fit_trapezoid(x: np.ndarray, flux: np.ndarray, depth0: float, duration0: float, min_duration: float = 0.0) -> dict:
    """Fit a trapezoid to phase-folded flux (x = time from mid-transit, days).

    Returns flat-bottom depth, total duration T14, ingress time T12 and the
    shape metric v_shape = 2 T12 / T14 (-> 1 for a V-shaped / grazing eclipse).
    """
    sel = np.abs(x) < 3.0 * duration0
    xs, fs = x[sel], flux[sel]
    depth0 = max(depth0, 1e-6)
    fallback = {"depth": depth0, "t14": duration0, "t12": 0.1 * duration0, "dt0": 0.0, "baseline": 1.0,
                "v_shape": 0.2, "converged": False}
    if len(xs) < 20:
        return fallback

    def resid(p):
        return trapezoid(xs, p[0], p[1], p[2] * 0.5 * p[1], p[3], p[4]) - fs

    best = None
    for frac0 in (0.2, 0.9):  # box-like and V-like starting points
        try:
            sol = least_squares(
                resid, [depth0, duration0, frac0, 0.0, 1.0],
                # a transit cannot be resolved below one cadence: without this floor an under-sampled dip is
                # fitted as an arbitrarily narrow, arbitrarily deep spike
                bounds=([0.0, min(max(0.2 * duration0, min_duration), 2.9 * duration0), 0.01, -duration0, 0.9],
                        [1.0, 3.0 * duration0, 1.0, duration0, 1.1]),
                loss="soft_l1", f_scale=max(float(np.std(fs)), 1e-6), x_scale=[depth0, duration0, 0.3, duration0, 1e-3],
            )
        except ValueError:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        return fallback
    depth, t14, frac, dt0, base = (float(v) for v in best.x)
    return {"depth": depth / base, "t14": t14, "t12": frac * 0.5 * t14, "dt0": dt0, "baseline": base,
            "v_shape": frac, "converged": bool(best.success)}


def refine_ephemeris_shape(time, flux, weights, period, t0, depth, t14, t12, dperiod,
                           n_period: int = 121, n_t0: int = 41) -> tuple[float, float]:
    """Least-squares (P, t0) for a fixed trapezoid shape.

    Unlike the box statistic, chi^2 against a trapezoid varies smoothly with
    P and t0, so the minimum is sharp enough for |dP|/P < 1e-4 (PRD §6).
    """
    baseline = time[-1] - time[0]
    y = flux - 1.0
    t12 = max(min(t12, 0.5 * t14), 1e-6)
    dt0 = 0.5 * t14
    for _ in range(4):
        margin = 0.5 * t14 + dt0 + dperiod * baseline / period
        near = np.abs((time - t0 + 0.5 * period) % period - 0.5 * period) < margin
        t_s, y_s, w_s = time[near], y[near], weights[near]
        if len(t_s) < 10:
            break
        p_grid = period + np.linspace(-dperiod, dperiod, n_period)
        t_grid = t0 + np.linspace(-dt0, dt0, n_t0)
        chi = np.empty((n_period, n_t0))
        for i, p in enumerate(p_grid):
            x = np.abs((t_s[None, :] - t_grid[:, None] + 0.5 * p) % p - 0.5 * p)
            model = depth * np.clip((0.5 * t14 - x) / t12, 0.0, 1.0)
            chi[i] = ((y_s[None, :] + model) ** 2) @ w_s
        i, j = np.unravel_index(int(chi.argmin()), chi.shape)
        period, t0 = float(p_grid[i]), float(t_grid[j])
        if 0 < i < n_period - 1:  # parabolic interpolation of the minimum along P
            a, b, c = chi[i - 1, j], chi[i, j], chi[i + 1, j]
            denom = a - 2 * b + c
            if denom > 0:
                period += float(0.5 * (a - c) / denom * (p_grid[1] - p_grid[0]))
        dperiod /= 6.0
        dt0 /= 4.0
    return period, t0


def fourier_comb_period(time: np.ndarray, flux: np.ndarray, pmin: float = 0.5, pmax: float = 30.0,
                        harmonics: int = 10) -> float:
    """Non-parametric Fourier comb filter: escalation path when BLS misses tolerance (PRD §6).

    A periodic transit train puts power at every harmonic of 1/P, so the comb
    C(f) = sum_h A(h f) of the amplitude spectrum peaks at the fundamental.
    """
    dt = float(np.median(np.diff(time)))
    grid = np.arange(time[0], time[-1] + dt, dt)
    y = np.interp(grid, time, flux - 1.0)
    idx = np.clip(np.searchsorted(time, grid), 1, len(time) - 1)
    y[(time[idx] - time[idx - 1]) > 5 * dt] = 0.0  # zero-fill data gaps
    n = int(2 ** np.ceil(np.log2(len(y) * 8)))  # zero-padding refines the frequency grid
    amp = np.abs(np.fft.rfft(y - y.mean(), n))
    freqs = np.fft.rfftfreq(n, dt)
    pmax = min(pmax, (time[-1] - time[0]) / 2.0)
    trial = np.arange(1.0 / pmax, 1.0 / pmin, freqs[1] / harmonics)
    comb = np.zeros_like(trial)
    for h in range(1, harmonics + 1):
        comb += np.interp(trial * h, freqs, amp, right=0.0)
    return float(1.0 / trial[int(comb.argmax())])
