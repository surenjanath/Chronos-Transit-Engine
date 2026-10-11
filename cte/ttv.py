"""Transit timing: when did each transit happen, against when a strictly periodic orbit says it should have?

A planet alone keeps perfect time. One pulled by a neighbour arrives early and late in a slow wave, so timing
variations are evidence of another body in the system. Each transit is timed by sliding the fitted transit shape
across it; a transit can only be timed when it stands out of the noise on its own, which rules out shallow signals.
"""
from __future__ import annotations

import numpy as np

MIN_EVENT_SNR = 3.0  # below this a single transit carries no usable time
MIN_TRANSITS = 8


def template(x: np.ndarray, t14: float, t12: float, depth: float) -> np.ndarray:
    """A trapezoid transit: full depth between the inner contacts, straight ramps of length `t12` either side."""
    half, ramp = 0.5 * t14, max(min(t12, 0.5 * t14), 1e-4)
    return -depth * np.clip((half - np.abs(x)) / ramp, 0.0, 1.0)


def transit_times(time, flat, period: float, t0: float, t14_hours: float, t12_hours: float, depth_ppm: float) -> dict:
    """Mid-time of every well-covered transit, its error, and how the set compares with a constant period."""
    t, f = np.asarray(time, dtype=float), np.asarray(flat, dtype=float)
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    t14, t12, depth = t14_hours / 24.0, t12_hours / 24.0, depth_ppm * 1e-6
    x = (t - t0 + 0.5 * period) % period - 0.5 * period
    sigma = float(1.4826 * np.median(np.abs(f[np.abs(x) > t14] - np.median(f[np.abs(x) > t14]))))
    cadence = float(np.median(np.diff(t)))
    per_transit = max(t14 / cadence, 1.0)
    event_snr = depth / sigma * np.sqrt(per_transit)
    out = {"period_days": period, "t0": t0, "sigma_ppm": sigma * 1e6, "event_snr": round(float(event_snr), 2), "measurable": False,
           "transits": []}
    if event_snr < MIN_EVENT_SNR:
        out["reason"] = (f"single transits are too shallow to time: each one stands only {event_snr:.1f} times above the noise "
                         f"(at least {MIN_EVENT_SNR:.0f} is needed)")
        return out
    epoch = np.round((t - t0) / period).astype(int)
    window, reach = 1.5 * t14, 0.75 * t14
    shifts = np.linspace(-reach, reach, 241)
    for n in np.unique(epoch[np.abs(x) < window]):
        m = (epoch == n) & (np.abs(x) < window)
        xs, ys = x[m], f[m]
        if m.sum() < 0.7 * (2 * window / cadence) or (np.abs(xs) < 0.5 * t14).sum() < 0.6 * per_transit:
            continue  # a transit with a gap in it gives a biased time
        chi = np.empty(len(shifts))
        for i, s in enumerate(shifts):
            model = template(xs - s, t14, t12, depth)
            chi[i] = np.sum((ys - model - np.mean(ys - model)) ** 2)
        chi /= sigma ** 2
        k = int(chi.argmin())
        if k < 3 or k > len(shifts) - 4:
            continue  # the best fit ran to the edge of the search: no transit was found in this window
        inside = np.flatnonzero(chi < chi[k] + 1.0)
        err = max(0.5 * (shifts[inside[-1]] - shifts[inside[0]]), shifts[1] - shifts[0])
        out["transits"].append({"epoch": int(n), "time": float(t0 + n * period + shifts[k]), "error_days": float(err)})
    tr = out["transits"]
    if len(tr) < MIN_TRANSITS:
        out["reason"] = f"only {len(tr)} transits are complete enough to time (at least {MIN_TRANSITS} are needed)"
        return out
    n, obs, err = (np.array([v[key] for v in tr]) for key in ("epoch", "time", "error_days"))
    w = 1.0 / err ** 2
    slope, intercept = np.polyfit(n, obs, 1, w=np.sqrt(w))  # the best constant period through the measured times
    resid = obs - (intercept + slope * n)
    for v, r in zip(tr, resid):
        v["o_minus_c_minutes"] = round(float(r * 1440.0), 3)
        v["error_minutes"] = round(v.pop("error_days") * 1440.0, 3)
    chi2 = float(np.sum(w * resid ** 2) / max(len(n) - 2, 1))
    rng = np.random.default_rng(20261011)

    def best_wave(r):
        """Largest drop in chi-square from fitting one sine wave to the residuals, over periods from 4 transits up."""
        best = (0.0, None, 0.0)
        for cycles in np.linspace(1.0, len(n) / 4.0, 120):
            p = (n.max() - n.min() + 1) / cycles
            a = np.c_[np.sin(2 * np.pi * n / p), np.cos(2 * np.pi * n / p), np.ones(len(n))] * np.sqrt(w)[:, None]
            coef, *_ = np.linalg.lstsq(a, r * np.sqrt(w), rcond=None)
            drop = float(np.sum(w * r ** 2) - np.sum((r * np.sqrt(w) - a @ coef) ** 2))
            if drop > best[0]:
                best = (drop, p, float(np.hypot(coef[0], coef[1])))
        return best

    drop, wave_period, amplitude = best_wave(resid)
    trials = np.array([best_wave(rng.permutation(resid))[0] for _ in range(200)])
    p_value = float((1 + np.sum(trials >= drop)) / (1 + len(trials)))
    out.update(measurable=True, n_timed=len(tr), refit_period_days=float(slope), refit_t0=float(intercept),
               scatter_minutes=round(float(np.std(resid) * 1440.0), 2), median_error_minutes=round(float(np.median(err) * 1440.0), 2),
               chi2_reduced=round(chi2, 2), wave={"period_transits": round(float(wave_period), 1) if wave_period else None,
                                                  "period_days": round(float(wave_period * period), 1) if wave_period else None,
                                                  "amplitude_minutes": round(amplitude * 1440.0, 2), "false_alarm": round(p_value, 3)})
    out["detected"] = bool(p_value < 0.01)
    out["reading"] = (
        f"A timing wave is detected: {amplitude * 1440.0:.1f} minutes in amplitude, repeating every {wave_period * period:.0f} days, "
        f"with a {p_value * 100:.1f}% chance of arising from noise. That is the signature of another body pulling on this one."
        if out["detected"] else
        f"No timing variations are detected. The {len(tr)} timed transits scatter by {np.std(resid) * 1440.0:.1f} minutes about a constant period, "
        f"with a formal error of {np.median(err) * 1440.0:.1f} minutes on each; the best-fitting wave ({amplitude * 1440.0:.1f} minutes) has a "
        f"{p_value * 100:.0f}% chance of arising from noise. This neither supports nor counts against other planets in the system: "
        f"small or non-resonant neighbours leave variations below this precision.")
    return out
