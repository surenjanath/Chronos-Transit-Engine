"""False-positive diagnostics run before a candidate reaches the vetting queue (PRD §1)."""
from __future__ import annotations

import numpy as np

from .conditioning import savgol_detrend


def robust_std(x: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _mean_err(x: np.ndarray, sigma: float) -> tuple[float, float]:
    return float(np.mean(x)), sigma / np.sqrt(max(len(x), 1))


def model_shift(time, flat, period, t0, t14, depth, a_over_rp: float | None = None) -> dict | None:
    """Look for a second dip at *any* orbital phase (the Kepler Robovetter's model-shift test).

    The folded curve is scanned with a transit-width window. Besides the primary this yields the deepest other dip
    (secondary), the next deepest (tertiary) and the strongest brightening (positive). The scatter of the scanned
    depths away from the primary is the noise actually achieved, red noise included. A secondary is significant when
    it clears the false-alarm level for the number of independent phases searched and stands clearly above both the
    tertiary and the positive feature, which are what noise alone produces.

    A significant secondary still fits a planet if the occultation is no brighter than a perfect mirror of the
    planet's size would be (geometric albedo <= 1). Anything brighter is self-luminous: a star.
    """
    from scipy.special import erfcinv

    width = t14 / period
    if not (0.0 < width < 0.2) or depth <= 0:
        return None
    nb = int(np.clip(round(6.0 / width), 120, 6000))  # six bins across a transit
    phase = ((time - t0) / period) % 1.0
    idx = np.minimum((phase * nb).astype(np.intp), nb - 1)
    count = np.bincount(idx, minlength=nb).astype(float)
    total = np.bincount(idx, weights=1.0 - flat, minlength=nb)
    w = max(3, int(round(width * nb)) | 1)  # window of one transit duration, odd number of bins
    kernel = np.ones(w)
    pad = w // 2
    wrap = lambda v: np.convolve(np.concatenate([v[-pad:], v, v[:pad]]), kernel, mode="valid")
    n_win, s_win = wrap(count), wrap(total)
    centres = (np.arange(nb) + 0.5) / nb
    dist = np.minimum(centres, 1.0 - centres)  # phase distance from the primary
    # stay three durations clear of the primary: its wings and the detrending bridge leave structure there
    usable = (n_win >= 0.3 * np.median(n_win[n_win > 0])) & (dist > 3.0 * width)
    if usable.sum() < 20:
        return None
    amp = np.where(n_win > 0, s_win / np.maximum(n_win, 1.0), 0.0)
    base = float(np.median(amp[usable]))
    amp = amp - base
    # scale every window to the typical window size so sparsely covered phases are not over-weighted
    z_scale = np.sqrt(n_win / np.median(n_win[usable]))
    stat = amp * z_scale
    sigma = robust_std(stat[usable])
    if not sigma > 0:
        return None
    oot = np.abs((time - t0 + 0.5 * period) % period - 0.5 * period) > t14
    white = robust_std(flat[oot]) / np.sqrt(np.median(n_win[usable]))
    cand = np.where(usable, stat, -np.inf)
    i_sec = int(cand.argmax())

    def away(i):
        d = np.abs(centres - centres[i])
        return np.minimum(d, 1.0 - d) > 2.0 * width

    rest = usable & away(i_sec)
    i_pos = int(np.where(usable, stat, np.inf).argmin())
    # with a long transit on a short orbit there may be no phase left that is clear of both dips
    sec, ter, pos = float(stat[i_sec]), float(stat[rest].max()) if rest.any() else 0.0, float(-stat[i_pos])
    n_indep = max(2.0, 1.0 / width)
    fa1 = float(np.sqrt(2.0) * erfcinv(1.0 / n_indep))        # deepest of n independent phases, by chance
    fa2 = float(np.sqrt(2.0) * erfcinv(min(0.9, 2.0 / n_indep)))  # margin over the next-deepest feature
    margin = min(sec / sigma - fa1, (sec - ter) / sigma - fa2, (sec - pos) / sigma - fa2)
    sec_depth = float(amp[i_sec])
    albedo = float(sec_depth * a_over_rp ** 2) if a_over_rp else float("nan")
    return {"phase": float(centres[i_sec]), "depth_ppm": sec_depth * 1e6, "sigma": sec / sigma,
            "tertiary_sigma": ter / sigma, "positive_sigma": pos / sigma, "red_noise_factor": float(sigma / white) if white > 0 else 1.0,
            "fa1": fa1, "fa2": fa2, "margin": float(margin), "ratio": sec_depth / depth, "albedo": albedo,
            "significant": bool(margin > 0)}


def diagnose(time, flat, period, t0, t14, depth, params: dict, centroids=None) -> tuple[dict, list[dict]]:
    """Return (diagnostic measurements, list of raised flags)."""
    x = (time - t0 + 0.5 * period) % period - 0.5 * period
    epoch = np.round((time - t0) / period).astype(int)
    in_tr = np.abs(x) < 0.5 * t14
    core = np.abs(x) < 0.25 * t14
    oot = np.abs(x) > t14
    sigma = robust_std(flat[oot])
    diag: dict = {"oot_scatter_ppm": sigma * 1e6}
    flags: list[dict] = []

    def flag(code, severity, message):
        flags.append({"code": code, "severity": severity, "message": message})

    # transit count & signal-to-noise
    epochs_hit = [e for e in np.unique(epoch[in_tr]) if (in_tr & (epoch == e)).sum() >= 3]
    n_in = int(in_tr.sum())
    snr = float(depth / sigma * np.sqrt(n_in)) if sigma > 0 and n_in else 0.0
    diag.update(n_transits=len(epochs_hit), n_in_transit=n_in, snr=snr)
    if snr < 7.1:
        flag("LOW_SNR", "high", f"Transit S/N {snr:.1f} is below the 7.1 detection threshold")
    if len(epochs_hit) < 3:
        flag("FEW_TRANSITS", "medium", f"Only {len(epochs_hit)} transit(s) observed; period is weakly constrained")

    # odd / even depth consistency (a mismatch means the true period is 2P: an eclipsing binary)
    odd = core & (epoch % 2 != 0)
    even = core & (epoch % 2 == 0)
    if odd.sum() >= 5 and even.sum() >= 5:
        d_odd, e_odd = _mean_err(1.0 - flat[odd], sigma)
        d_even, e_even = _mean_err(1.0 - flat[even], sigma)
        sig = abs(d_odd - d_even) / np.hypot(e_odd, e_even)
        diag["odd_even"] = {"depth_odd_ppm": d_odd * 1e6, "depth_even_ppm": d_even * 1e6, "sigma": float(sig)}
        if sig > 3.0 and abs(d_odd - d_even) > 0.1 * depth:
            flag("ODD_EVEN_MISMATCH", "high",
                 f"Odd and even transit depths differ by {sig:.1f} sigma - likely an eclipsing binary at twice the period")

    # secondary eclipse at phase 0.5
    sec = np.abs(np.abs(x) - 0.5 * period) < 0.5 * t14
    ref = (np.abs(np.abs(x) - 0.5 * period) > t14) & oot
    if sec.sum() >= 5 and ref.sum() >= 5:
        d_sec, e_sec = _mean_err(1.0 - flat[sec], sigma)
        d_sec -= float(np.mean(1.0 - flat[ref]))
        sig = d_sec / e_sec if e_sec > 0 else 0.0
        diag["secondary"] = {"depth_ppm": d_sec * 1e6, "sigma": float(sig), "ratio": float(d_sec / depth) if depth else 0.0}
        if sig > 5.0 and d_sec > 0.05 * depth:
            flag("SECONDARY_ECLIPSE", "high",
                 f"Secondary eclipse of {d_sec * 1e6:.0f} ppm at phase 0.5 ({sig:.1f} sigma) - self-luminous companion")

    # a second dip anywhere else in the orbit (an eccentric binary's secondary need not sit at phase 0.5)
    a_over_rp = None
    if params.get("semi_major_axis_au") and params.get("radius_earth"):
        a_over_rp = params["semi_major_axis_au"] * 23454.8 / params["radius_earth"]  # AU -> Earth radii
    shift = model_shift(time, flat, period, t0, t14, depth, a_over_rp)
    if shift is not None:
        diag["model_shift"] = shift
        too_bright = (np.isfinite(shift["albedo"]) and shift["albedo"] > 1.0) or shift["ratio"] > 0.1
        if shift["significant"] and too_bright and "SECONDARY_ECLIPSE" not in [f["code"] for f in flags]:
            flag("SECONDARY_ECLIPSE", "high",
                 f"Second dip of {shift['depth_ppm']:.0f} ppm at orbital phase {shift['phase']:.2f} ({shift['sigma']:.1f} sigma) - "
                 f"too bright to be a planet's occultation, so the companion is self-luminous")

    # physical plausibility
    if params["radius_earth"] > 25.0:
        flag("RADIUS_TOO_LARGE", "high",
             f"Derived radius {params['radius_earth']:.1f} R_earth exceeds 2.2 R_jup - companion is likely stellar")
    theory = params["transit_duration_hours"]
    ratio = (t14 * 24.0) / theory if theory > 0 else float("nan")
    diag["duration_ratio"] = float(ratio)
    if np.isfinite(ratio) and ratio > 1.6:
        flag("DURATION_MISMATCH", "medium",
             f"Observed duration is {ratio:.1f}x the central-transit prediction - host star parameters may be wrong or the signal is blended")

    # centroid motion in transit (background blend / off-target source)
    if centroids is not None and centroids[0] is not None and in_tr.sum() >= 5:
        shifts = []
        for c in centroids:
            good = np.isfinite(c)
            if good.sum() < 0.5 * len(c):
                continue
            resid = np.full(len(c), np.nan)
            trend, _ = savgol_detrend(time[good], c[good], 1.5, 3, transit_mask=(np.abs(x[good]) < t14))
            resid[good] = c[good] - trend
            s = robust_std(resid[oot & good])
            m_in, e_in = _mean_err(resid[in_tr & good], s)
            # Centroids carry correlated pointing systematics, so the white-noise error bar is far too small.
            # Measure the same statistic in transit-width windows at other orbital phases and use its scatter.
            null = []
            for shift in np.linspace(0.1, 0.9, 33):
                win = (np.abs((x + shift * period + 0.5 * period) % period - 0.5 * period) < 0.5 * t14) & good & oot
                if win.sum() >= max(5, 0.5 * (in_tr & good).sum()):
                    null.append(float(np.mean(resid[win])))
            if len(null) >= 8:
                e_in = max(e_in, float(np.std(null)))
            shifts.append((m_in - float(np.mean(resid[oot & good])), e_in))
        if shifts:
            offset = float(np.hypot(*[s[0] for s in shifts])) if len(shifts) > 1 else abs(shifts[0][0])
            err = float(np.hypot(*[s[1] for s in shifts])) if len(shifts) > 1 else shifts[0][1]
            sig = offset / err if err > 0 else 0.0
            # A dip of fractional depth d on a source s pixels from the target moves the photocentre by about d * s,
            # so shift / depth is the implied distance of the varying source. Significance alone is not enough:
            # a deep on-target transit shifts the centroid measurably without any background star being involved.
            separation = offset / depth if depth > 0 else 0.0
            diag["centroid"] = {"offset_mpix": offset * 1e3, "sigma": float(sig), "separation_px": float(separation)}
            if sig > 5.0 and separation > 0.5:
                flag("CENTROID_OFFSET", "high",
                     f"Photocentre shifts by {offset * 1e3:.2f} mpix in transit ({sig:.1f} sigma): the varying source lies "
                     f"about {separation:.1f} px from the target - likely a background star")
    return diag, flags
