"""Refined analysis: a limb-darkened physical transit fit with uncertainties, and a search for further signals."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .bls import bls_search
from .config import R_EARTH_M, R_SUN_M
from .synthetic import transit_flux

MAX_FIT_POINTS = 6000
HARMONICS = (1 / 6, 0.2, 0.25, 1 / 3, 0.5, 2 / 3, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)


def limb_darkening(t_eff_k: float) -> tuple[float, float]:
    """Quadratic coefficients for a broad red-optical band (Kepler / TESS), by effective temperature."""
    grid_t = np.array([3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8500.0])
    u1 = np.interp(t_eff_k, grid_t, [0.45, 0.55, 0.50, 0.38, 0.30, 0.25])
    u2 = np.interp(t_eff_k, grid_t, [0.30, 0.18, 0.20, 0.27, 0.30, 0.30])
    return float(u1), float(u2)


def fit_limb_darkened(x_days, flux, period, depth, t14_days, r_star_solar, t_eff_k) -> dict:
    """Least-squares fit of (Rp/R*, a/R*, b, mid-time, baseline) with fixed limb darkening.

    The PRD's delta = (Rp/R*)^2 ignores limb darkening, which makes a transit deeper at mid-point than the
    area ratio alone; fitting the darkened model removes that bias from the radius and yields 1-sigma errors
    from the fit covariance.
    """
    out = {"converged": False}
    sel = np.abs(x_days) < 2.0 * t14_days
    xs, fs = x_days[sel], flux[sel]
    if len(xs) < 30 or depth <= 0:
        return out
    if len(xs) > MAX_FIT_POINTS:
        pick = np.linspace(0, len(xs) - 1, MAX_FIT_POINTS).astype(int)
        order = np.argsort(xs)
        xs, fs = xs[order][pick], fs[order][pick]
    u = limb_darkening(t_eff_k)
    k0 = float(np.sqrt(depth) * 0.95)
    best = None
    for b0 in (0.2, 0.7):
        a0 = max(period / (np.pi * t14_days) * np.sqrt(max((1 + k0) ** 2 - b0 ** 2, 0.05)), 1.6)

        def resid(p):
            return p[4] * transit_flux(xs, period, p[3], p[0], p[1], p[2], u) - fs

        try:
            sol = least_squares(resid, [k0, a0, b0, 0.0, 1.0],
                                bounds=([0.2 * k0, 1.3, 0.0, -0.5 * t14_days, 0.95], [min(3 * k0, 0.95), 20 * a0, 1.3, 0.5 * t14_days, 1.05]),
                                x_scale=[k0, a0, 0.3, 0.1 * t14_days, 1e-3], loss="soft_l1", f_scale=max(float(np.std(fs)), 1e-6))
        except ValueError:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None or not best.success:
        return out
    k, a_rs, b, dt0, base = (float(v) for v in best.x)
    res = best.fun
    dof = max(len(res) - 5, 1)
    sigma2 = float(np.sum(res ** 2) / dof)
    try:
        cov = np.linalg.inv(best.jac.T @ best.jac) * sigma2
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        err = np.full(5, np.nan)
    fine = np.linspace(-2.0 * t14_days, 2.0 * t14_days, 241)
    model = transit_flux(fine, period, 0.0, k, a_rs, b, u)
    to_earth = r_star_solar * R_SUN_M / R_EARTH_M

    def clean(v, digits):
        return round(float(v), digits) if np.isfinite(v) else None

    out.update(
        converged=True, limb_darkening=[round(u[0], 3), round(u[1], 3)],
        radius_ratio=round(k, 5), radius_ratio_err=clean(err[0], 5),
        radius_earth=round(k * to_earth, 3), radius_earth_err=clean(err[0] * to_earth, 3),
        a_over_rstar=round(a_rs, 3), a_over_rstar_err=clean(err[1], 3),
        impact_param=round(b, 3), impact_param_err=clean(err[2], 3),
        dt0_days=dt0, depth_model_ppm=round(float((1.0 - model.min()) * 1e6), 1),
        area_ratio_ppm=round(k * k * 1e6, 1), density_solar=round(0.01342 * a_rs ** 3 / period ** 2, 4),
        rms_ppm=round(float(np.sqrt(sigma2)) * 1e6, 1), points=int(len(xs)),
        curve={"hours": np.round(fine * 24.0, 4).tolist(), "flux": np.round(model, 7).tolist()},
    )
    return out


def max_transit_duration(period_days: float, m_star: float, r_star: float) -> float:
    """Longest physically possible transit (days) at this period for this star: a central crossing by a large companion."""
    from .synthetic import a_over_rstar
    a_rs = max(a_over_rstar(period_days, m_star, r_star), 1.5)
    return float(period_days / np.pi * np.arcsin(min(1.0, 1.3 / a_rs)))


def _dip_fraction(t, f, period, t0, duration) -> float:
    """How much of a periodic dip survives once a smooth wave of the same period (two harmonics) is fitted and removed.

    A transit is narrow, so a smooth wave cannot follow it and most of the dip remains (about 60% at a 10% duty
    cycle, more for shorter ones). Rotation, pulsation and ellipsoidal variation *are* smooth waves and all but vanish.
    """
    ph = 2.0 * np.pi * (t - t0) / period
    basis = np.column_stack([np.ones(len(t)), np.cos(ph), np.sin(ph), np.cos(2 * ph), np.sin(2 * ph)])
    coef, *_ = np.linalg.lstsq(basis, f, rcond=None)
    resid = f - basis @ coef
    x = np.abs((t - t0 + 0.5 * period) % period - 0.5 * period)
    inside = x < 0.5 * duration
    if inside.sum() < 3 or (~inside).sum() < 10:
        return 1.0
    before = float(np.median(f[~inside]) - np.mean(f[inside]))
    after = float(np.median(resid[~inside]) - np.mean(resid[inside]))
    return after / before if before > 0 else 0.0


def phase_snr(t, f, period: float, t0: float, duration: float) -> float:
    """Depth of a periodic dip against the scatter the same measurement shows at every other phase of that period.

    The usual S/N assumes white noise. On a variable star the residual wiggles are correlated, and folding them at
    any period produces "dips" of similar size all round the orbit. Measuring the noise that way sets the bar where
    the star's own behaviour puts it.
    """
    width = duration / period
    if not (0.0 < width < 0.25):
        return 0.0
    nb = int(np.clip(round(4.0 / width), 40, 4000))
    idx = np.minimum((((t - t0) / period + 0.5 * width) % 1.0 * nb).astype(np.intp), nb - 1)
    count = np.bincount(idx, minlength=nb).astype(float)
    total = np.bincount(idx, weights=np.median(f) - f, minlength=nb)
    w = max(1, int(round(width * nb)))
    wrap = lambda v: np.convolve(np.concatenate([v, v[:w - 1]]), np.ones(w), mode="valid")
    n_win, s_win = wrap(count), wrap(total)
    ok = n_win >= 0.3 * np.median(n_win)
    mean = np.where(ok, s_win / np.maximum(n_win, 1.0), np.nan)
    stat = mean * np.sqrt(n_win / np.median(n_win[ok]))
    centres = np.arange(nb) / nb  # window k starts at phase k/nb (phase 0 is the start of the dip)
    off = ok & (np.minimum(centres, 1.0 - centres) > 2.0 * width)
    if off.sum() < 10 or not np.isfinite(stat[0]):
        return 0.0
    noise = float(1.4826 * np.median(np.abs(stat[off] - np.median(stat[off]))))
    return float((stat[0] - np.median(stat[off])) / noise) if noise > 0 else 0.0


def resolve_alias(t, f, period: float, t0: float, duration: float, sigma: float, p_min: float, p_max: float) -> tuple[float, float]:
    """Correct a detection made at twice or half the true period, using the dips themselves.

    Half the true period: every other "transit" is simply absent. Twice the true period: an equally deep dip sits
    exactly half-way between the transits. (A dip of clearly different depth there is an eclipsing binary's
    secondary, not an alias, and is left alone.)
    """
    def dip(centre_phase, parity=None):
        ph = ((t - t0) / period - centre_phase + 0.5) % 1.0 - 0.5
        m = np.abs(ph) < 0.3 * duration / period
        if parity is not None:
            m &= (np.round((t - t0) / period - centre_phase).astype(int) % 2) == parity
        if m.sum() < 4:
            return None
        return float(np.median(f) - f[m].mean()), sigma / np.sqrt(m.sum())

    even, odd = dip(0.0, 0), dip(0.0, 1)
    if even and odd and 2.0 * period <= p_max:
        strong, weak = (even, odd) if even[0] >= odd[0] else (odd, even)
        if strong[0] > 5.0 * strong[1] and weak[0] < 2.0 * weak[1] and weak[0] < 0.25 * strong[0]:
            return 2.0 * period, (t0 if strong is even else t0 + period)
    main, half = dip(0.0), dip(0.5)
    if main and half and period / 2.0 >= p_min and main[0] > 5.0 * main[1]:
        if half[0] > 4.0 * half[1] and abs(half[0] - main[0]) < 3.0 * np.hypot(main[1], half[1]) + 0.2 * main[0]:
            return period / 2.0, t0
    return period, t0


def _other_peaks(t, f, e, cfg: dict, periods, sr, skip_period: float, count: int = 4) -> list:
    """The next strongest, distinct peaks of a BLS spectrum, each resolved to a period, epoch and duration."""
    from .bls import BLSResult, bls_periodogram, normalized_weights, refine_ephemeris

    weights = normalized_weights(e, len(t))
    mean, spread = float(np.mean(sr)), float(np.std(sr))
    taken = [skip_period]
    out = []
    for j in np.argsort(sr)[::-1]:
        if len(out) >= count:
            break
        pj = float(periods[j])
        if any(abs(pj / q - h) / h < (0.03 if h == 1.0 else 0.004) for q in taken for h in HARMONICS):
            continue  # the same peak, or a harmonic of one already taken
        taken.append(pj)
        step = abs(1.0 / periods[min(j + 1, len(periods) - 1)] - 1.0 / periods[max(j - 1, 0)]) / 2.0 or 1e-6
        local_p = 1.0 / (1.0 / pj + np.arange(-1.0, 1.01, 0.25) * step)
        local_p = local_p[(local_p >= cfg["period_min"]) & (local_p <= periods.max())]
        if len(local_p) == 0:
            continue
        sr_l, st_l, wd_l = bls_periodogram(t, f, weights, local_p, cfg["q_min"], cfg["q_max"], 200)
        k = int(sr_l.argmax())
        period, q = float(local_p[k]), float(wd_l[k]) / 200
        t0 = float(t[0] + (st_l[k] + 0.5 * wd_l[k]) / 200 * period)
        period, t0 = refine_ephemeris(t, f, weights, period, t0, q * period, 1.5 * period * period * step)
        t0 = float(t0 + np.ceil((t[0] - t0) / period) * period)
        out.append(BLSResult(period=period, t0=t0, q=q, duration_days=q * period, depth=0.0, sr=float(sr_l[k]),
                             sde=float((sr[j] - mean) / spread) if spread > 0 else 0.0, n_periods=len(periods),
                             period_min=float(periods.min()), period_max=float(periods.max()), freq_step=float(step)))
    return out


def search_additional(time, flat, flux_err, period, t0, t14_days, cfg: dict, sigma: float, max_signals: int = 1,
                      m_star: float = 1.0, r_star: float = 1.0, trace: list | None = None,
                      flux_raw: np.ndarray | None = None) -> list[dict]:
    """Remove each detected signal in turn and search what is left for further periodic dips.

    A transit-like signal is masked; if it also has structure away from its dip (a secondary eclipse or
    out-of-eclipse variation) its whole repeating waveform at that period is divided out first. Each search then
    looks at the strongest few peaks of the spectrum, and each peak is one of three things:

    * a residue of something already removed (same period to within 3%, or an exact harmonic of it);
    * stellar variability - a smooth wave rather than a narrow dip, or a "dip" longer than any transit of this
      star could last;
    * a further transit-like signal.

    Only the last is reported and counted against `max_signals`. If none of the peaks is one, and the strongest
    is a residue or variability, its waveform is divided out and the search repeats - so a star's own rotation
    or pulsation can neither use up the budget nor hide a planet behind it.
    """
    from .conditioning import periodic_variability_model

    found: list[dict] = []
    if not (np.isfinite(sigma) and sigma > 0):
        sigma = float(1.4826 * np.median(np.abs(flat - np.median(flat))))
    # the first detection may itself be variability rather than a dip; then there is nothing to mask
    first_is_dip = t14_days < 0.15 * period and _dip_fraction(time, flat, period, t0, t14_days) >= 0.4
    known = [(period, t0, t14_days, "dip" if first_is_dip else "wave")]
    keep = np.ones(len(time), bool)
    work = np.array(flat, dtype=float)
    if not first_is_dip and flux_raw is not None:
        # `flat` was detrended with the first detection masked as if it were a transit. When that detection is
        # really the star's variability the mask covers much of the curve and the trend is poorly removed there,
        # so start again from the normalised flux with nothing masked.
        from .conditioning import savgol_detrend
        work = flux_raw / savgol_detrend(time, flux_raw, cfg["detrend_window_days"], int(cfg["detrend_polyorder"]))[0]
    spent, removed_other = 0, 0
    max_other = int(cfg.get("max_variability_removals", 5))
    while spent < max_signals and removed_other <= max_other:
        p_last, t_last, d_last, kind = known[-1]
        x = (time - t_last + 0.5 * p_last) % p_last - 0.5 * p_last
        if cfg.get("subtract_known_signals", 1):
            # Fine enough to follow the waveform, but never so fine that a bin holds too few cadences: sparse
            # bins fit noise and eat into other planets' transits (the dip itself is masked below anyway).
            nbins = int(min(np.clip(4.0 * p_last / max(d_last, 1e-3), 20, 400), max(20, len(time) // 40)))
            wave = periodic_variability_model(time, work, p_last, nbins)
            # A plain transit is fully handled by the mask, and subtracting a flat waveform would only eat into
            # planets in resonance with this one, whose transits recur at the same few phases. So subtract only
            # if the signal has structure away from its dip, or is not a dip at all.
            away = np.abs(x) > 1.5 * d_last
            per_bin_noise = sigma / np.sqrt(max(len(time) / nbins, 1.0))
            if kind != "dip" or (away.any() and float(np.ptp(wave[away])) > 6.0 * per_bin_noise):
                work = work / wave
        if kind == "dip":
            keep &= np.abs(x) > 0.9 * d_last
        if keep.sum() < 500:
            break
        t, f = time[keep], work[keep]
        e = flux_err[keep] if flux_err is not None else None
        resid_sigma = float(1.4826 * np.median(np.abs(f - np.median(f))))
        try:
            top, periods, sr = bls_search(t, f, e, cfg["period_min"], cfg["period_max"], cfg["q_min"], cfg["q_max"], int(cfg["bls_oversample"]))
        except ValueError:
            break
        judged = []
        for rank, res in enumerate([top] + _other_peaks(t, f, e, cfg, periods, sr, top.period)):
            x2 = np.abs((t - res.t0 + 0.5 * res.period) % res.period - 0.5 * res.period)
            inside = x2 < 0.5 * res.duration_days
            n_in = int(inside.sum())
            if n_in < 3:
                continue
            depth = float(np.median(f[~inside]) - np.mean(f[inside]))
            snr = depth / max(resid_sigma, 1e-12) * np.sqrt(n_in)
            epochs = len(np.unique(np.round((t[inside] - res.t0) / res.period)))
            step = {"period_days": round(res.period, 5), "snr": round(float(snr), 1), "sde": round(res.sde, 1),
                    "duration_hours": round(res.duration_days * 24.0, 2), "transits": int(epochs), "rank": rank + 1}
            # significant against white noise, and against what this star's residual does at other phases
            red_snr = phase_snr(t, f, res.period, res.t0, res.duration_days) if snr >= 7.1 else 0.0
            step["phase_snr"] = round(red_snr, 1)
            white = snr >= 7.1 and epochs >= 2 and depth > 0
            significant = white and red_snr >= float(cfg.get("min_phase_snr", 6.0))
            wiggle = white and red_snr < 5.0  # clearly the star's own behaviour: worth dividing out
            # A residue of a signal already found sits at an exact period ratio. Real planets in resonant chains
            # (TRAPPIST-1: 3:2, 5:3, 2:1 ...) are near those ratios but measurably off them, so that test must
            # be tight. No two bodies share nearly the same period, though, so close to 1:1 it can be loose:
            # that is the same signal again, drifting in phase or shape (spots, eclipse timing variations).
            residue = any(abs(res.period / p - 1.0) < 0.03 or any(abs(res.period / p - h) / h < 0.0025 for h in HARMONICS)
                          for p, _, _, _ in known)
            # A dip lasting far longer than any transit of this star could at this period is slow stellar
            # variability that survived detrending, and so is anything a smooth wave describes as well as a box.
            too_long = res.duration_days > 2.0 * max_transit_duration(res.period, m_star, r_star)
            wave_like = white and _dip_fraction(t, f, res.period, res.t0, res.duration_days) < 0.4
            # The periodogram peak of a long-period signal with few transits is never prominent, so a clearly
            # significant dip (S/N >= 10 over at least three transits) is accepted even when the SDE is modest.
            prominent = res.sde >= cfg["sde_threshold"] or (snr >= 10.0 and epochs >= 3)
            step["outcome"] = ("not significant" if not white else
                               "no deeper than this star's own wiggles at other phases" if wiggle else
                               "not significant" if not significant else
                               "residue of a signal already found" if residue else
                               "longer than any transit of this star could last" if too_long else
                               "smooth variability, not a dip" if wave_like else
                               "periodogram peak not prominent" if not prominent else "transit-like")
            judged.append((res, depth, snr, epochs, step))
        if trace is not None:
            trace.extend(j[4] for j in judged)
        if not judged:
            break
        good = [j for j in judged if j[4]["outcome"] == "transit-like"]
        # While the strongest peak is the star's own variability (or a residue), nothing weaker in the same
        # spectrum can be trusted: a strong wave throws up dip-like beats at many other periods. Remove it first.
        top_is_other = judged[0][0] is top and judged[0][4]["outcome"] in (
            "residue of a signal already found", "longer than any transit of this star could last", "smooth variability, not a dip",
            "no deeper than this star's own wiggles at other phases")
        if good and not (top_is_other and removed_other < max_other):
            res, depth, snr, epochs, step = max(good, key=lambda j: j[2])
            step["outcome"] = "further transit-like signal"
            p_fix, t_fix = resolve_alias(t, f, res.period, res.t0, res.duration_days, resid_sigma, cfg["period_min"], float(periods.max()))
            if p_fix != res.period:
                step["outcome"] += f" (found at {res.period:.4f} d, an alias; corrected)"
                res.period, res.t0 = p_fix, float(t_fix + np.ceil((t[0] - t_fix) / p_fix) * p_fix)
                inside = np.abs((t - res.t0 + 0.5 * res.period) % res.period - 0.5 * res.period) < 0.5 * res.duration_days
                if inside.sum() >= 3:
                    depth = float(np.median(f[~inside]) - np.mean(f[inside]))
                    snr = depth / max(resid_sigma, 1e-12) * np.sqrt(inside.sum())
                    epochs = len(np.unique(np.round((t[inside] - res.t0) / res.period)))
            known.append((res.period, res.t0, res.duration_days, "dip"))
            spent += 1
            found.append({"period_days": round(res.period, 6), "t0": round(res.t0, 5), "depth_ppm": round(depth * 1e6, 1),
                          "duration_hours": round(res.duration_days * 24.0, 3), "snr": round(float(snr), 1), "sde": round(res.sde, 1),
                          "transits": int(epochs), "period_ratio": round(res.period / period, 4), "phase_snr": step.get("phase_snr")})
            continue
        res, _, _, _, step = judged[0]
        if judged[0][0] is not top or step["outcome"] in ("not significant", "periodogram peak not prominent"):
            break  # nothing significant left
        # the strongest peak is a residue or variability: divide its waveform out and look again
        known.append((res.period, res.t0, res.duration_days, "wave"))
        removed_other += 1
        step["outcome"] += " - removed"
    return found
