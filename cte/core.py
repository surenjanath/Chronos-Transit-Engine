"""Analytical core: conditioning -> BLS -> physics -> CNN, plus Parquet archival (PRD §3).

Framework-free: the Django app in `engine/` supplies persistence and scheduling.
"""
from __future__ import annotations

import time as _time
from typing import Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import ingest, model, pixels, stack
from .bls import bls_search, fit_trapezoid, normalized_weights, refine_ephemeris_shape
from .conditioning import (choose_detrend_window, median_normalize, periodic_variability_model, savgol_detrend,
                           sigma_clip)
from .fit import fit_limb_darkened, search_additional
from .config import ARCHIVE_DIR, CLASS_NAMES
from .physics import StellarParameters, TransitAnalyticsEngine, derive, global_view, local_channels, local_view, shape_geometry, validate_transit
from .telemetry import RunLogger, clean
from .vetting import N_PIXEL, diagnose, extended
from .visuals import build_visuals

STAGES = ["INGEST", "NORMALIZATION", "OUTLIER_REJECTION", "DETRENDING", "BLS_SEARCH", "PHASE_FOLDING",
          "PHYSICS_DERIVATION", "VETTING_DIAGNOSTICS", "CNN_INFERENCE", "REFINED_ANALYSIS", "ARCHIVE"]
STAGE_PROGRESS = {s: i / len(STAGES) for i, s in enumerate(STAGES)}
SOLAR = {"t_eff_k": 5778.0, "r_star_solar": 1.0, "m_star_solar": 1.0}


def decide_disposition(prediction: str, p_planet: float, flags: list[dict], snr: float, cfg: dict) -> str:
    if snr < 7.1:
        return "NO_SIGNIFICANT_SIGNAL"
    if prediction == "ECLIPSING_BINARY":
        return "FALSE_POSITIVE_ECLIPSING_BINARY"
    if prediction == "STELLAR_OR_INSTRUMENT_NOISE":
        return "REJECTED_NOISE"
    if any(f["severity"] == "high" for f in flags):
        return "FALSE_POSITIVE_SUSPECT"
    return "CONFIRMED_VETTING_QUEUE" if p_planet >= cfg["planet_confidence"] else "MANUAL_REVIEW"


def process(data: ingest.LightCurveData, cfg: dict, log: RunLogger,
            on_stage: Callable[[str, float], None] | None = None,
            on_note: Callable[[str], None] | None = None) -> tuple[dict, dict]:
    """Run every analytical stage on one light curve. Returns (results, arrays for archival).

    `on_note` receives a plain-language line about what is happening right now, for whoever is watching."""
    timings: dict[str, float] = {}
    mark = _time.perf_counter()
    note = on_note or (lambda text: None)

    def stage(name: str, frac: float = 0.0):
        if on_stage:
            on_stage(name, STAGE_PROGRESS[name] + frac / len(STAGES))

    def lap(name: str):
        nonlocal mark
        now = _time.perf_counter()
        timings[name] = round(now - mark, 4)
        mark = now

    order = np.argsort(data.time, kind="stable")
    t_all, f_raw = data.time[order], data.flux[order]
    e_raw = data.flux_err[order] if data.flux_err is not None else None
    cents = [c[order] if c is not None else None for c in (data.centroid_col, data.centroid_row)]
    segment = data.segment[order] if data.segment is not None else None
    if len(t_all) < 200:
        raise ValueError(f"light curve has only {len(t_all)} cadences; at least 200 are required")
    cadence = float(np.median(np.diff(t_all)))
    baseline = float(t_all[-1] - t_all[0])

    # 2.1 median normalisation
    stage("NORMALIZATION")
    note(f"Read {len(t_all):,} brightness measurements. Scaling them so the star's normal brightness is 1.")
    f_norm, med = median_normalize(f_raw)
    e_norm = e_raw / med if e_raw is not None else None
    log.info("Median normalization complete", "NORMALIZATION", cadences=len(t_all), median_flux=med,
             baseline_days=baseline, cadence_minutes=cadence * 1440.0, variance=float(np.var(f_norm)))
    lap("NORMALIZATION")

    # 2.1 iterative rolling sigma clip
    stage("OUTLIER_REJECTION")
    note("Throwing out glitches: single measurements far from their neighbours (cosmic rays, pointing jitter).")
    keep, iterations = sigma_clip(f_norm, cfg["sigma_k"], int(cfg["clip_window"]))
    n_rej = int((~keep).sum())
    log.info("Iterative sigma-clipping complete", "OUTLIER_REJECTION", k=cfg["sigma_k"], window=int(cfg["clip_window"]),
             iterations=iterations, rejected=n_rej, retained=int(keep.sum()), rejected_fraction=n_rej / len(keep))
    t, f = t_all[keep], f_norm[keep]
    e = e_norm[keep] if e_norm is not None else None
    cents_k = [c[keep] if c is not None else None for c in cents]
    segment_k = segment[keep] if segment is not None else None
    lap("OUTLIER_REJECTION")

    # 2.2 Savitzky-Golay detrend (first pass)
    stage("DETRENDING")
    note("Removing the star's slow brightness changes so only short dips remain.")
    window_days, window_trials = cfg["detrend_window_days"], []
    if cfg.get("adaptive_detrend", 0):
        window_days, window_trials = choose_detrend_window(t, f, cfg["detrend_window_days"], int(cfg["detrend_polyorder"]))
    cfg = dict(cfg, detrend_window_days=window_days)  # every later detrend pass uses the chosen window
    trend, window = savgol_detrend(t, f, cfg["detrend_window_days"], int(cfg["detrend_polyorder"]))
    flat = f / trend
    log.info("Savitzky-Golay detrending complete", "DETRENDING", polyorder=int(cfg["detrend_polyorder"]),
             window_cadences=window, window_days=cfg["detrend_window_days"], pass_number=1, window_selection=window_trials,
             residual_std_ppm=float(np.std(flat) * 1e6))
    lap("DETRENDING")

    # 2.3 box least squares
    stage("BLS_SEARCH")
    note("Trying tens of thousands of possible orbital periods, looking for a dip that repeats on schedule.")
    removed: list[dict] = []
    for attempt in range(3):
        bls, periods, sr = bls_search(t, flat, e, cfg["period_min"], cfg["period_max"], cfg["q_min"], cfg["q_max"],
                                      int(cfg["bls_oversample"]), lambda frac: stage("BLS_SEARCH", frac))
        if attempt == 2 or not cfg.get("remove_periodic_variability", 0):
            break
        # Is the strongest periodicity a transit at all? A dip must be significant; a strong periodogram peak
        # without one is the star itself varying, and it hides any transit until it is taken out.
        x0 = np.abs((t - bls.t0 + 0.5 * bls.period) % bls.period - 0.5 * bls.period)
        inside, outside = x0 < 0.5 * bls.duration_days, x0 > bls.duration_days
        scatter = float(1.4826 * np.median(np.abs(flat[outside] - np.median(flat[outside])))) if outside.any() else 0.0
        dip = float(np.median(flat[outside]) - np.mean(flat[inside])) if inside.any() and outside.any() else 0.0
        quick_snr = dip / scatter * np.sqrt(inside.sum()) if scatter > 0 else 0.0
        if quick_snr >= 7.1 or bls.sde < cfg["sde_threshold"]:
            break
        wave = periodic_variability_model(t, f, bls.period)
        amplitude = float(np.ptp(wave))
        if amplitude <= 0:
            break
        f = f / wave
        trend, window = savgol_detrend(t, f, cfg["detrend_window_days"], int(cfg["detrend_polyorder"]))
        flat = f / trend
        removed.append({"period_days": round(bls.period, 5), "amplitude_ppm": round(amplitude * 1e6, 1), "sde": round(bls.sde, 1),
                        "box_snr": round(float(quick_snr), 2)})
        log.info("Periodic stellar variability removed before the transit search", "BLS_SEARCH", **removed[-1])
    if bls.period_max < cfg["period_max"] - 1e-9:
        log.info("Period search range truncated to half the observing baseline", "BLS_SEARCH",
                    requested_max=cfg["period_max"], searched_max=bls.period_max, baseline_days=baseline)
    # Empty-epoch check: if only every second (or third) "transit" is really there, the search has locked onto a
    # fraction of the true period. (An eclipsing binary at twice the period shows two unequal dips; here the
    # other epochs show nothing at all.)
    doubled = 0
    for _ in range(3):
        grown = False
        for m in (2, 3):
            if m * bls.period > min(cfg["period_max"], baseline / 2.0):
                continue
            xq = (t - bls.t0 + 0.5 * bls.period) % bls.period - 0.5 * bls.period
            ep = np.round((t - bls.t0) / bls.period).astype(int)
            core_q, out_q = np.abs(xq) < 0.3 * bls.duration_days, np.abs(xq) > bls.duration_days
            if out_q.sum() < 50:
                continue
            level, noise = float(np.median(flat[out_q])), float(1.4826 * np.median(np.abs(flat[out_q] - np.median(flat[out_q]))))
            groups = []
            for want in range(m):
                sel = core_q & (ep % m == want)
                groups.append((level - float(flat[sel].mean()), noise / np.sqrt(sel.sum()), want) if sel.sum() >= 4 else None)
            if None in groups:
                continue
            groups.sort(key=lambda v: -v[0])
            strong = groups[0]
            if strong[0] > 5.0 * strong[1] and all(w[0] < 2.0 * w[1] and w[0] < 0.25 * strong[0] for w in groups[1:]):
                bls.t0 += strong[2] * bls.period  # the epoch class that holds the real transits
                bls.period *= m
                bls.q = bls.duration_days / bls.period
                doubled += 1
                grown = True
                break
        if not grown:
            break
    if doubled:
        log.info("Period lengthened: the epochs in between show no transit", "BLS_SEARCH", steps=doubled, period_days=bls.period)
    # second detrend pass with the transit masked, so the trend cannot absorb transit depth
    x = (t - bls.t0 + 0.5 * bls.period) % bls.period - 0.5 * bls.period
    trend, _ = savgol_detrend(t, f, cfg["detrend_window_days"], int(cfg["detrend_polyorder"]),
                              transit_mask=np.abs(x) < bls.duration_days)
    flat = f / trend
    fit = fit_trapezoid(x, flat, bls.depth, bls.duration_days, cadence)
    validate_transit(bls.period, fit["t14"])
    if fit["converged"] and fit["depth"] > 0:
        # sharpen the ephemeris against the fitted transit shape, then re-fit the shape
        bls.period, bls.t0 = refine_ephemeris_shape(
            t, flat, normalized_weights(e, len(t)), bls.period, bls.t0 + fit["dt0"], fit["depth"], fit["t14"],
            fit["t12"], bls.period ** 2 * bls.freq_step)
        x = (t - bls.t0 + 0.5 * bls.period) % bls.period - 0.5 * bls.period
        trend, _ = savgol_detrend(t, f, cfg["detrend_window_days"], int(cfg["detrend_polyorder"]),
                                  transit_mask=np.abs(x) < fit["t14"])
        flat = f / trend
        fit = fit_trapezoid(x, flat, fit["depth"], fit["t14"], cadence)
    t0 = bls.t0 + fit["dt0"]
    t14 = fit["t14"]
    depth = fit["depth"] if fit["converged"] and fit["depth"] > 0 else bls.depth
    log.info("BLS periodogram search complete", "BLS_SEARCH", trial_periods=bls.n_periods,
             period_range_days=[bls.period_min, bls.period_max], best_period_days=bls.period, t0=t0, q_best=bls.q,
             sr_max=bls.sr, sde=bls.sde, bls_depth_ppm=bls.depth * 1e6, fitted_depth_ppm=depth * 1e6,
             fitted_duration_hours=t14 * 24.0, shape_v=fit["v_shape"], pass_2_residual_std_ppm=float(np.std(flat) * 1e6))
    lap("BLS_SEARCH")

    # phase folding + ML tensors
    stage("PHASE_FOLDING")
    note(f"Best repeating dip: every {bls.period:.4f} days. Stacking all its transits on top of each other.")
    q_local = float(np.clip(t14 / bls.period, 0.004, 0.15))
    phase, folded, fold_order = TransitAnalyticsEngine.phase_fold(t, flat, bls.period, t0)
    g_vec = global_view(phase, folded, int(cfg["global_bins"]))
    l_vec = local_view(phase, folded, q_local, int(cfg["local_bins"]))
    log.info("Phase-folding and feature aggregation complete", "PHASE_FOLDING", phase_bins=len(g_vec),
             local_bins=len(l_vec), local_phase_half_width=2.5 * q_local, min_flux_bin=float(g_vec.min()),
             max_flux_bin=float(g_vec.max()), min_local_bin=float(l_vec.min()))
    lap("PHASE_FOLDING")

    # 2.4 physical parameters
    stage("PHYSICS_DERIVATION")
    note("Turning the dip into a companion: its size from the depth, its orbit from the period.")
    stellar = {**SOLAR, **{k: v for k, v in data.stellar.items() if v is not None}}
    assumed = [k for k in SOLAR if data.stellar.get(k) is None]
    if assumed:
        log.warning("Stellar parameters missing; solar values assumed", "PHYSICS_DERIVATION", assumed=assumed)
    star = StellarParameters(stellar["t_eff_k"], stellar["r_star_solar"], stellar["m_star_solar"])
    params = derive(star, bls.period, depth, cfg["impact_param"], cfg["bond_albedo"])
    depth_info = {"adopted_ppm": depth * 1e6, "min_bin_ppm": float(1.0 - l_vec.min()) * 1e6, "bls_ppm": bls.depth * 1e6}
    log.info("Derived physical exoplanetary parameters", "PHYSICS_DERIVATION", results=params,
             inputs={"period_days": bls.period, "raw_depth": depth, "impact_param": cfg["impact_param"],
                     "bond_albedo": cfg["bond_albedo"], "stellar": stellar}, depth_estimates_ppm=depth_info)
    lap("PHYSICS_DERIVATION")

    # false-positive diagnostics
    stage("VETTING_DIAGNOSTICS")
    note("Checking for impostors: do odd and even transits match, is there a second eclipse, does the star's image shift?")
    diag, flags = diagnose(t, flat, bls.period, t0, t14, depth, params, cents_k)
    if fit["v_shape"] > 0.85 and depth > 0.003:
        flags.append({"code": "V_SHAPED", "severity": "medium",
                      "message": "Transit has no flat bottom (V-shaped) - consistent with a grazing eclipsing binary"})
    if bls.sde < cfg["sde_threshold"]:
        flags.append({"code": "LOW_SDE", "severity": "medium",
                      "message": f"BLS signal detection efficiency {bls.sde:.1f} is below {cfg['sde_threshold']:.1f}"})
    log.log("WARNING" if flags else "INFO", f"Vetting diagnostics complete: {len(flags)} flag(s) raised",
            "VETTING_DIAGNOSTICS", diagnostics=diag, flags=[f["code"] for f in flags])
    lap("VETTING_DIAGNOSTICS")

    # dual-branch CNN
    stage("CNN_INFERENCE")
    note("Showing the folded transit to five neural networks and asking each: planet, eclipsing binary, or noise?")
    l_stack = local_channels(t, flat, bls.period, t0, q_local, int(cfg["local_bins"]))
    cnn_probs = model.predict(g_vec, l_stack)
    probs = cnn_probs
    extra = extended(t, f, flat, bls.period, t0, t14, fit["t12"], depth, diag,
                     (stellar["t_eff_k"], stellar["r_star_solar"], stellar["m_star_solar"]), float(l_vec.min()), segment_k)
    kepid = data.meta.get("kepid")
    if kepid and data.meta.get("mission") == "Kepler":  # where on the detector the dip comes from
        note("Looking at the star's pixels: does the dip come from the target itself or from a neighbour?")
        try:
            extra[-N_PIXEL:] = pixels.measure_star(int(kepid), bls.period, t0, t14)
            diag["pixels"] = dict(zip(pixels.NAMES, [None if not np.isfinite(v) else round(v, 4) for v in extra[-N_PIXEL:]]))
        except Exception:
            pass  # the check is optional evidence; MAST being slow must not fail the run
    combined = stack.predict([cnn_probs[n] for n in CLASS_NAMES], stack.aux_from(depth, fit["v_shape"], diag, params["radius_earth"]), extra)
    if combined is not None:  # second stage: CNN shape evidence weighed with the physical diagnostics
        probs = {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, combined[0])}
    prediction = max(probs, key=probs.get)
    disposition = decide_disposition(prediction, probs["PLANET_CANDIDATE"], flags, diag["snr"], cfg)
    confident = prediction == "PLANET_CANDIDATE" and disposition == "CONFIRMED_VETTING_QUEUE"
    log.info("Inference complete: " + ("high-confidence transit detected" if confident else
                                       f"classified as {prediction.replace('_', ' ').lower()}"),
             "CNN_INFERENCE", prediction=prediction, probabilities=probs, cnn_probabilities=cnn_probs,
             second_stage=combined is not None, disposition=disposition)
    explain = model.explain(g_vec, l_stack)
    lap("CNN_INFERENCE")

    results = {
        "input": {"cadences": len(t_all), "baseline_days": baseline, "cadence_minutes": cadence * 1440.0,
                  "time_start": float(t_all[0]), "time_end": float(t_all[-1]), "median_flux": med,
                  "has_flux_err": e_raw is not None, "has_centroids": cents[0] is not None},
        "conditioning": {"rejected": n_rej, "clip_iterations": iterations, "detrend_window_cadences": window,
                         "detrend_window_days": cfg["detrend_window_days"], "window_selection": window_trials,
                         "variability_removed": removed,
                         "residual_std_ppm": float(np.std(flat) * 1e6)},
        "bls": bls.to_dict() | {"t0": t0},
        "fit": {"depth_ppm": fit["depth"] * 1e6, "t14_hours": t14 * 24.0, "t12_hours": fit["t12"] * 24.0,
                "v_shape": fit["v_shape"], "converged": fit["converged"], "q_local": q_local},
        "depth": depth_info,
        "stellar": stellar | {"assumed": assumed},
        "parameters": params,
        "diagnostics": diag,
        "flags": flags,
        "classification": {"prediction": prediction, "probabilities": probs, "disposition": disposition,
                           "cnn_probabilities": cnn_probs, "second_stage": combined is not None},
        "tensors": {"global": np.round(g_vec, 6), "local": np.round(l_vec, 6)},
        "explain": explain,
        "settings": cfg,
        "timings": timings,
    }
    # refined analysis: limb-darkened physical fit with uncertainties, and a search for a second signal
    stage("REFINED_ANALYSIS")
    note(f"Verdict so far: {prediction.replace('_', ' ').lower()} at {probs['PLANET_CANDIDATE'] * 100:.0f}% planet. "
         "Now fitting a physical transit model for a better radius.")
    x_final = (t - t0 + 0.5 * bls.period) % bls.period - 0.5 * bls.period
    refined = fit_limb_darkened(x_final, flat, bls.period, depth, t14, stellar["r_star_solar"], stellar["t_eff_k"])
    results["refined"] = refined
    search_trace: list[dict] = []
    results["additional_signals"] = search_additional(t, flat, e, bls.period, t0, t14, cfg, diag["oot_scatter_ppm"] * 1e-6,
                                                      int(cfg.get("max_additional_signals", 1)),
                                                      stellar["m_star_solar"], stellar["r_star_solar"], search_trace, flux_raw=f,
                                                      on_pass=lambda n, limit, text: (stage("REFINED_ANALYSIS", min(0.95, n / (limit + 1))), note(text)))
    results["additional_search"] = search_trace  # every peak examined and what was decided about it
    log.info("Refined analysis complete", "REFINED_ANALYSIS",
             limb_darkened_fit={k: v for k, v in refined.items() if k != "curve"}, additional_signals=results["additional_signals"],
             additional_search=search_trace)
    lap("REFINED_ANALYSIS")
    # geometry from the transit shape alone, as an independent check on the catalogued host star
    shape = shape_geometry(bls.period, depth, t14, fit["t12"])
    shape["density_catalogue_solar"] = round(stellar["m_star_solar"] / stellar["r_star_solar"] ** 3, 4)
    results["shape"] = shape
    results["visuals"] = build_visuals(t, flat, bls.period, t0, t14, depth, q_local, results["fit"] | {"baseline": fit["baseline"]},
                                       params, cfg["impact_param"], shape, refined)
    if data.truth:
        truth = dict(data.truth)
        if truth.get("period_days"):
            truth["period_rel_error"] = abs(bls.period - truth["period_days"]) / truth["period_days"]
        if truth.get("depth"):
            truth["depth_rel_error"] = (depth - truth["depth"]) / truth["depth"]
        if truth.get("period_days"):  # was the true period found, or one of its 1:2 / 2:1 aliases?
            ratio = bls.period / truth["period_days"]
            truth["period_match"] = next((name for name, r in (("exact", 1.0), ("half", 0.5), ("double", 2.0))
                                          if abs(ratio - r) / r < 2e-3), "none")
        results["truth"] = truth
    trend_full = np.full(len(t_all), np.nan)
    flat_full = np.full(len(t_all), np.nan)
    trend_full[keep], flat_full[keep] = trend, flat
    arrays = {
        "lightcurve": {"time": t_all, "flux_raw": f_raw, "flux_norm": f_norm, "kept": keep, "trend": trend_full,
                       "flux_flat": flat_full},
        "periodogram": {"period": periods, "sr": sr},
        "folded": {"phase": phase, "flux": folded, "time": t[fold_order]},
        "tensors": {"view": np.array(["global"] * len(g_vec) + ["local"] * len(l_vec)),
                    "index": np.concatenate((np.arange(len(g_vec)), np.arange(len(l_vec)))),
                    "phase": np.concatenate((np.linspace(-0.5, 0.5, len(g_vec)),
                                             np.linspace(-2.5 * q_local, 2.5 * q_local, len(l_vec)))),
                    "flux": np.concatenate((g_vec, l_vec))},
    }
    if e_raw is not None:
        arrays["lightcurve"]["flux_err"] = e_raw
    return clean(results), arrays


def archive(run_id: str, arrays: dict) -> dict:
    """Write columnar Parquet archives for a run."""
    out = ARCHIVE_DIR / run_id
    out.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name, cols in arrays.items():
        path = out / f"{name}.parquet"
        pq.write_table(pa.table({k: pa.array(v) for k, v in cols.items()}), path, compression="zstd")
        sizes[f"{name}.parquet"] = path.stat().st_size
    return sizes
