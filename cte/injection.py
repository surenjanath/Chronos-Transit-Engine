"""Injection and recovery: the completeness of the whole pipeline, measured the way Kepler and TESS measure theirs.

Limb-darkened (Mandel & Agol) transits of known period, depth and impact parameter are injected into real
Kepler light curves of quiet stars (labelled NTP, and with no periodic signal of their own detected by this
pipeline), the full blind pipeline is run, and the fraction
recovered is tabulated against the injected signal-to-noise  S/N = (depth / sigma) * sqrt(N_in_transit).
"""
from __future__ import annotations

import json
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from .config import DATA_DIR, DEFAULT_PIPELINE

RESULTS = DATA_DIR / "injection_recovery.json"
SNR_BINS = (3.0, 4.0, 5.0, 6.0, 7.1, 8.5, 10.0, 14.0, 20.0, 30.0)


def transit_model(time, period, t0, k, a_rs, b, u, exposure_days) -> tuple[np.ndarray, str]:
    """Mandel-Agol light curve from batman, integrated over the exposure; falls back to the built-in model."""
    try:
        import batman
        params = batman.TransitParams()
        params.t0, params.per, params.rp, params.a = t0, period, k, a_rs
        params.inc = float(np.degrees(np.arccos(min(b / a_rs, 1.0))))
        params.ecc, params.w, params.u, params.limb_dark = 0.0, 90.0, list(u), "quadratic"
        model = batman.TransitModel(params, np.ascontiguousarray(time, dtype=float), supersample_factor=7, exp_time=exposure_days)
        return model.light_curve(params), f"batman {batman.__version__}"
    except ImportError:
        from .synthetic import transit_flux
        return transit_flux(time, period, t0, k, a_rs, b, u), "built-in small-planet model"


QUIET = DATA_DIR / "kepler" / "quiet_stars.json"


def _baseline(kepid: int) -> dict:
    """Run the pipeline on the untouched star: it is 'quiet' only if nothing periodic is detected."""
    import logging

    from . import ingest, kepler
    from .core import SOLAR, process
    from .telemetry import RunLogger

    logging.getLogger("CTE-Core").setLevel(logging.ERROR)
    os.environ["CTE_BLS_WORKERS"] = "1"
    lc = kepler.download_star(int(kepid))
    order = np.argsort(lc["time"])
    data = ingest.LightCurveData(lc["time"][order], lc["flux"][order], lc["flux_err"][order], None, None,
                                 {"target_name": f"KIC {kepid}", "mission": "Kepler", "sector": None, "camera": None, "ccd": None}, dict(SOLAR))
    try:
        res, _ = process(data, dict(DEFAULT_PIPELINE, max_additional_signals=0), RunLogger(None, path=pathlib.Path(os.devnull)))
        return {"kepid": int(kepid), "snr": float(res["diagnostics"]["snr"]), "sde": float(res["bls"]["sde"])}
    except Exception:
        return {"kepid": int(kepid), "snr": 1e9, "sde": 1e9}


def quiet_stars(rows: list[dict], workers: int, report) -> list[dict]:
    """NTP stars with no detectable periodic signal of their own."""
    known = json.loads(QUIET.read_text()) if QUIET.exists() else {}
    todo = [r["kepid"] for r in rows if str(r["kepid"]) not in known]
    if todo:
        with ProcessPoolExecutor(workers) as pool:
            for i, fut in enumerate(as_completed([pool.submit(_baseline, k) for k in todo]), 1):
                b = fut.result()
                known[str(b["kepid"])] = b
                report(f"Screening for quiet stars {i}/{len(todo)}")
        QUIET.write_text(json.dumps(known))
    # quiet = no transit-like detection (S/N < 7.1) and no significant periodicity of its own (BLS SDE below threshold);
    # a star that pulsates or rotates with a strong period dominates the search whatever is injected into it
    return [r for r in rows if known[str(r["kepid"])]["snr"] < 7.1 and known[str(r["kepid"])]["sde"] < DEFAULT_PIPELINE["sde_threshold"]]


def _trial(args) -> dict:
    """One injection into one real star (runs in a worker process)."""
    kepid, target_snr, seed, row = args
    import logging

    from . import ingest, kepler
    from .conditioning import savgol_detrend
    from .core import SOLAR, process
    from .fit import limb_darkening
    from .synthetic import a_over_rstar
    from .telemetry import RunLogger

    logging.getLogger("CTE-Core").setLevel(logging.ERROR)
    os.environ["CTE_BLS_WORKERS"] = "1"  # the study already runs one trial per core
    rng = np.random.default_rng(seed)
    lc = kepler.download_star(int(kepid))
    order = np.argsort(lc["time"])
    time, flux, err = lc["time"][order], lc["flux"][order], lc["flux_err"][order]
    trend, _ = savgol_detrend(time, flux, 1.5, 3)
    resid = flux / trend - 1.0
    sigma = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
    r_star, m_star, t_eff = row.get("koi_srad") or 1.0, row.get("koi_smass") or 1.0, row.get("koi_steff") or 5778.0
    period = float(np.exp(rng.uniform(np.log(1.5), np.log(12.0))))
    b = float(rng.uniform(0.0, 0.6))
    a_rs = max(a_over_rstar(period, m_star, r_star), 2.5)
    t0 = float(time[0] + rng.uniform(0, period))
    exposure = float(np.median(np.diff(time)))
    u = limb_darkening(t_eff)
    k = 0.05
    for _ in range(4):  # scale the radius ratio until the injected S/N matches the target
        model, engine = transit_model(time, period, t0, k, a_rs, b, u, exposure)
        dip = 1.0 - model
        n_in = int((dip > 0.5 * dip.max()).sum()) if dip.max() > 0 else 0
        if n_in < 3:
            return {"snr": target_snr, "skipped": True}
        snr = float(dip[dip > 0.5 * dip.max()].mean() / sigma * np.sqrt(n_in))
        k = float(np.clip(k * np.sqrt(target_snr / snr), 0.003, 0.4))
    model, engine = transit_model(time, period, t0, k, a_rs, b, u, exposure)
    dip = 1.0 - model
    core = dip > 0.5 * dip.max()
    snr = float(dip[core].mean() / sigma * np.sqrt(core.sum()))
    data = ingest.LightCurveData(time, flux * model, err, None, None,
                                 {"target_name": f"KIC {kepid}", "mission": "Kepler", "sector": None, "camera": None, "ccd": None},
                                 {**SOLAR, "t_eff_k": t_eff, "r_star_solar": r_star, "m_star_solar": m_star})
    cfg = {**DEFAULT_PIPELINE, "max_additional_signals": 0, **(row.get("_cfg") or {})}
    try:
        res, _ = process(data, cfg, RunLogger(None, path=pathlib.Path(os.devnull)))
    except Exception as exc:
        return {"snr": target_snr, "injected_snr": snr, "kepid": int(kepid), "error": type(exc).__name__, "recovered": False,
                "detected": False, "planet": False}
    found = res["bls"]["period"]
    match = abs(found - period) / period < 2e-3
    # on a star with a signal of its own the injected planet can only be the second detection
    extra = [a for a in res.get("additional_signals", []) if abs(a["period_days"] - period) / period < 2e-3]
    return {"snr": target_snr, "injected_snr": round(snr, 2), "kepid": int(kepid), "period": round(period, 5),
            "found_period": round(found, 5), "depth_ppm": round(float(dip.max()) * 1e6, 1), "impact_param": round(b, 2),
            "measured_snr": round(res["diagnostics"]["snr"], 2), "period_match": bool(match),
            "detected": bool(match and res["diagnostics"]["snr"] >= 7.1),
            "recovered": bool(match), "planet": bool(match and res["classification"]["prediction"] == "PLANET_CANDIDATE"),
            "recovered_any": bool(match or extra), "variability_removed": len(res["conditioning"].get("variability_removed", [])),
            "engine": engine}


def run_study(trials_per_bin: int = 8, workers: int = 3, seed: int = 7,
              progress: Callable[[str], None] | None = None) -> dict:
    from . import kepler

    report = progress or (lambda msg: None)
    ntp = [r for r in kepler.load_sample() if r["label"] == 2 and (kepler.LC_DIR / f"{r['kepid']}.npz").exists()]
    if len(ntp) < 5:
        raise RuntimeError("the Kepler dataset has too few NTP stars; run build_kepler_dataset first")
    quiet = quiet_stars(ntp[:240], workers, report)
    if len(quiet) < 5:
        raise RuntimeError(f"only {len(quiet)} quiet stars found among {min(len(ntp), 240)} screened")
    rng = np.random.default_rng(seed)
    jobs = []
    for snr in SNR_BINS:
        for _ in range(trials_per_bin):
            row = quiet[int(rng.integers(len(quiet)))]
            jobs.append((row["kepid"], snr, int(rng.integers(1 << 31)), row))
    trials = []
    with ProcessPoolExecutor(workers) as pool:
        for i, fut in enumerate(as_completed([pool.submit(_trial, j) for j in jobs]), 1):
            trials.append(fut.result())
            report(f"Injection {i}/{len(jobs)}")
    bins = []
    for snr in SNR_BINS:
        t = [x for x in trials if x["snr"] == snr and not x.get("skipped")]
        n = len(t)
        bins.append({"snr": snr, "n": n, "period_recovered": sum(x["recovered"] for x in t), "detected": sum(x["detected"] for x in t),
                     "planet": sum(x["planet"] for x in t),
                     "recovery": sum(x["detected"] for x in t) / n if n else None})
    def pooled(lo, hi):
        t = [x for x in trials if lo <= x["snr"] < hi and not x.get("skipped")]
        return (sum(x["detected"] for x in t) / len(t), len(t)) if t else (None, 0)
    low, n_low = pooled(0, 4.5)
    high, n_high = pooled(10, 1e9)
    at, n_at = pooled(7.0, 7.2)
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "bins": bins, "trials": sorted(trials, key=lambda x: x["snr"]),
           "engine": next((x["engine"] for x in trials if x.get("engine")), "unknown"), "stars": len(quiet),
           "screened": min(len(ntp), 240),
           "recovery_below_4_5": low, "n_below_4_5": n_low, "recovery_at_7_1": at, "n_at_7_1": n_at,
           "recovery_from_10": high, "n_from_10": n_high, "trials_per_bin": trials_per_bin}
    RESULTS.write_text(json.dumps(out, indent=1))
    return out


def load_results() -> dict | None:
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else None
