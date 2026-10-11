"""Injection study on stars that are not quiet: is a planet behind a star's own variability found, and what turns
up when nothing is injected?

Each trial takes a real Kepler star with a signal of its own (rotation, pulsation, an eclipsing companion), injects
a limb-darkened transit strong enough to be an easy detection on a quiet star, and runs the whole pipeline blind.
"Detectable in principle" means the transit, measured at its true period and time, stands at least 7.1 times above
what the star does at other phases. The same stars are also run with nothing injected, to count false signals.

Needs the list of star behaviours written by the injection study (`manage.py injection_study`)."""
from __future__ import annotations

import json, os, pathlib, sys, logging, numpy as np
from concurrent.futures import ProcessPoolExecutor


def oracle_snr(time, flux, period, t0, t14, own):
    """S/N of the injected transit at its true ephemeris, against the scatter the same measurement shows at other
    phases of the same period (so red noise and the star's own signal count as noise)."""
    from cte.conditioning import savgol_detrend
    x = (time - t0 + 0.5 * period) % period - 0.5 * period
    mask = np.abs(x) < t14
    for (p2, t2, d2) in own:
        mask |= np.abs((time - t2 + 0.5 * p2) % p2 - 0.5 * p2) < d2
    flat = flux / savgol_detrend(time, flux, 1.5, 3, transit_mask=mask)[0]
    keep = np.ones(len(time), bool)
    for (p2, t2, d2) in own:
        keep &= np.abs((time - t2 + 0.5 * p2) % p2 - 0.5 * p2) > d2
    ph = ((time - t0) / period) % 1.0
    w = 0.5 * t14 / period

    def depth(c):
        d = np.abs(ph - c); d = np.minimum(d, 1 - d)
        m = (d < w) & keep
        return (1 - flat[m].mean(), m.sum()) if m.sum() > 5 else (np.nan, 0)
    d0, n0 = depth(0.0)
    null = np.array([depth(c)[0] for c in np.arange(4 * w, 1 - 4 * w, 2 * w)])
    null = null[np.isfinite(null)]
    return float(d0 / np.std(null)) if len(null) > 8 and np.std(null) > 0 else float("nan")


def trial(args):
    kepid, target_snr, seed, row, own_bls, noinj = args
    from cte import ingest, kepler, injection
    from cte.conditioning import savgol_detrend
    from cte.core import SOLAR, process
    from cte.config import DEFAULT_PIPELINE
    from cte.fit import limb_darkening
    from cte.synthetic import a_over_rstar
    from cte.telemetry import RunLogger
    logging.getLogger("CTE-Core").setLevel(logging.ERROR)
    os.environ["CTE_BLS_WORKERS"] = "1"
    rng = np.random.default_rng(seed)
    lc = kepler.download_star(int(kepid))
    order = np.argsort(lc["time"]); time, flux, err = lc["time"][order], lc["flux"][order], lc["flux_err"][order]
    resid = flux / savgol_detrend(time, flux, 1.5, 3)[0] - 1.0
    sigma = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
    r_star, m_star, t_eff = row.get("koi_srad") or 1.0, row.get("koi_smass") or 1.0, row.get("koi_steff") or 5778.0
    period = float(np.exp(rng.uniform(np.log(1.5), np.log(12.0)))); b = float(rng.uniform(0.0, 0.6))
    a_rs = max(a_over_rstar(period, m_star, r_star), 2.5); t0 = float(time[0] + rng.uniform(0, period))
    exposure = float(np.median(np.diff(time))); u = limb_darkening(t_eff); k = 0.05
    for _ in range(4):
        model, _e = injection.transit_model(time, period, t0, k, a_rs, b, u, exposure)
        dip = 1.0 - model; n_in = int((dip > 0.5 * dip.max()).sum()) if dip.max() > 0 else 0
        if n_in < 3: return None
        snr = float(dip[dip > 0.5 * dip.max()].mean() / sigma * np.sqrt(n_in))
        k = float(np.clip(k * np.sqrt(target_snr / snr), 0.003, 0.4))
    model, _e = injection.transit_model(time, period, t0, k, a_rs, b, u, exposure)
    dip = 1.0 - model; t14 = float((dip > 0.02 * dip.max()).sum() / max(1, len(np.unique(np.round((time[dip > 0] - t0) / period)))) * exposure)
    own = [(row["koi_period"], row["koi_time0bk"], row["koi_duration"] / 24.0)]
    osnr = oracle_snr(time, flux * model, period, t0, t14, own)
    data = ingest.LightCurveData(time, flux if noinj else flux * model, err, None, None, {"target_name": f"KIC {kepid}", "mission": "Kepler", "sector": None, "camera": None, "ccd": None},
                                 {**SOLAR, "t_eff_k": t_eff, "r_star_solar": r_star, "m_star_solar": m_star})
    cfg = dict(DEFAULT_PIPELINE)
    try:
        res, _ = process(data, cfg, RunLogger(None, path=pathlib.Path(os.devnull)))
    except Exception as exc:
        return {"kepid": int(kepid), "error": f"{type(exc).__name__}: {exc}"[:120], "found": False, "oracle_snr": osnr, "signals": []}
    sig = [(res["bls"]["period"], res["diagnostics"]["snr"])] + [(a["period_days"], a["snr"]) for a in res.get("additional_signals", [])]
    cls = [(res["bls"]["period"], res["classification"]["prediction"], res["classification"]["probabilities"]["PLANET_CANDIDATE"])] + [
        (a["period_days"], (a.get("classification") or {}).get("prediction"), ((a.get("classification") or {}).get("probabilities") or {}).get("PLANET_CANDIDATE"))
        for a in res.get("additional_signals", [])]
    found = any(abs(p / period - 1) < 2e-3 for p, _ in sig)
    as_planet = any(abs(p / period - 1) < 2e-3 and c == "PLANET_CANDIDATE" for p, c, _ in cls)
    extra_planets = [(round(p, 4), pp) for p, c, pp in cls[1:] if c == "PLANET_CANDIDATE" and abs(p / period - 1) >= 2e-3]
    extra_all = [(round(p, 4), c, pp) for p, c, pp in cls[1:]]
    cons = [(round(a["period_days"], 4), a.get("event_chi2"), a.get("events_brighter"), a["snr"], a["transits"], bool(abs(a["period_days"] / period - 1) < 2e-3)) for a in res.get("additional_signals", [])]
    alias = any(min(abs(p / period / m - 1) for m in (0.5, 2.0, 3.0, 1 / 3)) < 2e-3 for p, _ in sig)
    return {"kepid": int(kepid), "found": bool(found), "alias": bool(alias and not found), "oracle_snr": round(osnr, 1), "white_snr": round(snr, 1),
            "called_planet": bool(as_planet), "extra_planets": extra_planets, "extra_all": extra_all, "cons": cons, "inj_P": round(period, 4), "inj_t0": t0, "inj_depth_ppm": round(float(dip.max()) * 1e6), "t14_h": round(t14 * 24, 1), "own_P": round(row["koi_period"], 4),
            "own_snr": own_bls["snr"], "own_sde": own_bls["sde"], "signals": [(round(p, 4), round(s, 1)) for p, s in sig], "n_points": len(time),
            "trace": [(x["period_days"], x["snr"], x["sde"], x["duration_hours"], x.get("outcome", "")[:40], x.get("phase_snr")) for x in res.get("additional_search", [])],
            "prim": (round(res["bls"]["period"], 4), round(res["bls"]["sde"], 1), res["classification"]["disposition"][:12])}


def run(trials: int = 48, seed: int = 11, inject: bool = True, workers: int = 4) -> dict:
    """One set of trials. The same seed draws the same stars and planets, so a change to the search can be compared
    like for like; a new seed is a fresh test."""
    from cte import injection, kepler
    if not injection.QUIET.exists():
        raise RuntimeError("Run `manage.py injection_study` first: it records which stars are quiet and which vary.")
    q = json.load(open(injection.QUIET)); rows = {r["kepid"]: r for r in kepler.load_sample()}
    noisy = [rows[int(k)] for k, v in q.items() if (v["sde"] >= 7 or v["snr"] >= 7.1) and v["snr"] < 1e8 and int(k) in rows]
    rng = np.random.default_rng(seed)
    picks = [noisy[int(i)] for i in rng.integers(0, len(noisy), trials)]
    seeds = [int(x) for x in rng.integers(1 << 31, size=trials)]
    jobs = [(r["kepid"], 20.0, s, r, {"snr": round(q[str(r["kepid"])]["snr"], 1), "sde": round(q[str(r["kepid"])]["sde"], 1)}, not inject)
            for r, s in zip(picks, seeds)]
    with ProcessPoolExecutor(max(1, workers)) as pool:
        out = [o for o in pool.map(trial, jobs) if o]
    det = [o for o in out if o["oracle_snr"] >= 7.1]
    extra = {o["kepid"]: o.get("extra_all") for o in out if o.get("extra_all") and (not inject or o.get("extra_planets"))}
    return {"trials": len(out), "seed": seed, "injected": inject, "stars": len({o["kepid"] for o in out}),
            "found": sum(o["found"] for o in out), "detectable": len(det), "found_detectable": sum(o["found"] for o in det),
            "called_planet_detectable": sum(o.get("called_planet", False) for o in det),
            "stars_with_extra_signal": len({o["kepid"] for o in out if len(o["signals"]) > 1}) if not inject else None,
            "stars_with_extra_planet": len({o["kepid"] for o in out if o.get("extra_planets")}),
            "misses": [{k: o[k] for k in ("kepid", "inj_P", "inj_depth_ppm", "oracle_snr")} for o in det if not o["found"]] if inject else [],
            "results": out}
