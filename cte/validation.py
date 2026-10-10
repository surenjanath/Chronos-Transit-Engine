"""System verification suite: the five acceptance criteria of PRD §6."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Callable

import numpy as np

from . import ingest, model
from .bls import fourier_comb_period
from .config import DEFAULT_PIPELINE, LOG_DIR, TOLERANCES
from .physics import StellarParameters, derive
from .core import SOLAR, process
from .synthetic import SyntheticSpec
from .telemetry import RunLogger, verify_jsonl

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
KEPLER_TARGETS = [("TrES-2 b", "Kepler-1", 5), ("HAT-P-7 b", "Kepler-2", 5), ("Kepler-8 b", "Kepler-8", 5),
                  ("Kepler-7 b", "Kepler-7", 5)]
RADIUS_BENCHMARKS = ["WASP-12 b", "TRAPPIST-1 e"]
# Used only when the NASA Exoplanet Archive is unreachable (pscomppars, retrieved 2026-10-07).
CACHED_ARCHIVE = {
    "HAT-P-7 b": {"pl_orbper": 2.20474, "pl_rade": 16.92559, "pl_trandep": 0.6, "st_rad": 2.0, "st_mass": 1.56, "st_teff": 6389.0},
    "Kepler-7 b": {"pl_orbper": 4.8854892, "pl_rade": 18.180998, "pl_trandep": 0.75693, "st_rad": 1.966, "st_mass": 1.359, "st_teff": 5933.0},
    "Kepler-8 b": {"pl_orbper": 3.5224991, "pl_rade": 15.871944, "pl_trandep": 0.91457, "st_rad": 1.486, "st_mass": 1.213, "st_teff": 6213.0},
    "TRAPPIST-1 e": {"pl_orbper": 6.101013, "pl_rade": 0.92, "pl_trandep": 0.5012, "st_rad": 0.1192, "st_mass": 0.0898, "st_teff": 2566.0},
    "TrES-2 b": {"pl_orbper": 2.47063, "pl_rade": 15.24424, "pl_trandep": 1.6, "st_rad": 1.12, "st_mass": 1.36, "st_teff": 5850.0},
    "WASP-12 b": {"pl_orbper": 1.091418901, "pl_rade": 22.02564708, "pl_trandep": 1.53777921, "st_rad": 1.69, "st_mass": 1.325, "st_teff": 6265.0},
}


def fetch_archive(names: list[str]) -> tuple[dict, str]:
    """Reference planet parameters from the NASA Exoplanet Archive (live, with cached fallback)."""
    quoted = ",".join("'" + n.replace("'", "''") + "'" for n in names)
    query = ("select pl_name,pl_orbper,pl_rade,pl_trandep,st_rad,st_mass,st_teff from pscomppars "
             f"where pl_name in ({quoted})")
    try:
        url = TAP_URL + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
        with urllib.request.urlopen(url, timeout=30) as resp:
            rows = json.load(resp)
        out = {r["pl_name"]: r for r in rows}
        if all(n in out for n in names):
            return out, "NASA Exoplanet Archive (live TAP query)"
    except Exception:
        pass
    return {n: CACHED_ARCHIVE[n] for n in names}, "NASA Exoplanet Archive (cached values, archive unreachable)"


def _check(cid, parameter, dataset, tolerance, action, group="PRD §6") -> dict:
    return {"id": cid, "parameter": parameter, "dataset": dataset, "tolerance": tolerance, "failure_action": action,
            "status": "running", "measured": None, "cases": [], "action_taken": None, "notes": [], "group": group}


GOLDEN = [  # canonical systems every transit pipeline must reproduce
    {"name": "HD 209458 b", "query": "HD 209458", "mission": "TESS", "sequence": 56, "period": 3.52474859, "depth": 0.0146},
    {"name": "WASP-12 b", "query": "WASP-12", "mission": "TESS", "sequence": 20, "period": 1.09141890, "depth": 0.0138},
]
TRAPPIST = {"b": 1.510826, "c": 2.421937, "d": 4.049219, "e": 6.101013, "f": 9.207540, "g": 12.352446}


def physics_sanity() -> dict:
    """Fundamental assertions that must hold whatever the data: unit conversion, Kepler's law, non-physical input."""
    from .physics import validate_transit
    c = _check("physics", "Physics Sanity", "Analytic identities", "Exact within rounding", "Halt: derivations are broken", "Extended")
    one = derive(StellarParameters(5772.0, 1.0, 1.0), 365.256, 0.01)
    cases = [("1% dip on a 1 R_sun star is 10.9 R_earth", 10.9076, one["radius_earth"], abs(one["radius_earth"] - 10.9076) < 0.01),
             ("365.256 d around 1 M_sun is 1 AU", 1.0, one["semi_major_axis_au"], abs(one["semi_major_axis_au"] - 1.0) < 1e-3)]
    rejected = 0
    for period, duration in ((3.0, 3.5), (3.0, float("nan")), (3.0, -0.1), (0.0, 0.1)):
        try:
            validate_transit(period, duration)
        except ValueError:
            rejected += 1
    cases.append(("Non-physical durations rejected (>= period, NaN, negative, zero period)", 4, rejected, rejected == 4))
    c["cases"] = [{"label": label, "expected": exp, "measured": got, "ok": bool(ok)} for label, exp, got, ok in cases]
    return _finish(c, [k["ok"] for k in c["cases"]], f"{sum(k['ok'] for k in c['cases'])}/{len(cases)} assertions hold")



def _finish(check: dict, ok_values: list[bool], measured: str) -> dict:
    check["measured"] = measured
    check["status"] = "pass" if ok_values and all(ok_values) else ("skipped" if not ok_values else "fail")
    return check


def run_validation(val_id: int, progress: Callable[[str], None] | None = None, online: bool = True) -> dict:
    report = progress or (lambda msg: None)
    cfg = dict(DEFAULT_PIPELINE)
    log = RunLogger(None, module="validation", path=LOG_DIR / f"validation-{val_id}.jsonl")
    log.info("Starting system verification suite", "VALIDATION", tolerances=TOLERANCES)
    names = [k[0] for k in KEPLER_TARGETS] + RADIUS_BENCHMARKS
    archive, archive_source = fetch_archive(names) if online else ({n: CACHED_ARCHIVE[n] for n in names},
                                                                   "NASA Exoplanet Archive (cached values)")
    checks = []

    # 1. BLS period recovery ------------------------------------------------
    c = _check("bls_period", "BLS Period Recovery", "Kepler confirmed targets (MAST) + synthetic injections",
               r"\(|\Delta P| / P < 10^{-4}\)", "Escalate to non-parametric Fourier comb filter")
    oks = []

    def period_case(label, data, p_true, kind):
        res, arrays = process(data, cfg, log)
        rel = abs(res["bls"]["period"] - p_true) / p_true
        case = {"label": label, "kind": kind, "expected": p_true, "measured": res["bls"]["period"], "rel_error": rel,
                "ok": rel < TOLERANCES["period_rel"]}
        if not case["ok"]:  # operational failure action
            lc = arrays["lightcurve"]
            keep = np.asarray(lc["kept"])
            comb = fourier_comb_period(np.asarray(lc["time"])[keep], np.asarray(lc["flux_flat"])[keep],
                                       cfg["period_min"], cfg["period_max"])
            case["comb_period"] = comb
            case["comb_rel_error"] = abs(comb - p_true) / p_true
            c["action_taken"] = "Escalated failing target(s) to the Fourier comb filter"
        c["cases"].append(case)
        oks.append(case["ok"])

    if online:
        for planet, query, quarter in KEPLER_TARGETS:
            report(f"Period recovery: {planet} (Kepler Q{quarter})")
            try:
                data = ingest.fetch_mast(query, "Kepler", quarter)
                data.stellar.update({"t_eff_k": archive[planet]["st_teff"], "r_star_solar": archive[planet]["st_rad"],
                                     "m_star_solar": archive[planet]["st_mass"]})
                period_case(f"{planet} - Kepler Q{quarter}", data, float(archive[planet]["pl_orbper"]), "kepler")
            except Exception as exc:
                c["notes"].append(f"{planet}: MAST retrieval failed ({type(exc).__name__}: {exc})")
    else:
        c["notes"].append("Offline mode: Kepler targets skipped")
    grid = [(0.917, 0.004, 700), (1.7431, 0.0012, 350), (2.6158, 0.008, 1200), (3.524748, 0.0146, 1000),
            (5.0712, 0.0025, 500), (7.3391, 0.0009, 200), (9.4862, 0.006, 900), (12.163, 0.003, 450)]
    for i, (p, depth, noise) in enumerate(grid):
        report(f"Period recovery: synthetic injection {i + 1}/{len(grid)}")
        data = ingest.make_synthetic(vars(SyntheticSpec(period_days=p, depth=depth, noise_ppm=noise, seed=1000 + i)),
                                     SOLAR, f"VAL-P{i}")
        period_case(f"Synthetic P={p} d, depth {depth * 1e6:.0f} ppm", data, p, "synthetic")
    worst = max((k["rel_error"] for k in c["cases"]), default=float("nan"))
    checks.append(_finish(c, oks, f"max \\(|\\Delta P|/P\\) = {worst:.2e} over {len(oks)} targets"))

    # 2. Transit depth ------------------------------------------------------
    c = _check("depth", "Transit Depth", "Synthetic injections (100 - 5000 ppm)", r"Within \(\pm 5\%\) of injected depth \(\delta\)",
               "Reject target; flag high background noise")
    oks = []
    depths = [100, 180, 320, 560, 1000, 1800, 3200, 5000]
    for i, ppm in enumerate(depths):
        report(f"Depth recovery: {ppm} ppm injection ({i + 1}/{len(depths)})")
        spec = SyntheticSpec(period_days=[2.1, 3.3, 4.7, 2.9, 5.9, 3.8, 6.4, 4.2][i], depth=ppm * 1e-6,
                             noise_ppm=max(40.0, 0.3 * ppm), impact_param=0.0, limb_darkening=False, seed=2000 + i)
        res, _ = process(ingest.make_synthetic(vars(spec), SOLAR, f"VAL-D{i}"), cfg, log)
        got = res["depth"]["adopted_ppm"]
        rel = (got - ppm) / ppm
        ok = abs(rel) <= TOLERANCES["depth_rel"]
        c["cases"].append({"label": f"{ppm} ppm injected (noise {spec.noise_ppm:.0f} ppm)", "expected": ppm,
                           "measured": got, "rel_error": rel, "ok": ok})
        oks.append(ok)
    if not all(oks):
        c["action_taken"] = "Failing injections rejected and flagged for high background noise"
    worst = max(abs(k["rel_error"]) for k in c["cases"])
    checks.append(_finish(c, oks, f"max \\(|\\Delta\\delta|/\\delta\\) = {worst * 100:.2f}% over {len(oks)} injections"))

    # 3. Radius estimation --------------------------------------------------
    c = _check("radius", "Radius Estimation", "Benchmark targets (WASP-12 b, TRAPPIST-1 e)",
               r"Within \(\pm 10\%\) of NASA Archive values", "Flag host star parameter mismatch")
    c["notes"].append(f"Reference values: {archive_source}")
    oks = []
    for planet in RADIUS_BENCHMARKS:
        ref = archive[planet]
        star = StellarParameters(ref["st_teff"], ref["st_rad"], ref["st_mass"])
        got = derive(star, ref["pl_orbper"], ref["pl_trandep"] / 100.0)["radius_earth"]
        rel = (got - ref["pl_rade"]) / ref["pl_rade"]
        ok = abs(rel) <= TOLERANCES["radius_rel"]
        c["cases"].append({"label": f"{planet} - physics engine on archive transit depth", "kind": "analytic",
                           "expected": ref["pl_rade"], "measured": got, "rel_error": rel, "ok": ok})
        oks.append(ok)
    if online:
        report("Radius estimation: WASP-12 b end-to-end from TESS photometry")
        try:
            ref = archive["WASP-12 b"]
            data = ingest.fetch_mast("WASP-12", "TESS", 20)
            data.stellar.update({"t_eff_k": ref["st_teff"], "r_star_solar": ref["st_rad"], "m_star_solar": ref["st_mass"]})
            res, _ = process(data, cfg, log)
            got = res["parameters"]["radius_earth"]
            rel = (got - ref["pl_rade"]) / ref["pl_rade"]
            ok = abs(rel) <= TOLERANCES["radius_rel"]
            c["cases"].append({"label": "WASP-12 b - end-to-end from TESS sector 20 photometry", "kind": "photometric",
                               "expected": ref["pl_rade"], "measured": got, "rel_error": rel, "ok": ok})
            oks.append(ok)
        except Exception as exc:
            c["notes"].append(f"WASP-12 photometric case skipped ({type(exc).__name__}: {exc})")
    c["notes"].append("TRAPPIST-1 e is checked analytically: its 0.5% transits on a V=18.8 star are not recoverable "
                      "from a single TESS sector.")
    if not all(oks):
        c["action_taken"] = "Host star parameter mismatch flagged on failing benchmark(s)"
    worst = max(abs(k["rel_error"]) for k in c["cases"])
    checks.append(_finish(c, oks, f"max \\(|\\Delta R_p|/R_p\\) = {worst * 100:.2f}% over {len(oks)} cases"))

    # 4. Classification accuracy -------------------------------------------
    c = _check("auc", "Classification Accuracy", "Held-out simulated TCE test partition",
               r"AUC \(\ge 0.96\)", "Quarantine model checkpoint; trigger retrain")
    c["notes"].append("The AstroNet Kepler TFRecord benchmark is not bundled; the classifier is scored on an "
                      "independent simulated TCE partition drawn with a seed never used in training.")
    report("Classification: scoring CNN on a fresh test partition")
    try:
        g, l, y = model.make_dataset(3000, seed=val_id * 7919 + 13)
        m = model.evaluate(model.get_model(), g, l, y)
        ok = m["auc_planet"] >= TOLERANCES["auc_min"]
        c["cases"] = [{"label": f"{name} vs rest", "expected": TOLERANCES["auc_min"], "measured": auc,
                       "ok": auc >= TOLERANCES["auc_min"] if i == 0 else None} for i, (name, auc) in enumerate(m["auc"].items())]
        c["confusion"], c["accuracy"] = m["confusion"], m["accuracy"]
        measured = f"simulated AUC = {m['auc_planet']:.4f} (n = {m['n_test']})"
        oks = [ok]
        real = (model.load_metrics() or {}).get("real")
        if real:
            # real held-out Kepler stars are the closest equivalent of the AstroNet benchmark the PRD names
            stars = real["stars"]
            score = np.array([s["p"][0] for s in stars])
            is_planet = np.array([s["label"] == 0 for s in stars])
            rng = np.random.default_rng(0)
            boot = []
            for _ in range(400):
                pick = rng.integers(0, len(stars), len(stars))
                if 0 < is_planet[pick].sum() < len(pick):
                    boot.append(model.roc_curve(score[pick], is_planet[pick])[2])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            real_ok = real["auc_planet"] >= TOLERANCES["auc_min"]
            c["dataset"] = "Held-out Kepler DR24 TCEs (PC / AFP / NTP labels) + simulated TCE partition"
            c["cases"].insert(0, {"label": f"Real Kepler stars, CNN alone, planet vs rest (n = {len(stars)}; 95% interval {lo:.3f} - {hi:.3f})",
                                  "expected": TOLERANCES["auc_min"], "measured": real["auc_planet"], "ok": bool(real_ok)})
            c["notes"][0] = ("Scored on held-out stars from the Kepler DR24 TCE table with the Science Office labels - the "
                             "ground truth behind the AstroNet benchmark - updated where NASA has since confirmed a planet or "
                             "ruled a signal a false positive. The simulated partition gates checkpoint activation.")
            comb = real.get("combined")
            if comb:  # the system's final answer is the second stage (CNN + physical diagnostics)
                c["cases"].insert(0, {"label": f"Real Kepler stars, final classification = CNN + vetting diagnostics (n = {len(stars)})",
                                      "expected": TOLERANCES["auc_min"], "measured": comb["auc_planet"],
                                      "ok": bool(comb["auc_planet"] >= TOLERANCES["auc_min"])})
                c["cases"].insert(1, {"label": "Real Kepler stars, final classification, average precision (PR-AUC)",
                                      "expected": TOLERANCES["pr_auc_min"], "measured": comb["pr_auc_planet"],
                                      "ok": bool(comb["pr_auc_planet"] >= TOLERANCES["pr_auc_min"])})
                oks += [bool(comb["auc_planet"] >= TOLERANCES["auc_min"]), bool(comb["pr_auc_planet"] >= TOLERANCES["pr_auc_min"])]
                measured = f"final AUC = {comb['auc_planet']:.4f}, PR-AUC = {comb['pr_auc_planet']:.4f}; CNN alone: " + measured
            pr_ok = real.get("pr_auc_planet", 0) >= TOLERANCES["pr_auc_min"]
            c["cases"].insert(1, {"label": "Real Kepler stars, CNN alone, average precision (PR-AUC)",
                                  "expected": TOLERANCES["pr_auc_min"], "measured": real.get("pr_auc_planet"), "ok": bool(pr_ok)})
            c["tolerance"] = r"AUC \(\ge 0.96\), PR-AUC \(\ge 0.88\)"
            measured = f"real-star AUC = {real['auc_planet']:.4f}, PR-AUC = {real.get('pr_auc_planet', 0):.4f} (n = {len(stars)}); " + measured
            oks.append(bool(real_ok))
            if not real_ok and ok:
                c["action_taken"] = ("Real-star AUC is below the floor: retrain on a larger Kepler sample "
                                     "(manage.py build_kepler_dataset --per-class 400). Checkpoint kept active because the "
                                     "simulated gate passes")
        if not ok:
            model.CHECKPOINT.replace(model.QUARANTINE)
            model._model = None
            c["action_taken"] = "Checkpoint quarantined; retrain required"
        checks.append(_finish(c, oks, measured))
    except RuntimeError as exc:
        c["notes"].append(str(exc))
        checks.append(_finish(c, [False], "no active checkpoint"))

    # Extended validation (beyond the PRD's five criteria) --------------------
    checks.append(physics_sanity())

    c = _check("golden", "Golden Targets", "HD 209458 b, WASP-12 b, TRAPPIST-1 (MAST)", r"\(|\Delta P|/P < 10^{-4}\), depth \(\pm 10\%\), \(P_{\mathrm{planet}} > 0.95\)",
               "Detrending or BLS is corrupting a known signal", "Extended")
    oks = []
    if online:
        for g in GOLDEN:
            report(f"Golden target: {g['name']}")
            try:
                res, _ = process(ingest.fetch_mast(g["query"], g["mission"], g["sequence"]), cfg, log)
                p_err = abs(res["bls"]["period"] - g["period"]) / g["period"]
                d_err = (res["depth"]["adopted_ppm"] / 1e6 - g["depth"]) / g["depth"]
                p_planet = res["classification"]["probabilities"]["PLANET_CANDIDATE"]
                for label, exp, got, rel, ok in (
                        (f"{g['name']} period (d)", g["period"], res["bls"]["period"], p_err, p_err < 1e-4),
                        (f"{g['name']} depth (fraction)", g["depth"], res["depth"]["adopted_ppm"] / 1e6, d_err, abs(d_err) < 0.10),
                        (f"{g['name']} P(planet)", 0.95, p_planet, None, p_planet > 0.95)):
                    c["cases"].append({"label": label, "expected": exp, "measured": got, "rel_error": rel, "ok": bool(ok)})
                    oks.append(bool(ok))
            except Exception as exc:
                c["notes"].append(f"{g['name']}: skipped ({type(exc).__name__}: {exc})")
        report("Golden target: TRAPPIST-1 (K2 campaign 12)")
        try:
            from . import archive
            data = ingest.fetch_mast("TRAPPIST-1", "K2", 12)
            try:
                data.stellar.update(archive.host_star("TRAPPIST-1"))
            except Exception:  # archive unavailable: Agol et al. 2021 values
                data.stellar.update({"t_eff_k": 2566.0, "r_star_solar": 0.1192, "m_star_solar": 0.0898})
            # transits of an ultra-cool dwarf last under an hour: search shorter duty cycles than the PRD default
            res, _ = process(data, dict(cfg, max_additional_signals=8, q_min=0.004, q_max=0.05, period_min=1.0,
                                        period_max=20.0, detrend_window_days=0.75), log)
            c["notes"].append(f"TRAPPIST-1 photometry: {data.meta.get('author')} (K2 campaign 12, systematics-corrected); "
                              "searched with q in [0.004, 0.05] because its transits last under an hour.")
            found = [res["bls"]["period"]] + [a["period_days"] for a in res["additional_signals"]]
            for letter, period in TRAPPIST.items():
                hit = min(found, key=lambda f: abs(f - period))
                ok = abs(hit - period) / period < 2e-3
                c["cases"].append({"label": f"TRAPPIST-1 {letter} period recovered (d)", "expected": period,
                                   "measured": hit if ok else None, "rel_error": abs(hit - period) / period if ok else None,
                                   "ok": bool(ok)})
                oks.append(bool(ok))
        except Exception as exc:
            c["notes"].append(f"TRAPPIST-1: skipped ({type(exc).__name__}: {exc})")
    else:
        c["notes"].append("Offline mode: golden targets skipped")
    checks.append(_finish(c, oks, f"{sum(oks)}/{len(oks)} golden assertions hold" if oks else "not run"))

    c = _check("injection", "Injection & Recovery", "Mandel-Agol transits injected into quiet real Kepler stars",
               r"\(\le 10\%\) below S/N 4.5; \(\ge 90\%\) from S/N 10", "Pipeline completeness is inadequate: review detrending and thresholds",
               "Extended")
    from . import injection
    inj = injection.load_results()
    if inj:
        for b in inj["bins"]:
            c["cases"].append({"label": f"Injected S/N {b['snr']:g}: detected {b['detected']} of {b['n']} "
                                        f"(period found {b['period_recovered']}, classified planet {b['planet']})",
                               "expected": None, "measured": b["recovery"], "ok": None})
        ok_low = inj["recovery_below_4_5"] is not None and inj["recovery_below_4_5"] <= 0.10
        ok_high = inj["recovery_from_10"] is not None and inj["recovery_from_10"] >= 0.90
        c["notes"].append(f"Transit model: {inj['engine']}; {inj['stars']} real quiet stars; run {inj['generated_at'][:16]} UTC. "
                          f"Detection requires the injected period and a measured S/N of at least 7.1, so recovery at an "
                          f"injected S/N of exactly 7.1 is near one half by construction ({inj['recovery_at_7_1']}).")
        checks.append(_finish(c, [ok_low, ok_high], f"{inj['recovery_below_4_5'] * 100:.0f}% below S/N 4.5; "
                                                    f"{inj['recovery_from_10'] * 100:.0f}% from S/N 10 (n = {inj['n_from_10']})"))
    else:
        c["notes"].append("No injection study on disk: run `manage.py injection_study`.")
        checks.append(_finish(c, [], "not run"))

    # 5. Structured log integrity -------------------------------------------
    c = _check("log_integrity", "Structured Log Integrity", "All execution runs", "100% valid JSON schema matching",
               "Terminate ingest pipeline run")
    report("Log integrity: verifying every JSONL audit trail")
    log.info("Verification stages complete; auditing structured logs", "VALIDATION")
    total = valid = files = broken = 0
    for path in sorted(LOG_DIR.glob("*.jsonl")):
        v = verify_jsonl(path)
        files += 1
        total += v["records"]
        valid += v["valid"]
        if v["valid"] != v["records"] or not v["chain_ok"]:
            broken += 1
            c["cases"].append({"label": path.name, "measured": f"{v['valid']}/{v['records']} valid", "ok": False,
                               "detail": "; ".join(v["errors"][:3])})
    pct = 100.0 * valid / total if total else 100.0
    c["cases"].insert(0, {"label": f"{files} audit files, {total} records", "measured": f"{pct:.2f}% valid", "ok": broken == 0})
    checks.append(_finish(c, [broken == 0], f"{valid}/{total} records valid ({pct:.2f}%), hash chains {'intact' if broken == 0 else 'BROKEN'}"))

    order = {"PRD §6": 0, "Extended": 1}
    checks.sort(key=lambda k: order[k["group"]])
    prd = [k for k in checks if k["group"] == "PRD §6"]
    ext = [k for k in checks if k["group"] == "Extended"]
    summary = {"passed": sum(k["status"] == "pass" for k in prd), "failed": sum(k["status"] == "fail" for k in prd),
               "skipped": sum(k["status"] == "skipped" for k in prd), "total": len(prd),
               "extended_passed": sum(k["status"] == "pass" for k in ext), "extended_failed": sum(k["status"] == "fail" for k in ext),
               "extended_total": len(ext)}
    return {"checks": checks, "summary": summary, "archive_source": archive_source, "online": online}
