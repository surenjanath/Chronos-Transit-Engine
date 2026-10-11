"""Research: a possible planet from the survey, analysed on its own, and the report written from a finished run."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from cte import survey
from cte.config import DATA_DIR

from . import services
from .models import ResearchCase, Run

REPORT_DIR = DATA_DIR / "reports"
BROWSERS = ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge", "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
            "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "chrome")


def same_period(a: float, b: float, tolerance: float = 0.01) -> bool:
    return abs(a / b - 1.0) < tolerance


def find_signal(kepid: int, period: float) -> dict | None:
    return next((s for s in survey.possible_planets() if s["kepid"] == int(kepid) and same_period(s["period_days"], period, 1e-4)), None)


def open_case(kepid: int, period: float) -> ResearchCase:
    """The case for this signal, started if it is new or failed before."""
    signal = find_signal(kepid, period)
    if signal is None:
        raise LookupError("That signal is not in the survey's list of possible planets.")
    case = next((c for c in ResearchCase.objects.filter(kepid=kepid) if same_period(c.period_days, period, 1e-4)), None)
    if case is None:
        case = ResearchCase.objects.create(kepid=kepid, period_days=signal["period_days"], host=signal["host"], signal=signal)
    elif case.status != "failed":
        return case
    ResearchCase.objects.filter(pk=case.pk).update(status="queued", step="Waiting for a free worker", error="")
    services._spawn(investigate, case.pk)
    return case


ROUNDS = 4


def signals_of(results: dict) -> list[dict]:
    """Every signal of a finished run, the main one first, as period, epoch, duration and depth."""
    out = [{"period_days": results["bls"]["period"], "t0": results["bls"]["t0"], "duration_hours": results["fit"]["t14_hours"],
            "depth_ppm": results["parameters"]["transit_depth_ppm"]}]
    out += [{"period_days": a["period_days"], "t0": a["t0"], "duration_hours": a["duration_hours"], "depth_ppm": a["depth_ppm"]}
            for a in results.get("additional_signals", [])]
    return out


def describe(run: Run) -> dict:
    """The main signal of a run in a line, for the record of what each round found."""
    res = run.results
    return {"run": run.id, "period_days": run.period_days, "depth_ppm": run.depth_ppm, "snr": res["diagnostics"]["snr"],
            "prediction": run.prediction, "p_planet": run.p_planet, "flags": [f["code"] for f in res.get("flags", [])]}


def investigate(case_id: int) -> None:
    """Run the star, then run it again with everything but this signal cut out. If something else is still stronger,
    cut that out too and try again; after `ROUNDS` tries the signal is recorded as not recovered on its own."""
    def update(**fields):
        ResearchCase.objects.filter(pk=case_id).update(**fields)

    case = ResearchCase.objects.get(pk=case_id)
    try:
        base = Run.objects.filter(source="research", status="completed", config__kepid=case.kepid, config__remove__isnull=True).first()
        if base is None:
            update(status="running", step="Analysing the whole light curve of the star")
            base_id = services.create_run({"source": "research", "kepid": case.kepid})
            update(base_run_id=base_id)
            services.execute_run(base_id)
            base = Run.objects.get(pk=base_id)
            if base.status != "completed":
                raise RuntimeError(f"the analysis of the star failed: {base.error}")
        update(base_run_id=base.id, run_id=None, outcome={})
        signals = signals_of(base.results)
        star = describe(base)
        if not any(same_period(s["period_days"], case.period_days) for s in signals):
            update(status="done", step="", outcome={"recovered": False, "star": star, "rounds": [], "seen_as_further": False,
                                                    "reason": "a fresh analysis of the whole star did not find this signal again"})
            return
        if same_period(signals[0]["period_days"], case.period_days):
            update(status="done", step="", run_id=base.id, outcome={"recovered": True, "star": star, "rounds": []})
            return
        removed = [s for s in signals if not same_period(s["period_days"], case.period_days)]
        rounds, seen = [], True
        for n in range(1, ROUNDS + 1):
            update(status="running", step=f"Removing {len(removed)} other signal{'s' if len(removed) != 1 else ''} and analysing this one on its own"
                                          + (f" (try {n})" if n > 1 else ""))
            run_id = services.create_run({"source": "research", "kepid": case.kepid, "remove": removed, "focus_period": case.period_days})
            services.execute_run(run_id)
            run = Run.objects.get(pk=run_id)
            if run.status != "completed":
                raise RuntimeError(f"the analysis of the signal failed: {run.error}")
            found = signals_of(run.results)
            rounds.append(describe(run) | {"removed": len(removed)})
            if same_period(found[0]["period_days"], case.period_days):
                update(status="done", step="", run_id=run.id, outcome={"recovered": True, "star": star, "rounds": rounds})
                return
            seen = any(same_period(s["period_days"], case.period_days) for s in found)
            removed = removed + [s for s in found if not same_period(s["period_days"], case.period_days)]
            if run.disposition == "NO_SIGNIFICANT_SIGNAL":
                break  # nothing left above the noise: removing it would not bring this signal out
        update(status="done", step="", outcome={"recovered": False, "star": star, "rounds": rounds, "seen_as_further": seen,
                                                "reason": "with the star's other signals removed, this one was not the strongest thing left"})
    except Exception as exc:
        update(status="failed", step="", error=str(exc)[:1000])


def browser() -> str | None:
    """A Chromium-family browser to print the report with, if the machine has one."""
    for name in (os.environ.get("CTE_BROWSER"), *BROWSERS):
        if name and (os.path.isfile(name) or shutil.which(name)):
            return name if os.path.isfile(name) else shutil.which(name)
    return None


def render_pdf(url: str, run_id: str) -> bytes:
    """The report page printed to PDF by a headless browser, so it keeps the page's own fonts and charts."""
    exe = browser()
    if exe is None:
        raise RuntimeError("No Chrome, Chromium, Edge or Brave was found to print the report. Open the report and use the browser's Print, Save as PDF.")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"{run_id}.pdf"
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={profile}",
                        "--force-device-scale-factor=2", "--virtual-time-budget=20000", "--run-all-compositor-stages-before-draw",
                        f"--print-to-pdf={out}", url], check=True, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out.read_bytes()


# ---------------------------------------------------------------------------
# Transit timing, the data packet, and statistical validation
# ---------------------------------------------------------------------------

VALIDATION_DIR = DATA_DIR / "validation"
PACKET_README = """Chronos Transit Engine - data packet for one signal
====================================================

summary.json         Target, ephemeris, transit parameters, host star, every vetting measurement, the classifier's
                     verdict, what NASA lists on the star, and the run and audit-trail hash this came from.
folded_transit.csv   The detrended light curve folded on the ephemeris, within the stated window of mid-transit.
                     time_days is the time from mid-transit; flux is normalised to 1 out of transit; flux_err is the
                     robust scatter of the out-of-transit points (one value, repeated).
transit_times.csv    Mid-time of each transit that could be timed on its own, with its error and its difference from
                     a constant period (present only when single transits stand out of the noise).
validation.json      False-positive probability from TRICERATOPS, when it has been run for this signal.

folded_transit.csv is in the form TRICERATOPS, DAVE and similar tools take: time from mid-transit in days, flux, error.
To run TRICERATOPS on it:  python tools/triceratops_validate.py <this packet, unzipped>

Nothing in this packet is a confirmation. It is what one pipeline measured, laid out so that it can be checked.
"""


def _series(run: Run):
    import numpy as np
    import pyarrow.parquet as pq

    from cte.config import ARCHIVE_DIR
    lc = pq.read_table(ARCHIVE_DIR / run.id / "lightcurve.parquet").to_pydict()
    kept = np.asarray(lc["kept"], dtype=bool)
    return np.asarray(lc["time"], dtype=float)[kept], np.asarray(lc["flux_flat"], dtype=float)[kept]


def timing(run: Run) -> dict | None:
    """Transit times of a finished run, measured once and kept beside its archive."""
    import json

    from cte import ttv
    from cte.config import ARCHIVE_DIR
    path = ARCHIVE_DIR / run.id / "timing.json"
    if path.exists():
        return json.loads(path.read_text())
    if not (ARCHIVE_DIR / run.id / "lightcurve.parquet").exists():
        return None
    res = run.results
    t, flat = _series(run)
    out = ttv.transit_times(t, flat, res["bls"]["period"], res["bls"]["t0"], res["fit"]["t14_hours"], res["fit"]["t12_hours"],
                            res["fit"]["depth_ppm"])
    path.write_text(json.dumps(out))
    return out


def validation(run: Run) -> dict | None:
    import json
    path = VALIDATION_DIR / f"{run.id}.json"
    return json.loads(path.read_text()) if path.exists() else None


def summary(run: Run) -> dict:
    """Everything measured about the signal of a run, as one machine-readable record."""
    res = run.results
    p, d, fit, meta = res["parameters"], res["diagnostics"], res["fit"], res.get("metadata") or {}
    case = run.research_cases.first()
    last = run.logs.order_by("seq").last()
    archive_match = res.get("archive_match") or {}
    t = timing(run)
    return {
        "target": {"name": case.host if case else run.target.display, "catalogue_id": run.target.name, "kepid": meta.get("kepid"),
                   "mission": run.target.mission, "ra_deg": run.target.ra, "dec_deg": run.target.dec},
        "ephemeris": {"period_days": res["bls"]["period"], "t0": res["bls"]["t0"], "time_system": meta.get("time_system"),
                      "note": "BKJD = BJD - 2454833" if meta.get("time_system") == "bkjd" else None},
        "transit": {"depth_ppm": p["transit_depth_ppm"], "duration_hours": fit["t14_hours"], "ingress_hours": fit["t12_hours"],
                    "radius_ratio": p["radius_ratio"], "radius_earth": p["radius_earth"], "a_over_rstar": p["a_over_rstar"],
                    "semi_major_axis_au": p["semi_major_axis_au"], "equilibrium_temp_k": p["equilibrium_temp_k"],
                    "snr": d["snr"], "transits_observed": d["n_transits"], "sde": res["bls"]["sde"]},
        "host_star": {k: res["stellar"].get(k) for k in ("t_eff_k", "r_star_solar", "m_star_solar")},
        "vetting": {"odd_even": d.get("odd_even"), "secondary": d.get("secondary"), "centroid": d.get("centroid"),
                    "pixels": d.get("pixels"), "duration_ratio": d.get("duration_ratio"), "v_shape": fit.get("v_shape"),
                    "model_shift": d.get("model_shift"), "oot_scatter_ppm": d.get("oot_scatter_ppm"),
                    "flags": [f["code"] for f in res.get("flags", [])]},
        "classifier": {"prediction": run.prediction, "probabilities": res["classification"]["probabilities"],
                       "disposition": run.disposition},
        "catalogue": {"objects_on_star": archive_match.get("objects") or [], "match": archive_match.get("match")},
        "removed_before_search": meta.get("removed_signals") or [],
        "survey": {k: case.signal.get(k) for k in ("rank", "signals_on_star", "snr", "p_planet", "event_chi2", "notes")} if case else None,
        "timing": {k: v for k, v in t.items() if k != "transits"} if t else None,
        "validation": validation(run),
        "provenance": {"run": run.id, "analysed_utc": run.finished_at.isoformat() if run.finished_at else None,
                       "audit_records": run.logs.count(), "audit_last_sha256": last.hash if last else None,
                       "settings": res.get("settings"), "software": "Chronos Transit Engine, github.com/surenjanath/Chronos-Transit-Engine"},
        "status": "candidate signal; not a confirmation",
    }


def folded(run: Run) -> tuple[list[tuple[float, float, float]], float]:
    """(time from mid-transit in days, flux, error) within three durations of mid-transit, and that window in days."""
    import numpy as np
    res = run.results
    t, flat = _series(run)
    period, t14 = res["bls"]["period"], res["fit"]["t14_hours"] / 24.0
    x = (t - res["bls"]["t0"] + 0.5 * period) % period - 0.5 * period
    window = min(max(3.0 * t14, 0.25), 0.45 * period)
    out = np.abs(x) > t14
    sigma = float(1.4826 * np.median(np.abs(flat[out] - np.median(flat[out]))))
    near = np.flatnonzero(np.abs(x) < window)
    near = near[np.argsort(x[near])]
    return [(float(x[i]), float(flat[i]), sigma) for i in near], window


def packet(run: Run) -> bytes:
    """A zip of the summary, the folded light curve, the transit times and any validation result."""
    import io
    import json
    import zipfile
    rows, window = folded(run)
    info = summary(run)
    info["folded_transit_window_days"] = window
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", PACKET_README)
        z.writestr("summary.json", json.dumps(info, indent=1, default=str))
        z.writestr("folded_transit.csv", "time_days,flux,flux_err\n" + "".join(f"{a:.6f},{b:.7f},{c:.7f}\n" for a, b, c in rows))
        t = timing(run)
        if t and t.get("measurable"):
            z.writestr("transit_times.csv", "epoch,time,error_minutes,o_minus_c_minutes\n" + "".join(
                f"{v['epoch']},{v['time']:.6f},{v['error_minutes']},{v['o_minus_c_minutes']}\n" for v in t["transits"]))
        if info["validation"]:
            z.writestr("validation.json", json.dumps(info["validation"], indent=1))
    return buf.getvalue()
