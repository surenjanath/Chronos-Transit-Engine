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
