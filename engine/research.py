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


def signals_of(results: dict) -> list[dict]:
    """Every signal of a finished run, the main one first, as period, epoch and duration."""
    out = [{"period_days": results["bls"]["period"], "t0": results["bls"]["t0"], "duration_hours": results["fit"]["t14_hours"]}]
    out += [{"period_days": a["period_days"], "t0": a["t0"], "duration_hours": a["duration_hours"]} for a in results.get("additional_signals", [])]
    return out


def investigate(case_id: int) -> None:
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
        update(base_run_id=base.id)
        signals = signals_of(base.results)
        if not any(same_period(s["period_days"], case.period_days) for s in signals):
            raise RuntimeError("a fresh analysis of the star did not find this signal again; it found "
                               + ", ".join(f"{s['period_days']:.4f} d" for s in signals))
        if same_period(signals[0]["period_days"], case.period_days):
            update(status="done", step="", run_id=base.id)  # already the main signal of the star
            return
        update(status="running", step="Removing the star's other transits and analysing this signal on its own")
        others = [s for s in signals if not same_period(s["period_days"], case.period_days)]
        run_id = services.create_run({"source": "research", "kepid": case.kepid, "remove": others, "focus_period": case.period_days})
        update(run_id=run_id)
        services.execute_run(run_id)
        run = Run.objects.get(pk=run_id)
        if run.status != "completed":
            raise RuntimeError(f"the analysis of the signal failed: {run.error}")
        update(status="done", step="")
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
