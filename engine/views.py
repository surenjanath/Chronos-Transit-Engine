"""Server-rendered pages. JSON endpoints exist only to refresh live status and feed the plots."""
import csv
import json

import numpy as np
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Count, Q
from django.http import FileResponse, Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_GET, require_POST

from cte import core, galaxy, ingest, model, signalcheck
from cte.config import ARCHIVE_DIR, DEFAULT_PIPELINE, LOG_DIR, TOLERANCES
from cte.physics import StellarParameters, derive
from cte.synthetic import a_over_rstar, transit_flux
from cte.telemetry import verify_jsonl

from . import autopilot, services
from . import research as research_jobs
from .forms import CalculatorForm, NewRunForm, SettingsForm, SignalCheckForm, VettingForm
from .models import CLASS_LABELS, DISPOSITIONS, AutomationCycle, LogEvent, ResearchCase, Run, Target, ValidationRun


def _page(request, queryset, per_page):
    return Paginator(queryset, per_page).get_page(request.GET.get("page"))


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def overview(request):
    data = services.overview()
    return render(request, "pages/overview.html", {
        "o": data, "page_title": "Overview",
        "chart": {"daily": data["daily"], "classes": [{"label": c["label"], "count": c["count"]} for c in data["classes"]]},
    })


@require_GET
def live(request):
    """Header status: lets open pages notice finished runs without a manual refresh."""
    active = list(Run.objects.filter(status__in=("queued", "running")).values("id", "stage", "progress"))
    latest = Run.objects.order_by("-finished_at").exclude(finished_at=None).values_list("id", flat=True).first()
    val = ValidationRun.objects.first()
    return JsonResponse({"active": active, "latest_finished": latest, "runs": Run.objects.count(),
                         "pending": Run.objects.filter(services.queue_filter(), vetting_status="pending").count(),
                         "training": services.train_state, "benchmark": services.benchmark_state,
                         "autopilot": AutomationCycle.objects.filter(status="running").values_list("step", flat=True).first(),
                         "validation": val.status if val else None,
                         "validation_progress": val.progress if val else ""})


@require_GET
def log_tail(request):
    rows = list(LogEvent.objects.all()[:40])[::-1]
    return JsonResponse({"lines": [{"time": e.timestamp.strftime("%H:%M:%S"), "level": e.level, "stage": e.stage,
                                    "message": e.message + (f" [{e.run_id}]" if e.run_id else "")} for e in rows]})


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def targets(request):
    q, mission = request.GET.get("q", "").strip(), request.GET.get("mission", "")
    qs = Target.objects.annotate(n_runs=Count("runs"))
    if q:
        qs = qs.filter(Q(name__icontains=q) | Q(alias__icontains=q))
    if mission:
        qs = qs.filter(mission=mission)
    rows = list(qs)
    latest = {}
    for run in Run.objects.filter(status="completed", target__in=rows).defer("results", "config").order_by("created_at"):
        latest[run.target_id] = run
    for t in rows:
        t.latest = latest.get(t.id)
    return render(request, "pages/targets.html", {"targets": rows, "q": q, "mission": mission, "page_title": "Targets",
                                                  "missions": ["TESS", "Kepler", "K2", "Synthetic", "Upload"]})


def _place(t: Target) -> dict | None:
    """Where a target is, if the archive lists it as a planet host with a measured distance."""
    host = galaxy.locate([t.name, t.alias], t.ra, t.dec)
    if not host:
        return None
    return {"host": host["name"], "distance_ly": host["d"], "planets": len(host["planets"]), "travel": galaxy.travel(host["d"])}


def target(request, pk):
    t = get_object_or_404(Target, pk=pk)
    runs = list(t.runs.defer("results", "config"))
    latest = next((r for r in runs if r.status == "completed"), None)
    return render(request, "pages/target.html", {"t": t, "runs": runs, "latest": latest, "page_title": t.display,
                                                 "place": _place(t)})


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def runs(request):
    status, disposition, q = request.GET.get("status", ""), request.GET.get("disposition", ""), request.GET.get("q", "").strip()
    qs = Run.objects.select_related("target").defer("results", "config")
    if status in ("completed", "failed"):
        qs = qs.filter(status=status)
    elif status == "running":
        qs = qs.filter(status__in=("queued", "running"))
    if disposition in DISPOSITIONS:
        qs = qs.filter(disposition=disposition)
    if q:
        qs = qs.filter(Q(target__name__icontains=q) | Q(target__alias__icontains=q) | Q(id__icontains=q))
    counts = Run.objects.aggregate(total=Count("id"), done=Count("id", filter=Q(status="completed")),
                                   failed=Count("id", filter=Q(status="failed")),
                                   live=Count("id", filter=Q(status__in=("queued", "running"))))
    durations = sorted(Run.objects.filter(status="completed").exclude(duration_s=None).values_list("duration_s", flat=True))
    return render(request, "pages/runs.html", {
        "runs": _page(request, qs, 25), "status": status, "disposition": disposition, "q": q, "counts": counts,
        "median": durations[len(durations) // 2] if durations else None, "dispositions": DISPOSITIONS, "page_title": "Runs",
        "statuses": [("", "All"), ("completed", "Completed"), ("running", "Running"), ("failed", "Failed")],
    })


def run(request, pk):
    r = get_object_or_404(Run.objects.select_related("target"), pk=pk)
    res = r.results or {}
    probs = [{"key": k, "label": CLASS_LABELS[k], "p": p, "pct": round(p * 100, 2), "top": k == r.prediction}
             for k, p in (res.get("classification", {}).get("probabilities") or {}).items()]
    stage_i = core.STAGES.index(r.stage) if r.stage in core.STAGES else -1
    stages = [{"name": s.replace("_", " ").lower(), "state": "done" if i < stage_i or r.status == "completed" else
               ("now" if i == stage_i and r.is_live else "")} for i, s in enumerate(core.STAGES)]
    return render(request, "pages/run.html", {
        "run": r, "res": res, "probs": probs, "stages": stages, "logs": r.logs.order_by("seq"),
        "page_title": r.target.display, "downloads": ["lightcurve", "periodogram", "folded", "tensors"] if res else [],
        "system": system_view(r, r.research_cases.first()) if r.status == "completed" else None,
        "place": _place(r.target),
    })


@require_GET
def run_status(request, pk):
    r = get_object_or_404(Run.objects.defer("results", "config"), pk=pk)
    from django.utils import timezone
    started = r.started_at or r.created_at
    events = [{"time": e.timestamp.strftime("%H:%M:%S"), "stage": e.stage.replace("_", " ").lower(), "message": e.message,
               # the ingest record of a benchmark star carries the catalogue's answer, which is not something found
               "facts": [] if e.stage == "INGEST" else _facts(e.telemetry or {})} for e in r.logs.order_by("-seq")[:8]][::-1]
    return JsonResponse({"status": r.status, "stage": r.stage, "stage_label": r.stage_label, "percent": r.percent,
                         "stage_index": core.STAGES.index(r.stage) if r.stage in core.STAGES else -1,
                         "elapsed_s": round((timezone.now() - started).total_seconds()),
                         "note": services.live_notes.get(r.id, "") if r.status == "running" else
                                 ("Waiting for a free worker." if r.status == "queued" else ""),
                         "events": events})


FACT_LABELS = (("cadences", "measurements", "{:,.0f}"), ("baseline_days", "days observed", "{:,.1f}"), ("rejected", "glitches removed", "{:,.0f}"),
               ("trial_periods", "periods tried", "{:,.0f}"), ("period_days", "period (days)", "{:.5f}"), ("sde", "peak strength (SDE)", "{:.1f}"),
               ("depth_ppm", "depth (ppm)", "{:,.0f}"), ("radius_earth", "radius (Earths)", "{:.2f}"), ("snr", "S/N", "{:.1f}"),
               ("n_transits", "transits seen", "{:.0f}"), ("prediction", "verdict", "{}"), ("p_planet", "P(planet)", "{:.3f}"))


def _facts(telemetry: dict) -> list[str]:
    """The few numbers in a log record worth showing to someone watching a run."""
    flat = dict(telemetry)
    for v in telemetry.values():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                flat.setdefault(k2, v2)
    out = []
    for key, label, fmt in FACT_LABELS:
        v = flat.get(key)
        if v is None or isinstance(v, (dict, list)):
            continue
        try:
            out.append(f"{label} {fmt.format(v)}".replace("_", " ").lower() if key == "prediction" else f"{label} {fmt.format(v)}")
        except (ValueError, TypeError):
            continue
    return out[:5]


@require_GET
def run_series(request, pk):
    series = services.run_series(get_object_or_404(Run, pk=pk))
    if series is None:
        raise Http404("No archived series for this run")
    return JsonResponse(series)


@require_POST
def run_vet(request, pk):
    r = get_object_or_404(Run, pk=pk)
    form = VettingForm(request.POST)
    if r.status != "completed":
        messages.error(request, "Only completed runs can be vetted.")
    elif not form.is_valid():
        messages.error(request, "Invalid vetting decision.")
    else:
        services.vet_run(r, form.cleaned_data["status"], form.cleaned_data["note"])
        messages.success(request, {"confirmed": f"{r.target.display} confirmed as a candidate.",
                                   "false_positive": f"{r.target.display} marked as a false positive.",
                                   "pending": f"{r.target.display} returned to the queue."}[r.vetting_status])
    nxt = request.POST.get("next", "")
    if url_has_allowed_host_and_scheme(nxt, allowed_hosts={request.get_host()}):
        return redirect(nxt)
    return redirect("run", pk=pk)


@require_POST
def run_rerun(request, pk):
    r = get_object_or_404(Run, pk=pk)
    new_id = services.submit_run(json.loads(json.dumps(r.config)))
    messages.success(request, "Re-run submitted with the same configuration.")
    return redirect("run", pk=new_id)


@require_POST
def run_delete(request, pk):
    r = get_object_or_404(Run.objects.select_related("target"), pk=pk)
    if r.is_live:
        messages.error(request, "This run is still in progress.")
        return redirect("run", pk=pk)
    services.delete_run(r)
    messages.success(request, f"Run {pk} and its archived products were deleted.")
    return redirect("runs")


@require_POST
def run_verify(request, pk):
    get_object_or_404(Run, pk=pk)
    v = verify_jsonl(LOG_DIR / f"{pk}.jsonl")
    if v["chain_ok"] and v["valid"] == v["records"]:
        messages.success(request, f"Audit trail intact: {v['records']} records valid, SHA-256 hash chain unbroken.")
    else:
        messages.error(request, f"Audit trail integrity failure: {'; '.join(v['errors'][:2]) or 'invalid records'}")
    return redirect("run", pk=pk)


@require_GET
def run_download(request, pk, name):
    r = get_object_or_404(Run.objects.select_related("target"), pk=pk)
    files = {"audit.jsonl": LOG_DIR / f"{pk}.jsonl"} | {
        f"{n}.parquet": ARCHIVE_DIR / pk / f"{n}.parquet" for n in ("lightcurve", "periodogram", "folded", "tensors")}
    path = files.get(name)
    if path is None or not path.exists():
        raise Http404("File not found")
    return FileResponse(path.open("rb"), as_attachment=True, filename=f"{r.target.name.replace(' ', '_')}_{pk}_{name}")


# ---------------------------------------------------------------------------
# Vetting queue
# ---------------------------------------------------------------------------

def vetting(request):
    tab = request.GET.get("tab", "pending")
    if tab not in ("pending", "confirmed", "false_positive"):
        tab = "pending"
    qs = Run.objects.select_related("target").defer("results", "config").filter(status="completed", vetting_status=tab)
    if tab == "pending":
        qs = qs.filter(services.queue_filter())
    return render(request, "pages/vetting.html", {
        "runs": list(qs), "tab": tab, "page_title": "Vetting",
        "tabs": [("pending", "Awaiting decision"), ("confirmed", "Confirmed"), ("false_positive", "False positives")],
    })


# ---------------------------------------------------------------------------
# New analysis
# ---------------------------------------------------------------------------

def new(request):
    products, search_error = None, ""
    if request.method == "POST":
        form = NewRunForm(request.POST, request.FILES)
        if not model.CHECKPOINT.exists():
            messages.error(request, "No active CNN checkpoint. Train the classifier on the Validation page first.")
        elif form.is_valid():
            config = form.cleaned_data["config"]
            batch = form.cleaned_data.get("batch_targets") or []
            if len(batch) > 1:  # several MAST targets at once: one run each
                for name in batch:
                    services.submit_run(json.loads(json.dumps(config | {"mast": config["mast"] | {"target": name}})))
                messages.success(request, f"{len(batch)} pipeline runs submitted: {', '.join(batch)}.")
                return redirect("runs")
            run_id = services.submit_run(config)
            messages.success(request, "Pipeline run submitted.")
            return redirect("run", pk=run_id)
        source = request.POST.get("source", "mast")
    else:
        source = request.GET.get("source", "mast")
        initial = {"source": source, "mast_target": request.GET.get("mast_target", "WASP-12"),
                   "mission": request.GET.get("mission", "TESS"), **NewRunForm.SYN_DEFAULTS}
        form = NewRunForm(initial=initial)
        if request.GET.get("search") and initial["mast_target"].strip():
            try:
                products = ingest.search_mast(initial["mast_target"].strip(), initial["mission"])
            except Exception as exc:
                search_error = f"MAST search failed: {type(exc).__name__}: {exc}"
    if source not in ("mast", "synthetic", "upload"):
        source = "mast"
    return render(request, "pages/new.html", {"form": form, "source": source, "products": products,
                                              "search_error": search_error, "defaults": services.pipeline_settings(),
                                              "page_title": "New analysis"})


# ---------------------------------------------------------------------------
# Calculator (PRD §7)
# ---------------------------------------------------------------------------

CALC_PRESETS = [
    ("HD 209458 b", {"r_star_solar": 1.155, "m_star_solar": 1.119, "t_eff_k": 6065, "period_days": 3.524748, "depth_ppm": 14600, "impact_param": 0.5, "bond_albedo": 0.3}),
    ("WASP-12 b", {"r_star_solar": 1.69, "m_star_solar": 1.325, "t_eff_k": 6265, "period_days": 1.091419, "depth_ppm": 13780, "impact_param": 0.4, "bond_albedo": 0.3}),
    ("TRAPPIST-1 e", {"r_star_solar": 0.1192, "m_star_solar": 0.0898, "t_eff_k": 2566, "period_days": 6.101013, "depth_ppm": 5012, "impact_param": 0.2, "bond_albedo": 0.3}),
    ("Earth – Sun", {"r_star_solar": 1.0, "m_star_solar": 1.0, "t_eff_k": 5772, "period_days": 365.256, "depth_ppm": 84, "impact_param": 0.0, "bond_albedo": 0.3}),
]
CALC_SLIDERS = [  # key, label, unit, min, max, log scale?
    ("r_star_solar", r"Stellar radius \(R_\star\)", r"\(R_\odot\)", 0.08, 3, False), ("m_star_solar", r"Stellar mass \(M_\star\)", r"\(M_\odot\)", 0.08, 3, False),
    ("t_eff_k", r"Effective temperature \(T_{\mathrm{eff}}\)", "K", 2300, 10000, False), ("period_days", r"Orbital period \(P\)", "d", 0.3, 400, True),
    ("depth_ppm", r"Transit depth \(\delta\)", "ppm", 20, 50000, True), ("impact_param", r"Impact parameter \(b\)", "", 0, 1.2, False),
    ("bond_albedo", r"Bond albedo \(A_B\)", "", 0, 0.9, False),
]


def compute_physics(v: dict) -> dict:
    depth = v["depth_ppm"] / 1e6
    params = derive(StellarParameters(v["t_eff_k"], v["r_star_solar"], v["m_star_solar"]), v["period_days"], depth,
                    v["impact_param"], v["bond_albedo"])
    k, b = params["radius_ratio"], v["impact_param"]
    a_rs = a_over_rstar(v["period_days"], v["m_star_solar"], v["r_star_solar"])
    half = max(params["transit_duration_hours"], 0.2) * 1.6 / 24.0
    x = np.linspace(-half, half, 241)
    transits = bool(a_rs > 1.0 + k and b < 1.0 + k)
    flux = transit_flux(x, v["period_days"], 0.0, k, max(a_rs, 1.0 + k), b) if transits else np.ones_like(x)
    warnings = []
    if a_rs <= 1.0 + k:
        warnings.append("Orbit lies inside the star (a ≤ R★ + Rp): these parameters are unphysical.")
    if b >= 1.0 + k:
        warnings.append("Impact parameter exceeds 1 + k: the planet does not transit.")
    elif b > 1.0 - k:
        warnings.append("Grazing geometry (b > 1 − k): the transit is V-shaped and depth underestimates (Rp/R★)².")
    if params["radius_earth"] > 25.0:
        warnings.append("Radius above 2.2 R_Jup: a stellar companion is more likely than a planet.")
    params["equilibrium_temp_c"] = round(params["equilibrium_temp_k"] - 273.15, 1)
    return {"parameters": params, "warnings": warnings,
            "geometry": {"k": k, "a_rs": max(a_rs, 1.0 + k), "b": b, "period_days": v["period_days"]},
            "curve": {"hours": np.round(x * 24.0, 4).tolist(), "flux": np.round(flux, 7).tolist()}}


def calculator(request):
    values = dict(CALC_PRESETS[0][1])
    form = CalculatorForm(request.GET) if request.GET else None
    if form is not None:
        if form.is_valid():
            values = form.cleaned_data
        else:
            messages.error(request, "Some inputs were out of range; showing the default system.")
    out = compute_physics(values)
    if request.GET.get("record") and form is not None and form.is_valid():
        services.system_logger().info("Derived physical exoplanetary parameters", "PHYSICS_DERIVATION",
                                      results=out["parameters"], inputs=values, origin="interactive_vetting_tool")
        messages.success(request, "Calculation written to the audit log.")
    sliders = [{"key": k, "label": label, "unit": unit, "min": lo, "max": hi, "log": is_log, "value": values[k]}
               for k, label, unit, lo, hi, is_log in CALC_SLIDERS]
    return render(request, "pages/calculator.html", {"sliders": sliders, "out": out, "presets": CALC_PRESETS,
                                                     "page_title": "Calculator"})


@require_GET
def calculator_compute(request):
    form = CalculatorForm(request.GET)
    if not form.is_valid():
        return JsonResponse({"errors": form.errors}, status=422)
    return JsonResponse(compute_physics(form.cleaned_data))


# ---------------------------------------------------------------------------
# Signal check: a verdict from catalogue parameters alone
# ---------------------------------------------------------------------------

CHECK_PRESETS = [
    ("Hot Jupiter", {"period_days": 3.5247, "depth_ppm": 14600, "duration_hours": 3.0, "snr": 300, "n_transits": 100, "t_eff_k": 6065,
                     "r_star_solar": 1.155, "m_star_solar": 1.119, "odd_even_sigma": 0.5, "secondary_sigma": 0.5, "secondary_depth_ppm": 20, "centroid_sigma": 0.4}),
    ("Small planet", {"period_days": 12.3, "depth_ppm": 420, "duration_hours": 3.9, "snr": 28, "n_transits": 110, "t_eff_k": 5600,
                      "r_star_solar": 0.95, "m_star_solar": 0.93, "odd_even_sigma": 0.6, "secondary_sigma": 0.3, "secondary_depth_ppm": 5, "centroid_sigma": 0.5}),
    ("Eclipsing binary", {"period_days": 5.318, "depth_ppm": 85000, "duration_hours": 4.6, "snr": 900, "n_transits": 250, "t_eff_k": 5900,
                          "r_star_solar": 1.1, "m_star_solar": 1.05, "odd_even_sigma": 14, "secondary_sigma": 40, "secondary_depth_ppm": 21000, "centroid_sigma": 0.8}),
    ("Neighbouring star", {"period_days": 4.1057, "depth_ppm": 3000, "duration_hours": 3.4, "snr": 80, "n_transits": 300, "t_eff_k": 5700,
                           "r_star_solar": 1.0, "m_star_solar": 1.0, "odd_even_sigma": 1.0, "secondary_sigma": 6, "secondary_depth_ppm": 400, "centroid_sigma": 9}),
    ("Three numbers only", {"period_days": 9.4862, "depth_ppm": 2200, "duration_hours": 2.8}),
]
# column names accepted in an uploaded table: ours, and those of NASA's KOI and TCE tables
CHECK_COLUMNS = {
    "period_days": ("period_days", "period", "koi_period", "tce_period", "pl_orbper"),
    "depth_ppm": ("depth_ppm", "depth", "koi_depth", "tce_depth"),
    "duration_hours": ("duration_hours", "duration", "koi_duration", "tce_duration"),
    "snr": ("snr", "koi_model_snr", "tce_model_snr"), "n_transits": ("n_transits", "koi_num_transits", "tce_num_transits"),
    "t_eff_k": ("t_eff_k", "teff", "koi_steff", "tce_steff", "st_teff"), "r_star_solar": ("r_star_solar", "koi_srad", "tce_sradius", "st_rad"),
    "m_star_solar": ("m_star_solar", "koi_smass", "tce_smass", "st_mass"), "odd_even_sigma": ("odd_even_sigma",),
    "secondary_sigma": ("secondary_sigma",), "secondary_depth_ppm": ("secondary_depth_ppm",), "centroid_sigma": ("centroid_sigma",),
}
CHECK_NAME_COLUMNS = ("name", "kepoi_name", "kepler_name", "pl_name", "toi", "kepid", "id")
CHECK_BATCH_LIMIT = 5000


def _check_inputs(cleaned: dict) -> dict:
    return {k: v for k, v in cleaned.items() if v is not None}


def signal_check(request):
    ready = signalcheck.available()
    form = SignalCheckForm(request.GET) if request.GET else SignalCheckForm()
    result = None
    if ready and form.is_bound and form.is_valid():
        inputs = _check_inputs(form.cleaned_data)
        result = signalcheck.assess(**inputs)
        result["bars"] = [{"label": CLASS_LABELS[c], "p": p, "width": round(p * 100, 1)} for c, p in result["probabilities"].items()]
        if request.GET.get("record"):
            services.system_logger().info("Signal check from catalogue parameters", "SIGNAL_CHECK", inputs=inputs,
                                          prediction=result["prediction"], probabilities=result["probabilities"],
                                          origin="signal_check")
            messages.success(request, "Assessment written to the audit log.")
    return render(request, "pages/check.html", {"form": form, "result": result, "ready": ready, "presets": CHECK_PRESETS,
                                                "metrics": signalcheck.load_metrics(), "batch_limit": CHECK_BATCH_LIMIT,
                                                "page_title": "Signal check"})


@require_GET
def signal_check_score(request):
    if not signalcheck.available():
        return JsonResponse({"error": "The signal-check model has not been trained."}, status=503)
    form = SignalCheckForm(request.GET)
    if not form.is_valid():
        return JsonResponse({"errors": form.errors}, status=422)
    return JsonResponse(signalcheck.assess(**_check_inputs(form.cleaned_data)))


@require_POST
def signal_check_batch(request):
    """Score every row of an uploaded table and send it back with the verdict added."""
    upload = request.FILES.get("table")
    if not signalcheck.available() or upload is None:
        messages.error(request, "Choose a CSV file to score." if upload is None else "The signal-check model has not been trained.")
        return redirect("signal_check")
    try:
        text = upload.read(8_000_000).decode("utf-8-sig")
    except UnicodeDecodeError:
        messages.error(request, "That file is not text. Upload a CSV.")
        return redirect("signal_check")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]  # NASA archive files open with # comments
    reader = csv.DictReader(lines)
    names = {(n or "").strip().lower(): n for n in (reader.fieldnames or [])}
    column = {key: next((names[a] for a in aliases if a in names), None) for key, aliases in CHECK_COLUMNS.items()}
    absent = [k for k in SignalCheckForm.REQUIRED if column[k] is None]
    if absent:
        messages.error(request, f"The table needs columns for {', '.join(absent)}. Found: {', '.join(reader.fieldnames or []) or 'none'}.")
        return redirect("signal_check")
    label = next((names[n] for n in CHECK_NAME_COLUMNS if n in names), None)
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="signal-check.csv"'
    out = csv.writer(response)
    out.writerow(["row", "name", *CHECK_COLUMNS, "verdict", "p_planet", "p_binary", "p_noise", "problem"])
    for i, rec in enumerate(reader, 1):
        if i > CHECK_BATCH_LIMIT:
            break
        form = SignalCheckForm({k: (rec.get(c) or "").strip() for k, c in column.items() if c})
        name = (rec.get(label) or "").strip() if label else ""
        if not form.is_valid():
            problem = "; ".join(f"{k}: {' '.join(v)}" if k != "__all__" else " ".join(v) for k, v in form.errors.items())
            out.writerow([i, name, *[(rec.get(c) or "") if c else "" for c in column.values()], "", "", "", "", problem])
            continue
        r = signalcheck.assess(**_check_inputs(form.cleaned_data))
        p = r["probabilities"]
        out.writerow([i, name, *[form.cleaned_data.get(k) if form.cleaned_data.get(k) is not None else "" for k in CHECK_COLUMNS],
                      r["disposition"], p["PLANET_CANDIDATE"], p["ECLIPSING_BINARY"], p["STELLAR_OR_INSTRUMENT_NOISE"], ""])
    return response


# ---------------------------------------------------------------------------
# Validation and model
# ---------------------------------------------------------------------------

def validation(request):
    current = ValidationRun.objects.first()
    done = ValidationRun.objects.filter(status="completed").first()
    metrics = model.load_metrics()
    confusion = []
    if metrics:
        top = max(max(row) for row in metrics["confusion"]) or 1
        short = ["Planet", "Binary", "Noise"]
        confusion = [{"label": short[i], "cells": [{"n": n, "alpha": round(n / top, 3)} for n in row]}
                     for i, row in enumerate(metrics["confusion"])]
    return render(request, "pages/validation.html", {
        "current": current, "done": done, "metrics": metrics, "confusion": confusion, "train": services.train_state,
        "tolerances": TOLERANCES, "history": ValidationRun.objects.all()[:8], "page_title": "Validation",
        "chart": {"roc": {CLASS_LABELS[k]: v for k, v in metrics["roc"].items()}, "auc": {CLASS_LABELS[k]: v for k, v in metrics["auc"].items()},
                  "history": metrics["history"]} if metrics else None,
    })


@require_POST
def validation_start(request):
    if services.start_validation() is None:
        messages.error(request, "A verification run is already in progress.")
    else:
        messages.success(request, "Verification suite started. This takes about three minutes.")
    return redirect("validation")


@require_POST
def model_train(request):
    if services.start_training():
        messages.success(request, "Training started. The current checkpoint stays active until the new one passes the AUC threshold.")
    else:
        messages.error(request, "Training is already running.")
    return redirect("validation")


# ---------------------------------------------------------------------------
# Benchmark on real stars, and the atlas
# ---------------------------------------------------------------------------

def benchmark(request):
    from cte import kepler
    metrics = model.load_metrics() or {}
    real = metrics.get("real")
    confusion = []
    if real:
        top = max(max(row) for row in real["confusion"]) or 1
        confusion = [{"label": ["Planet", "Binary", "Noise"][i], "cells": [{"n": n, "alpha": round(n / top, 3)} for n in row]}
                     for i, row in enumerate(real["confusion"])]
    blind = services.benchmark_summary()
    remaining = 0
    if kepler.SAMPLE.exists():
        done = set(Run.objects.filter(source="benchmark").values_list("target__name", flat=True))
        remaining = sum(r["split"] == "test" and f"KIC {r['kepid']}" not in done for r in kepler.load_sample())
    from cte import injection
    inj = injection.load_results()
    return render(request, "pages/benchmark.html", {
        "inj": inj, "dataset": kepler.dataset_summary(), "real": real, "confusion": confusion, "blind": blind, "remaining": remaining,
        "state": services.benchmark_state, "labels": CLASS_LABELS, "page_title": "Benchmark",
        "chart": {"roc": {CLASS_LABELS[k]: v for k, v in real["roc"].items()} if real else None,
                  "auc": {CLASS_LABELS[k]: v for k, v in real["auc"].items()} if real else None,
                  "scatter": blind["scatter"], "injection": {"bins": inj["bins"]} if inj else None},
    })


@require_POST
def benchmark_start(request):
    try:
        per_class = min(max(int(request.POST.get("per_class", 10)), 1), 40)
    except ValueError:
        per_class = 10
    if not model.CHECKPOINT.exists():
        messages.error(request, "Train the classifier before running the benchmark.")
    else:
        n = services.start_benchmark(per_class)
        if n:
            messages.success(request, f"Blind benchmark started on {n} held-out Kepler stars. Each takes 15–30 s.")
        else:
            messages.error(request, "Nothing to run: a benchmark is already in progress, every test star is done, or the dataset has not been built.")
    return redirect("benchmark")


ATLAS_CACHE = LOG_DIR.parent / "atlas_reference.json"


def _atlas_reference() -> list:
    """Period and radius of every confirmed transiting planet, cached from the NASA Exoplanet Archive."""
    if ATLAS_CACHE.exists():
        return json.loads(ATLAS_CACHE.read_text())
    try:
        from cte.archive import _tap
        rows = _tap("select pl_orbper,pl_rade from pscomppars where tran_flag=1 and pl_orbper is not null and pl_rade is not null "
                    "and pl_orbper < 400", timeout=40)
        ref = [[round(r["pl_orbper"], 4), round(r["pl_rade"], 3)] for r in rows]
        ATLAS_CACHE.write_text(json.dumps(ref))
        return ref
    except Exception:
        return []


def atlas(request):
    ref = _atlas_reference()
    runs = list(Run.objects.select_related("target").defer("config").filter(status="completed").exclude(
        disposition__in=("NO_SIGNIFICANT_SIGNAL", "REJECTED_NOISE")))
    groups = {"queue": [], "review": [], "binary": []}
    for r in runs:
        key = "queue" if r.disposition == "CONFIRMED_VETTING_QUEUE" else ("binary" if r.disposition == "FALSE_POSITIVE_ECLIPSING_BINARY" else "review")
        groups[key].append([r.period_days, r.radius_earth])
        r.teq = r.results["parameters"]["equilibrium_temp_k"]
        r.insolation = r.results["parameters"]["insolation_earth"]
    return render(request, "pages/atlas.html", {"runs": sorted(runs, key=lambda r: r.radius_earth or 0), "n_ref": len(ref),
                                                "chart": {"reference": ref, **groups}, "page_title": "Atlas"})


# ---------------------------------------------------------------------------
# Map: where the known planetary systems are, relative to Earth
# ---------------------------------------------------------------------------

VOYAGER1 = {"ra": 258.4, "dec": 12.1, "au_per_year": 3.573,           # heading (Ophiuchus) and speed, JPL
            "one_light_day": (2026, 11, 15), "light_day_au": 173.145}  # the date NASA gives for reaching one light-day


def _voyager1_au(today) -> float:
    from datetime import date
    return VOYAGER1["light_day_au"] - VOYAGER1["au_per_year"] * (date(*VOYAGER1["one_light_day"]) - today).days / 365.25


def _map_targets(catalogue) -> tuple[list[dict], list[dict]]:
    """Stars this engine has analysed, split into those the archive gives a distance for and those it does not."""
    placed, unplaced = [], []
    for t in Target.objects.annotate(n_runs=Count("runs")).order_by("name"):
        latest = t.runs.filter(status="completed").order_by("-finished_at").defer("results", "config").first()
        row = {"id": t.id, "name": t.display, "url": f"/targets/{t.id}/", "runs": t.n_runs,
               "disposition": latest.disposition if latest else "", "p_planet": latest.p_planet if latest else None,
               "run_url": f"/runs/{latest.id}/" if latest else ""}
        if t.mission in ("Synthetic", "Upload") and t.ra is None:
            continue  # a simulated star is nowhere
        host = galaxy.locate([t.name, t.alias], t.ra, t.dec, catalogue) if catalogue else None
        if host:
            placed.append({**row, "host": host["name"], "distance_ly": host["d"], "planets": len(host["planets"])})
        else:
            unplaced.append(row)
    return sorted(placed, key=lambda r: r["distance_ly"]), unplaced


def starmap(request):
    from datetime import date
    catalogue = galaxy.load_systems()
    today = date.today()
    voyager_au = _voyager1_au(today)
    vl, vb = galaxy.equatorial_to_galactic(VOYAGER1["ra"], VOYAGER1["dec"])
    vx, vy, vz = galaxy.galactic_xyz(vl, vb, voyager_au / galaxy.AU_PER_LY)
    placed, unplaced = _map_targets(catalogue)
    focus = asked = request.GET.get("focus", "").strip()  # a star or one of its planets, by name
    if focus and catalogue:
        hit = galaxy.locate([focus], catalogue=catalogue)
        if hit is None:  # a planet's name: find the system that holds it
            wanted = focus.lower().replace(" ", "")
            hit = next((h for h in catalogue["systems"] if any(p["name"].lower().replace(" ", "") == wanted for p in h["planets"])), None)
        focus = hit["name"] if hit else ""
    ctx = {
        "year": today.year, "focus": focus, "focus_name": asked if focus else "",
        "voyager1": {"au": round(voyager_au, 1), "x": vx, "y": vy, "z": vz, "ra": VOYAGER1["ra"], "dec": VOYAGER1["dec"]},
        "radio_ly": today.year - 1906,  # Fessenden's first voice broadcast, December 1906
        "galactic_centre_ly": galaxy.GALACTIC_CENTRE_LY,
        "galactic_plane": [[round(v, 3) for v in galaxy.galactic_to_equatorial(l, 0.0)] for l in range(0, 361, 2)],
        "mine": [{"host": r["host"], "name": r["name"], "url": r["url"]} for r in placed],
        "data_url": "/map/systems.json", "new_url": "/new/",
    }
    nearest = [{"name": h["name"], "d": h["d"], "n": len(h["planets"]), "vmag": h["vmag"], "travel": galaxy.travel(h["d"], today.year)}
               for h in (catalogue["systems"][:12] if catalogue else [])]
    return render(request, "pages/map.html", {
        "catalogue": {k: v for k, v in catalogue.items() if k not in ("systems", "_index")} if catalogue else None,
        "ctx": ctx, "placed": placed, "unplaced": unplaced, "nearest": nearest, "voyager_au": voyager_au, "page_title": "Map",
    })


@require_GET
def starmap_data(request):
    catalogue = galaxy.load_systems(fetch=False)
    if not catalogue:
        return JsonResponse({"systems": []})
    response = JsonResponse({"systems": catalogue["systems"]})
    response["Cache-Control"] = "max-age=600"
    return response


@require_POST
def starmap_refresh(request):
    try:
        c = galaxy.fetch_systems()
        messages.success(request, f"Catalogue refreshed from the NASA Exoplanet Archive: {c['n_planets']:,} planets in {c['n_systems']:,} systems.")
    except Exception as exc:
        messages.error(request, f"The archive could not be reached ({type(exc).__name__}); the map keeps the copy already on disk.")
    return redirect("map")


# ---------------------------------------------------------------------------
# Automation
# ---------------------------------------------------------------------------

def automation(request):
    from cte import kepler
    cfg = autopilot.config()
    if request.method == "POST":
        values = {"enabled": request.POST.get("enabled") == "on"}
        errors = []
        for key, (lo, hi) in autopilot.LIMITS.items():
            try:
                v = int(request.POST.get(key, cfg[key]))
            except ValueError:
                errors.append(f"{key.replace('_', ' ')} must be a whole number")
                continue
            if not lo <= v <= hi:
                errors.append(f"{key.replace('_', ' ')} must be between {lo} and {hi}")
            values[key] = v
        if errors:
            messages.error(request, "; ".join(errors).capitalize() + ".")
        else:
            autopilot.save_config(values)
            messages.success(request, "Autopilot " + ("enabled." if values["enabled"] else "settings saved; it is switched off."))
        return redirect("automation")
    history = model.load_history()
    cycles = list(AutomationCycle.objects.all()[:15])
    return render(request, "pages/automation.html", {
        "cfg": cfg, "limits": autopilot.LIMITS, "cycles": cycles, "running": autopilot.is_running(),
        "next_due": autopilot.next_due(), "dataset": kepler.dataset_summary(), "history": history[::-1][:12],
        "metrics": model.load_metrics(), "page_title": "Automation",
        "chart": {"labels": [h["trained_at"][:16].replace("T", " ") for h in history],
                  "real": [h["auc_real"] for h in history], "sim": [h["auc_simulated"] for h in history],
                  "stars": [h["real_stars"] for h in history]},
    })


@require_POST
def automation_run(request):
    if not model.CHECKPOINT.exists():
        messages.error(request, "Train the classifier before running the autopilot.")
    elif autopilot.start_cycle_async("manual"):
        messages.success(request, "Autopilot cycle started.")
    else:
        messages.error(request, "A cycle is already running.")
    return redirect("automation")


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def _log_query(request):
    level, stage, q, run_id = (request.GET.get(k, "").strip() for k in ("level", "stage", "q", "run"))
    qs = LogEvent.objects.all()
    if level in ("INFO", "WARNING", "ERROR"):
        qs = qs.filter(level=level)
    if stage:
        qs = qs.filter(stage=stage)
    if run_id:
        qs = qs.filter(run_id=run_id)
    if q:
        qs = qs.filter(Q(message__icontains=q) | Q(run_id__icontains=q) | Q(stage__icontains=q))
    return qs, {"level": level, "stage": stage, "q": q, "run": run_id}


def logs(request):
    qs, filters = _log_query(request)
    levels = dict(LogEvent.objects.values_list("level").annotate(n=Count("id")))
    stages = sorted(s for s in LogEvent.objects.values_list("stage", flat=True).distinct() if s)
    return render(request, "pages/logs.html", {"logs": _page(request, qs, 100), "f": filters, "levels": levels, "stages": stages,
                                               "level_tabs": [("", "All levels"), ("INFO", "Info"), ("WARNING", "Warning"), ("ERROR", "Error")],
                                               "page_title": "Audit log"})


@require_GET
def export_logs(request):
    qs, _ = _log_query(request)
    body = "".join(json.dumps(e.as_record()) + "\n" for e in qs.order_by("timestamp", "id").iterator())
    resp = HttpResponse(body, content_type="application/x-ndjson")
    resp["Content-Disposition"] = "attachment; filename=cte_audit_log.jsonl"
    return resp


def _csv(rows, fields, filename):
    resp = HttpResponse(content_type="text/csv")
    resp["Content-Disposition"] = f"attachment; filename={filename}"
    writer = csv.DictWriter(resp, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return resp


@require_GET
def export_runs(request):
    fields = ["id", "target", "alias", "mission", "source", "sector", "status", "created_at", "duration_s", "prediction",
              "p_planet", "disposition", "period_days", "depth_ppm", "radius_earth", "sde", "n_flags", "vetting_status", "vetting_note"]
    rows = [{**{f: getattr(r, f, None) for f in fields}, "target": r.target.name, "alias": r.target.alias,
             "mission": r.target.mission, "created_at": r.created_at.isoformat()}
            for r in Run.objects.select_related("target").defer("results", "config")]
    return _csv(rows, fields, "cte_runs.csv")


@require_GET
def export_candidates(request):
    fields = ["target", "alias", "mission", "sector", "run_id", "period_days", "t0", "transit_depth_ppm", "radius_ratio",
              "radius_earth", "semi_major_axis_au", "transit_duration_hours", "equilibrium_temp_k", "snr", "sde", "p_planet",
              "disposition", "vetting_status", "t_eff_k", "r_star_solar", "m_star_solar"]
    return _csv(services.candidate_rows(), fields, "cte_candidates_citizen_science.csv")


# ---------------------------------------------------------------------------
# Settings and method
# ---------------------------------------------------------------------------

def settings_page(request):
    if request.method == "POST":
        if "reset" in request.POST:
            services.reset_pipeline_settings()
            messages.success(request, "Restored the PRD defaults.")
            return redirect("settings")
        form = SettingsForm(request.POST)
        if form.is_valid():
            services.save_pipeline_settings(form.pipeline_values())
            messages.success(request, "Pipeline settings saved.")
            return redirect("settings")
    else:
        form = SettingsForm(initial=services.pipeline_settings())
    from django.conf import settings as dj
    engine = dj.DATABASES["default"]["ENGINE"].rsplit(".", 1)[-1]
    return render(request, "pages/settings.html", {"form": form, "defaults": DEFAULT_PIPELINE, "tolerances": TOLERANCES,
                                                   "database": {"sqlite3": "SQLite (local file)", "postgresql": "PostgreSQL"}.get(engine, engine),
                                                   "data_dir": LOG_DIR.parent, "page_title": "Settings"})


def method(request):
    flags = [
        ("LOW_SNR", "Transit signal-to-noise below 7.1; the run is dispositioned as no significant signal."),
        ("LOW_SDE", "The BLS peak does not stand clear of the periodogram background."),
        ("FEW_TRANSITS", "Fewer than three transits observed; the period is weakly constrained."),
        ("ODD_EVEN_MISMATCH", "Alternate transits differ in depth: an eclipsing binary at twice the detected period."),
        ("SECONDARY_ECLIPSE", "A significant dip at phase 0.5: the companion is self-luminous."),
        ("V_SHAPED", "No flat bottom; consistent with a grazing stellar eclipse."),
        ("RADIUS_TOO_LARGE", "Derived radius above 2.2 Jupiter radii; the companion is likely a star."),
        ("DURATION_MISMATCH", "Fitted duration far exceeds the prediction for the host star: wrong stellar parameters or a blend."),
        ("CENTROID_OFFSET", "The photocentre moves during transit by more than a source on the target could explain: the signal comes from a background star at least half a pixel away."),
    ]
    meanings = {
        "CONFIRMED_VETTING_QUEUE": "Planet probability above the queue threshold and no high-severity flag. Ready for human vetting.",
        "MANUAL_REVIEW": "Classified as a planet but below the confidence threshold.",
        "FALSE_POSITIVE_SUSPECT": "Classified as a planet, but a high-severity diagnostic disagrees.",
        "FALSE_POSITIVE_ECLIPSING_BINARY": "Classified as an eclipsing binary. Excluded from export.",
        "REJECTED_NOISE": "Classified as stellar or instrumental noise.",
        "NO_SIGNIFICANT_SIGNAL": "Transit signal-to-noise below the detection threshold.",
    }
    return render(request, "pages/method.html", {"flags": flags, "page_title": "Method",
                                                 "dispositions": [(DISPOSITIONS[k], v, k) for k, v in meanings.items()]})


# ---------------------------------------------------------------------------
# Research: possible planets from the survey, and the report on a run
# ---------------------------------------------------------------------------

def research(request):
    from cte import survey
    show = request.GET.get("show", "clean")
    rows = survey.possible_planets()
    cases = {(c.kepid, round(c.period_days, 4)): c for c in ResearchCase.objects.select_related("run")}
    for r in rows:
        r["case"] = cases.get((r["kepid"], round(r["period_days"], 4)))
    clean = [r for r in rows if not r["notes"]]
    return render(request, "pages/research.html", {
        "rows": clean if show == "clean" else rows, "show": show, "n_all": len(rows), "n_clean": len(clean),
        "stars": survey.stars_surveyed(), "cases": ResearchCase.objects.select_related("run")[:12], "page_title": "Research",
    })


@require_POST
def research_open(request):
    try:
        case = research_jobs.open_case(int(request.POST["kepid"]), float(request.POST["period"]))
    except (LookupError, KeyError, ValueError) as exc:
        messages.error(request, str(exc))
        return redirect("research")
    return redirect("research_case", pk=case.pk)


def _checks(res: dict) -> list[dict]:
    """Each automated test of a run as a row: what was measured and which way it points."""
    d, out = res.get("diagnostics") or {}, []

    def add(name, measured, verdict, why):
        out.append({"name": name, "measured": measured, "verdict": verdict, "why": why})

    oe = d.get("odd_even")
    if oe:
        add("Odd against even transits", f"{oe['depth_odd_ppm']:.0f} and {oe['depth_even_ppm']:.0f} ppm, {abs(oe['sigma']):.1f}σ apart",
            "against" if abs(oe["sigma"]) >= 3 else "for", "Two unequal stars eclipsing each other give alternating depths; a planet gives the same depth every time.")
    else:
        add("Odd against even transits", "not measurable", "none", "Too few transits to split by parity.")
    sec = d.get("secondary")
    if sec:
        add("Secondary eclipse, half an orbit later", f"{sec['depth_ppm']:.0f} ppm, {sec['sigma']:.1f}σ",
            "against" if sec["sigma"] >= 3 and sec.get("ratio", 0) >= 0.05 else "for", "A dip here means the companion shines by its own light: a star, not a planet.")
    else:
        add("Secondary eclipse, half an orbit later", "no coverage", "none", "The light curve has no data at phase 0.5.")
    cen = d.get("centroid")
    if cen:
        add("Movement of the star's image in transit", f"{cen['offset_mpix']:.2f} millipixels, {cen['sigma']:.1f}σ",
            "against" if abs(cen["sigma"]) >= 3 else "for", "If the image shifts during the dip, the dimming comes from a neighbouring star.")
    else:
        add("Movement of the star's image in transit", "no centroid data", "none", "This light curve carries no image positions.")
    pix = d.get("pixels")
    if pix and pix.get("pix_quarters"):
        sigma = pix.get("pix_sigma")
        add("Where on the detector the dip comes from", f"{pix['pix_offset']:.2f} pixels from the target"
            + (f", {sigma:.1f}σ" if sigma is not None else "") + f", {pix['pix_quarters']:.0f} quarters",
            "against" if sigma is not None and sigma >= 3 and pix["pix_offset"] >= 0.5 else ("weak" if (res["parameters"]["transit_depth_ppm"] or 0) < 150 else "for"),
            "The image in transit is subtracted from the image just outside it; the difference shows the source of the dip. Below about 150 ppm the difference image is noise-limited and the test has little power.")
    else:
        add("Where on the detector the dip comes from", "not measured", "none", "No target pixel files were read for this run.")
    ratio = d.get("duration_ratio")
    if ratio is not None:
        add("Duration against a transit across this star", f"{ratio:.2f} of the expected {res['parameters']['transit_duration_hours']:.2f} h",
            "against" if ratio > 1.6 or ratio < 0.2 else "for", "A dip much longer than the star allows points to a different, larger star; a much shorter one to a grazing eclipse.")
    v = (res.get("fit") or {}).get("v_shape")
    if v is not None:
        add("Shape of the dip", f"{v:.2f} on a scale from 0 (flat-bottomed) to 1 (V-shaped)", "against" if v > 0.9 else "for",
            "Planets cross the whole disc and give flat-bottomed dips; grazing binaries give V shapes.")
    ms = d.get("model_shift")
    if ms:
        add("Any other dip at this period", f"strongest elsewhere {ms['depth_ppm']:.0f} ppm at phase {ms['phase']:.2f}, {ms['sigma']:.1f}σ",
            "against" if ms.get("significant") else "for", "A second significant dip at another phase is the other eclipse of a binary on an eccentric orbit.")
    radius = res["parameters"].get("radius_earth")
    if radius:
        add("Size of the companion", f"{radius:.2f} Earth radii", "against" if radius > 25 else "for", "Anything much larger than Jupiter (11.2 Earth radii) is a star.")
    return out


SCENARIOS = {  # TRICERATOPS' scenario codes, in words
    "TP": "Planet on the target star", "EB": "Eclipsing binary: the target itself", "EBx2P": "Eclipsing binary on the target, at twice the period",
    "PTP": "Planet on the target, which has an unseen bound companion", "PEB": "Eclipsing binary on the target, which has an unseen bound companion",
    "PEBx2P": "The same, at twice the period", "STP": "Planet on an unseen bound companion star", "SEB": "Eclipsing binary: an unseen bound companion",
    "SEBx2P": "The same, at twice the period", "DTP": "Planet on the target, with an unseen star in the foreground or background",
    "DEB": "Eclipsing binary on the target, with an unseen star in the line of sight", "DEBx2P": "The same, at twice the period",
    "BTP": "Planet on an unseen background star", "BEB": "Eclipsing binary: an unseen background star", "BEBx2P": "The same, at twice the period",
    "NTP": "Planet on a known nearby star", "NEB": "Eclipsing binary: a known nearby star", "NEBx2P": "The same, at twice the period",
}


def _prob(p: float) -> str:
    return "below 0.01%" if p < 1e-4 else (f"{p * 100:.2f}%" if p < 0.1 else f"{p * 100:.0f}%")


def run_packet(request, pk):
    r = get_object_or_404(Run.objects.select_related("target"), pk=pk)
    if r.status != "completed" or not r.results or not (ARCHIVE_DIR / r.id / "lightcurve.parquet").exists():
        raise Http404("Only a finished run with its archive has a data packet")
    case = r.research_cases.first()
    name = "".join(ch if ch.isalnum() else "-" for ch in (case.host if case else r.target.display)).strip("-")
    response = HttpResponse(research_jobs.packet(r), content_type="application/zip")
    response["Content-Disposition"] = f'attachment; filename="{name}-{r.period_days:.4f}d-packet.zip"'
    return response


def _report_context(r: Run) -> dict:
    from cte import model as cnn
    res = r.results
    p, d, st = res["parameters"], res["diagnostics"], res["stellar"]
    case = r.research_cases.first()
    removed = (res.get("metadata") or {}).get("removed_signals") or []
    archive_match = res.get("archive_match") or {}
    objects, match = archive_match.get("objects") or [], archive_match.get("match")
    probs = [{"label": CLASS_LABELS[k], "p": v, "pct": round(v * 100, 2), "top": k == r.prediction}
             for k, v in res["classification"]["probabilities"].items()]
    checks = _checks(res)
    against = [c for c in checks if c["verdict"] == "against"]
    star = case.host if case else (r.target.name if r.source == "research" else r.target.display)
    summary = [f"The engine finds a dip of {p['transit_depth_ppm']:.0f} parts per million in the light of {star} that repeats every "
               f"{p['period_days']:.4f} days and lasts {res['fit']['t14_hours']:.1f} hours. It was seen {d['n_transits']} times, "
               f"at a signal-to-noise of {d['snr']:.1f}."]
    summary.append(f"The classifier puts the probability that this is a planet at {r.p_planet * 100:.1f}%. If it is one, it is "
                   f"{p['radius_earth']:.2f} times the size of Earth and orbits {p['semi_major_axis_au']:.3f} AU from its star, where it would be at about "
                   f"{p['equilibrium_temp_k']:.0f} K.")
    if removed:
        summary.append(f"This analysis was made after cutting out the transits of {len(removed)} other signal{'s' if len(removed) != 1 else ''} "
                       f"on the same star (at {', '.join(format(s['period_days'], '.4f') + ' d' for s in removed)}), so that this one could be measured on its own.")
    if match:
        summary.append(f"The period matches {match['name']}, which NASA lists as {match['status'].lower()}.")
    elif objects:
        summary.append(f"The period matches none of the {len(objects)} object{'s' if len(objects) != 1 else ''} NASA lists for this star.")
    elif archive_match:
        summary.append("NASA lists nothing for this star.")
    summary.append(f"Of {len(checks)} automated tests, {len(against)} count{'s' if len(against) == 1 else ''} against a planet"
                   + (": " + "; ".join(c["name"].lower() for c in against) + "." if against else "."))
    logs = r.logs.order_by("seq")
    last = logs.last()
    timing, validation = research_jobs.timing(r), research_jobs.validation(r)
    if validation:
        for row in validation["scenarios"]:
            row["label"] = SCENARIOS.get(row["scenario"], row["scenario"])
            row["planet"] = row["scenario"] in ("TP", "PTP", "DTP")
        summary.append(f"TRICERATOPS puts the false-positive probability at {_prob(validation['fpp'])} "
                       f"(from nearby stars alone: {_prob(validation['nfpp'])}): {validation['class']}.")
    if timing and timing.get("measurable"):
        summary.append("Transit timing: " + ("a timing wave is detected." if timing["detected"] else "no variations detected."))
    return {"system": system_view(r, case), "timing": timing, "validation": validation, "run": r, "res": res, "star": star, "probs": probs, "checks": checks, "summary": summary, "case": case, "removed": removed,
            "objects": objects, "match": match, "has_archive": bool(archive_match), "series": services.run_series(r),
            "n_logs": logs.count(), "last_hash": last.hash if last else "", "metrics": cnn.load_metrics() or {},
            "flags": res.get("flags") or [], "further": res.get("additional_signals") or [],
            "generated": __import__("django.utils.timezone", fromlist=["now"]).now(), "page_title": f"Report · {star}"}


def run_report(request, pk):
    r = get_object_or_404(Run.objects.select_related("target"), pk=pk)
    if r.status != "completed" or not r.results:
        raise Http404("Only a finished run has a report")
    return render(request, "pages/report.html", _report_context(r) | {"printing": request.GET.get("print") == "1"})


def run_report_pdf(request, pk):
    r = get_object_or_404(Run.objects.select_related("target"), pk=pk)
    if r.status != "completed" or not r.results:
        raise Http404("Only a finished run has a report")
    from django.urls import reverse
    try:
        pdf = research_jobs.render_pdf(request.build_absolute_uri(reverse("run_report", args=[r.id])) + "?print=1", r.id)
    except Exception as exc:
        messages.error(request, f"The PDF could not be made: {exc}")
        return redirect("run_report", pk=r.id)
    case = r.research_cases.first()
    name = "".join(ch if ch.isalnum() else "-" for ch in (case.host if case else r.target.display)).strip("-")
    response = HttpResponse(pdf, content_type="application/pdf")
    response["Content-Disposition"] = f'{"inline" if request.GET.get("view") else "attachment"}; filename="{name}-{r.period_days:.4f}d-report.pdf"'
    return response


def research_case(request, pk):
    case = get_object_or_404(ResearchCase.objects.select_related("run__target", "base_run"), pk=pk)
    ctx = {"case": case, "signal": case.signal, "page_title": str(case)}
    if case.status == "done" and case.run and case.run.results:
        ctx |= _report_context(case.run) | {"page_title": str(case)}
    return render(request, "pages/research_case.html", ctx)


@require_GET
def research_status(request, pk):
    case = get_object_or_404(ResearchCase.objects.select_related("run", "base_run"), pk=pk)
    live = next((r for r in (case.run, case.base_run) if r is not None and r.is_live), None)
    return JsonResponse({"status": case.status, "step": case.step, "error": case.error,
                         "stage": live.stage_label if live else "", "percent": live.percent if live else None})


# ---------------------------------------------------------------------------
# System view: the star, its planets' orbits to scale, and where this run's signal sits among them
# ---------------------------------------------------------------------------

BKJD_ZERO = 2454833.0
SPECTRAL = ((30000, "O"), (10000, "B"), (7500, "A"), (6000, "F"), (5200, "G"), (3700, "K"), (0, "M"))


def _planet_kind(radius_earth) -> str:
    """NASA's size classes, from the radius alone."""
    if not radius_earth:
        return ""
    return "Terrestrial" if radius_earth < 1.25 else "Super-Earth" if radius_earth < 2.0 else "Neptune-like" if radius_earth < 6.0 else "Gas giant"


def system_view(r: Run, case=None) -> dict | None:
    """Everything the system view draws: the host star, every planet NASA lists on it, and this run's own signal,
    with orbit sizes in AU. Planets come from the archive's confirmed list when the host is on the map; candidates
    and this run's signal are added from the run itself."""
    res = r.results or {}
    if not res.get("parameters") or not res.get("stellar"):
        return None
    st, p, meta = res["stellar"], res["parameters"], res.get("metadata") or {}
    mass, radius, teff = st.get("m_star_solar") or 1.0, st.get("r_star_solar") or 1.0, st.get("t_eff_k") or 5772.0
    archive_match = res.get("archive_match") or {}
    names = [r.target.name, r.target.alias] + [o["name"] for o in archive_match.get("objects") or []]
    names += [n.rsplit(" ", 1)[0] for n in names if n and len(n.rsplit(" ", 1)) == 2 and len(n.rsplit(" ", 1)[1]) == 1]  # "Kepler-783 c" -> its host
    host = galaxy.locate(([case.host] if case else []) + names, r.target.ra, r.target.dec)
    lum = radius ** 2 * (teff / 5772.0) ** 4

    def orbit_au(period_days):  # Kepler's third law
        return round((mass * (period_days / 365.25) ** 2) ** (1.0 / 3.0), 5)

    planets = []
    for q in (host or {}).get("planets", []):
        if q.get("P"):
            planets.append({"name": q["name"], "status": "confirmed", "period_days": q["P"], "a_au": q.get("a") or orbit_au(q["P"]),
                            "radius_earth": q.get("r"), "mass_earth": q.get("m"), "temp_k": q.get("T"), "year": q.get("year"),
                            "t0_jd": q.get("t0"), "period_err": q.get("Pe"), "tilt_deg": q.get("i")})
    for o in archive_match.get("objects") or []:
        status = str(o.get("status", "")).lower()
        if status in ("confirmed", "candidate") and not any(abs(q["period_days"] / o["period_days"] - 1) < 0.01 for q in planets):
            planets.append({"name": o["name"], "status": "confirmed" if status == "confirmed" else "NASA candidate", "period_days": o["period_days"],
                            "a_au": orbit_au(o["period_days"]), "radius_earth": o.get("radius_earth")})
    mine = next((q for q in planets if abs(q["period_days"] / p["period_days"] - 1) < 0.01), None)
    if mine is None and r.prediction == "PLANET_CANDIDATE":
        mine = {"name": f"{(case.host if case else (host or {}).get('name') or r.target.display)} candidate", "status": "candidate found by this engine",
                "period_days": p["period_days"], "a_au": p["semi_major_axis_au"], "radius_earth": p["radius_earth"], "temp_k": p.get("equilibrium_temp_k"),
                "t0_jd": res["bls"]["t0"] + BKJD_ZERO if meta.get("time_system") == "bkjd" else None, "p_planet": r.p_planet}
        planets.append(mine)
    if mine is not None:
        mine["focus"] = True
        mine.setdefault("measured_here", {"period_days": p["period_days"], "radius_earth": p["radius_earth"], "depth_ppm": p["transit_depth_ppm"]})
    if not planets:
        return None
    for q in planets:
        q["kind"] = _planet_kind(q.get("radius_earth"))
        if q.get("temp_k") is None:  # equilibrium temperature for the same albedo the run assumes
            q["temp_k"] = round(teff * (radius * 0.00465047 / (2.0 * q["a_au"])) ** 0.5 * (1.0 - res["settings"]["bond_albedo"]) ** 0.25)
    planets.sort(key=lambda q: q["a_au"])
    name = case.host if case else (host["name"] if host else r.target.display)
    confirmed = sum(q["status"] == "confirmed" for q in planets)
    others = len(planets) - confirmed
    letter = next(s for limit, s in SPECTRAL if teff >= limit)
    distance = host["d"] if host else None
    line = (f"{confirmed} confirmed planet{'s' if confirmed != 1 else ''}" if confirmed else "")
    line += (" and " if confirmed and others else "") + (f"{others} candidate{'s' if others != 1 else ''}" if others else "")
    line += f" orbiting a {letter}-type star" + (f", {distance:,.0f} light-years from Earth." if distance else ".")
    return {"host": name, "headline": line, "distance_ly": distance,
            "star": {"teff": teff, "radius_solar": radius, "mass_solar": mass, "luminosity_solar": round(lum, 3), "type": letter},
            # conservative habitable zone (Kopparapu et al. 2013: runaway greenhouse to maximum greenhouse, as stellar flux)
            "habitable_zone_au": [round((lum / 1.107) ** 0.5, 4), round((lum / 0.356) ** 0.5, 4)],
            "planets": planets}
