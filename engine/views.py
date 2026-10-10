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
from .forms import CalculatorForm, NewRunForm, SettingsForm, SignalCheckForm, VettingForm
from .models import CLASS_LABELS, DISPOSITIONS, AutomationCycle, LogEvent, Run, Target, ValidationRun


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
    focus = request.GET.get("focus", "").strip()
    if focus and catalogue:
        hit = galaxy.locate([focus], catalogue=catalogue)
        focus = hit["name"] if hit else ""
    ctx = {
        "year": today.year, "focus": focus,
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
