"""Persistence and scheduling around the analytical core in `cte/`."""
from __future__ import annotations

import os
import threading
import time
import traceback
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import pyarrow.parquet as pq
from django.conf import settings
from django.db import close_old_connections
from django.db.models import Q
from django.utils import timezone

from cte import archive, core, ingest, model
from cte.config import ARCHIVE_DIR, CLASS_NAMES, DEFAULT_PIPELINE, LOG_DIR, MAX_WORKERS, TOLERANCES
from cte.telemetry import LogIntegrityError, RunLogger, clean

from .models import CLASS_LABELS, NEEDS_HUMAN, LogEvent, Run, Setting, Target, ValidationRun

# ---------------------------------------------------------------------------
# Settings and logging
# ---------------------------------------------------------------------------


def pipeline_settings(overrides: dict | None = None) -> dict:
    cfg = dict(DEFAULT_PIPELINE)
    saved = Setting.objects.filter(key="pipeline").first()
    if saved:
        cfg.update({k: v for k, v in saved.value.items() if k in cfg})
    cfg.update({k: v for k, v in (overrides or {}).items() if k in cfg and v is not None})
    return cfg


def save_pipeline_settings(values: dict) -> dict:
    merged = pipeline_settings(values)
    Setting.objects.update_or_create(key="pipeline", defaults={"value": merged})
    system_logger().info("Pipeline settings updated", "SYSTEM", changes=values)
    return merged


def reset_pipeline_settings() -> None:
    Setting.objects.filter(key="pipeline").delete()
    system_logger().info("Pipeline settings reset to PRD defaults", "SYSTEM")


def db_log_sink(run_id: str | None, record: dict, stage: str | None) -> None:
    LogEvent.objects.create(run_id=run_id, seq=record["seq"], timestamp=datetime.fromisoformat(record["timestamp"]),
                            level=record["level"], module=record["module"], message=record["message"],
                            stage=stage or "", telemetry=record.get("telemetry"), hash=record["hash"])


_system_logger: RunLogger | None = None
_system_lock = threading.Lock()


def system_logger() -> RunLogger:
    global _system_logger
    with _system_lock:
        if _system_logger is None:
            _system_logger = RunLogger(None, module="cte_system", sink=db_log_sink)
        return _system_logger


_started = False


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True


def startup() -> None:
    """Once per process: fail jobs whose owning process has gone (a job owned by a live CLI command is left alone)."""
    global _started
    if _started:
        return
    _started = True
    stale = 0
    for run in Run.objects.filter(status__in=("running", "queued")).only("id", "config"):
        if not _alive(run.config.get("_pid")):
            Run.objects.filter(id=run.id).update(status="failed", error="Interrupted by server restart", finished_at=timezone.now())
            stale += 1
    for val in ValidationRun.objects.filter(status="running"):
        if not _alive((val.results or {}).get("pid")):
            ValidationRun.objects.filter(id=val.id).update(status="failed", progress="Interrupted by server restart")
    if stale:
        system_logger().warning("Recovered interrupted runs after restart", "SYSTEM", interrupted_runs=stale)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def get_or_create_target(name: str, mission: str, stellar: dict) -> Target:
    target, _ = Target.objects.get_or_create(name=name, defaults={"mission": mission})
    changed = False
    for key in ("t_eff_k", "r_star_solar", "m_star_solar", "v_mag", "ra", "dec"):
        if stellar.get(key) is not None and getattr(target, key) != stellar[key]:
            setattr(target, key, stellar[key])
            changed = True
    if changed:
        target.save()
    return target


def create_run(config: dict) -> str:
    """Register a queued run. `config` = {source, target_name?, mast?, synthetic?, upload_id?, stellar_params?, pipeline?}."""
    source = config.get("source")
    if source not in ("synthetic", "mast", "upload", "benchmark"):
        raise ValueError("source must be one of synthetic, mast, upload, benchmark")
    stellar = {k: v for k, v in (config.get("stellar_params") or {}).items() if v is not None}
    if source == "mast":
        mast = config.get("mast") or {}
        if not str(mast.get("target", "")).strip():
            raise ValueError("A MAST target is required")
        name, mission = str(mast["target"]).strip(), mast.get("mission", "TESS")
    elif source == "benchmark":
        name, mission = f"KIC {int(config['kepid'])}", "Kepler"
    else:
        name = (config.get("target_name") or "").strip() or f"{'SYN' if source == 'synthetic' else 'UPL'}-{uuid.uuid4().hex[:6].upper()}"
        mission = "Synthetic" if source == "synthetic" else "Upload"
        config["target_name"] = name
    run_id = uuid.uuid4().hex[:12]
    config["_pid"] = os.getpid()  # lets a restarted server tell orphaned runs from ones a live process still owns
    target = get_or_create_target(name, mission, stellar)
    Run.objects.create(id=run_id, target=target, source=source, config=config, sector=(config.get("mast") or {}).get("sequence"))
    return run_id


def execute_run(run_id: str) -> None:
    log = RunLogger(run_id, sink=db_log_sink)
    started = time.time()

    def update(**fields):
        Run.objects.filter(id=run_id).update(**fields)

    try:
        run = Run.objects.select_related("target").get(id=run_id)
        config, target_name = dict(run.config), run.target.name
        update(status="running", started_at=timezone.now(), stage="INGEST", progress=0.0)
        cfg = pipeline_settings(config.get("pipeline"))
        log.info("Initializing Chronos Transit Pipeline execution", "INGEST", target=target_name,
                 source=config["source"], settings=cfg)
        stellar = {k: v for k, v in (config.get("stellar_params") or {}).items() if v is not None}
        if config["source"] == "mast":
            mast = config["mast"]
            data = ingest.fetch_mast(mast["target"], mast.get("mission", "TESS"), mast.get("sequence"), mast.get("exptime"))
            if not all(data.stellar.get(k) for k in ("t_eff_k", "r_star_solar", "m_star_solar")):
                try:  # K2 / faint targets often carry no stellar parameters: ask the Planetary Systems table
                    found = archive.host_star(mast["target"])
                    for key, value in found.items():
                        data.stellar[key] = data.stellar.get(key) or value
                    if found:
                        data.warnings.append("stellar parameters completed from the NASA Exoplanet Archive host entry")
                except Exception:
                    pass
            data.stellar.update(stellar)  # operator overrides win over the catalogue
            message = "Ingested photometric timeseries from MAST"
        elif config["source"] == "benchmark":
            data = ingest.load_benchmark_star(config["kepid"])
            message = "Ingested held-out Kepler KOI photometry (blind benchmark)"
        elif config["source"] == "upload":
            data = ingest.load_upload(config["upload_id"], stellar, target_name)
            message = "Ingested uploaded photometric timeseries"
        else:
            data = ingest.make_synthetic(config.get("synthetic") or {}, {**core.SOLAR, **stellar}, target_name)
            message = "Synthesized photometric timeseries"
        for w in data.warnings:
            log.warning(w, "INGEST")
        log.info(message, "INGEST", cadences=len(data.time),
                 baseline_variance=float(np.var(data.flux / np.median(data.flux))), metadata=data.meta,
                 stellar_params=data.stellar, injected_truth=data.truth)

        # the query alias (e.g. "WASP-12") resolves to a canonical catalogue name ("TIC 86396382")
        target = get_or_create_target(data.meta["target_name"], data.meta["mission"], data.stellar)
        old = run.target
        if data.meta.get("alias") and not target.alias:
            target.alias = data.meta["alias"]
            target.save(update_fields=["alias"])
        if old.id != target.id:
            if not target.alias:
                target.alias = old.name
                target.save(update_fields=["alias"])
        update(target=target, sector=data.meta.get("sector"), camera=data.meta.get("camera"), ccd=data.meta.get("ccd"))
        if old.id != target.id and not old.runs.exists():
            old.delete()

        results, arrays = core.process(data, cfg, log, lambda name, frac: update(stage=name, progress=round(frac, 3)),
                                       lambda text: live_notes.__setitem__(run_id, text))
        results["metadata"] = clean(data.meta)
        if config["source"] in ("mast", "benchmark"):  # what does the NASA Exoplanet Archive already know about this star?
            try:
                match = archive.crossmatch(data.meta["target_name"], results["bls"]["period"],
                                           results["parameters"]["radius_earth"], results["parameters"]["transit_depth_ppm"])
                results["archive_match"] = clean(match)
                m = match["match"]
                log.info("Archive cross-match complete", "ARCHIVE_CROSSMATCH", catalogued_objects=len(match["objects"]),
                         matched=m["name"] if m else None, status=m["status"] if m else None,
                         period_rel_error=m["period_rel_error"] if m else None, alias=m["alias"] if m else None)
            except Exception as exc:
                log.warning(f"Archive cross-match unavailable: {type(exc).__name__}", "ARCHIVE_CROSSMATCH")

        update(stage="ARCHIVE", progress=core.STAGE_PROGRESS["ARCHIVE"])
        sizes = core.archive(run_id, arrays)
        results["archive"] = sizes
        duration = time.time() - started
        log.info("Archived run products", "ARCHIVE", parquet_bytes=sizes, jsonl=log.path.name, wall_seconds=round(duration, 3))
        c, p = results["classification"], results["parameters"]
        update(status="completed", stage="COMPLETE", progress=1.0, finished_at=timezone.now(), duration_s=duration,
               results=results, prediction=c["prediction"], p_planet=c["probabilities"]["PLANET_CANDIDATE"],
               disposition=c["disposition"], period_days=p["period_days"], depth_ppm=p["transit_depth_ppm"],
               radius_earth=p["radius_earth"], sde=results["bls"]["sde"], n_flags=len(results["flags"]))
    except LogIntegrityError as exc:  # PRD §6: terminate the ingest pipeline run
        update(status="failed", finished_at=timezone.now(), duration_s=time.time() - started,
               error=f"Structured log integrity violation - run terminated: {exc}")
    except Exception as exc:
        try:
            log.error(f"Pipeline run failed: {exc}", "FAILED", error_type=type(exc).__name__,
                      traceback=traceback.format_exc(limit=6))
        finally:
            update(status="failed", finished_at=timezone.now(), duration_s=time.time() - started,
                   error=f"{type(exc).__name__}: {exc}")


_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="cte-worker")


def _in_worker(fn, *args):
    try:
        fn(*args)
    finally:
        close_old_connections()


def _spawn(fn, *args, pool: bool = False) -> None:
    if settings.CTE_RUN_INLINE:
        fn(*args)
    elif pool:
        _executor.submit(_in_worker, fn, *args)
    else:
        threading.Thread(target=_in_worker, args=(fn, *args), daemon=True).start()


def submit_run(config: dict) -> str:
    run_id = create_run(config)
    _spawn(execute_run, run_id, pool=True)
    return run_id


def delete_run(run: Run) -> None:
    import shutil
    target, run_id = run.target, run.id
    run.delete()
    if not target.runs.exists():
        target.delete()
    shutil.rmtree(ARCHIVE_DIR / run_id, ignore_errors=True)
    (LOG_DIR / f"{run_id}.jsonl").unlink(missing_ok=True)
    system_logger().warning("Run deleted by operator", "SYSTEM", deleted_run=run_id)


def vet_run(run: Run, status: str, note: str) -> None:
    run.vetting_status, run.vetting_note = status, note
    run.vetted_at = timezone.now() if status != "pending" else None
    run.save(update_fields=["vetting_status", "vetting_note", "vetted_at"])
    RunLogger(run.id, module="vetting", sink=db_log_sink).info(
        f"Operator vetting decision: {status}", "HUMAN_VETTING", decision=status, note=note or None)


def queue_filter() -> Q:
    return Q(status="completed", disposition__in=NEEDS_HUMAN)


def candidate_rows() -> list[dict]:
    """Citizen-science export: planet candidates with every flagged false positive removed (PRD §1)."""
    runs = Run.objects.select_related("target").filter(status="completed").exclude(vetting_status="false_positive").filter(
        Q(vetting_status="confirmed") | Q(disposition="CONFIRMED_VETTING_QUEUE")).order_by("-p_planet")
    out = []
    for r in runs:
        res = r.results
        out.append({"target": r.target.name, "alias": r.target.alias, "mission": r.target.mission, "sector": r.sector,
                    "run_id": r.id, "t0": res["bls"]["t0"], "snr": round(res["diagnostics"]["snr"], 2),
                    "sde": round(res["bls"]["sde"], 2), "p_planet": r.p_planet, "disposition": r.disposition,
                    "vetting_status": r.vetting_status, **res["parameters"],
                    **{k: res["stellar"][k] for k in ("t_eff_k", "r_star_solar", "m_star_solar")}})
    return out


# ---------------------------------------------------------------------------
# Plot series
# ---------------------------------------------------------------------------

def _decimate(n: int, limit: int) -> np.ndarray:
    return np.arange(n) if n <= limit else np.unique(np.linspace(0, n - 1, limit).astype(int))


def _round(a, digits=6) -> list:
    return clean(np.round(np.asarray(a, dtype=float), digits))


def run_series(run: Run, max_points: int = 6000) -> dict | None:
    base = ARCHIVE_DIR / run.id
    if run.status != "completed" or not (base / "lightcurve.parquet").exists():
        return None
    res = run.results
    lc = pq.read_table(base / "lightcurve.parquet").to_pydict()
    idx = _decimate(len(lc["time"]), max_points)
    rejected = np.flatnonzero(~np.asarray(lc["kept"]))
    pg = pq.read_table(base / "periodogram.parquet").to_pydict()
    period, sr = np.asarray(pg["period"]), np.asarray(pg["sr"])
    edges = np.linspace(0, len(period), min(2500, len(period)) + 1).astype(int)
    pk = np.array([lo + int(sr[lo:hi].argmax()) for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo])  # keep peaks
    fo = pq.read_table(base / "folded.parquet").to_pydict()
    phase, flux = np.asarray(fo["phase"]), np.asarray(fo["flux"])
    fi = _decimate(len(phase), max_points)
    q_local, p_days = res["fit"]["q_local"], res["bls"]["period"]
    near = np.flatnonzero(np.abs(phase) < 2.5 * q_local)
    ni = near[_decimate(len(near), 3000)]
    n_g, n_l = len(res["tensors"]["global"]), len(res["tensors"]["local"])
    transits = np.arange(res["bls"]["t0"], res["input"]["time_end"], p_days)[:400]
    return {
        "time": _round(np.asarray(lc["time"])[idx], 5), "flux_norm": _round(np.asarray(lc["flux_norm"])[idx]),
        "trend": _round(np.asarray(lc["trend"], dtype=float)[idx]),
        "flux_flat": _round(np.asarray(lc["flux_flat"], dtype=float)[idx]),
        "rejected_time": _round(np.asarray(lc["time"])[rejected], 5),
        "rejected_flux": _round(np.asarray(lc["flux_norm"])[rejected]),
        "transits": _round(transits, 5),
        "period": _round(period[pk], 6), "sr": _round(sr[pk] * 1e3, 5), "best_period": p_days,
        "fold_phase": _round(phase[fi]), "fold_flux": _round(flux[fi]),
        "local_hours": _round(phase[ni] * p_days * 24.0, 4), "local_flux": _round(flux[ni]),
        "global_x": _round((np.arange(n_g) + 0.5) / n_g - 0.5, 5), "global_view": res["tensors"]["global"],
        "local_x": _round(((np.arange(n_l) + 0.5) / n_l - 0.5) * 5 * q_local * p_days * 24.0, 4),
        "local_view": res["tensors"]["local"],
        "visuals": res.get("visuals"), "explain": res.get("explain"),
    }


# ---------------------------------------------------------------------------
# Overview aggregates
# ---------------------------------------------------------------------------

def overview() -> dict:
    now = timezone.now()
    week = now - timedelta(days=7)
    runs = list(Run.objects.select_related("target").defer("results", "config"))
    done = [r for r in runs if r.status == "completed"]
    cand = [r for r in done if r.disposition == "CONFIRMED_VETTING_QUEUE"]
    fps = [r for r in done if r.disposition.startswith("FALSE_POSITIVE") or r.vetting_status == "false_positive"]
    pending = [r for r in done if r.vetting_status == "pending" and r.disposition in NEEDS_HUMAN]
    live = [r for r in runs if r.is_live]
    n = len(done)

    def share(count):
        return round(100 * count / n) if n else 0

    preds = Counter(r.prediction for r in done)
    classes = [{"key": k, "label": CLASS_LABELS[k], "count": preds.get(k, 0), "pct": share(preds.get(k, 0))} for k in CLASS_NAMES]
    groups = [("Planet candidates", ("CONFIRMED_VETTING_QUEUE",)), ("Needs review", ("MANUAL_REVIEW", "FALSE_POSITIVE_SUSPECT")),
              ("Eclipsing binaries", ("FALSE_POSITIVE_ECLIPSING_BINARY",)), ("Noise / no signal", ("REJECTED_NOISE", "NO_SIGNIFICANT_SIGNAL"))]
    dispositions = [{"label": label, "count": sum(r.disposition in keys for r in done),
                     "pct": share(sum(r.disposition in keys for r in done)), "param": keys[0]} for label, keys in groups]
    sources = Counter(r.source for r in runs)
    days = [(now - timedelta(days=i)).date() for i in range(13, -1, -1)]
    by_day = {d: [0, 0] for d in days}
    for r in done:
        d = r.created_at.date()
        if d in by_day:
            by_day[d][0] += 1
            by_day[d][1] += r.disposition == "CONFIRMED_VETTING_QUEUE"
    stage_time: dict[str, list[float]] = {}
    for results in Run.objects.filter(status="completed").values_list("results", flat=True):
        for k, v in ((results or {}).get("timings") or {}).items():
            stage_time.setdefault(k, []).append(v)
    timings = [{"stage": s.replace("_", " ").lower(), "seconds": float(np.mean(stage_time[s]))} for s in core.STAGES if s in stage_time]
    tmax = max((t["seconds"] for t in timings), default=0) or 1
    for t in timings:
        t["pct"] = round(100 * t["seconds"] / tmax)
    durations = sorted(r.duration_s for r in done if r.duration_s)
    metrics = model.load_metrics()
    recent = LogEvent.objects.select_related("run__target").filter(
        Q(stage__in=("CNN_INFERENCE", "HUMAN_VETTING", "FAILED")) | Q(stage="INGEST", message__startswith="Initializing"))[:7]
    return {
        "n_targets": Target.objects.count(), "targets_week": Target.objects.filter(created_at__gte=week).count(),
        "n_done": n, "runs_week": sum(r.created_at >= week for r in done), "n_failed": sum(r.status == "failed" for r in runs),
        "n_candidates": len(cand), "n_fps": len(fps), "yield_pct": share(len(cand)),
        "pending": sorted(pending, key=lambda r: r.created_at, reverse=True), "live": live,
        "top": sorted((r for r in done if r.prediction == "PLANET_CANDIDATE"), key=lambda r: -(r.p_planet or 0))[:5],
        "classes": classes, "dispositions": dispositions,
        "sources": [{"label": {"mast": "MAST archive", "synthetic": "Synthetic", "upload": "CSV upload",
                               "benchmark": "Blind benchmark"}.get(k, k), "count": v}
                    for k, v in sources.most_common()],
        "daily": {"labels": [d.strftime("%m-%d") for d in days], "runs": [by_day[d][0] for d in days],
                  "candidates": [by_day[d][1] for d in days]},
        "timings": timings, "median_runtime": durations[len(durations) // 2] if durations else None,
        "metrics": metrics, "recent": recent,
        "vetted_confirmed": sum(r.vetting_status == "confirmed" for r in done),
        "vetted_fp": sum(r.vetting_status == "false_positive" for r in done),
    }


# ---------------------------------------------------------------------------
# Model training and system verification jobs
# ---------------------------------------------------------------------------

live_notes: dict[str, str] = {}  # run id -> what the pipeline is doing right now, in plain words

train_state = {"status": "idle", "stage": "", "progress": 0.0, "error": ""}
_job_lock = threading.Lock()


def _train_job() -> None:
    log = system_logger()
    try:
        log.info("CNN training started", "MODEL_TRAINING")
        metrics = model.train(progress=lambda stage, frac: train_state.update(stage=stage, progress=round(frac, 3)))
        ok = metrics["status"] == "active"
        log.log("INFO" if ok else "ERROR", "CNN checkpoint activated" if ok else
                "CNN checkpoint quarantined: AUC below acceptance threshold", "MODEL_TRAINING",
                auc_planet=metrics["auc_planet"], accuracy=metrics["accuracy"], status=metrics["status"],
                train_seconds=metrics["train_seconds"])
        train_state.update(status="idle", stage="", progress=0.0, error="" if ok else
                           f"Checkpoint quarantined (AUC {metrics['auc_planet']:.4f} < {TOLERANCES['auc_min']})")
    except Exception as exc:
        log.error(f"CNN training failed: {exc}", "MODEL_TRAINING")
        train_state.update(status="idle", stage="", progress=0.0, error=str(exc))


def start_training() -> bool:
    with _job_lock:
        if train_state["status"] == "running":
            return False
        train_state.update(status="running", stage="Starting", progress=0.0, error="")
    _spawn(_train_job)
    return True


def run_validation_job(val_id: int, online: bool = True) -> None:
    from cte.validation import run_validation
    log = system_logger()
    try:
        results = run_validation(val_id, lambda msg: ValidationRun.objects.filter(id=val_id).update(progress=msg[:200]), online)
        ValidationRun.objects.filter(id=val_id).update(status="completed", results=clean(results),
                                                       finished_at=timezone.now(), progress="")
        log.log("INFO" if results["summary"]["failed"] == 0 else "ERROR", "System verification suite finished",
                "VALIDATION", **results["summary"])
    except Exception as exc:
        ValidationRun.objects.filter(id=val_id).update(status="failed", progress=f"{type(exc).__name__}: {exc}"[:200],
                                                       finished_at=timezone.now())
        log.error(f"System verification suite crashed: {exc}", "VALIDATION")


def start_validation() -> ValidationRun | None:
    with _job_lock:
        if ValidationRun.objects.filter(status="running").exists():
            return None
        val = ValidationRun.objects.create(status="running", progress="Starting", results={"pid": os.getpid()})
    _spawn(run_validation_job, val.id)
    return val


# ---------------------------------------------------------------------------
# Blind benchmark on held-out real stars
# ---------------------------------------------------------------------------

benchmark_state = {"status": "idle", "done": 0, "total": 0, "current": "", "error": ""}


def _benchmark_job(kepids: list[int]) -> None:
    log = system_logger()
    log.info("Blind benchmark started", "BENCHMARK", stars=len(kepids))
    try:
        for i, kepid in enumerate(kepids):
            benchmark_state.update(done=i, current=f"KIC {kepid}")
            run_id = create_run({"source": "benchmark", "kepid": int(kepid)})
            execute_run(run_id)
        benchmark_state.update(status="idle", done=len(kepids), current="", error="")
        log.info("Blind benchmark finished", "BENCHMARK", **{k: v for k, v in benchmark_summary().items() if not isinstance(v, (list, dict))})
    except Exception as exc:
        benchmark_state.update(status="idle", current="", error=str(exc))
        log.error(f"Blind benchmark crashed: {exc}", "BENCHMARK")


def start_benchmark(per_class: int = 15) -> int:
    """Run the whole pipeline, blind, on held-out test stars that have not been benchmarked yet."""
    from cte import kepler
    with _job_lock:
        if benchmark_state["status"] == "running":
            return 0
        done = set(Run.objects.filter(source="benchmark").values_list("target__name", flat=True))
        picks = []
        for cls in range(3):
            pool = [r for r in kepler.load_sample() if r["split"] == "test" and r["label"] == cls
                    and f"KIC {r['kepid']}" not in done and (kepler.LC_DIR / f"{r['kepid']}.npz").exists()]
            picks += [r["kepid"] for r in pool[:per_class]]
        if not picks:
            return 0
        order = np.random.default_rng(len(done)).permutation(len(picks))  # interleave the classes
        picks = [picks[i] for i in order]
        benchmark_state.update(status="running", done=0, total=len(picks), current="", error="")
    _spawn(_benchmark_job, picks)
    return len(picks)


def benchmark_summary() -> dict:
    """Score the blind runs against the NASA Exoplanet Archive dispositions."""
    latest: dict[int, Run] = {}  # a star benchmarked more than once counts once, by its most recent run
    for r in Run.objects.select_related("target").filter(source="benchmark", status="completed").order_by("created_at"):
        if r.results.get("truth"):
            latest[r.target_id] = r
    runs = list(latest.values())
    rows = []
    confusion = [[0] * 3 for _ in range(3)]
    for r in runs:
        tr, cls = r.results["truth"], r.results["classification"]
        pred = CLASS_NAMES.index(cls["prediction"])
        if r.disposition == "NO_SIGNIFICANT_SIGNAL":
            pred = 2  # the pipeline's answer is "nothing detected", whatever the classifier made of the noise
        confusion[tr["label"]][pred] += 1
        rows.append({"run": r, "label": tr["label"], "label_name": CLASS_LABELS[CLASS_NAMES[tr["label"]]], "pred": pred,
                     "correct": pred == tr["label"], "p_planet": r.p_planet, "period_true": tr["period_days"],
                     "period_match": tr.get("period_match", "none"), "period_rel_error": tr.get("period_rel_error"),
                     "queued": r.disposition == "CONFIRMED_VETTING_QUEUE", "koi": tr["koi"], "snr": r.results["diagnostics"]["snr"]})
    n = len(rows)
    planets = [x for x in rows if x["label"] == 0]
    others = [x for x in rows if x["label"] != 0]
    auc = None
    if planets and others:
        _, _, auc = model.roc_curve(np.array([x["p_planet"] for x in rows]), np.array([x["label"] == 0 for x in rows]))
    hist = []
    for cls in range(3):
        counts = np.histogram([x["p_planet"] for x in rows if x["label"] == cls], bins=10, range=(0, 1))[0]
        hist.append({"label": CLASS_LABELS[CLASS_NAMES[cls]], "counts": counts.tolist()})
    top = max((max(row) for row in confusion), default=0) or 1
    short = ["Planet", "Binary", "Noise"]
    return {
        "n": n, "rows": sorted(rows, key=lambda x: (x["label"], -(x["p_planet"] or 0))),
        "accuracy": sum(x["correct"] for x in rows) / n if n else None, "auc_planet": auc,
        "confusion": [{"label": short[i], "cells": [{"n": v, "alpha": round(v / top, 3)} for v in row]} for i, row in enumerate(confusion)],
        "period_exact": sum(x["period_match"] == "exact" for x in rows if x["label"] != 2),
        "period_alias": sum(x["period_match"] in ("half", "double") for x in rows if x["label"] != 2),
        "period_total": sum(x["label"] != 2 for x in rows),
        "planets": len(planets), "planets_queued": sum(x["queued"] for x in planets),
        "planets_called_planet": sum(x["pred"] == 0 for x in planets),
        "others": len(others), "others_queued": sum(x["queued"] for x in others),
        "hist": hist, "hist_max": max((max(h["counts"]) for h in hist), default=0) or 1,
        "scatter": {"true": [x["period_true"] for x in rows if x["label"] != 2],
                    "measured": [x["run"].period_days for x in rows if x["label"] != 2]},
    }
