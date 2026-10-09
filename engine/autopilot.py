"""Autopilot: pull newly listed targets, grow the labelled dataset, retrain when there is enough new data.

A cycle does three things, each recorded in the audit log and in an AutomationCycle row:
  1. New targets  - the newest TESS Objects of Interest not yet in the catalogue are fetched from MAST and analysed.
  2. New labels   - more Kepler KOIs with settled archive dispositions are downloaded into the real dataset.
  3. Retraining   - once enough new labelled stars have accumulated the CNN is retrained; the challenger replaces
                    the active model only if it is not worse on the fixed held-out real stars.
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from datetime import timedelta

from django.conf import settings
from django.db import close_old_connections
from django.utils import timezone

from cte import archive, ingest, kepler, model
from cte.telemetry import clean

from . import services
from .models import AutomationCycle, Run, Setting, Target

DEFAULTS = {"enabled": False, "interval_minutes": 360, "tois_per_cycle": 3, "stars_per_cycle": 30, "retrain_after_stars": 60}
LIMITS = {"interval_minutes": (15, 10080), "tois_per_cycle": (0, 10), "stars_per_cycle": (0, 150), "retrain_after_stars": (15, 600)}
_lock = threading.Lock()


def config() -> dict:
    saved = Setting.objects.filter(key="autopilot").first()
    return {**DEFAULTS, **(saved.value if saved else {})}


def save_config(values: dict) -> dict:
    cfg = {**config(), **{k: v for k, v in values.items() if k in DEFAULTS}}
    Setting.objects.update_or_create(key="autopilot", defaults={"value": cfg})
    services.system_logger().info("Autopilot settings updated", "AUTOPILOT", **cfg)
    return cfg


def _memory(key: str) -> list:
    row = Setting.objects.filter(key=key).first()
    return list(row.value.get("items", [])) if row else []


def _remember(key: str, items: list) -> None:
    Setting.objects.update_or_create(key=key, defaults={"value": {"items": items[-2000:]}})


def newest_tois(limit: int = 150) -> list[dict]:
    """Most recently listed TOIs. (The archive applies TOP before ORDER BY, so filter by date and sort here.)"""
    since = (timezone.now() - timedelta(days=540)).strftime("%Y-%m-%d")
    rows = archive._tap("select toi,tid,tfopwg_disp,pl_orbper,toi_created from toi "
                        f"where toi_created > '{since}'", timeout=60)
    rows.sort(key=lambda r: r["toi_created"], reverse=True)
    return rows[:limit]


def pull_new_targets(count: int, step) -> dict:
    """Analyse the newest TOIs that are not in the catalogue yet and have pipeline photometry at MAST."""
    out = {"requested": count, "runs": [], "skipped": []}
    if count <= 0:
        return out
    step("Querying the NASA Exoplanet Archive for newly listed TOIs")
    seen = set(_memory("autopilot_seen_tics"))
    known = set(Target.objects.values_list("name", flat=True))
    tried = 0
    for row in newest_tois():
        name = f"TIC {row['tid']}"
        if name in known or row["tid"] in seen:
            continue
        if len(out["runs"]) >= count or tried >= count * 4:
            break
        tried += 1
        seen.add(row["tid"])
        step(f"Checking MAST for TOI-{row['toi']} ({name})")
        try:
            products = ingest.search_mast(name, "TESS")
        except Exception as exc:
            out["skipped"].append({"toi": row["toi"], "reason": f"MAST search failed: {type(exc).__name__}"})
            seen.discard(row["tid"])  # transient: try again next cycle
            continue
        usable = [p for p in products if p["exptime"] == 120] or [p for p in products if 120 < p["exptime"] <= 1800]
        if not usable:
            out["skipped"].append({"toi": row["toi"], "reason": "no pipeline light curve at MAST yet"})
            continue
        latest = max(usable, key=lambda p: p["sequence"] or 0)
        step(f"Analysing TOI-{row['toi']} ({name}), sector {latest['sequence']}")
        run_id = services.create_run({"source": "mast", "origin": "autopilot", "toi": row["toi"],
                                      "mast": {"target": name, "mission": "TESS", "sequence": latest["sequence"],
                                               "exptime": latest["exptime"]}})
        services.execute_run(run_id)
        run = Run.objects.select_related("target").get(id=run_id)
        known.add(run.target.name)
        out["runs"].append({"run": run_id, "toi": row["toi"], "target": run.target.name, "status": run.status,
                            "listed": row["toi_created"], "archive_disposition": row["tfopwg_disp"],
                            "disposition": run.disposition, "period_days": run.period_days,
                            "archive_period_days": row["pl_orbper"], "p_planet": run.p_planet})
    _remember("autopilot_seen_tics", sorted(seen))
    return out


def grow_dataset(count: int, step) -> dict:
    """Download more labelled Kepler stars (about a third per class) and rebuild the view tensors."""
    before = kepler.dataset_summary()
    if count <= 0:
        return {"added": 0, "stars": before["stars"] if before else 0}
    per_class = (len(kepler.load_sample()) + 2) // 3 + max(1, count // 3) if before else max(20, count // 3)
    step(f"Downloading labelled Kepler stars from MAST (target {per_class} per class)")
    after = kepler.build_dataset(per_class, workers=6, progress=lambda stage, frac: step(f"{stage}"))
    return {"added": after["stars"] - (before["stars"] if before else 0), "stars": after["stars"], "test": after["test"],
            "per_class": per_class}


def maybe_retrain(cfg: dict, step) -> dict:
    summary = kepler.dataset_summary()
    metrics = model.load_metrics() or {}
    real_m = metrics.get("real") or {}
    trained_on = real_m.get("cnn_trained_on_stars") or (real_m.get("summary") or {}).get("stars", 0)
    stars = summary["stars"] if summary else 0
    new = stars - trained_on
    if not summary or new < cfg["retrain_after_stars"]:
        return {"retrained": False, "new_stars_since_training": new, "needed": cfg["retrain_after_stars"]}
    step(f"Retraining the classifier on {stars} real stars ({new} new)")
    m = model.train(progress=lambda stage, frac: step(f"Retraining: {stage} {frac * 100:.0f}%"), keep_champion_if_better=True)
    real = m.get("real") or {}
    if m["status"] != "active":  # champion kept: still refit the second stage and re-score on today's held-out stars
        model.refresh_real_metrics()
    return {"retrained": True, "status": m["status"], "auc_real": real.get("auc_planet"), "accuracy_real": real.get("accuracy"),
            "champion_auc_real": (real.get("baseline") or {}).get("auc_planet"), "auc_simulated": m["auc_planet"],
            "real_stars": stars, "new_stars_since_training": new}


def run_cycle(trigger: str = "manual") -> AutomationCycle | None:
    """Run one full cycle. Returns None if a cycle is already running in this process."""
    if not _lock.acquire(blocking=False):
        return None
    cycle = AutomationCycle.objects.create(trigger=trigger, summary={"pid": os.getpid()})
    log = services.system_logger()
    cfg = config()
    summary: dict = {"pid": os.getpid(), "config": cfg}

    def step(text: str):
        AutomationCycle.objects.filter(id=cycle.id).update(step=text[:200])

    try:
        log.info("Autopilot cycle started", "AUTOPILOT", cycle=cycle.id, trigger=trigger)
        for key, fn in (("targets", lambda: pull_new_targets(cfg["tois_per_cycle"], step)),
                        ("dataset", lambda: grow_dataset(cfg["stars_per_cycle"], step)),
                        ("training", lambda: maybe_retrain(cfg, step))):
            try:
                summary[key] = fn()
            except Exception as exc:  # one failed stage must not stop the others
                summary[key] = {"error": f"{type(exc).__name__}: {exc}"}
                log.error(f"Autopilot stage '{key}' failed: {exc}", "AUTOPILOT", cycle=cycle.id,
                          traceback=traceback.format_exc(limit=4))
        failed = [k for k in ("targets", "dataset", "training") if "error" in summary[k]]
        status = "completed" if not failed else ("failed" if len(failed) == 3 else "partial")
        AutomationCycle.objects.filter(id=cycle.id).update(status=status, finished_at=timezone.now(), step="",
                                                           summary=clean(summary))
        log.log("INFO" if not failed else "WARNING", "Autopilot cycle finished", "AUTOPILOT", cycle=cycle.id, status=status,
                new_runs=len(summary["targets"].get("runs", [])), stars_added=summary["dataset"].get("added"),
                retrained=summary["training"].get("retrained"), model_status=summary["training"].get("status"))
    finally:
        _lock.release()
    cycle.refresh_from_db()
    return cycle


def next_due() -> timezone.datetime | None:
    cfg = config()
    if not cfg["enabled"]:
        return None
    last = AutomationCycle.objects.exclude(finished_at=None).first()
    return (last.finished_at + timedelta(minutes=cfg["interval_minutes"])) if last else timezone.now()


def is_running() -> bool:
    return AutomationCycle.objects.filter(status="running").exists()


def _scheduler() -> None:
    """Background loop in the web process: start a cycle whenever one is due."""
    while True:
        time.sleep(30)
        try:
            due = next_due()
            if due and due <= timezone.now() and not is_running():
                run_cycle("schedule")
        except Exception:
            pass
        finally:
            close_old_connections()


_scheduler_started = False


def start_scheduler() -> None:
    global _scheduler_started
    if _scheduler_started or settings.CTE_RUN_INLINE:
        return
    _scheduler_started = True
    for cycle in AutomationCycle.objects.filter(status="running"):
        if not services._alive(cycle.summary.get("pid")):
            AutomationCycle.objects.filter(id=cycle.id).update(status="failed", step="Interrupted by server restart",
                                                               finished_at=timezone.now())
    threading.Thread(target=_scheduler, daemon=True, name="cte-autopilot").start()


def start_cycle_async(trigger: str = "manual") -> bool:
    if is_running():
        return False
    if settings.CTE_RUN_INLINE:
        run_cycle(trigger)
    else:
        threading.Thread(target=lambda: (run_cycle(trigger), close_old_connections()), daemon=True).start()
    return True
