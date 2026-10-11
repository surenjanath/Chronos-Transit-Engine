"""Survey: run the whole pipeline blind on stars already on disk and list every signal no NASA table contains.

For each star the engine finds its strongest signal, then searches for further ones and classifies each. Every
signal is compared with what NASA lists for that star - the Kepler pipeline's Threshold Crossing Events and the
current Objects of Interest - at the same period or a simple multiple of it. What is left is a signal this engine
sees and NASA's tables do not. That is a list for a person to inspect, not a list of planets.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import DATA_DIR

RESULTS = DATA_DIR / "survey.jsonl"
UNLISTED = DATA_DIR / "survey_unlisted.csv"
MULTIPLES = (1.0, 0.5, 2.0, 1.0 / 3.0, 3.0)


def catalogued_periods() -> dict[int, list[tuple[float, str]]]:
    """Every period NASA lists for each star: (days, where it is listed)."""
    from . import kepler

    out: dict[int, list[tuple[float, str]]] = {}
    for r in kepler.fetch_koi_table():
        out.setdefault(int(r["kepid"]), []).append((float(r["koi_period"]), f"{r['kepoi_name']} ({r['koi_disposition']})"))
    for kepid, rows in kepler.current_dispositions().items():
        for r in rows:
            name = r.get("kepler_name") or r["kepoi_name"]
            out.setdefault(int(kepid), []).append((float(r["koi_period"]), f"{name} ({r['koi_disposition']})"))
    return out


def match(period: float, listed: list[tuple[float, str]], tolerance: float = 0.01) -> str | None:
    """The catalogued signal this period belongs to, allowing for the usual period aliases."""
    for known, name in listed:
        for m in MULTIPLES:
            if abs(period / known - m) / m < tolerance:
                return name if m == 1.0 else f"{name} at {m:g}x its period"
    return None


def one_star(kepid: int) -> dict:
    """Blind run of one star; every signal found, with its verdict."""
    from . import ingest
    from .config import DEFAULT_PIPELINE
    from .core import process
    from .telemetry import RunLogger

    logging.getLogger("CTE-Core").setLevel(logging.ERROR)
    os.environ["CTE_BLS_WORKERS"] = "1"  # the stars run side by side instead
    try:
        data = ingest.load_benchmark_star(int(kepid))
        res, _ = process(data, dict(DEFAULT_PIPELINE), RunLogger(None, path=pathlib.Path(os.devnull)))
    except Exception as exc:
        return {"kepid": int(kepid), "error": f"{type(exc).__name__}: {exc}"[:200], "signals": []}
    first = res["classification"]
    signals = [{"rank": 1, "period_days": res["bls"]["period"], "depth_ppm": res["fit"]["depth_ppm"], "snr": res["diagnostics"]["snr"],
                "duration_hours": res["fit"]["t14_hours"], "radius_earth": res["parameters"]["radius_earth"],
                "prediction": first["prediction"], "p_planet": first["probabilities"]["PLANET_CANDIDATE"],
                "disposition": first["disposition"]}]
    for i, a in enumerate(res.get("additional_signals", []), 2):
        c = a.get("classification") or {}
        signals.append({"rank": i, "period_days": a["period_days"], "depth_ppm": a["depth_ppm"], "snr": a["snr"],
                        "duration_hours": a["duration_hours"], "radius_earth": c.get("radius_earth"), "transits": a["transits"],
                        "event_chi2": a.get("event_chi2"), "prediction": c.get("prediction"),
                        "p_planet": (c.get("probabilities") or {}).get("PLANET_CANDIDATE"), "disposition": c.get("disposition")})
    return {"kepid": int(kepid), "catalogue": data.truth.get("catalogue_disposition"), "quarters": len(data.meta.get("quarters") or []),
            "signals": signals}


def run(kepids: list[int] | None = None, workers: int = 4, limit: int | None = None, report=lambda *_: None) -> dict:
    """Survey the given stars (default: every star of the dataset), resuming where an earlier survey stopped."""
    from . import kepler

    listed = catalogued_periods()
    todo = [int(k) for k in (kepids or [r["kepid"] for r in kepler.load_sample()])]
    done = {}
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            rec = json.loads(line)
            done[rec["kepid"]] = rec
    todo = [k for k in todo if k not in done][: limit or None]
    if todo:
        with ProcessPoolExecutor(max(1, workers)) as pool, RESULTS.open("a") as fh:
            futures = {pool.submit(one_star, k): k for k in todo}
            for n, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                for s in rec["signals"]:
                    s["listed_as"] = match(s["period_days"], listed.get(rec["kepid"], []))
                done[rec["kepid"]] = rec
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                report(f"Surveyed {n}/{len(todo)} stars", n / len(todo))
    return summarise(list(done.values()))


def summarise(records: list[dict]) -> dict:
    """Counts, and the unlisted signals written out as a table, strongest planet verdicts first."""
    unlisted = [dict(s, kepid=r["kepid"], star_catalogue=r.get("catalogue"), quarters=r.get("quarters"))
                for r in records for s in r["signals"] if s.get("listed_as") is None]
    unlisted.sort(key=lambda s: -(s.get("p_planet") or 0.0))
    fields = ["kepid", "rank", "period_days", "depth_ppm", "duration_hours", "snr", "radius_earth", "transits", "event_chi2",
              "prediction", "p_planet", "disposition", "star_catalogue", "quarters"]
    with UNLISTED.open("w", newline="") as fh:
        out = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        out.writeheader()
        out.writerows(unlisted)
    planets = [s for s in unlisted if s.get("prediction") == "PLANET_CANDIDATE"]
    return {"stars": len(records), "errors": sum(1 for r in records if r.get("error")),
            "signals": sum(len(r["signals"]) for r in records), "unlisted": len(unlisted),
            "unlisted_called_planet": len(planets), "unlisted_in_vetting_queue": sum(1 for s in planets if (s.get("p_planet") or 0) >= 0.9),
            "top": planets[:20], "table": str(UNLISTED)}
