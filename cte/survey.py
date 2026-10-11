"""Survey: run the whole pipeline blind on stars already on disk and list every signal no NASA table contains.

For each star the engine finds its strongest signal, then searches for further ones and classifies each. Every
signal is compared with what NASA lists for that star - the Kepler pipeline's Threshold Crossing Events and the
current Objects of Interest - at the same period or a simple multiple of it. What is left is a signal this engine
sees and NASA's tables do not. That is a list for a person to inspect, not a list of planets.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import DATA_DIR

RESULTS = DATA_DIR / "survey.jsonl"
UNLISTED = DATA_DIR / "survey_unlisted.csv"
DETECTIONS = DATA_DIR / "kepler" / "tce_all.json"
TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
TCE_TABLES = ("q1_q12_tce", "q1_q16_tce", "q1_q17_dr24_tce", "q1_q17_dr25_tce")
MULTIPLES = (1.0, 0.5, 2.0, 1.0 / 3.0, 3.0, 2.0 / 3.0, 1.5)
MAX_FRACTION = 40  # the search stops at 30 days, so a planet with a longer period turns up at period / n
FRACTION_TOLERANCE = 0.002


def all_detections(refresh: bool = False) -> dict[int, list[tuple[float, str]]]:
    """Every Threshold Crossing Event of NASA's four Kepler detection lists, vetted or not. Cached; empty when the
    archive cannot be reached and nothing is cached."""
    import urllib.parse
    import urllib.request

    if not DETECTIONS.exists() or refresh:
        rows = []
        try:
            for table in TCE_TABLES:
                url = TAP + "?" + urllib.parse.urlencode({"query": f"select kepid,tce_plnt_num,tce_period from {table}", "format": "csv"})
                with urllib.request.urlopen(url, timeout=300) as resp:
                    for r in csv.DictReader(io.StringIO(resp.read().decode())):
                        rows.append([int(r["kepid"]), float(r["tce_period"]), f"{table} TCE {r['tce_plnt_num']}"])
            if len(rows) < 10000:
                raise RuntimeError("short answer")
            DETECTIONS.parent.mkdir(parents=True, exist_ok=True)
            DETECTIONS.write_text(json.dumps(rows))
        except Exception:
            if not DETECTIONS.exists():
                return {}
    out: dict[int, list[tuple[float, str]]] = {}
    for kepid, period, name in json.loads(DETECTIONS.read_text()):
        out.setdefault(int(kepid), []).append((period, name))
    return out


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
    for kepid, rows in all_detections().items():  # last, so a named planet is preferred as the match
        out.setdefault(kepid, []).extend(rows)
    return out


def match(period: float, listed: list[tuple[float, str]], tolerance: float = 0.01) -> str | None:
    """The catalogued signal this period belongs to, allowing for the usual period aliases."""
    for known, name in listed:
        for m in MULTIPLES:
            if abs(period / known - m) / m < tolerance:
                return name if m == 1.0 else f"{name} at {m:g}x its period"
    for known, name in listed:  # a longer catalogued period seen at a whole fraction of itself
        n = round(known / period)
        if 4 <= n <= MAX_FRACTION and abs(known / (n * period) - 1.0) < FRACTION_TOLERANCE:
            return f"{name} at 1/{n} of its period"
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


def surveyed() -> list[dict]:
    """Every star surveyed so far, the latest record of each."""
    done = {}
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["kepid"]] = rec
    return list(done.values())


def run(kepids: list[int] | None = None, workers: int = 4, limit: int | None = None, report=lambda *_: None) -> dict:
    """Survey the given stars (default: every star of the dataset), resuming where an earlier survey stopped."""
    from . import kepler

    listed = catalogued_periods()
    todo = [int(k) for k in (kepids or [r["kepid"] for r in kepler.load_sample()])]
    done = {rec["kepid"]: rec for rec in surveyed()}
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
    listed = catalogued_periods()
    for r in records:  # matched again here, so a survey made before the tables or the aliases grew is brought up to date
        for s in r["signals"]:
            s["listed_as"] = match(s["period_days"], listed.get(r["kepid"], []))
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


# the ratios at which one signal leaves an echo of itself: whole multiples and fractions, 2:3 and 3:2
RATIOS = sorted({float(n) for n in range(2, 9)} | {1.0 / n for n in range(2, 9)} | {2.0 / 3.0, 1.5})


def triage(signal: dict, objects: list[dict], main: dict | None = None) -> list[str]:
    """Reasons to trust an unlisted signal less. Empty when it passes all of them. `main` is the star's strongest signal."""
    notes = []
    if main is not None and main is not signal and main.get("prediction") == "ECLIPSING_BINARY":
        notes.append("the strongest signal on this star is classed as an eclipsing binary")
    kinds = {o["koi_disposition"] for o in objects}
    if kinds and not kinds & {"CONFIRMED", "CANDIDATE"}:
        notes.append("the only object NASA lists on this star is a false positive")
    if not objects:
        notes.append("NASA lists no object of interest on this star")
    if (signal.get("event_chi2") or 0) > 10:
        notes.append("its single transits disagree with each other")
    if signal["rank"] == 1 and objects:
        notes.append("it is the strongest signal on the star, yet not the catalogued one")
    for o in objects:
        ratio = signal["period_days"] / float(o["koi_period"])
        if abs(ratio - 1) < 0.05:
            notes.append(f"its period is within 5% of {o.get('kepler_name') or o['kepoi_name']}'s")
        elif any(abs(ratio / r - 1) < 0.01 for r in RATIOS):
            notes.append(f"its period is close to a simple ratio of {o.get('kepler_name') or o['kepoi_name']}'s")
    return notes


_possible: dict = {}


def possible_planets() -> list[dict]:
    """Every surveyed signal that is in no NASA table and that the classifier calls a planet, most probable first, each
    with what NASA lists on its star and the triage notes against it."""
    from . import kepler

    stamp = RESULTS.stat().st_mtime if RESULTS.exists() else None
    if _possible.get("stamp") != stamp:
        listed, current = catalogued_periods(), kepler.current_dispositions()
        out = []
        for r in surveyed():
            objects = sorted(current.get(r["kepid"], []), key=lambda o: float(o["koi_period"]))
            for s in r["signals"]:
                if s.get("prediction") != "PLANET_CANDIDATE" or match(s["period_days"], listed.get(r["kepid"], [])):
                    continue
                names = sorted({o["kepler_name"].rsplit(" ", 1)[0] for o in objects if o.get("kepler_name")})
                koi = sorted({o["kepoi_name"].split(".")[0].replace("K0", "KOI-").replace("KOI-0", "KOI-") for o in objects})
                out.append(dict(s, kepid=r["kepid"], host=(names or koi or [f"KIC {r['kepid']}"])[0], signals_on_star=len(r["signals"]),
                                objects=[{"name": o.get("kepler_name") or o["kepoi_name"], "period_days": float(o["koi_period"]),
                                          "status": o["koi_disposition"]} for o in objects],
                                notes=triage(s, objects, r["signals"][0])))
        out.sort(key=lambda s: (bool(s["notes"]), -(s.get("p_planet") or 0.0)))
        _possible.update(stamp=stamp, rows=out, stars=len(surveyed()))
    return _possible["rows"]


def stars_surveyed() -> int:
    possible_planets()
    return _possible.get("stars", 0)
