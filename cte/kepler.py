"""Real labelled data: the Kepler DR24 Threshold Crossing Event table (Q1-Q17) with MAST long-cadence photometry.

Labels are the Kepler Science Office training labels in the NASA Exoplanet Archive (`av_training_set`),
the same ground truth AstroNet was trained and scored on:
  PC   planet candidate / confirmed planet      -> PLANET_CANDIDATE
  AFP  astrophysical false positive (binaries)   -> ECLIPSING_BINARY
  NTP  non-transiting phenomenon (systematics)   -> STELLAR_OR_INSTRUMENT_NOISE
Those labels date from 2015. Where NASA's current KOI table has since settled a signal - confirmed it as a planet,
or ruled it a false positive - the current verdict replaces the old label.
"""
from __future__ import annotations

import io
import json
import time
import urllib.parse
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np

from .conditioning import savgol_detrend, sigma_clip
from .config import CLASS_NAMES, DATA_DIR
from . import pixels
from .physics import TransitAnalyticsEngine, global_view, local_channels, local_view, normalize_view
from .vetting import N_PIXEL

KEPLER_DIR = DATA_DIR / "kepler"
LC_DIR = KEPLER_DIR / "lightcurves"
DATASET = KEPLER_DIR / "dataset.npz"
SAMPLE = KEPLER_DIR / "sample.json"
TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
LC_URL = "https://archive.stsci.edu/missions/kepler/lightcurves/{p}/{k}/kplr{k}-{stamp}_llc.fits"
QUARTER_STAMPS = {1: "2009166043257", 2: "2009259160929", 3: "2009350155506", 4: "2010078095331", 5: "2010174085026",
                  6: "2010265121752", 7: "2010355172524", 8: "2011073133259", 9: "2011177032512", 10: "2011271113734",
                  11: "2012004120508", 12: "2012088054726", 13: "2012179063303", 14: "2012277125453",
                  15: "2013011073258", 16: "2013098041711", 17: "2013131215648"}
ALL_QUARTERS = tuple(range(1, 18))
DEFAULT_QUARTERS = (3, 4, 5, 6, 7, 8)
TCE_API = ("https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=q1_q17_dr24_tce"
           "&select=kepid,tce_plnt_num,tce_period,tce_time0bk,tce_duration,tce_depth,tce_model_snr,tce_prad,tce_steff,"
           "tce_sradius,tce_slogg,av_training_set&where=tce_period<30+and+tce_period>0.5&format=csv")
TCE_LABELS = {"PC": 0, "AFP": 1, "NTP": 2}
TABLE = KEPLER_DIR / "tce_table.json"


def _f(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_koi_table(refresh: bool = False) -> list[dict]:
    """The labelled DR24 TCEs, with field names shared by the rest of the code base."""
    import csv

    KEPLER_DIR.mkdir(parents=True, exist_ok=True)
    if TABLE.exists() and not refresh:
        return apply_current(json.loads(TABLE.read_text()))
    with urllib.request.urlopen(TCE_API, timeout=180) as resp:
        text = resp.read().decode()
    if text.lstrip().startswith("ERROR"):
        raise RuntimeError("NASA Exoplanet Archive rejected the TCE table query")
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        if r["av_training_set"] not in TCE_LABELS:
            continue  # UNK: never vetted
        srad, logg = _f(r["tce_sradius"]), _f(r["tce_slogg"])
        rows.append({
            "kepid": int(r["kepid"]), "kepoi_name": f"TCE {int(r['kepid'])}-{int(r['tce_plnt_num'])}", "kepler_name": None,
            "koi_disposition": r["av_training_set"], "tce_label": r["av_training_set"],
            "koi_period": _f(r["tce_period"]), "koi_time0bk": _f(r["tce_time0bk"]), "koi_duration": _f(r["tce_duration"]),
            "koi_depth": _f(r["tce_depth"]), "koi_prad": _f(r["tce_prad"]), "koi_model_snr": _f(r["tce_model_snr"]),
            "koi_steff": _f(r["tce_steff"]), "koi_srad": srad,
            "koi_smass": round(10 ** (logg - 4.438) * srad ** 2, 3) if srad and logg else None, "koi_kepmag": None,
        })
    TABLE.write_text(json.dumps(rows))
    return apply_current(rows)


CURRENT = KEPLER_DIR / "koi_current.json"
CURRENT_QUERY = ("select kepid,kepoi_name,kepler_name,koi_disposition,koi_period,koi_fpflag_nt from cumulative")


def current_dispositions(refresh: bool = False) -> dict[int, list[dict]]:
    """NASA's present verdict on every Kepler Object of Interest, by star. Empty when the archive cannot be reached
    and nothing is cached: the 2015 labels are then used unchanged."""
    import csv

    if not CURRENT.exists() or refresh:
        try:
            url = TAP_URL + "?" + urllib.parse.urlencode({"query": CURRENT_QUERY, "format": "csv"})
            with urllib.request.urlopen(url, timeout=180) as resp:
                rows = [r for r in csv.DictReader(io.StringIO(resp.read().decode())) if r.get("koi_period")]
            if len(rows) < 1000:
                raise RuntimeError("short answer")
            KEPLER_DIR.mkdir(parents=True, exist_ok=True)
            CURRENT.write_text(json.dumps(rows))
        except Exception:
            if not CURRENT.exists():
                return {}
    by_star: dict[int, list[dict]] = {}
    for r in json.loads(CURRENT.read_text()):
        by_star.setdefault(int(r["kepid"]), []).append(r)
    return by_star


def apply_current(rows: list[dict]) -> list[dict]:
    """Set each row's `label` and `koi_disposition`: the DR24 label, overridden only where the current table is
    definitive for the same signal (same star, same period). A signal still listed as an unresolved candidate
    keeps its DR24 label."""
    koi = current_dispositions()
    for r in rows:
        r["label"] = TCE_LABELS[r["tce_label"]]
        r["koi_disposition"] = r["tce_label"]
        for k in koi.get(r["kepid"], []) if r["koi_period"] else []:
            if abs(float(k["koi_period"]) / r["koi_period"] - 1.0) < 0.002:
                r["kepler_name"] = k["kepler_name"] or None
                r["koi_disposition"] = f"{r['tce_label']}, now {k['koi_disposition']}"
                if k["koi_disposition"] == "CONFIRMED":
                    r["label"] = 0
                elif k["koi_disposition"] == "FALSE POSITIVE" and r["label"] == 0:
                    r["label"] = 2 if k["koi_fpflag_nt"] == "1" else 1
                break
    return rows


def label_of(row: dict) -> int | None:
    return row["label"] if "label" in row else TCE_LABELS.get(row.get("tce_label"))


MIN_EXPECTED_SNR = 10.0  # in the quarters actually downloaded
KEPLER_QUARTERS_TOTAL = 17


def min_catalogue_snr(quarters=DEFAULT_QUARTERS) -> float:
    """The catalogue S/N is for the whole four-year mission; scale it to the quarters the model is shown."""
    return MIN_EXPECTED_SNR / (len(quarters) / KEPLER_QUARTERS_TOTAL) ** 0.5


def select_sample(per_class: int = 200, seed: int = 2026, quarters=DEFAULT_QUARTERS) -> list[dict]:
    """A class-balanced sample of stars from the labelled TCE table.

    Planets and binaries must be detectable in the photometry supplied (expected S/N >= 10 in the downloaded
    quarters); otherwise the label describes a signal the classifier cannot see. A star's split depends only on
    its KIC number, so the held-out fifth never changes when the dataset is rebuilt or enlarged.
    """
    rows = fetch_koi_table()
    floor = min_catalogue_snr(quarters)
    strongest: dict[int, dict] = {}
    for r in rows:  # one signal per star: its strongest labelled TCE
        if r["koi_period"] and r["koi_time0bk"] and r["koi_duration"]:
            best = strongest.get(r["kepid"])
            if best is None or (r["koi_model_snr"] or 0) > (best["koi_model_snr"] or 0):
                strongest[r["kepid"]] = r
    cached = {int(f.stem) for f in LC_DIR.glob("*.npz")} if LC_DIR.exists() else set()
    rng = np.random.default_rng(seed)
    out = []
    for cls in range(3):
        pool = [r for r in strongest.values() if label_of(r) == cls and (cls == 2 or (r["koi_model_snr"] or 0) >= floor)]
        pool.sort(key=lambda r: r["kepoi_name"])
        order = [pool[int(i)] for i in rng.permutation(len(pool))]
        order.sort(key=lambda r: r["kepid"] not in cached)  # stable: reuse photometry already on disk first
        for r in order[:per_class]:
            out.append({**r, "label": cls, "split": "test" if r["kepid"] % 5 == 0 else "train"})
    chosen = {r["kepid"] for r in out}
    for kepid in sorted(cached - chosen):  # photometry already on disk is never left unused
        r = strongest.get(kepid)
        if r is not None and label_of(r) is not None:
            out.append({**r, "label": label_of(r), "split": "test" if kepid % 5 == 0 else "train"})
    return out


class QuarterUnavailable(Exception):
    """The archive failed to answer. Unlike a 404 this says nothing about whether the quarter exists."""


def _download_quarter(kepid: int, quarter: int, timeout: float) -> tuple | None:
    from astropy.io import fits

    k = f"{kepid:09d}"
    url = LC_URL.format(p=k[:4], k=k, stamp=QUARTER_STAMPS[quarter])
    raw = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                raw = resp.read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None  # the star was not observed in this quarter
        except Exception:
            pass
        time.sleep(2.0 * (attempt + 1))
    if raw is None:
        raise QuarterUnavailable(f"KIC {kepid} quarter {quarter}: the archive did not answer")
    with fits.open(io.BytesIO(raw)) as hdul:
        d = hdul[1].data
        t, f, e, quality = (np.asarray(d[c], dtype=float) for c in ("TIME", "PDCSAP_FLUX", "PDCSAP_FLUX_ERR", "SAP_QUALITY"))
        cc, cr = np.asarray(d["MOM_CENTR1"], dtype=float), np.asarray(d["MOM_CENTR2"], dtype=float)
    ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (e > 0) & (quality == 0)
    if ok.sum() < 500:
        return None
    med = np.median(f[ok])
    return (t[ok], f[ok] / med, e[ok] / med, cc[ok] - np.nanmedian(cc[ok]), cr[ok] - np.nanmedian(cr[ok]), np.full(ok.sum(), float(quarter)))


FIELDS = ("time", "flux", "flux_err", "centroid_col", "centroid_row", "quarter")


def download_star(kepid: int, quarters=DEFAULT_QUARTERS, timeout: float = 30.0, recheck: bool = False) -> dict | None:
    """PDCSAP long-cadence photometry for the given quarters, each quarter median-normalised. Cached as .npz;
    quarters missing from an existing cache file are fetched and merged in."""
    LC_DIR.mkdir(parents=True, exist_ok=True)
    cache = LC_DIR / f"{kepid}.npz"
    have: dict | None = None
    tried: set[int] = set()
    if cache.exists():
        with np.load(cache) as z:
            have = {k: z[k] for k in FIELDS}
            tried = {int(q) for q in z["tried"]} if "tried" in z.files else {int(q) for q in np.unique(z["quarter"])} | {4, 5, 6}
    def only(d):  # a cache file may hold more quarters than this caller asked for
        keep = np.isin(d["quarter"], list(quarters))
        return {k: v[keep] for k, v in d.items()} if keep.any() else None

    if recheck and have is not None:  # ask again for every quarter that is absent, whatever was recorded before
        tried = {int(q) for q in np.unique(have["quarter"])}
    missing = [q for q in quarters if q not in tried]
    if have is not None and not missing:
        return only(have)
    parts = [tuple(have[k] for k in FIELDS)] if have is not None else []
    resolved = set()
    for q in missing:
        try:
            part = _download_quarter(kepid, q, timeout)
        except QuarterUnavailable:
            continue  # left unresolved, so the next call asks for it again
        resolved.add(q)
        if part is not None:
            parts.append(part)
    if not parts:
        return None
    quarters_done = tried | resolved
    out = {name: np.concatenate([p[i] for p in parts]) for i, name in enumerate(FIELDS)}
    order = np.argsort(out["time"], kind="stable")
    out = {k: v[order] for k, v in out.items()}
    tmp = cache.with_suffix(".tmp.npz")  # written aside and renamed, so a reader never sees a half-written file
    np.savez_compressed(tmp, tried=np.array(sorted(quarters_done)), **out)
    tmp.replace(cache)
    return only(out)


def koi_views(row: dict, lc: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Global and local views folded on the catalogue ephemeris - how AstroNet built its training set - with the
    two sets of second-stage measurements."""
    out = koi_analysis(row, lc)
    return None if out is None else (out["g"], out["l"], out["aux"], out["ext"])


def koi_analysis(row: dict, lc: dict) -> dict | None:
    """Views plus the full diagnostics for one catalogued signal, folded on the catalogue ephemeris."""
    t, f = lc["time"], lc["flux"]
    period, t0, dur = row["koi_period"], row["koi_time0bk"], row["koi_duration"] / 24.0
    x = (t - t0 + 0.5 * period) % period - 0.5 * period
    in_tr = np.abs(x) < dur
    keep = np.ones(len(t), bool)
    keep_oot, _ = sigma_clip(f[~in_tr], 5.0, 101)  # never clip the transit itself
    keep[np.flatnonzero(~in_tr)[~keep_oot]] = False
    cents = [lc[c][keep] for c in ("centroid_col", "centroid_row")] if "centroid_col" in lc else None
    segment = lc["quarter"][keep] if "quarter" in lc else None
    t, f, x, in_tr = t[keep], f[keep], x[keep], in_tr[keep]
    if in_tr.sum() < 5 or len(t) < 500:
        return None
    trend, _ = savgol_detrend(t, f, 1.5, 3, transit_mask=in_tr)
    flat = f / trend
    phase, folded, _ = TransitAnalyticsEngine.phase_fold(t, flat, period, t0)
    q = float(np.clip(dur / period, 0.004, 0.15))
    g, l = global_view(phase, folded, 201), local_view(phase, folded, q, 61)
    if not (np.isfinite(g).all() and np.isfinite(l).all()) or l.min() >= np.median(l):
        return None
    stack = local_channels(t, flat, period, t0, q, 61)
    if not np.isfinite(stack).all():
        return None
    # the same diagnostics a blind run measures, for the second-stage classifier
    from .bls import fit_trapezoid
    from .stack import aux_from
    from .vetting import diagnose, extended
    depth0 = max(float(1.0 - np.median(flat[np.abs(x) < 0.25 * dur])) if (np.abs(x) < 0.25 * dur).any() else 1e-4, 1e-5)
    fit = fit_trapezoid(x, flat, depth0, dur, float(np.median(np.diff(t))))
    depth = fit["depth"] if fit["converged"] and fit["depth"] > 0 else depth0
    from .physics import StellarParameters, derive
    star = (row.get("koi_steff") or 5778.0, row.get("koi_srad") or 1.0, row.get("koi_smass") or 1.0)
    params = derive(StellarParameters(*star), period, depth)
    diag, _ = diagnose(t, flat, period, t0 + fit["dt0"], fit["t14"], depth, params, cents)
    aux = aux_from(depth, fit["v_shape"], diag, params["radius_earth"])
    ext = extended(t, f, flat, period, t0 + fit["dt0"], fit["t14"], fit["t12"], depth, diag, star, float(l.min()), segment)
    return {"g": normalize_view(g), "l": stack, "aux": np.array(aux, dtype=np.float32), "ext": np.array(ext, dtype=np.float32),
            "diag": diag, "depth": depth, "t14": fit["t14"], "params": params}


# Independent stretches of the mission: each gives the classifier a separate real observation of the same star.
VIEW_SUBSETS = ((1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12), (13, 14, 15, 16, 17), ALL_QUARTERS)


def _subset(lc: dict, quarters) -> dict | None:
    keep = np.isin(lc["quarter"], list(quarters))
    return {k: v[keep] for k, v in lc.items()} if keep.sum() >= 1000 else None


def _star_examples(args) -> list[tuple]:
    """Every example one star contributes (top level, so it can run in a worker process): each labelled signal on
    it, strongest first, with the transits of the stronger ones removed before a weaker one is measured - as a
    blind run removes each signal it finds before searching for the next."""
    signals, test = args
    try:
        with np.load(LC_DIR / f"{signals[0]['kepid']}.npz") as z:
            lc = {k: z[k] for k in FIELDS}
    except Exception:
        return []
    out = []
    removed = np.zeros(len(lc["time"]), bool)
    for rank, row in enumerate(signals):
        base = {k: v[~removed] for k, v in lc.items()}
        pix = pixels.cached(row["kepid"], row["kepoi_name"])
        for qs in ((ALL_QUARTERS,) if test else VIEW_SUBSETS):
            part = base if qs == ALL_QUARTERS else _subset(base, qs)
            try:
                v = koi_views(row, part) if part is not None else None
            except Exception:
                v = None
            if v is not None:
                v[3][-N_PIXEL:] = pix  # the same pixels whichever stretch of the light curve is used
                out.append((*v, row["label"], rank, qs == ALL_QUARTERS))
        x = (lc["time"] - row["koi_time0bk"] + 0.5 * row["koi_period"]) % row["koi_period"] - 0.5 * row["koi_period"]
        removed |= np.abs(x) < row["koi_duration"] / 24.0
    return out


def build_dataset(per_class: int = 200, quarters=ALL_QUARTERS, workers: int = 12,
                  progress: Callable[[str, float], None] | None = None) -> dict:
    """Download photometry for the sample and write the labelled examples to `dataset.npz`.

    Every labelled signal on a sampled star is used, not only its strongest. Held-out stars contribute one example
    per signal, built from every quarter available. Training stars contribute one example per stretch of the
    mission as well (and one from the whole record), so the network sees the same real signal under several
    independent noise realisations. A star's `rank` 0 example is its strongest signal, the one listed in the sample.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    report = progress or (lambda stage, frac: None)
    report("Querying the NASA Exoplanet Archive TCE table", 0.0)
    sample = select_sample(per_class)
    done = 0
    with ThreadPoolExecutor(workers) as pool:
        for fut in as_completed([pool.submit(download_star, r["kepid"], quarters) for r in sample]):
            done += 1
            if done % 50 == 0:
                report(f"Fetching Kepler photometry ({done}/{len(sample)} stars)", 0.5 * done / len(sample))
    on_star: dict[int, list[dict]] = {}
    for r in fetch_koi_table():
        if r["koi_period"] and r["koi_time0bk"] and r["koi_duration"]:
            on_star.setdefault(r["kepid"], []).append(r)
    jobs = []
    for r in sample:  # the sampled (strongest) signal first, then the star's other labelled signals
        others = sorted((o for o in on_star.get(r["kepid"], []) if o["kepoi_name"] != r["kepoi_name"]),
                        key=lambda o: -(o["koi_model_snr"] or 0))
        jobs.append(([r] + others, r["split"] == "test"))
    examples: dict[int, list] = {}
    with ProcessPoolExecutor(max(1, min(workers, (os.cpu_count() or 2) - 2))) as pool:
        for i, v in enumerate(pool.map(_star_examples, jobs, chunksize=8)):
            if v and any(e[5] == 0 for e in v):
                examples[i] = v
            if (i + 1) % 50 == 0:
                report(f"Building views from Kepler photometry ({i + 1}/{len(sample)} stars)", 0.5 + 0.5 * (i + 1) / len(sample))
    idx = sorted(examples)
    rows = [sample[i] for i in idx]
    flat = [(i, v) for i in idx for v in examples[i]]
    np.savez_compressed(DATASET, g=np.stack([v[0] for _, v in flat]), l=np.stack([v[1] for _, v in flat]),
                        aux=np.stack([v[2] for _, v in flat]), ext=np.stack([v[3] for _, v in flat]),
                        y=np.array([v[4] for _, v in flat]), rank=np.array([v[5] for _, v in flat]),
                        whole=np.array([v[6] for _, v in flat]),
                        test=np.array([sample[i]["split"] == "test" for i, _ in flat]),
                        kepid=np.array([sample[i]["kepid"] for i, _ in flat]))
    SAMPLE.write_text(json.dumps(rows))
    return dataset_summary()


def load_dataset() -> dict | None:
    if not DATASET.exists():
        return None
    with np.load(DATASET) as z:
        return {k: z[k] for k in z.files}


def load_sample() -> list[dict]:
    return json.loads(SAMPLE.read_text()) if SAMPLE.exists() else []


def dataset_summary() -> dict | None:
    d = load_dataset()
    if d is None:
        return None
    stars = np.unique(d["kepid"])
    first = np.unique(d["kepid"], return_index=True)[1]  # one row per star (its strongest signal) for the class counts
    y1, t1 = d["y"][first], d["test"][first]
    return {"stars": int(len(stars)), "train": int((~t1).sum()), "test": int(t1.sum()),
            "examples": int(len(d["y"])), "train_examples": int((~d["test"]).sum()),
            "classes": [{"name": n, "train": int(((y1 == c) & ~t1).sum()), "test": int(((y1 == c) & t1).sum())}
                        for c, n in enumerate(CLASS_NAMES)],
            "quarters": list(ALL_QUARTERS), "min_expected_snr": MIN_EXPECTED_SNR,
            "min_catalogue_snr": round(min_catalogue_snr(), 1),
            "source": "Kepler DR24 TCE table (Q1-Q17), Science Office labels PC / AFP / NTP updated to NASA's current KOI "
                      "dispositions, NASA Exoplanet Archive + MAST long cadence"}
