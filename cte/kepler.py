"""Real labelled data: the Kepler DR24 Threshold Crossing Event table (Q1-Q17) with MAST long-cadence photometry.

Labels are the Kepler Science Office training labels in the NASA Exoplanet Archive (`av_training_set`),
the same ground truth AstroNet was trained and scored on:
  PC   planet candidate / confirmed planet      -> PLANET_CANDIDATE
  AFP  astrophysical false positive (binaries)   -> ECLIPSING_BINARY
  NTP  non-transiting phenomenon (systematics)   -> STELLAR_OR_INSTRUMENT_NOISE
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
from .physics import TransitAnalyticsEngine, global_view, local_channels, local_view, normalize_view

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
        return json.loads(TABLE.read_text())
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
    return rows


def label_of(row: dict) -> int | None:
    return TCE_LABELS.get(row.get("tce_label"))


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


def koi_views(row: dict, lc: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Global and local views folded on the catalogue ephemeris - how AstroNet built its training set."""
    out = koi_analysis(row, lc)
    return None if out is None else (out["g"], out["l"], out["aux"])


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
    from .vetting import diagnose
    depth0 = max(float(1.0 - np.median(flat[np.abs(x) < 0.25 * dur])) if (np.abs(x) < 0.25 * dur).any() else 1e-4, 1e-5)
    fit = fit_trapezoid(x, flat, depth0, dur, float(np.median(np.diff(t))))
    depth = fit["depth"] if fit["converged"] and fit["depth"] > 0 else depth0
    from .physics import StellarParameters, derive
    params = derive(StellarParameters(row.get("koi_steff") or 5778.0, row.get("koi_srad") or 1.0, row.get("koi_smass") or 1.0),
                    period, depth)
    diag, _ = diagnose(t, flat, period, t0 + fit["dt0"], fit["t14"], depth, params, cents)
    aux = aux_from(depth, fit["v_shape"], diag, params["radius_earth"])
    return {"g": normalize_view(g), "l": stack, "aux": np.array(aux, dtype=np.float32), "diag": diag,
            "depth": depth, "t14": fit["t14"], "params": params}


# Independent stretches of the mission: each gives the classifier a separate real observation of the same star.
VIEW_SUBSETS = ((1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12), (13, 14, 15, 16, 17), ALL_QUARTERS)


def _subset(lc: dict, quarters) -> dict | None:
    keep = np.isin(lc["quarter"], list(quarters))
    return {k: v[keep] for k, v in lc.items()} if keep.sum() >= 1000 else None


def build_dataset(per_class: int = 200, quarters=ALL_QUARTERS, workers: int = 12,
                  progress: Callable[[str, float], None] | None = None) -> dict:
    """Download photometry for the sample and write the labelled view tensors to `dataset.npz`.

    Held-out stars contribute one example each, built from every quarter available. Training stars contribute
    one example per stretch of the mission (and one from the whole record), so the network sees the same real
    signal under several independent noise realisations. The sample itself is chosen as before, by
    detectability in six quarters, so every example shows a visible signal.
    """
    report = progress or (lambda stage, frac: None)
    report("Querying the NASA Exoplanet Archive TCE table", 0.0)
    sample = select_sample(per_class)
    examples: dict[int, list] = {}
    done = 0

    def work(i):
        lc = download_star(sample[i]["kepid"], quarters)
        if lc is None:
            return i, []
        subsets = (ALL_QUARTERS,) if sample[i]["split"] == "test" or len(quarters) < 12 else VIEW_SUBSETS
        out = []
        for qs in subsets:
            part = lc if qs == ALL_QUARTERS else _subset(lc, qs)
            v = koi_views(sample[i], part) if part is not None else None
            if v is not None:
                out.append(v)
        return i, out

    with ThreadPoolExecutor(workers) as pool:
        for fut in as_completed([pool.submit(work, i) for i in range(len(sample))]):
            try:
                i, v = fut.result()
                if v:
                    examples[i] = v
            except Exception:
                pass
            done += 1
            if done % 10 == 0:
                report(f"Building views from Kepler photometry ({done}/{len(sample)} stars)", done / len(sample))
    idx = sorted(examples)
    rows = [sample[i] for i in idx]
    flat = [(i, v) for i in idx for v in examples[i]]
    np.savez_compressed(DATASET, g=np.stack([v[0] for _, v in flat]), l=np.stack([v[1] for _, v in flat]),
                        aux=np.stack([v[2] for _, v in flat]),
                        y=np.array([sample[i]["label"] for i, _ in flat]),
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
    first = np.unique(d["kepid"], return_index=True)[1]  # one row per star for the class counts
    y1, t1 = d["y"][first], d["test"][first]
    return {"stars": int(len(stars)), "train": int((~t1).sum()), "test": int(t1.sum()),
            "examples": int(len(d["y"])), "train_examples": int((~d["test"]).sum()),
            "classes": [{"name": n, "train": int(((y1 == c) & ~t1).sum()), "test": int(((y1 == c) & t1).sum())}
                        for c, n in enumerate(CLASS_NAMES)],
            "quarters": list(ALL_QUARTERS), "min_expected_snr": MIN_EXPECTED_SNR,
            "min_catalogue_snr": round(min_catalogue_snr(), 1),
            "source": "Kepler DR24 TCE table (Q1-Q17), Science Office labels PC / AFP / NTP, NASA Exoplanet Archive + MAST long cadence"}
