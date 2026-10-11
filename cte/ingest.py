"""Light-curve ingestion: MAST archive (TESS / Kepler / K2), CSV upload, synthetic injection."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import numpy as np

from .config import MAST_CACHE_DIR, UPLOAD_DIR
from .synthetic import SyntheticSpec, synthesize

# TESS: 2-minute SPOC products, plus the TESS-SPOC light curves extracted from full-frame images
# K2: after the reaction-wheel failure the pipeline photometry carries strong pointing systematics, so the
# community's systematics-corrected light curves (EVEREST, K2SFF) are preferred when MAST has them
MISSION_AUTHORS = {"TESS": ("SPOC", "TESS-SPOC"), "Kepler": "Kepler", "K2": ("EVEREST", "K2SFF", "K2")}


@dataclass
class LightCurveData:
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray | None
    centroid_col: np.ndarray | None = None
    centroid_row: np.ndarray | None = None
    meta: dict = field(default_factory=dict)
    stellar: dict = field(default_factory=dict)
    truth: dict | None = None
    warnings: list[str] = field(default_factory=list)
    segment: np.ndarray | None = None  # observing quarter / sector of each sample, when several are stitched together


def _arr(col) -> np.ndarray:
    values = getattr(col, "value", col)
    return np.ma.filled(np.ma.asarray(values, dtype=float), np.nan)


def _num(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


MAST_TIMEOUT_S = 150


def _bounded(fn, what: str, seconds: float = MAST_TIMEOUT_S, attempts: int = 2):
    """Run a network call that has no timeout of its own; give up (and retry once) if the archive stalls."""
    import threading

    last: BaseException | None = None
    for _ in range(attempts):
        box: dict = {}

        def work():
            try:
                box["value"] = fn()
            except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread
                box["error"] = exc

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        thread.join(seconds)
        if "value" in box:
            return box["value"]
        last = box.get("error") or TimeoutError(f"{what} did not respond within {seconds:.0f} s")
        if not isinstance(last, (TimeoutError, OSError, ConnectionError)):
            break  # a real answer (e.g. target not found): retrying will not help
    raise last


def search_mast(target: str, mission: str = "TESS") -> list[dict]:
    """List light curves available at MAST for a target (pipeline-author products only)."""
    import lightkurve as lk

    author = MISSION_AUTHORS.get(mission)
    if author is None:
        raise ValueError(f"unsupported mission '{mission}'")
    author = list(author) if isinstance(author, tuple) else author
    result = _bounded(lambda: lk.search_lightcurve(target, mission=mission, author=author), "MAST search", 60)
    rows = []
    for i in range(len(result)):
        table = result.table[i]
        label = str(table["mission"])
        seq = re.search(r"(\d+)\s*$", label)
        rows.append({
            "label": label,
            "sequence": int(seq.group(1)) if seq else None,
            "author": str(table["author"]),
            "exptime": float(table["exptime"]),
            "target_name": str(table["target_name"]),
            "year": int(table["year"]) if "year" in table.colnames else None,
        })
    return rows


def query_tic(tic_id: int) -> dict:
    from astroquery.mast import Catalogs

    rows = Catalogs.query_criteria(catalog="TIC", ID=int(tic_id))
    if len(rows) == 0:
        return {}
    row = rows[0]
    return {"t_eff_k": _num(row["Teff"]), "r_star_solar": _num(row["rad"]), "m_star_solar": _num(row["mass"]),
            "v_mag": _num(row["Vmag"]), "ra": _num(row["ra"]), "dec": _num(row["dec"])}


def fetch_mast(target: str, mission: str = "TESS", sequence: int | None = None, exptime: float | None = None) -> LightCurveData:
    """Download one sector/quarter of PDCSAP photometry from MAST."""
    import lightkurve as lk

    author = MISSION_AUTHORS.get(mission)
    if author is None:
        raise ValueError(f"unsupported mission '{mission}'")
    kwargs = {"mission": mission, "author": list(author) if isinstance(author, tuple) else author}
    if sequence is not None:
        kwargs[{"TESS": "sector", "Kepler": "quarter", "K2": "campaign"}[mission]] = sequence
    if exptime:
        kwargs["exptime"] = exptime
    else:  # default to the standard survey cadence: 2 min for TESS, 30 min for Kepler / K2
        kwargs["exptime"] = 120 if mission == "TESS" else "long"
    result = _bounded(lambda: lk.search_lightcurve(target, **kwargs), "MAST search")
    if len(result) == 0 and "exptime" in kwargs and not exptime:
        kwargs.pop("exptime")
        result = _bounded(lambda: lk.search_lightcurve(target, **kwargs), "MAST search")
    if len(result) == 0:
        raise LookupError(f"MAST has no pipeline light curve for '{target}' ({mission}"
                          + (f", sequence {sequence}" if sequence is not None else "") + ")")
    pick = 0
    if isinstance(author, tuple) and not exptime:  # take the preferred author's product at the standard cadence
        rows = [(str(a), float(x)) for a, x in zip(result.table["author"], result.table["exptime"])]
        standard = kwargs.get("exptime") if isinstance(kwargs.get("exptime"), (int, float)) else (1800.0 if mission != "TESS" else 120.0)
        ranked = sorted(range(len(rows)), key=lambda i: (author.index(rows[i][0]) if rows[i][0] in author else 99,
                                                         abs(rows[i][1] - standard)))
        pick = ranked[0]
    lc = _bounded(lambda: result[pick].download(download_dir=str(MAST_CACHE_DIR), quality_bitmask="default"), "MAST download")
    time, flux, err = _arr(lc.time), _arr(lc.flux), _arr(lc.flux_err)
    if np.isfinite(err).mean() < 0.5 or np.nanmedian(err) <= 0:  # community products often carry no usable errors
        err = np.full(len(flux), 1.4826 * np.nanmedian(np.abs(np.diff(flux))) / np.sqrt(2.0))
    cols = lc.colnames
    ccol = _arr(lc["centroid_col"]) if "centroid_col" in cols else None
    crow = _arr(lc["centroid_row"]) if "centroid_row" in cols else None
    ok = np.isfinite(time) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
    if ok.sum() < 200:
        raise ValueError(f"only {int(ok.sum())} valid cadences in the MAST light curve")
    m = lc.meta
    warnings: list[str] = []
    stellar = {"t_eff_k": _num(m.get("TEFF")), "r_star_solar": _num(m.get("RADIUS")), "m_star_solar": None,
               "v_mag": _num(m.get("TESSMAG") or m.get("KEPMAG")), "ra": _num(m.get("RA_OBJ")), "dec": _num(m.get("DEC_OBJ"))}
    tic = m.get("TICID")
    if tic:
        try:
            for key, value in _bounded(lambda: query_tic(int(tic)), "TIC catalogue", 45, 1).items():
                if value is not None:
                    stellar[key] = value
        except Exception as exc:  # catalogue outage must not block photometry
            warnings.append(f"TIC catalogue query failed ({type(exc).__name__}); using FITS header stellar parameters")
    if stellar["m_star_solar"] is None and stellar["r_star_solar"]:
        logg = _num(m.get("LOGG"))
        if logg:
            stellar["m_star_solar"] = round(10 ** (logg - 4.438) * stellar["r_star_solar"] ** 2, 3)
            warnings.append("stellar mass derived from log g and radius")
        else:
            stellar["m_star_solar"] = round(stellar["r_star_solar"] ** 1.25, 3)
            warnings.append("stellar mass estimated from a main-sequence mass-radius relation")
    name = f"TIC {tic}" if tic else str(m.get("OBJECT") or target)
    if mission != "TESS" and m.get("KEPLERID"):
        name = f"{'KIC' if mission == 'Kepler' else 'EPIC'} {m['KEPLERID']}"
    meta = {
        "target_name": name, "query": target, "mission": mission,
        "kepid": int(m["KEPLERID"]) if mission == "Kepler" and m.get("KEPLERID") else None,
        "sector": int(m["SECTOR"]) if m.get("SECTOR") else (int(m["QUARTER"]) if m.get("QUARTER") is not None else
                                                             (int(m["CAMPAIGN"]) if m.get("CAMPAIGN") is not None else None)),
        "camera": int(m["CAMERA"]) if m.get("CAMERA") else None, "ccd": int(m["CCD"]) if m.get("CCD") else None,
        "product": str(result.table[pick]["productFilename"]) if "productFilename" in result.table.colnames else None,
        "author": str(result.table[pick]["author"]), "flux_column": "pdcsap_flux", "time_system": str(getattr(lc.time, "format", "bjd")),
    }
    return LightCurveData(time[ok], flux[ok], err[ok], ccol[ok] if ccol is not None else None,
                          crow[ok] if crow is not None else None, meta, stellar, None, warnings)


def make_synthetic(spec: dict, stellar: dict, name: str) -> LightCurveData:
    known = {k: v for k, v in spec.items() if k in SyntheticSpec.__dataclass_fields__ and v is not None}
    s = SyntheticSpec(**known)
    out = synthesize(s, stellar.get("r_star_solar") or 1.0, stellar.get("m_star_solar") or 1.0)
    meta = {"target_name": name, "mission": "Synthetic", "sector": None, "camera": None, "ccd": None,
            "flux_column": "synthetic", "time_system": "btjd", "seed": s.seed}
    return LightCurveData(out["time"], out["flux"], out["flux_err"], out["centroid_col"], out["centroid_row"],
                          meta, dict(stellar), out["truth"])


TIME_NAMES = ("time", "t", "bjd", "btjd", "bkjd", "jd", "date")
FLUX_NAMES = ("flux", "pdcsap_flux", "sap_flux", "f", "flux_raw")
ERR_NAMES = ("flux_err", "pdcsap_flux_err", "sap_flux_err", "err", "error", "flux_error", "e_flux")


def parse_csv(content: bytes) -> dict:
    """Parse an uploaded CSV with time and flux columns (optional flux error)."""
    text = content.decode("utf-8-sig", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if len(lines) < 50:
        raise ValueError("CSV needs a header row and at least 50 data rows")
    delim = "," if "," in lines[0] else ("\t" if "\t" in lines[0] else None)
    header = [h.strip().strip('"').lower() for h in (lines[0].split(delim) if delim else lines[0].split())]
    has_header = any(re.search(r"[a-z]", h) for h in header)
    data = np.genfromtxt(io.StringIO("\n".join(lines[1 if has_header else 0:])), delimiter=delim)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("CSV must have at least two numeric columns (time, flux)")

    def find(names, default):
        if has_header:
            for n in names:
                if n in header:
                    return header.index(n)
        return default

    ti, fi = find(TIME_NAMES, 0), find(FLUX_NAMES, 1)
    ei = find(ERR_NAMES, 2 if (not has_header and data.shape[1] > 2) else None)
    time, flux = data[:, ti], data[:, fi]
    err = data[:, ei] if ei is not None and ei < data.shape[1] else None
    ok = np.isfinite(time) & np.isfinite(flux)
    if err is not None:
        ok &= np.isfinite(err) & (err > 0)
    order = np.argsort(time[ok])
    if ok.sum() < 50:
        raise ValueError("fewer than 50 finite (time, flux) rows")
    return {"time": time[ok][order], "flux": flux[ok][order], "flux_err": err[ok][order] if err is not None else None}


def load_upload(upload_id: str, stellar: dict, name: str) -> LightCurveData:
    if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
        raise ValueError("invalid upload id")
    path = UPLOAD_DIR / f"{upload_id}.csv"
    if not path.exists():
        raise FileNotFoundError("uploaded file not found; upload it again")
    parsed = parse_csv(path.read_bytes())
    meta = {"target_name": name, "mission": "Upload", "sector": None, "camera": None, "ccd": None,
            "flux_column": "csv", "time_system": "user"}
    return LightCurveData(parsed["time"], parsed["flux"], parsed["flux_err"], None, None, meta, dict(stellar))


def load_benchmark_star(kepid: int) -> LightCurveData:
    """A held-out Kepler KOI star from the real dataset, with its catalogue values attached as ground truth."""
    from . import kepler
    from .config import CLASS_NAMES

    row = next((r for r in kepler.load_sample() if r["kepid"] == int(kepid)), None)
    if row is None:
        raise LookupError(f"KIC {kepid} is not in the Kepler dataset; build it with `manage.py build_kepler_dataset`")
    lc = kepler.download_star(int(kepid), kepler.ALL_QUARTERS)  # the whole four-year record
    if lc is None:
        raise LookupError(f"no Kepler photometry available for KIC {kepid}")
    order = np.argsort(lc["time"])
    stellar = {"t_eff_k": _num(row["koi_steff"]), "r_star_solar": _num(row["koi_srad"]), "m_star_solar": _num(row["koi_smass"]),
               "v_mag": _num(row["koi_kepmag"])}
    truth = {"kind": CLASS_NAMES[row["label"]], "label": row["label"], "koi": row["kepoi_name"],
             "catalogue_disposition": row["koi_disposition"], "period_days": row["koi_period"],
             "depth": row["koi_depth"] / 1e6 if row["koi_depth"] else None, "radius_earth": row["koi_prad"],
             "duration_hours": row["koi_duration"], "split": row["split"]}
    meta = {"target_name": f"KIC {kepid}", "alias": row.get("kepler_name") or row["kepoi_name"], "mission": "Kepler",
            "kepid": int(kepid), "sector": None, "camera": None, "ccd": None, "flux_column": "pdcsap_flux", "time_system": "bkjd",
            "quarters": sorted({int(q) for q in lc["quarter"]})}
    return LightCurveData(lc["time"][order], lc["flux"][order], lc["flux_err"][order], lc["centroid_col"][order],
                          lc["centroid_row"][order], meta, stellar, truth, segment=lc["quarter"][order])


def remove_transits(data: LightCurveData, signals: list[dict]) -> LightCurveData:
    """The same light curve with every transit of the given signals cut out (`period_days`, `t0`, `duration_hours`
    each), half a duration of margin either side, more for a deep eclipse. What is left is searched as if those signals were not there."""
    keep = np.ones(len(data.time), dtype=bool)
    for s in signals:
        p = float(s["period_days"])
        days = float(s["duration_hours"]) / 24.0
        # a deep eclipse has wings well outside its fitted duration, and what is left of them is found as a signal
        half = 1.5 * days + 0.06 if float(s.get("depth_ppm") or 0.0) > 2000.0 else 0.75 * days + 0.02
        keep &= np.abs((data.time - float(s["t0"]) + 0.5 * p) % p - 0.5 * p) > half

    def cut(a):
        return None if a is None else a[keep]

    meta = dict(data.meta, removed_signals=[{k: s[k] for k in ("period_days", "t0", "duration_hours")} for s in signals],
                removed_cadences=int((~keep).sum()))
    periods = ", ".join(f"{float(s['period_days']):.4f} d" for s in signals)
    return LightCurveData(data.time[keep], data.flux[keep], cut(data.flux_err), cut(data.centroid_col), cut(data.centroid_row),
                          meta, dict(data.stellar), None,
                          data.warnings + [f"transits of {len(signals)} other signal(s) removed before the search ({periods})"],
                          segment=cut(data.segment))
