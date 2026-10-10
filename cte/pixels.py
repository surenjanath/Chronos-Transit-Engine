"""Pixel-level check: where on the detector does the dip come from?

A background eclipsing binary blended into the target's light produces a dip that looks like a small planet's. The
light curve cannot tell them apart; the pixels can. For each Kepler quarter the mean image in transit is subtracted
from the mean image just outside it. That difference image shows the source of the dip, and its centroid is compared
with the centroid of the out-of-transit image, which is the target. Target pixel files are read from MAST and
discarded; only the per-quarter measurements are cached.
"""
from __future__ import annotations

import gzip
import io
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .config import DATA_DIR

PIX_DIR = DATA_DIR / "kepler" / "pix"
URL = "https://archive.stsci.edu/missions/kepler/target_pixel_files/{p}/{k}/kplr{k}-{stamp}_lpd-targ.fits.gz"
QUARTERS = (3, 4, 5, 6, 7, 8, 9, 10)  # two of each observing season
NAMES = ["pix_offset", "pix_sigma", "pix_peak_dist", "pix_quarters"]


def fetch(kepid: int, quarter: int, timeout: float = 40.0):
    """(time, flux cube, aperture mask) for one quarter, or None when the star was not observed or MAST did not answer."""
    from astropy.io import fits
    from .kepler import QUARTER_STAMPS

    k = f"{kepid:09d}"
    url = URL.format(p=k[:4], k=k, stamp=QUARTER_STAMPS[quarter])
    for _ in range(2):
        try:
            raw = urllib.request.urlopen(url, timeout=timeout).read()
            with fits.open(io.BytesIO(gzip.decompress(raw))) as h:
                d = h[1].data
                t, flux, quality = np.asarray(d["TIME"], float), np.asarray(d["FLUX"], float), np.asarray(d["QUALITY"])
                aperture = np.asarray(h[2].data)
            ok = np.isfinite(t) & (quality == 0) & np.isfinite(flux).any(axis=(1, 2))
            return t[ok], flux[ok], aperture
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
        except Exception:
            time.sleep(1.5)
    return None


def _centroid(img, mask, clip=True):
    w = np.where(mask & np.isfinite(img), img, 0.0)
    if clip:
        w = np.clip(w, 0, None)
    total = w.sum()
    if total <= 0:
        return None
    yy, xx = np.indices(img.shape)
    return float((w * xx).sum() / total), float((w * yy).sum() / total)


def measure(t, flux, aperture, period: float, t0: float, duration: float):
    """One quarter: (dx, dy) in pixels between the difference-image centroid and the out-of-transit centroid, a weight
    for the quarter, and the distance between the brightest pixel of each image. None when no transit was covered."""
    x = (t - t0 + 0.5 * period) % period - 0.5 * period
    inside = np.abs(x) < 0.35 * duration
    near = (np.abs(x) > 0.75 * duration) & (np.abs(x) < 2.5 * duration)
    epoch = np.round((t - t0) / period).astype(int)
    diffs, outs = [], []
    with np.errstate(all="ignore"):
        for e in np.unique(epoch[inside]):  # per transit: local out-of-transit minus in-transit, so slow drifts cancel
            a, b = inside & (epoch == e), near & (epoch == e)
            if a.sum() >= 2 and b.sum() >= 4:
                diffs.append(np.nanmean(flux[b], 0) - np.nanmean(flux[a], 0))
                outs.append(np.nanmean(flux[b], 0))
        if not diffs:
            return None
        diff, out = np.nanmean(diffs, 0), np.nanmean(outs, 0)
        valid = np.isfinite(out) & (aperture > 0)
        noise = np.nanstd(np.diff(flux[near], axis=0), axis=0) / np.sqrt(2) / np.sqrt(max(inside.sum(), 1))
        weight = float(np.nansum(np.where(valid, diff, 0)) / np.sqrt(np.nansum(np.where(valid, noise, 0) ** 2) + 1e-12))
    c_out = _centroid(out, valid)
    c_diff = _centroid(diff, valid, clip=False)  # signed: clipping the noise would pull the centroid to the centre
    if c_out is None or c_diff is None:
        return None
    peak_out = np.unravel_index(np.nanargmax(np.where(valid, out, -np.inf)), out.shape)
    peak_diff = np.unravel_index(np.nanargmax(np.where(valid, diff, -np.inf)), diff.shape)
    return (c_diff[0] - c_out[0], c_diff[1] - c_out[1], weight,
            float(np.hypot(peak_out[0] - peak_diff[0], peak_out[1] - peak_diff[1])), len(diffs))


def summarise(per_quarter: list) -> list[float]:
    """Across quarters, in the order of `NAMES`: mean offset in pixels, its significance from the quarter-to-quarter
    scatter, the median distance between the two brightest pixels, and how many quarters went in. NaN when unmeasured."""
    nan = float("nan")
    if not per_quarter:
        return [nan, nan, nan, 0.0]
    a = np.array(per_quarter, dtype=float)[:, -5:]  # dx, dy, weight, peak distance, transits
    w = np.clip(a[:, 2], 0.5, None) ** 2
    dx, dy = np.average(a[:, 0], weights=w), np.average(a[:, 1], weights=w)
    offset = float(np.hypot(dx, dy))
    sigma = nan
    if len(a) >= 3:
        ex = np.sqrt(np.average((a[:, 0] - dx) ** 2, weights=w) / (len(a) - 1))
        ey = np.sqrt(np.average((a[:, 1] - dy) ** 2, weights=w) / (len(a) - 1))
        sigma = offset / max(float(np.hypot(ex, ey)), 1e-3)
    return [offset, sigma, float(np.median(a[:, 3])), float(len(a))]


def cached(kepid: int, signal: str) -> list[float]:
    """Summary for a catalogued signal whose pixels were measured when the dataset was assembled."""
    path = PIX_DIR / f"{kepid}.json"
    if not path.exists():
        return summarise([])
    try:
        return summarise(json.loads(path.read_text()).get(signal) or [])
    except (json.JSONDecodeError, ValueError):
        return summarise([])


def measure_star(kepid: int, period: float, t0: float, duration: float, quarters=QUARTERS, budget_s: float = 90.0) -> list[float]:
    """Fetch the pixels and measure one signal (a blind run). Whatever MAST does not deliver in time is left out."""
    started = time.time()

    def one(q):
        if time.time() - started > budget_s:
            return None
        got = fetch(kepid, q, timeout=25.0)
        return None if got is None else measure(*got, period, t0, duration)

    with ThreadPoolExecutor(len(quarters)) as pool:
        return summarise([m for m in pool.map(one, quarters) if m])
