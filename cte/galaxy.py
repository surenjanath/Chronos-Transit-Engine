"""Where the known planetary systems are, in real space.

Every confirmed planet in the NASA Exoplanet Archive (Planetary Systems Composite Parameters) that has a measured
distance is placed in Sun-centred galactic Cartesian coordinates:

    x  towards the Galactic centre        (l = 0,  b = 0)
    y  in the direction of Galactic rotation (l = 90, b = 0)
    z  towards the North Galactic Pole     (b = 90)

Distances are in light-years. Nothing here is simulated: a system without a distance in the archive is counted and
left off the map rather than given a made-up position.
"""
from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime, timezone

from .config import DATA_DIR

CATALOG_DIR = DATA_DIR / "catalog"
SYSTEMS = CATALOG_DIR / "systems.json"
LY_PER_PC = 3.2615638
AU_PER_LY = 63241.077
GALACTIC_CENTRE_LY = 26670.0   # 8.178 kpc (GRAVITY Collaboration 2019)
SOURCE = "NASA Exoplanet Archive · Planetary Systems Composite Parameters"

COLUMNS = ("pl_name,hostname,ra,dec,glon,glat,sy_dist,sy_pnum,sy_snum,pl_rade,pl_bmasse,pl_orbper,pl_orbsmax,pl_eqt,pl_insol,"
           "st_teff,st_rad,st_mass,st_spectype,disc_year,discoverymethod,disc_facility,sy_vmag,tic_id,hd_name,hip_name")

# J2000 equatorial -> galactic rotation (IAU 1958 system, Hipparcos values)
_TO_GALACTIC = ((-0.0548755604162154, -0.8734370902348850, -0.4838350155487132),
                (+0.4941094278755837, -0.4448296299600112, +0.7469822444972189),
                (-0.8676661490190047, -0.1980763734312015, +0.4559837761750669))


def equatorial_to_galactic(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    """(l, b) in degrees for a J2000 right ascension and declination."""
    ra, dec = math.radians(ra_deg), math.radians(dec_deg)
    v = (math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec))
    g = [sum(m * c for m, c in zip(row, v)) for row in _TO_GALACTIC]
    return math.degrees(math.atan2(g[1], g[0])) % 360.0, math.degrees(math.asin(max(-1.0, min(1.0, g[2]))))


def galactic_to_equatorial(l_deg: float, b_deg: float) -> tuple[float, float]:
    l, b = math.radians(l_deg), math.radians(b_deg)
    g = (math.cos(b) * math.cos(l), math.cos(b) * math.sin(l), math.sin(b))
    v = [sum(_TO_GALACTIC[r][c] * g[r] for r in range(3)) for c in range(3)]  # transpose = inverse
    return math.degrees(math.atan2(v[1], v[0])) % 360.0, math.degrees(math.asin(max(-1.0, min(1.0, v[2]))))


def galactic_xyz(l_deg: float, b_deg: float, distance_ly: float) -> tuple[float, float, float]:
    l, b = math.radians(l_deg), math.radians(b_deg)
    return (distance_ly * math.cos(b) * math.cos(l), distance_ly * math.cos(b) * math.sin(l), distance_ly * math.sin(b))


def _round(v, digits):
    return None if v is None else round(float(v), digits)


def build_systems(rows: list[dict]) -> dict:
    """Group archive planet rows by host star and place each host in space."""
    hosts: dict[str, dict] = {}
    unplaced: dict[str, int] = {}
    for r in rows:
        name = r["hostname"]
        if r.get("sy_dist") is None or r.get("ra") is None or r.get("dec") is None:
            unplaced[name] = unplaced.get(name, 0) + 1
            continue
        h = hosts.get(name)
        if h is None:
            d_ly = float(r["sy_dist"]) * LY_PER_PC
            if r.get("glon") is not None and r.get("glat") is not None:
                l, b = float(r["glon"]), float(r["glat"])
            else:
                l, b = equatorial_to_galactic(float(r["ra"]), float(r["dec"]))
            x, y, z = galactic_xyz(l, b, d_ly)
            h = hosts[name] = {
                "name": name, "x": round(x, 3), "y": round(y, 3), "z": round(z, 3), "d": round(d_ly, 3),
                "ra": round(float(r["ra"]), 5), "dec": round(float(r["dec"]), 5), "l": round(l, 4), "b": round(b, 4),
                "teff": _round(r.get("st_teff"), 0), "rad": _round(r.get("st_rad"), 3), "mass": _round(r.get("st_mass"), 3),
                "type": r.get("st_spectype"), "vmag": _round(r.get("sy_vmag"), 2), "stars": r.get("sy_snum"),
                "ids": [v for v in (r.get("tic_id"), r.get("hd_name"), r.get("hip_name")) if v], "planets": [],
            }
        h["planets"].append({
            "name": r["pl_name"], "r": _round(r.get("pl_rade"), 3), "m": _round(r.get("pl_bmasse"), 2),
            "P": _round(r.get("pl_orbper"), 5), "a": _round(r.get("pl_orbsmax"), 5), "T": _round(r.get("pl_eqt"), 0),
            "S": _round(r.get("pl_insol"), 2), "year": r.get("disc_year"), "method": r.get("discoverymethod"),
            "by": r.get("disc_facility"),
        })
    systems = sorted(hosts.values(), key=lambda h: h["d"])
    for h in systems:
        h["planets"].sort(key=lambda p: (p["P"] is None, p["P"] or 0.0))
    methods: dict[str, int] = {}
    for h in systems:
        for p in h["planets"]:
            methods[p["method"] or "Unknown"] = methods.get(p["method"] or "Unknown", 0) + 1
    return {
        "as_of": datetime.now(timezone.utc).isoformat(), "source": SOURCE,
        "n_planets": sum(len(h["planets"]) for h in systems), "n_systems": len(systems),
        "unplaced_planets": sum(unplaced.values()), "unplaced_systems": len(unplaced),
        "methods": dict(sorted(methods.items(), key=lambda kv: -kv[1])),
        "farthest_ly": systems[-1]["d"] if systems else 0.0, "nearest": systems[0]["name"] if systems else None,
        "systems": systems,
    }


def fetch_systems(timeout: float = 40.0) -> dict:
    """Download every confirmed planet from the archive and write the placed systems to disk."""
    from .archive import _tap
    rows = _tap(f"select {COLUMNS} from pscomppars", timeout)
    if len(rows) < 1000:
        raise RuntimeError(f"the archive returned only {len(rows)} planets; keeping the previous catalogue")
    out = build_systems(rows)
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SYSTEMS.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, separators=(",", ":")))
    tmp.replace(SYSTEMS)
    _cache.clear()
    return out


_cache: dict = {}


def load_systems(max_age_days: float | None = None, fetch: bool = True) -> dict | None:
    """The catalogue from disk; downloaded first if it is missing (or older than `max_age_days`).

    Returns None only when there is no copy on disk and the archive cannot be reached.
    """
    stale = not SYSTEMS.exists() or (max_age_days is not None and time.time() - SYSTEMS.stat().st_mtime > max_age_days * 86400)
    if stale and fetch:
        try:
            return fetch_systems()
        except Exception:
            if not SYSTEMS.exists():
                return None
    if not SYSTEMS.exists():
        return None
    mtime = SYSTEMS.stat().st_mtime
    if _cache.get("mtime") != mtime:
        _cache.update(mtime=mtime, data=json.loads(SYSTEMS.read_text()))
    return _cache["data"]


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def locate(names: list[str], ra: float | None = None, dec: float | None = None, catalogue: dict | None = None) -> dict | None:
    """Find a star in the catalogue by any of its names, or failing that by position (within 20 arcseconds)."""
    cat = catalogue or load_systems(fetch=False)
    if not cat:
        return None
    index = cat.get("_index")
    if index is None:
        index = {}
        for h in cat["systems"]:
            for n in [h["name"], *h["ids"]]:
                index.setdefault(_key(n), h)
        cat["_index"] = index
    for n in names:
        hit = index.get(_key(n))
        if hit:
            return hit
    if ra is None or dec is None:
        return None
    best, best_sep = None, 20.0 / 3600.0
    cos_dec = math.cos(math.radians(dec))
    for h in cat["systems"]:
        if abs(h["dec"] - dec) > best_sep:
            continue
        d_ra = (h["ra"] - ra + 180.0) % 360.0 - 180.0
        sep = math.hypot(d_ra * cos_dec, h["dec"] - dec)
        if sep < best_sep:
            best, best_sep = h, sep
    return best


def travel(distance_ly: float, now_year: int | None = None) -> dict:
    """What a distance means in travel time: for light, and for the fastest things people have built."""
    year = now_year or datetime.now(timezone.utc).year
    c_km_s = 299792.458
    return {
        "light_years": distance_ly, "parsecs": distance_ly / LY_PER_PC, "au": distance_ly * AU_PER_LY,
        "km": distance_ly * 9.4607304725808e12,
        "light_left_year": year - distance_ly,                     # the light arriving tonight left then
        "voyager1_years": distance_ly * c_km_s / 16.95,            # Voyager 1, 16.95 km/s relative to the Sun
        "parker_years": distance_ly * c_km_s / 192.0,              # Parker Solar Probe at perihelion, 192 km/s
        "jet_years": distance_ly * c_km_s / 0.25,                  # an airliner at 900 km/h
    }
