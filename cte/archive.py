"""Cross-match against the NASA Exoplanet Archive: what is already known about a target, and does the detection agree?"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
ALIASES = (("same period", 1.0), ("half the catalogued period", 2.0), ("twice the catalogued period", 0.5))


def _tap(query: str, timeout: float = 25.0) -> list[dict]:
    url = TAP_URL + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def known_objects(target_name: str) -> list[dict]:
    """Catalogued planets, TESS Objects of Interest or Kepler Objects of Interest for a TIC / KIC identifier."""
    m = re.fullmatch(r"(TIC|KIC)\s*(\d+)", target_name.strip())
    if not m:
        return []
    kind, num = m.group(1), int(m.group(2))
    out = []
    if kind == "TIC":
        for r in _tap(f"select pl_name,pl_orbper,pl_rade,pl_trandep from pscomppars where tic_id='TIC {num}'"):
            out.append({"name": r["pl_name"], "status": "Confirmed planet", "period_days": r["pl_orbper"], "radius_earth": r["pl_rade"],
                        "depth_ppm": r["pl_trandep"] * 1e4 if r["pl_trandep"] else None, "catalogue": "Planetary Systems"})
        labels = {"CP": "Confirmed planet", "KP": "Known planet", "PC": "Planet candidate", "APC": "Ambiguous candidate",
                  "FP": "False positive", "FA": "False alarm"}
        for r in _tap(f"select toi,tfopwg_disp,pl_orbper,pl_rade,pl_trandep from toi where tid={num}"):
            out.append({"name": f"TOI-{r['toi']}", "status": labels.get(r["tfopwg_disp"], r["tfopwg_disp"] or "Unclassified"),
                        "period_days": r["pl_orbper"], "radius_earth": r["pl_rade"], "depth_ppm": r["pl_trandep"],
                        "catalogue": "TESS Objects of Interest"})
    else:
        for r in _tap("select kepoi_name,kepler_name,koi_disposition,koi_period,koi_prad,koi_depth,koi_fpflag_ss,koi_fpflag_co,"
                      f"koi_fpflag_nt from cumulative where kepid={num}"):
            why = [w for flag, w in (("koi_fpflag_ss", "stellar eclipse"), ("koi_fpflag_co", "centroid offset"),
                                     ("koi_fpflag_nt", "not transit-like")) if r[flag]]
            status = r["koi_disposition"].capitalize() + (f" ({', '.join(why)})" if why else "")
            out.append({"name": r["kepler_name"] or r["kepoi_name"], "status": status, "period_days": r["koi_period"],
                        "radius_earth": r["koi_prad"], "depth_ppm": r["koi_depth"], "catalogue": "Kepler Objects of Interest"})
    return [o for o in out if o["period_days"]]


def crossmatch(target_name: str, period_days: float, radius_earth: float, depth_ppm: float) -> dict:
    """Find the catalogued object whose period (or its 1:2 / 2:1 alias) matches the detection."""
    objects = known_objects(target_name)
    best = None
    for o in objects:
        for label, factor in ALIASES:
            err = abs(period_days * factor - o["period_days"]) / o["period_days"]
            rank = (o["catalogue"] != "Planetary Systems", err)  # a confirmed-planet entry outranks its own TOI row
            if err < 0.01 and (best is None or rank < best["_rank"]):
                best = {**o, "_rank": rank, "alias": label, "period_rel_error": err,
                        "radius_rel_error": (radius_earth - o["radius_earth"]) / o["radius_earth"] if o["radius_earth"] else None,
                        "depth_rel_error": (depth_ppm - o["depth_ppm"]) / o["depth_ppm"] if o["depth_ppm"] else None}
    if best:
        best.pop("_rank")
    return {"objects": objects, "match": best}


def host_star(name: str) -> dict:
    """Stellar parameters for a named host from the Planetary Systems table (used when the photometry carries none)."""
    safe = name.replace("'", "''").strip()
    rows = _tap(f"select top 1 st_teff,st_rad,st_mass from pscomppars where hostname='{safe}' or hd_name='{safe}' "
                f"or tic_id='{safe}'")
    if not rows:
        return {}
    r = rows[0]
    return {k: v for k, v in (("t_eff_k", r["st_teff"]), ("r_star_solar", r["st_rad"]), ("m_star_solar", r["st_mass"])) if v}
