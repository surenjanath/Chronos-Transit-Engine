"""Statistical validation of one signal with TRICERATOPS (Giacalone et al. 2021), from a Chronos data packet.

    python tools/triceratops_validate.py PACKET_DIR [--quarters 4,5,6,7] [--runs 5] [--draws 1000000]

PACKET_DIR is an unzipped data packet (summary.json and folded_transit.csv). TRICERATOPS weighs the transit against
every way a star in or near the aperture could imitate it and returns the false-positive probability (FPP) and the
part of it owed to known nearby stars (NFPP). The result is written to PACKET_DIR/validation.json.

TRICERATOPS has its own heavy dependencies; install it in an environment of its own (python 3.11: `pip install
triceratops "setuptools<81"`) and run this script with that interpreter. It needs the network: TIC, Gaia and Kepler
target pixel files are read from MAST and VizieR.
"""
import argparse
import json
import pathlib
import sys
import time
import warnings

import numpy as np

CADENCE_DAYS = 0.0204336  # Kepler long cadence


def binned(x, y, width):
    edges = np.arange(x.min(), x.max() + width, width)
    i = np.digitize(x, edges) - 1
    keep = [k for k in range(len(edges) - 1) if (i == k).sum() >= 5]
    return (np.array([x[i == k].mean() for k in keep]), np.array([y[i == k].mean() for k in keep]),
            np.array([y[i == k].std() / np.sqrt((i == k).sum()) for k in keep]))


def apertures(kepid, quarters, cache):
    """The pipeline's optimal aperture of each quarter as (column, row) pixel pairs, as TRICERATOPS expects them."""
    import lightkurve
    out = []
    for q in quarters:
        tpf = lightkurve.search_targetpixelfile(f"KIC {kepid}", mission="Kepler", quarter=int(q)).download_all(download_dir=cache)[0]
        rows, cols = np.nonzero(tpf.pipeline_mask)
        out.append(np.c_[cols + tpf.column, rows + tpf.row])
    return out


GAIA_TAP = ("https://gaia.ari.uni-heidelberg.de/tap/sync", "https://gea.esac.esa.int/tap-server/tap/sync")


def direct_gaia():
    """TRICERATOPS reads its field stars through astroquery's Gaia job queue, which can wait for ever on a busy
    archive. Answer the same queries with one plain request to a Gaia DR3 server instead, with a time limit."""
    import io

    import requests
    from astropy.table import Table
    from astroquery.gaia import Gaia

    class Done:
        def __init__(self, table):
            self.table = table

        def get_results(self):
            return self.table

    def job(query, **_):
        last = None
        for url in GAIA_TAP:
            try:
                r = requests.post(url, data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}, timeout=240)
                r.raise_for_status()
                return Done(Table.read(io.StringIO(r.text), format="ascii.csv"))
            except Exception as exc:
                last = exc
        raise RuntimeError(f"no Gaia DR3 server answered: {last}")

    Gaia.launch_job = Gaia.launch_job_async = job


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("packet")
    ap.add_argument("--quarters", default="4,5,6,7", help="Kepler quarters whose pixels and apertures are used (one per season by default)")
    ap.add_argument("--runs", type=int, default=5, help="independent repeats; the FPP is a Monte Carlo estimate and varies between them")
    ap.add_argument("--draws", type=int, default=1000000, help="simulated systems per scenario")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    import triceratops
    import triceratops.triceratops as tr
    direct_gaia()

    folder = pathlib.Path(args.packet)
    info = json.loads((folder / "summary.json").read_text())
    if info["target"]["mission"] != "Kepler" or not info["target"].get("kepid"):
        sys.exit("This script handles Kepler targets (a packet whose summary carries a kepid).")
    data = np.genfromtxt(folder / "folded_transit.csv", delimiter=",", names=True)
    x, y, e = binned(data["time_days"], data["flux"], CADENCE_DAYS)
    outside = np.abs(x) > info["transit"]["duration_hours"] / 24.0
    error = float(np.std(y[outside]))
    quarters = np.array([int(q) for q in args.quarters.split(",")])
    kepid, depth, period = int(info["target"]["kepid"]), info["transit"]["depth_ppm"] * 1e-6, info["ephemeris"]["period_days"]
    cache = str(folder / "mast")
    print(f"KIC {kepid}: {len(x)} binned points, error {error * 1e6:.0f} ppm, depth {depth * 1e6:.0f} ppm, P = {period:.5f} d", flush=True)

    started = time.time()
    target = tr.target(ID=kepid, sectors=quarters, mission="Kepler", lightkurve_cache_dir=cache)
    aps = apertures(kepid, quarters, cache)
    target.calc_depths(tdepth=depth, all_ap_pixels=aps)
    stars = target.stars
    nearby = stars[stars["tdepth"] > 0] if "tdepth" in stars else stars
    print(f"{len(stars)} catalogued stars within the search radius; {max(len(nearby) - 1, 0)} besides the target could produce the dip", flush=True)

    runs, table = [], None
    for n in range(args.runs):
        target.calc_probs(time=x, flux_0=y, flux_err_0=error, P_orb=period, N=args.draws, exptime=CADENCE_DAYS, nsamples=10, verbose=0)
        runs.append({"FPP": float(target.FPP), "NFPP": float(target.NFPP)})
        probs = target.probs[["ID", "scenario", "prob"]].copy()
        table = probs if table is None else table.assign(prob=table["prob"] + probs["prob"])
        print(f"run {n + 1}/{args.runs}: FPP {target.FPP:.4g}  NFPP {target.NFPP:.4g}  ({time.time() - started:.0f} s)", flush=True)
    table["prob"] = table["prob"] / len(runs)
    by_scenario = table.groupby("scenario")["prob"].sum().sort_values(ascending=False)
    fpp, nfpp = np.array([r["FPP"] for r in runs]), np.array([r["NFPP"] for r in runs])
    siblings = [o for o in info["catalogue"]["objects_on_star"] if str(o.get("status", "")).lower() in ("confirmed", "candidate")]
    boost = 0 if not siblings else (25 if len(siblings) == 1 else 50)  # Lissauer et al. 2012, as applied by Rowe et al. 2014
    out = {
        "tool": f"TRICERATOPS {getattr(triceratops, '__version__', '')}".strip(), "reference": "Giacalone et al. 2021, AJ 161, 24",
        "fpp": float(fpp.mean()), "fpp_std": float(fpp.std()), "nfpp": float(nfpp.mean()), "nfpp_std": float(nfpp.std()), "runs": runs,
        "draws_per_scenario": args.draws, "quarters": quarters.tolist(), "binned_points": int(len(x)), "flux_error_ppm": error * 1e6,
        "stars_in_field": int(len(stars)), "stars_able_to_produce_dip": int(max(len(nearby) - 1, 0)),
        "scenarios": [{"scenario": k, "probability": float(v)} for k, v in by_scenario.items() if v > 1e-6],
        "siblings": len(siblings), "multiplicity_factor": boost,
        "fpp_with_multiplicity": float(fpp.mean() / boost) if boost else None,
        "thresholds": {"validated": "FPP < 0.015 and NFPP < 0.001", "likely_planet": "FPP < 0.5 and NFPP < 0.001",
                       "likely_nearby_false_positive": "NFPP > 0.1"},
        "no_high_resolution_imaging": True, "seconds": round(time.time() - started),
    }
    f, nf = out["fpp"], out["nfpp"]
    out["class"] = ("likely nearby false positive" if nf > 0.1 else "validated planet" if f < 0.015 and nf < 0.001
                    else "likely planet" if f < 0.5 and nf < 0.001 else "not validated")
    (folder / "validation.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("runs", "scenarios")}, indent=1))


if __name__ == "__main__":
    main()
