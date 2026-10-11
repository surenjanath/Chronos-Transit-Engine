import uuid

import numpy as np
from django.core.management.base import BaseCommand, CommandError

from cte import ingest
from engine import services
from engine.models import Run


class Command(BaseCommand):
    help = ("Run a Kepler star again with the transits of every other signal of an earlier run removed, so one of its "
            "further signals becomes the main one and gets the full analysis and plots. The new run has no pixel or "
            "centroid check: it is fed the trimmed light curve as time and brightness only.")

    def add_arguments(self, parser):
        parser.add_argument("run_id", help="a finished run of a Kepler dataset star (source: benchmark)")
        parser.add_argument("period", type=float, help="period in days of the further signal to keep")

    def handle(self, *args, run_id, period, **options):
        run = Run.objects.filter(id=run_id).first()
        if run is None or not run.results or run.source != "benchmark":
            raise CommandError("Give the id of a finished run of a Kepler dataset star.")
        res = run.results
        signals = [(res["bls"]["period"], res["bls"]["t0"], res["fit"]["t14_hours"])]
        signals += [(a["period_days"], a["t0"], a["duration_hours"]) for a in res.get("additional_signals", [])]
        others = [s for s in signals if abs(s[0] / period - 1) > 0.01]
        if len(others) == len(signals):
            raise CommandError(f"That run has no signal near {period} d. It has: " + ", ".join(f"{s[0]:.4f}" for s in signals))
        data = ingest.load_benchmark_star(int(run.config["kepid"]))
        keep = np.isfinite(data.time) & np.isfinite(data.flux)
        for p, t0, hours in others:  # drop each transit with a margin of half its duration either side
            keep &= np.abs((data.time - t0 + 0.5 * p) % p - 0.5 * p) > 0.75 * hours / 24.0 + 0.02
        err = data.flux_err if data.flux_err is not None else np.full(len(data.time), float(np.nanstd(data.flux)))
        upload_id = uuid.uuid4().hex
        ingest.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        np.savetxt(ingest.UPLOAD_DIR / f"{upload_id}.csv", np.c_[data.time, data.flux, err][keep], delimiter=",",
                   header="time,flux,flux_err", comments="", fmt="%.8f")
        stellar = {k: data.stellar.get(k) for k in ("t_eff_k", "r_star_solar", "m_star_solar")}
        new_id = services.create_run({"source": "upload", "upload_id": upload_id, "stellar_params": stellar,
                                      "target_name": f"{run.target.name} · known transits removed"})
        self.stdout.write(f"removed {len(others)} signal(s), kept {int(keep.sum())} of {len(keep)} cadences; run {new_id} started")
        services.execute_run(new_id)
        done = Run.objects.get(id=new_id)
        self.stdout.write(f"{done.status}: P = {done.period_days} d, {done.depth_ppm} ppm, P(planet) {done.p_planet} -> /runs/{new_id}/")
