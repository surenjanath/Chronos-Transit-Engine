from django.core.management.base import BaseCommand, CommandError

from engine import services
from engine.models import Run


class Command(BaseCommand):
    help = ("Run a Kepler star again with the transits of every other signal of an earlier run removed, so one of its "
            "further signals becomes the main one and gets the full analysis, pixel check included. The Research page does "
            "the same from the browser.")

    def add_arguments(self, parser):
        parser.add_argument("run_id", help="a finished run of a Kepler dataset star (source: benchmark)")
        parser.add_argument("period", type=float, help="period in days of the further signal to keep")

    def handle(self, *args, run_id, period, **options):
        run = Run.objects.filter(id=run_id).first()
        if run is None or not run.results or run.source not in ("benchmark", "research"):
            raise CommandError("Give the id of a finished run of a Kepler dataset star.")
        res = run.results
        signals = [(res["bls"]["period"], res["bls"]["t0"], res["fit"]["t14_hours"])]
        signals += [(a["period_days"], a["t0"], a["duration_hours"]) for a in res.get("additional_signals", [])]
        others = [s for s in signals if abs(s[0] / period - 1) > 0.01]
        if len(others) == len(signals):
            raise CommandError(f"That run has no signal near {period} d. It has: " + ", ".join(f"{s[0]:.4f}" for s in signals))
        remove = [{"period_days": p, "t0": t0, "duration_hours": hours} for p, t0, hours in others]
        new_id = services.create_run({"source": "research", "kepid": int(run.config["kepid"]), "remove": remove, "focus_period": period})
        self.stdout.write(f"removing {len(others)} signal(s); run {new_id} started")
        services.execute_run(new_id)
        done = Run.objects.get(id=new_id)
        self.stdout.write(f"{done.status}: P = {done.period_days} d, {done.depth_ppm} ppm, P(planet) {done.p_planet} -> /runs/{new_id}/")
