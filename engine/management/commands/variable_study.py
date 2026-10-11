import json

from django.core.management.base import BaseCommand, CommandError

from cte import variable_study
from cte.config import DATA_DIR


class Command(BaseCommand):
    help = "Inject planets into variable Kepler stars and measure how many the full pipeline finds, and what it finds when nothing is injected."

    def add_arguments(self, parser):
        parser.add_argument("--trials", type=int, default=48)
        parser.add_argument("--seed", type=int, default=11, help="the same seed repeats the same stars and planets; a new one is a fresh test")
        parser.add_argument("--workers", type=int, default=4)
        parser.add_argument("--no-injection", action="store_true", help="run the same stars with nothing injected, to count false signals")

    def handle(self, *args, trials, seed, workers, no_injection, **options):
        try:
            out = variable_study.run(trials, seed, not no_injection, workers)
        except RuntimeError as exc:
            raise CommandError(str(exc)) from exc
        path = DATA_DIR / f"variable_study_{seed}{'_noinj' if no_injection else ''}.json"
        path.write_text(json.dumps(out, indent=1))
        if no_injection:
            self.stdout.write(f"stars {out['stars']} | given an extra signal: {out['stars_with_extra_signal']} | one the classifier calls a planet: {out['stars_with_extra_planet']}")
        else:
            self.stdout.write(f"trials {out['trials']} | found {out['found']} | detectable in principle {out['detectable']} | "
                              f"found among those {out['found_detectable']} | also called a planet {out['called_planet_detectable']}")
            for m in out["misses"]:
                self.stdout.write(f"  missed: KIC {m['kepid']}  P = {m['inj_P']} d  {m['inj_depth_ppm']} ppm  S/N against the star {m['oracle_snr']}")
        self.stdout.write(f"details: {path}")
