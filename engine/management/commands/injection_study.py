import sys

from django.core.management.base import BaseCommand

from cte import injection


class Command(BaseCommand):
    help = "Inject Mandel-Agol transits into real quiet Kepler stars and measure pipeline recovery against S/N."

    def add_arguments(self, parser):
        parser.add_argument("--trials", type=int, default=8, help="injections per S/N bin")
        parser.add_argument("--workers", type=int, default=3)

    def handle(self, *args, trials, workers, **options):
        out = injection.run_study(trials, workers, progress=lambda m: print(f"\r{m:<40}", end="", file=sys.stderr))
        self.stderr.write("")
        for b in out["bins"]:
            self.stderr.write(f"S/N {b['snr']:>5}: detected {b['detected']}/{b['n']}  (period found {b['period_recovered']}, called planet {b['planet']})")
        self.stderr.write(f"engine {out['engine']}; recovery below 4.5: {out['recovery_below_4_5']}, at 7.1: {out['recovery_at_7_1']}, from 10: {out['recovery_from_10']}")
