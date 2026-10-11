import sys

from django.core.management.base import BaseCommand

from cte import survey


class Command(BaseCommand):
    help = ("Run the pipeline blind on the Kepler stars on disk and list every signal that no NASA table contains. "
            "Resumes where it stopped; results in data/survey.jsonl and data/survey_unlisted.csv.")

    def add_arguments(self, parser):
        parser.add_argument("--kepids", type=str, default="", help="comma-separated KIC numbers (default: every star of the dataset)")
        parser.add_argument("--limit", type=int, default=None, help="survey at most this many stars not yet done")
        parser.add_argument("--workers", type=int, default=4, help="stars analysed side by side")

    def handle(self, *args, kepids, limit, workers, **options):
        ids = [int(k) for k in kepids.split(",") if k.strip()] or None
        out = survey.run(ids, workers, limit, lambda m, f: print(f"\r{m:<40} {f * 100:5.1f}%", end="", file=sys.stderr))
        self.stderr.write("")
        self.stdout.write(f"stars {out['stars']} (errors {out['errors']}) | signals {out['signals']} | not in any NASA table: {out['unlisted']} | "
                          f"of those called a planet: {out['unlisted_called_planet']} | in the vetting queue (P >= 0.9): {out['unlisted_in_vetting_queue']}")
        for s in out["top"]:
            self.stdout.write(f"  KIC {s['kepid']}  P = {s['period_days']:.4f} d  {s['depth_ppm']:.0f} ppm  S/N {s['snr']:.1f}  "
                              f"P(planet) {s['p_planet']:.3f}  signal {s['rank']} of its star")
        self.stdout.write(f"full table: {out['table']}")
