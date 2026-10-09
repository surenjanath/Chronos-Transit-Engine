import sys

from django.core.management.base import BaseCommand

from cte import kepler


class Command(BaseCommand):
    help = "Download labelled Kepler KOI photometry from MAST and build the real training / test dataset."

    def add_arguments(self, parser):
        parser.add_argument("--per-class", type=int, default=200, help="stars per class (planet, binary, noise)")
        parser.add_argument("--workers", type=int, default=6)

    def handle(self, *args, per_class, workers, **options):
        summary = kepler.build_dataset(per_class, workers=workers,
                                       progress=lambda stage, frac: print(f"\r{stage:<70} {frac * 100:5.1f}%", end="", file=sys.stderr))
        self.stderr.write(f"\n{summary}")
