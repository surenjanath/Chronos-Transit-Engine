import time

from django.core.management.base import BaseCommand

from engine import autopilot


class Command(BaseCommand):
    help = "Run autopilot cycles: pull new TOIs, grow the labelled dataset, retrain when warranted."

    def add_arguments(self, parser):
        parser.add_argument("--loop", action="store_true", help="keep running, one cycle per configured interval")

    def handle(self, *args, loop, **options):
        while True:
            cycle = autopilot.run_cycle("command")
            if cycle:
                s = cycle.summary
                self.stderr.write(f"cycle {cycle.id} {cycle.status}: {len(s.get('targets', {}).get('runs', []))} new runs, "
                                  f"{s.get('dataset', {}).get('added')} stars added, training {s.get('training')}")
            if not loop:
                return
            time.sleep(autopilot.config()["interval_minutes"] * 60)
