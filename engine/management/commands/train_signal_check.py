import json

from django.core.management.base import BaseCommand, CommandError

from cte import signalcheck


class Command(BaseCommand):
    help = "Fit the signal-check model (catalogue parameters only, no light curve) on the Kepler dataset."

    def handle(self, *args, **options):
        def report(message, fraction):
            self.stdout.write(f"{message:60s} {fraction * 100:5.1f}%")

        try:
            metrics = signalcheck.train(report)
        except RuntimeError as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(json.dumps({k: v for k, v in metrics.items() if k != "features"}, indent=1))
