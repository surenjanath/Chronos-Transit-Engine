import sys

from django.core.management.base import BaseCommand, CommandError

from cte import model


class Command(BaseCommand):
    help = "Train the dual-branch CNN and activate the checkpoint if it meets the AUC floor."

    def handle(self, *args, **options):
        metrics = model.train(progress=lambda stage, frac: print(f"\r{stage:<40} {frac * 100:5.1f}%", end="", file=sys.stderr))
        self.stderr.write(f"\nstatus={metrics['status']} auc_planet={metrics['auc_planet']} accuracy={metrics['accuracy']}")
        if metrics["status"] != "active":
            raise CommandError("checkpoint quarantined: AUC below the acceptance threshold")
