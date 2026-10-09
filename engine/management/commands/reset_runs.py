import shutil

from django.core.management.base import BaseCommand

from cte.config import ARCHIVE_DIR, LOG_DIR
from engine.models import AutomationCycle, LogEvent, Run, Target, ValidationRun


class Command(BaseCommand):
    help = "Delete every run, target, log record and archived product (keeps the classifier and downloaded photometry)."

    def add_arguments(self, parser):
        parser.add_argument("--yes", action="store_true", help="confirm deletion")

    def handle(self, *args, yes, **options):
        if not yes:
            self.stderr.write("Refusing to delete without --yes")
            return
        for model in (LogEvent, Run, Target, ValidationRun):
            model.objects.all().delete()
        for cycle in AutomationCycle.objects.all():  # keep the cycle history, drop links to runs that no longer exist
            for r in (cycle.summary.get("targets") or {}).get("runs", []):
                r["run"] = None
            cycle.save(update_fields=["summary"])
        for d in (ARCHIVE_DIR, LOG_DIR):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
        self.stderr.write("Runs, targets, logs and archives cleared.")
