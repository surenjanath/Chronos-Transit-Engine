import os

from django.core.management.base import BaseCommand, CommandError

from engine import services
from engine.models import ValidationRun


class Command(BaseCommand):
    help = "Run the PRD section 6 system verification suite."

    def add_arguments(self, parser):
        parser.add_argument("--offline", action="store_true", help="skip checks that need MAST")

    def handle(self, *args, offline=False, **options):
        val = ValidationRun.objects.create(status="running", progress="Starting", results={"pid": os.getpid()})
        services.run_validation_job(val.id, online=not offline)
        val.refresh_from_db()
        if val.status != "completed":
            raise CommandError(val.progress)
        for c in val.results["checks"]:
            self.stderr.write(f"{c['status'].upper():<8} [{c['group']:<8}] {c['parameter']:<26} {c['measured']}")
            for k in c["cases"]:
                if k.get("ok") is False:
                    self.stderr.write(f"           - not met: {k['label']} (measured {k.get('measured')})")
        if val.results["summary"]["failed"]:
            raise CommandError("one or more PRD acceptance criteria failed")
