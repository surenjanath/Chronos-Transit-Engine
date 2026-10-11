import json
import os
import pathlib
import subprocess
import zipfile
import io

from django.core.management.base import BaseCommand, CommandError

BASE_DIR = pathlib.Path(__file__).resolve().parents[3]
from engine import research
from engine.models import Run


class Command(BaseCommand):
    help = ("Statistical validation of a finished Kepler run with TRICERATOPS: the false-positive probability, kept with the "
            "run and shown in its report. TRICERATOPS runs in an environment of its own (see tools/triceratops_validate.py); "
            "CTE_TRICERATOPS_PYTHON names its interpreter, default data/validation/venv/bin/python.")

    def add_arguments(self, parser):
        parser.add_argument("run_id")
        parser.add_argument("--runs", type=int, default=5)
        parser.add_argument("--quarters", default="4,5,6,7")

    def handle(self, *args, run_id, runs, quarters, **options):
        run = Run.objects.filter(id=run_id, status="completed").first()
        if run is None:
            raise CommandError("Give the id of a finished run.")
        python = pathlib.Path(os.environ.get("CTE_TRICERATOPS_PYTHON") or research.VALIDATION_DIR / "venv" / "bin" / "python")
        if not python.exists():
            raise CommandError(f"No TRICERATOPS environment at {python}. Make one: python3.11 -m venv data/validation/venv && "
                               'data/validation/venv/bin/pip install triceratops "setuptools<81"')
        folder = research.VALIDATION_DIR / run.id
        folder.mkdir(parents=True, exist_ok=True)
        zipfile.ZipFile(io.BytesIO(research.packet(run))).extractall(folder)
        subprocess.run([str(python), "-u", str(BASE_DIR / "tools" / "triceratops_validate.py"), str(folder), "--runs", str(runs),
                        "--quarters", quarters], check=True)
        result = json.loads((folder / "validation.json").read_text())
        (research.VALIDATION_DIR / f"{run.id}.json").write_text(json.dumps(result, indent=1))
        self.stdout.write(f"FPP {result['fpp']:.4g} ± {result['fpp_std']:.2g}, NFPP {result['nfpp']:.4g}: {result['class']}")
