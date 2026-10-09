import json

from django.core.management.base import BaseCommand

from engine import services
from engine.models import Run

DEMO_RUNS = [
    {"source": "mast", "mast": {"target": "WASP-12", "mission": "TESS", "sequence": 20}},
    {"source": "mast", "mast": {"target": "TIC 261136679", "mission": "TESS", "sequence": 1}},
    {"source": "mast", "mast": {"target": "WASP-18", "mission": "TESS", "sequence": 2}},
    {"source": "mast", "mast": {"target": "Kepler-8", "mission": "Kepler", "sequence": 5}},
    {"source": "synthetic", "target_name": "SYN HD 209458 b analogue",
     "synthetic": {"kind": "planet", "period_days": 3.524748, "depth": 0.0146, "noise_ppm": 1000, "seed": 11}},
    {"source": "synthetic", "target_name": "SYN super-Earth",
     "synthetic": {"kind": "planet", "period_days": 6.2679, "depth": 0.00042, "noise_ppm": 130, "impact_param": 0.3, "seed": 12}},
    {"source": "synthetic", "target_name": "SYN warm Neptune",
     "synthetic": {"kind": "planet", "period_days": 9.4862, "depth": 0.0022, "noise_ppm": 500, "impact_param": 0.2, "seed": 13}},
    {"source": "synthetic", "target_name": "SYN eclipsing binary",
     "synthetic": {"kind": "eclipsing_binary", "period_days": 2.7412, "depth": 0.02, "noise_ppm": 800, "impact_param": 0.7, "seed": 14}},
    {"source": "synthetic", "target_name": "SYN grazing binary",
     "synthetic": {"kind": "eclipsing_binary", "period_days": 5.318, "depth": 0.05, "noise_ppm": 600, "impact_param": 1.1,
                   "sb_ratio": 0.15, "seed": 15}},
    {"source": "synthetic", "target_name": "SYN background blend",
     "synthetic": {"kind": "blend", "period_days": 4.1057, "depth": 0.03, "noise_ppm": 300, "seed": 16}},
    {"source": "synthetic", "target_name": "SYN quiet star", "synthetic": {"kind": "noise", "noise_ppm": 900, "seed": 17}},
    {"source": "synthetic", "target_name": "SYN active star",
     "synthetic": {"kind": "noise", "noise_ppm": 700, "variability_amp": 0.006, "seed": 18}},
]


class Command(BaseCommand):
    help = "Run a demonstration set of analyses (real MAST targets plus synthetic injections)."

    def add_arguments(self, parser):
        parser.add_argument("--offline", action="store_true", help="skip MAST targets")

    def handle(self, *args, offline=False, **options):
        for cfg in DEMO_RUNS:
            if offline and cfg["source"] == "mast":
                continue
            run_id = services.create_run(json.loads(json.dumps(cfg)))
            services.execute_run(run_id)
            run = Run.objects.select_related("target").get(id=run_id)
            self.stderr.write(f"{run.target.display:<28} {run.status:<10} {run.disposition or run.error}")
