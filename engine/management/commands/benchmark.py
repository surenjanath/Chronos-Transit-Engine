from django.core.management.base import BaseCommand

from engine import services


class Command(BaseCommand):
    help = "Run the full pipeline, blind, on held-out real Kepler stars and score it against the archive dispositions."

    def add_arguments(self, parser):
        parser.add_argument("--per-class", type=int, default=10)

    def handle(self, *args, per_class, **options):
        from django.conf import settings
        settings.CTE_RUN_INLINE = True
        n = services.start_benchmark(per_class)
        s = services.benchmark_summary()
        self.stderr.write(f"ran {n} stars; total scored {s['n']}; accuracy {s['accuracy']}; AUC {s['auc_planet']}; "
                          f"period exact {s['period_exact']}/{s['period_total']} (+{s['period_alias']} alias); "
                          f"planets called planet {s['planets_called_planet']}/{s['planets']}; FPs queued {s['others_queued']}/{s['others']}")
