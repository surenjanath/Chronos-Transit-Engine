from django.urls import reverse

from cte import model

from . import autopilot, services
from .models import Run, Target

NAV = [("overview", "Overview"), ("targets", "Targets"), ("runs", "Runs"), ("vetting", "Vetting"), ("research", "Research"), ("map", "Map"),
       ("new", "New analysis"), ("signal_check", "Signal check"),
       ("atlas", "Atlas"), ("benchmark", "Benchmark"), ("automation", "Automation"), ("validation", "Validation"), ("logs", "Audit log"),
       ("calculator", "Calculator"), ("settings", "Settings"), ("method", "Method")]
MORE = {"atlas", "benchmark", "automation", "validation", "logs", "calculator", "settings", "method"}  # kept under "More" in the header
PAGE_OF = {"target": "targets", "run": "runs", "research_case": "research"}


def shell(request):
    """Shared shell context: navigation, live status and health marks."""
    services.startup()
    autopilot.start_scheduler()
    name = getattr(request.resolver_match, "url_name", "") or ""
    page = PAGE_OF.get(name, name)
    keys = [k for k, _ in NAV]
    i = keys.index(page) if page in keys else 0
    live = Run.objects.filter(status__in=("queued", "running")).count()
    pending = Run.objects.filter(services.queue_filter(), vetting_status="pending").count()
    return {
        "nav": [{"key": k, "label": label, "url": reverse(k)} for k, label in NAV if k not in MORE],
        "nav_more": [{"key": k, "label": label, "url": reverse(k)} for k, label in NAV if k in MORE],
        "nav_more_label": next((label for k, label in NAV if k == page and k in MORE), ""),
        "page": page,
        "cycle_prev": reverse(keys[(i - 1) % len(keys)]), "cycle_next": reverse(keys[(i + 1) % len(keys)]),
        "shell": {"live": live, "pending": pending, "targets": Target.objects.count(), "runs": Run.objects.count(),
                  "model_active": model.CHECKPOINT.exists(),
                  "training": services.train_state["status"] == "running"},
    }
