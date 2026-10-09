import json
import math

from django import template
from django.utils import timezone
from django.utils.html import format_html

register = template.Library()
DASH = "—"


@register.filter
def num(value, digits=2):
    """Thousands-separated fixed-point number; an em dash when the value is missing."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return DASH
    if not math.isfinite(f):
        return DASH
    return f"{f:,.{int(digits)}f}"


@register.filter
def pct(value, digits=1):
    try:
        return f"{float(value) * 100:.{int(digits)}f}%"
    except (TypeError, ValueError):
        return DASH


@register.filter
def sci(value):
    try:
        return f"{float(value):.1e}"
    except (TypeError, ValueError):
        return DASH


@register.filter
def relerr(value):
    """Relative error as a percentage, or scientific notation when tiny."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return DASH
    return f"{f:.2e}" if abs(f) < 1e-3 else f"{f * 100:+.2f}%"


@register.filter
def dur(seconds):
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return DASH
    return f"{s:.1f} s" if s < 60 else f"{int(s // 60)}m {round(s % 60)}s"


@register.filter
def ago(dt):
    if not dt:
        return DASH
    if isinstance(dt, str):
        from datetime import datetime
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    s = (timezone.now() - dt).total_seconds()
    if s < 60:
        return "just now"
    if s < 3600:
        return f"{int(s // 60)} min ago"
    if s < 86400:
        return f"{int(s // 3600)} h ago"
    return f"{int(s // 86400)} d ago"


@register.filter
def get(mapping, key):
    try:
        return mapping.get(key)
    except AttributeError:
        return None


@register.filter
def times(value, factor):
    try:
        return float(value) * float(factor)
    except (TypeError, ValueError):
        return None


@register.filter
def lower_words(value):
    return str(value or "").replace("_", " ").lower()


@register.filter
def pretty(value):
    return json.dumps(value, indent=2, ensure_ascii=False)


@register.filter
def case_value(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:,.5f}" if abs(value) < 20 else f"{value:,.1f}"
    return DASH if value is None else value


@register.simple_tag
def pill(run):
    """Disposition pill in the ledger's monochrome vocabulary."""
    if not run.disposition:
        return DASH
    cls = {"CONFIRMED_VETTING_QUEUE": "High", "MANUAL_REVIEW": "Medium", "FALSE_POSITIVE_SUSPECT": "Medium",
           "FALSE_POSITIVE_ECLIPSING_BINARY": "Low", "REJECTED_NOISE": "Long", "NO_SIGNIFICANT_SIGNAL": "Long"}.get(run.disposition, "")
    return format_html('<span class="pill {}">{}</span>', cls, run.disposition_label)


@register.simple_tag(takes_context=True)
def qs(context, **changes):
    """Current query string with some parameters replaced (for filters and pagination)."""
    params = context["request"].GET.copy()
    for key, value in changes.items():
        if value in (None, ""):
            params.pop(key, None)
        else:
            params[key] = value
    encoded = params.urlencode()
    return f"?{encoded}" if encoded else "?"


@register.filter
def disposition_label(code):
    from engine.models import DISPOSITIONS
    return DISPOSITIONS.get(code, code) or DASH


@register.filter
def bigyears(years):
    """A long span of years in words: 4,600 years, 1.2 million years, 3.4 billion years."""
    try:
        y = float(years)
    except (TypeError, ValueError):
        return DASH
    if y < 1e4:
        return f"{y:,.0f} years"
    if y < 1e6:
        return f"{round(y, -3):,.0f} years"
    if y < 1e9:
        return f"{y / 1e6:,.1f} million years"
    return f"{y / 1e9:,.1f} billion years"
