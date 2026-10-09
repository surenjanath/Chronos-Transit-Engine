"""Relational archive (PostgreSQL in production, SQLite locally)."""
from django.db import models
from django.utils import timezone

DISPOSITIONS = {
    "CONFIRMED_VETTING_QUEUE": "Vetting queue",
    "MANUAL_REVIEW": "Manual review",
    "FALSE_POSITIVE_SUSPECT": "FP suspect",
    "FALSE_POSITIVE_ECLIPSING_BINARY": "Eclipsing binary",
    "REJECTED_NOISE": "Noise",
    "NO_SIGNIFICANT_SIGNAL": "No signal",
}
NEEDS_HUMAN = ("CONFIRMED_VETTING_QUEUE", "MANUAL_REVIEW", "FALSE_POSITIVE_SUSPECT")
CLASS_LABELS = {
    "PLANET_CANDIDATE": "Planet candidate",
    "ECLIPSING_BINARY": "Eclipsing binary",
    "STELLAR_OR_INSTRUMENT_NOISE": "Stellar / instrument noise",
}


class Target(models.Model):
    name = models.CharField(max_length=120, unique=True)
    alias = models.CharField(max_length=120, blank=True, default="")
    mission = models.CharField(max_length=40, default="TESS")
    t_eff_k = models.FloatField(null=True)
    r_star_solar = models.FloatField(null=True)
    m_star_solar = models.FloatField(null=True)
    v_mag = models.FloatField(null=True)
    ra = models.FloatField(null=True)
    dec = models.FloatField(null=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.display

    @property
    def display(self) -> str:
        return self.alias or self.name


class Run(models.Model):
    class Status(models.TextChoices):
        QUEUED = "queued"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"

    class Vetting(models.TextChoices):
        PENDING = "pending"
        CONFIRMED = "confirmed"
        FALSE_POSITIVE = "false_positive", "False positive"

    id = models.CharField(primary_key=True, max_length=40)
    target = models.ForeignKey(Target, on_delete=models.CASCADE, related_name="runs")
    source = models.CharField(max_length=20)
    sector = models.IntegerField(null=True)
    camera = models.IntegerField(null=True)
    ccd = models.IntegerField(null=True)
    status = models.CharField(max_length=20, choices=Status, default=Status.QUEUED, db_index=True)
    stage = models.CharField(max_length=40, default="QUEUED")
    progress = models.FloatField(default=0.0)
    config = models.JSONField(default=dict)
    results = models.JSONField(null=True)
    error = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    started_at = models.DateTimeField(null=True)
    finished_at = models.DateTimeField(null=True)
    duration_s = models.FloatField(null=True)

    # Denormalised headline results for fast listing and filtering.
    prediction = models.CharField(max_length=40, blank=True, default="")
    p_planet = models.FloatField(null=True)
    disposition = models.CharField(max_length=40, blank=True, default="", db_index=True)
    period_days = models.FloatField(null=True)
    depth_ppm = models.FloatField(null=True)
    radius_earth = models.FloatField(null=True)
    sde = models.FloatField(null=True)
    n_flags = models.IntegerField(default=0)

    # Human vetting.
    vetting_status = models.CharField(max_length=20, choices=Vetting, default=Vetting.PENDING)
    vetting_note = models.TextField(blank=True, default="")
    vetted_at = models.DateTimeField(null=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.target.display} · {self.id}"

    @property
    def is_live(self) -> bool:
        return self.status in (self.Status.QUEUED, self.Status.RUNNING)

    @property
    def disposition_label(self) -> str:
        return DISPOSITIONS.get(self.disposition, self.disposition)

    @property
    def needs_human(self) -> bool:
        return self.status == self.Status.COMPLETED and self.disposition in NEEDS_HUMAN

    @property
    def stage_label(self) -> str:
        return self.stage.replace("_", " ").lower()

    @property
    def percent(self) -> int:
        return round(self.progress * 100)


class LogEvent(models.Model):
    run = models.ForeignKey(Run, null=True, on_delete=models.CASCADE, related_name="logs")
    seq = models.IntegerField()
    timestamp = models.DateTimeField(db_index=True)
    level = models.CharField(max_length=10, db_index=True)
    module = models.CharField(max_length=60)
    message = models.TextField()
    stage = models.CharField(max_length=40, blank=True, default="", db_index=True)
    telemetry = models.JSONField(null=True)
    hash = models.CharField(max_length=64)

    class Meta:
        ordering = ["-timestamp", "-id"]

    def as_record(self) -> dict:
        """The record exactly as written to the JSONL audit file."""
        rec = {"timestamp": self.timestamp.isoformat(), "level": self.level, "module": self.module, "message": self.message}
        if self.telemetry:
            rec["telemetry"] = self.telemetry
        rec |= {"seq": self.seq, "hash": self.hash}
        return rec


class ValidationRun(models.Model):
    status = models.CharField(max_length=20, default="running")
    progress = models.CharField(max_length=200, blank=True, default="")
    created_at = models.DateTimeField(default=timezone.now)
    finished_at = models.DateTimeField(null=True)
    results = models.JSONField(null=True)

    class Meta:
        ordering = ["-id"]


class Setting(models.Model):
    key = models.CharField(primary_key=True, max_length=60)
    value = models.JSONField()


class AutomationCycle(models.Model):
    """One pass of the autopilot: pull new targets, grow the labelled dataset, retrain if warranted."""
    started_at = models.DateTimeField(default=timezone.now)
    finished_at = models.DateTimeField(null=True)
    status = models.CharField(max_length=20, default="running")
    trigger = models.CharField(max_length=20, default="schedule")
    step = models.CharField(max_length=200, blank=True, default="")
    summary = models.JSONField(default=dict)

    class Meta:
        ordering = ["-id"]
