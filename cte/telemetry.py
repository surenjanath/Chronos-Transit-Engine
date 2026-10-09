"""Structured scientific logging (PRD §4, §5).

Every record is newline-delimited JSON with the PRD schema
(`timestamp`, `level`, `module`, `message`, optional `telemetry`) plus a
sequence number and SHA-256 hash chain so an audit trail cannot be edited
without detection. Records are validated before they are written; an invalid
record raises `LogIntegrityError`, which terminates the pipeline run (§6).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import LOG_DIR

LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
REQUIRED_KEYS = ("timestamp", "level", "module", "message")
ALLOWED_KEYS = set(REQUIRED_KEYS) | {"telemetry", "seq", "hash"}
GENESIS_HASH = "0" * 64


class LogIntegrityError(RuntimeError):
    """A log record failed schema validation."""


class JSONTelemetryFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if hasattr(record, "cte_record"):
            return json.dumps(record.cte_record, allow_nan=False)
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if hasattr(record, "telemetry"):
            log_obj["telemetry"] = clean(record.telemetry)
        return json.dumps(log_obj, allow_nan=False)


logger = logging.getLogger("CTE-Core")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(JSONTelemetryFormatter())
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def clean(value: Any) -> Any:
    """Convert numpy types to plain JSON types; non-finite floats become null."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, (int, str)):
        return value
    return str(value)


def validate_record(obj: Any) -> None:
    """Raise LogIntegrityError unless `obj` matches the telemetry schema."""
    if not isinstance(obj, dict):
        raise LogIntegrityError("record is not a JSON object")
    for key in REQUIRED_KEYS:
        if not isinstance(obj.get(key), str) or not obj[key]:
            raise LogIntegrityError(f"missing or non-string field '{key}'")
    extra = set(obj) - ALLOWED_KEYS
    if extra:
        raise LogIntegrityError(f"unexpected fields: {sorted(extra)}")
    if obj["level"] not in LEVELS:
        raise LogIntegrityError(f"invalid level '{obj['level']}'")
    try:
        datetime.fromisoformat(obj["timestamp"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise LogIntegrityError(f"invalid timestamp: {exc}") from exc
    if "telemetry" in obj and not isinstance(obj["telemetry"], dict):
        raise LogIntegrityError("'telemetry' must be an object")
    if "seq" in obj and not isinstance(obj["seq"], int):
        raise LogIntegrityError("'seq' must be an integer")
    try:
        json.dumps(obj, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise LogIntegrityError(f"record is not valid JSON: {exc}") from exc


def record_hash(prev_hash: str, record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "hash"}
    payload = prev_hash + json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def verify_jsonl(path: Path) -> dict:
    """Validate schema and hash chain of a JSONL audit file."""
    total = valid = 0
    errors: list[str] = []
    prev = GENESIS_HASH
    chain_ok = True
    if not path.exists():
        return {"records": 0, "valid": 0, "chain_ok": False, "errors": ["file missing"]}
    with path.open() as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            total += 1
            try:
                obj = json.loads(line)
                validate_record(obj)
                valid += 1
            except (json.JSONDecodeError, LogIntegrityError) as exc:
                errors.append(f"line {line_no}: {exc}")
                chain_ok = False
                continue
            if obj.get("hash") != record_hash(prev, obj):
                chain_ok = False
                errors.append(f"line {line_no}: hash chain mismatch")
            prev = obj.get("hash", prev)
    return {"records": total, "valid": valid, "chain_ok": chain_ok, "errors": errors[:20]}


class RunLogger:
    """Writes validated telemetry to stdout, a per-run JSONL file and the database."""

    def __init__(self, run_id: str | None, module: str = "transit_engine", sink=None, path: Path | None = None):
        self.run_id = run_id
        self.module = module
        self.path = path or LOG_DIR / f"{run_id or 'system'}.jsonl"
        self._sink = sink
        self._lock = threading.Lock()
        self._seq = 0
        self._prev = GENESIS_HASH
        if self.path.exists():  # resume an existing chain (system log)
            with self.path.open() as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        self._seq, self._prev = obj["seq"], obj["hash"]
                    except (json.JSONDecodeError, KeyError):
                        continue

    def log(self, level: str, message: str, stage: str | None = None, **telemetry: Any) -> dict:
        with self._lock:
            tele = clean(telemetry)
            if stage:
                tele = {"stage": stage, **tele}
            if self.run_id:
                tele = {"run_id": self.run_id, **tele}
            self._seq += 1
            record: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": level,
                "module": self.module,
                "message": message,
            }
            if tele:
                record["telemetry"] = tele
            record["seq"] = self._seq
            record["hash"] = record_hash(self._prev, record)
            validate_record(record)
            self._prev = record["hash"]
            with self.path.open("a") as fh:
                fh.write(json.dumps(record, allow_nan=False) + "\n")
            logger.log(getattr(logging, level), message, extra={"cte_record": record})
            if self._sink:
                self._sink(self.run_id, record, stage)
            return record

    def info(self, message: str, stage: str | None = None, **telemetry: Any) -> dict:
        return self.log("INFO", message, stage, **telemetry)

    def warning(self, message: str, stage: str | None = None, **telemetry: Any) -> dict:
        return self.log("WARNING", message, stage, **telemetry)

    def error(self, message: str, stage: str | None = None, **telemetry: Any) -> dict:
        return self.log("ERROR", message, stage, **telemetry)
