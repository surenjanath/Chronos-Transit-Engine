"""Runtime configuration for the Chronos Transit Engine."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CTE_DATA_DIR", PROJECT_DIR / "data"))
LOG_DIR = DATA_DIR / "logs"
ARCHIVE_DIR = DATA_DIR / "archive"
MODEL_DIR = DATA_DIR / "models"
UPLOAD_DIR = DATA_DIR / "uploads"
MAST_CACHE_DIR = DATA_DIR / "mast_cache"

for _d in (DATA_DIR, LOG_DIR, ARCHIVE_DIR, MODEL_DIR, UPLOAD_DIR, MAST_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = int(os.environ.get("CTE_MAX_WORKERS", "2"))

# Physical constants (PRD §2.4). R_sun / R_earth = 109.076 as specified.
R_SUN_M = 6.957e8
AU_M = 1.496e11
R_EARTH_M = R_SUN_M / 109.076
DAYS_PER_YEAR = 365.256

# Pipeline defaults (PRD §2.1 - §2.3, §3.2). Editable from the Settings page.
DEFAULT_PIPELINE = {
    "sigma_k": 5.0,
    "clip_window": 101,
    "detrend_window_days": 1.5,
    "detrend_polyorder": 3,
    "period_min": 0.5,
    "period_max": 30.0,
    "q_min": 0.01,
    "q_max": 0.1,
    "bls_oversample": 3,
    "impact_param": 0.0,
    "bond_albedo": 0.3,
    "global_bins": 201,
    "local_bins": 61,
    "sde_threshold": 7.0,
    "planet_confidence": 0.9,
    "max_additional_signals": 2,
    # Experimental, off by default: on 36 injections into variable Kepler stars neither changed recovery (17/36).
    "adaptive_detrend": 0,
    "remove_periodic_variability": 0,
}

# Acceptance tolerances (PRD §6).
TOLERANCES = {
    "period_rel": 1e-4,
    "depth_rel": 0.05,
    "radius_rel": 0.10,
    "auc_min": 0.96,
    "pr_auc_min": 0.88,
}

CLASS_NAMES = ["PLANET_CANDIDATE", "ECLIPSING_BINARY", "STELLAR_OR_INSTRUMENT_NOISE"]
