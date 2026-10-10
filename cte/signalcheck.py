"""Signal check: classify a transit signal from its catalogue parameters alone, with no light curve.

The full classifier needs the photometry. This one answers a narrower question - given only the numbers a
catalogue or a paper lists for a signal (period, depth, duration, and whatever else is known), how do signals
like it on real Kepler stars turn out? It is a boosted-tree model fitted on the same labelled Kepler stars as the
main classifier and scored on the same held-out stars. Every optional input may be left blank: the model is
trained with those inputs withheld at random, so a missing value is something it has seen.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

import numpy as np

from .config import MODEL_DIR
from .physics import StellarParameters, derive

MODEL = MODEL_DIR / "signalcheck.joblib"
METRICS = MODEL_DIR / "signalcheck.json"
CLASSES = ["PLANET_CANDIDATE", "ECLIPSING_BINARY", "STELLAR_OR_INSTRUMENT_NOISE"]
FEATURES = ["log_period", "log_duration_h", "duration_over_period", "log_depth", "log_snr", "n_transits", "snr_per_transit",
            "teff", "log_rstar", "log_mstar", "log_density", "log_radius_earth", "duration_ratio",
            "odd_even_sigma", "secondary_sigma", "secondary_ratio", "centroid_sigma"]
# optional inputs, and the columns that go missing with each
GROUPS = {
    "snr": ("Signal-to-noise", ["log_snr", "snr_per_transit"]),
    "n_transits": ("Number of transits", ["n_transits", "snr_per_transit"]),
    "star": ("The star", ["teff", "log_rstar", "log_mstar", "log_density", "log_radius_earth", "duration_ratio"]),
    "odd_even": ("Odd/even depth difference", ["odd_even_sigma"]),
    "secondary": ("Secondary eclipse", ["secondary_sigma", "secondary_ratio"]),
    "centroid": ("Centroid shift", ["centroid_sigma"]),
}
SEEDS = 3
_lock = threading.Lock()
_cache: dict = {}


def row(period_days, depth_ppm, duration_hours, snr=None, n_transits=None, t_eff_k=None, r_star_solar=None,
        m_star_solar=None, odd_even_sigma=None, secondary_sigma=None, secondary_depth_ppm=None,
        centroid_sigma=None) -> tuple[list[float], dict]:
    """One signal as the model's feature row, plus the physical quantities derived on the way."""
    nan = float("nan")

    def val(v):
        return nan if v is None or not np.isfinite(v) else float(v)

    p, d, t = float(period_days), float(depth_ppm) * 1e-6, float(duration_hours)
    snr, n_tr = val(snr), val(n_transits)
    teff, rs, ms = val(t_eff_k), val(r_star_solar), val(m_star_solar)
    physics: dict = {}
    radius = ratio = nan
    if np.isfinite(rs) and np.isfinite(ms) and rs > 0 and ms > 0:
        physics = derive(StellarParameters(teff if np.isfinite(teff) else 5772.0, rs, ms), p, d)
        radius = physics["radius_earth"]
        theory = physics["transit_duration_hours"]
        ratio = t / theory if theory > 0 else nan
        physics["duration_ratio"] = round(ratio, 3) if np.isfinite(ratio) else None
        if not np.isfinite(teff):
            for k in ("equilibrium_temp_k", "insolation_earth"):
                physics.pop(k, None)
    sec_ratio = val(secondary_depth_ppm) * 1e-6 / d if secondary_depth_ppm is not None else nan
    x = [np.log10(p), np.log10(max(t, 0.1)), t / 24.0 / p, np.log10(np.clip(d, 1e-6, 1.0)),
         np.log10(np.clip(snr, 1.0, 1e4)) if np.isfinite(snr) else nan, n_tr,
         snr / np.sqrt(max(n_tr, 1.0)) if np.isfinite(snr) and np.isfinite(n_tr) else nan,
         teff, np.log10(rs) if rs > 0 else nan, np.log10(ms) if ms > 0 else nan,
         np.log10(ms / rs ** 3) if rs > 0 and ms > 0 else nan,
         np.log10(max(radius, 0.05)) if np.isfinite(radius) else nan, float(np.clip(ratio, 0.0, 10.0)) if np.isfinite(ratio) else nan,
         val(odd_even_sigma), val(secondary_sigma), sec_ratio, val(centroid_sigma)]
    return [float(v) for v in x], physics


def _training_rows(d: dict) -> np.ndarray:
    """The same feature row for every example of the Kepler dataset, from what was measured on its light curve."""
    from .vetting import EXTRA
    e = {n: d["ext"][:, i].astype(float) for i, n in enumerate(EXTRA)}
    aux = d["aux"].astype(float)
    out = []
    for i in range(len(aux)):
        out.append(row(10 ** e["log_period"][i], aux[i, 0] * 1e6, 10 ** e["log_t14h"][i], aux[i, 5], e["n_transits"][i],
                       e["teff"][i], 10 ** e["log_srad"][i], 10 ** e["log_smass"][i], aux[i, 2], e["sec_sigma_raw"][i],
                       e["sec_ratio_raw"][i] * aux[i, 0] * 1e6, e["cen_sigma_raw"][i])[0])
    return np.array(out)


def _mask(x: np.ndarray, groups) -> np.ndarray:
    x = x.copy()
    for g in groups:
        for name in GROUPS[g][1]:
            x[..., FEATURES.index(name)] = np.nan
    return x


def _predict(members, x: np.ndarray) -> np.ndarray:
    return np.mean([m.predict_proba(x) for m in members], axis=0)


def train(report=lambda *_: None) -> dict:
    """Fit on the training stars of the Kepler dataset; score on its held-out stars, with all inputs and with few."""
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier

    from . import kepler
    from .model import average_precision, roc_curve

    d = kepler.load_dataset()
    if d is None:
        raise RuntimeError("No Kepler dataset on disk. Build it first (python manage.py build_kepler_dataset).")
    report("Measuring catalogue parameters of every training example", 0.1)
    x, y = _training_rows(d), d["y"].astype(int)
    train_rows, test_rows = ~d["test"], d["test"] & (d["rank"] == 0)
    rng = np.random.default_rng(20261010)
    xt, yt = x[train_rows], y[train_rows]
    parts = [xt]
    for _ in range(3):  # the same examples with optional inputs withheld at random, as a user would leave them blank
        m = xt.copy()
        for g in GROUPS:
            drop = rng.random(len(m)) < 0.5
            m[drop] = _mask(m[drop], [g])
        parts.append(m)
    xa, ya = np.concatenate(parts), np.tile(yt, len(parts))
    members = []
    for s in range(SEEDS):
        report(f"Fitting model {s + 1} of {SEEDS}", 0.2 + 0.6 * s / SEEDS)
        members.append(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=40,
                                                      l2_regularization=1.0, class_weight="balanced", early_stopping=False,
                                                      random_state=s).fit(xa, ya))
    planet = y[test_rows] == 0

    def score(groups_missing):
        p = _predict(members, _mask(x[test_rows], groups_missing))
        call = p.argmax(1) == 0
        return {"auc_planet": round(float(roc_curve(p[:, 0], planet)[2]), 4),
                "pr_auc_planet": round(float(average_precision(p[:, 0], planet)), 4),
                "accuracy": round(float((p.argmax(1) == y[test_rows]).mean()), 4),
                "planet_or_not": round(float((call == planet).mean()), 4),
                "planets_called": [int(call[planet].sum()), int(planet.sum())],
                "others_called_planet": [int(call[~planet].sum()), int((~planet).sum())]}

    report("Scoring on held-out stars", 0.9)
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(), "features": FEATURES,
        "n_train_examples": int(train_rows.sum()), "n_train_stars": int(len(np.unique(d["kepid"][train_rows]))),
        "n_test": int(test_rows.sum()),
        "levels": [
            {"name": "Everything filled in", "inputs": "all twelve inputs", **score([])},
            {"name": "No vetting measurements", "inputs": "period, depth, duration, S/N, transits, star", **score(["odd_even", "secondary", "centroid"])},
            {"name": "Period, depth, duration and the star", "inputs": "six inputs", **score(["odd_even", "secondary", "centroid", "snr", "n_transits"])},
            {"name": "Period, depth and duration only", "inputs": "three inputs", **score(list(GROUPS))},
        ],
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"features": FEATURES, "members": members}, MODEL)
    METRICS.write_text(json.dumps(metrics, indent=1))
    with _lock:
        _cache.clear()
    report("Signal check model saved", 1.0)
    return metrics


def available() -> bool:
    return MODEL.exists()


def load_metrics() -> dict | None:
    return json.loads(METRICS.read_text()) if METRICS.exists() else None


def _members():
    with _lock:
        stamp = MODEL.stat().st_mtime
        if _cache.get("stamp") != stamp:
            import joblib
            saved = joblib.load(MODEL)
            if saved["features"] != FEATURES:
                raise RuntimeError("The saved signal-check model does not match this version; retrain it (python manage.py train_signal_check).")
            _cache.update(stamp=stamp, members=saved["members"])
        return _cache["members"]


def assess(**inputs) -> dict:
    """Class probabilities for one signal, what each optional input contributed, and plain-language notes."""
    if not available():
        raise RuntimeError("The signal-check model has not been trained (python manage.py train_signal_check).")
    x, physics = row(**inputs)
    xa = np.array([x])
    members = _members()
    p = _predict(members, xa)[0]
    given = [g for g, (_, cols) in GROUPS.items() if all(np.isfinite(x[FEATURES.index(c)]) for c in cols if c != "snr_per_transit")]
    moved = []
    for g in given:
        without = _predict(members, _mask(xa, [g]))[0]
        moved.append({"key": g, "label": GROUPS[g][0], "delta_planet": round(float(p[0] - without[0]), 4)})
    moved.sort(key=lambda m: -abs(m["delta_planet"]))
    notes = []
    radius = physics.get("radius_earth")
    if radius is not None and radius > 25.0:
        notes.append(f"A dip this deep on a star this size needs a body of {radius / 11.209:.1f} Jupiter radii: larger than any planet, so the companion is probably a star.")
    ratio = physics.get("duration_ratio")
    if ratio is not None and ratio > 1.6:
        notes.append(f"The dip lasts {ratio:.1f} times longer than a transit across the middle of this star would. The star may be larger than listed, or the dip may be on a different star.")
    if ratio is not None and ratio < 0.35:
        notes.append(f"The dip lasts only {ratio:.2f} of a central transit of this star: a grazing crossing, which is more often a binary than a planet.")
    sec_ratio = x[FEATURES.index("secondary_ratio")]
    if np.isfinite(sec_ratio) and sec_ratio >= 0.05 and (inputs.get("secondary_sigma") or 0) >= 3:
        notes.append(f"The secondary eclipse is {sec_ratio * 100:.0f}% as deep as the main dip. A planet's is at most a few percent; this points to a second star.")
    if (inputs.get("odd_even_sigma") or 0) >= 3:
        notes.append("Odd and even dips differ in depth. That is the signature of two unequal stars eclipsing each other at twice the stated period.")
    if (inputs.get("centroid_sigma") or 0) >= 3:
        notes.append("The star's image moves during the dip, so the dimming comes at least partly from a neighbouring star.")
    if inputs.get("snr") is not None and inputs["snr"] < 7.1:
        notes.append("The signal-to-noise is below 7.1, the usual minimum for calling a detection at all.")
    missing = [GROUPS[g][0].lower() for g in GROUPS if g not in given]
    best = int(np.argmax(p))
    return {"prediction": CLASSES[best], "probabilities": {c: round(float(v), 4) for c, v in zip(CLASSES, p)},
            "p_planet": round(float(p[0]), 4), "moved": moved, "notes": notes, "missing": missing,
            "physics": {k: v for k, v in physics.items() if v is not None},
            "disposition": "planet candidate" if best == 0 else ("likely eclipsing binary" if best == 1 else "likely noise or stellar variability")}
