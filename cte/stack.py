"""Second-stage classifier: the CNN's opinion combined with the physical vetting evidence.

The CNN sees only the normalised shape of the folded curve. It cannot know how deep the transit is, whether odd
and even transits differ, whether the depth changes from one observing season to the next or whether the star
brightens and dims in step with the orbit - the evidence that separates a planet from an eclipsing binary or a
blend. Gradient-boosted trees, fitted on real labelled stars, weigh the CNN probabilities together with those
measurements and produce the final class probabilities.
"""
from __future__ import annotations

import numpy as np

from .config import MODEL_DIR
from .vetting import EXTRA

STACKER = MODEL_DIR / "stacker.joblib"
LEGACY = MODEL_DIR / "stacker.json"  # the earlier small-network second stage
FEATURES = ["cnn_logit_planet", "cnn_logit_binary", "log10_depth", "shape_v", "odd_even_sigma", "secondary_sigma",
            "secondary_ratio", "log10_snr", "centroid_sigma", "centroid_separation_px", "radius_excess_over_2rj", "duration_ratio",
            "shift_margin", "shift_ratio", "shift_log10_albedo"]
AUX = ["depth", "v_shape", "odd_even_sigma", "secondary_sigma", "secondary_ratio", "snr", "centroid_sigma",
       "centroid_separation_px", "radius_earth", "duration_ratio", "shift_margin", "shift_ratio", "shift_albedo"]


def aux_from(depth: float, v_shape: float, diag: dict, radius_earth: float) -> list[float]:
    """The diagnostic inputs, taken the same way for a blind run and for a catalogue-folded training star."""
    oe, sec, cen = diag.get("odd_even") or {}, diag.get("secondary") or {}, diag.get("centroid") or {}
    shift = diag.get("model_shift") or {}
    # As with the phase-0.5 eclipse: a second dip argues for a star only if no planet could produce it. An
    # ultra-hot Jupiter's occultation is real and detectable (WASP-12 b, WASP-18 b) and must not count against it.
    albedo = shift.get("albedo", 0.0)
    too_bright = (albedo is not None and np.isfinite(albedo) and albedo > 1.0) or float(shift.get("ratio", 0.0)) > 0.1
    significant = bool(shift.get("significant")) and too_bright
    # A secondary eclipse argues for a stellar companion only when it is too deep for a planet: hot Jupiters show
    # genuine occultations of a few percent of the transit depth. Same 5% rule as the SECONDARY_ECLIPSE flag.
    stellar_secondary = float(sec.get("sigma", 0.0)) if float(sec.get("ratio", 0.0)) >= 0.05 else 0.0
    return [float(depth), float(v_shape), float(oe.get("sigma", 0.0)), stellar_secondary,
            float(sec.get("ratio", 0.0)), float(diag.get("snr", 0.0)), float(cen.get("sigma", 0.0)),
            # shift / depth is meaningless unless the shift itself is significant
            float(cen.get("separation_px", 0.0)) if cen.get("sigma", 0.0) >= 3.0 else 0.0,
            float(radius_earth or 0.0), float(diag.get("duration_ratio") or 1.0),
            # second dip at any orbital phase: how far it clears the false-alarm level, its size relative to the
            # transit, and the albedo a planet would need to produce it (above 1 means self-luminous)
            float(shift.get("margin", -3.0)) if too_bright else -3.0, float(shift.get("ratio", 0.0)) if significant else 0.0,
            float(shift.get("albedo", 0.0)) if significant and np.isfinite(shift.get("albedo", 0.0)) else 0.0]


def features(probs: np.ndarray, aux: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
    probs = np.atleast_2d(np.asarray(probs, dtype=float))
    aux = np.nan_to_num(np.atleast_2d(np.asarray(aux, dtype=float)), nan=0.0, posinf=1e4, neginf=0.0)  # undefined -> neutral
    extra = np.full((len(aux), len(EXTRA)), np.nan) if extra is None else np.atleast_2d(np.asarray(extra, dtype=float))
    p = np.clip(np.nan_to_num(probs, nan=1 / 3), 1e-4, 1 - 1e-4)
    logit = np.log(p / (1 - p))
    return np.column_stack([
        logit[:, 0], logit[:, 1], np.log10(np.clip(aux[:, 0], 1e-6, 1.0)), np.clip(aux[:, 1], 0, 1),
        np.clip(aux[:, 2], 0, 10), np.clip(aux[:, 3], 0, 10), np.clip(aux[:, 4], 0, 1), np.log10(np.clip(aux[:, 5], 1.0, 1e4)),
        np.clip(aux[:, 6], 0, 20), np.clip(aux[:, 7], 0, 5), # radius counts against a planet only beyond ~2 Jupiter radii (physics), not merely for being a giant
        # (which would just encode how rare giants are among Kepler's planets)
        np.log10(np.clip(aux[:, 8], 20.0, 300.0) / 20.0), np.clip(aux[:, 9], 0.2, 3.0),
        np.clip(aux[:, 10], -3.0, 15.0), np.clip(aux[:, 11], 0.0, 1.0), np.log10(np.clip(aux[:, 12], 0.1, 1000.0)),
        extra,  # trees need no scaling, and take a missing measurement (NaN) as it is
    ])


MEMBERS = 5


def fit(probs: np.ndarray, aux: np.ndarray, extra: np.ndarray, y: np.ndarray) -> dict:
    """Several boosted-tree models that differ only in their random seed; the answer is their mean. Trees suit
    this evidence: each measurement matters only in combination with others (a deep dip is a planet only if
    everything else is clean), and several of them are absent for some stars."""
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier

    x = features(probs, aux, extra)
    # classes weighted equally, so the answer does not drift with how many binaries happen to be in the training set
    members = [HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                              l2_regularization=1.0, early_stopping=False, class_weight="balanced",
                                              random_state=seed).fit(x, y)
               for seed in range(MEMBERS)]
    return {"features": FEATURES + EXTRA, "n": int(len(y)), "members": members, "sklearn": sklearn.__version__}


def save(params: dict) -> None:
    global _cached
    import joblib
    joblib.dump(params, STACKER)
    LEGACY.unlink(missing_ok=True)
    _cached = params


_cached: dict | None = None


def load() -> dict | None:
    global _cached
    if _cached is None and STACKER.exists():
        import joblib
        try:
            _cached = joblib.load(STACKER)
        except Exception:
            return None  # written by an incompatible library version: retrain to refresh
    return _cached


def reset_cache() -> None:
    global _cached
    _cached = None


def predict(probs, aux, extra=None, params: dict | None = None) -> np.ndarray | None:
    """Final class probabilities, or None when no second stage has been fitted (the CNN output is then used as is)."""
    params = params or load()
    names = (params or {}).get("features") or []
    if params is None or not names or names != (FEATURES + EXTRA)[:len(names)]:
        return None  # written by an older version: retrain to refresh
    x = features(probs, aux, extra)[:, :len(names)]  # a model fitted before the newest measurements simply ignores them
    return np.mean([m.predict_proba(x) for m in params["members"]], axis=0)
