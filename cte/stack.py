"""Second-stage classifier: the CNN's opinion combined with the physical vetting evidence.

The CNN sees only the normalised shape of the folded curve. It cannot know how deep the transit is, whether odd
and even transits differ, or whether there is a secondary eclipse - exactly the evidence that separates a deep
hot Jupiter from an eclipsing binary. A small multinomial logistic model, fitted on real labelled stars, weighs
the CNN probabilities together with those diagnostics and produces the final class probabilities.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from .config import CLASS_NAMES, MODEL_DIR

STACKER = MODEL_DIR / "stacker.json"
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


def features(probs: np.ndarray, aux: np.ndarray) -> np.ndarray:
    probs = np.atleast_2d(np.asarray(probs, dtype=float))
    aux = np.nan_to_num(np.atleast_2d(np.asarray(aux, dtype=float)), nan=0.0, posinf=1e4, neginf=0.0)  # undefined -> neutral
    p = np.clip(np.nan_to_num(probs, nan=1 / 3), 1e-4, 1 - 1e-4)
    logit = np.log(p / (1 - p))
    return np.column_stack([
        logit[:, 0], logit[:, 1], np.log10(np.clip(aux[:, 0], 1e-6, 1.0)), np.clip(aux[:, 1], 0, 1),
        np.clip(aux[:, 2], 0, 10), np.clip(aux[:, 3], 0, 10), np.clip(aux[:, 4], 0, 1), np.log10(np.clip(aux[:, 5], 1.0, 1e4)),
        np.clip(aux[:, 6], 0, 20), np.clip(aux[:, 7], 0, 5), # radius counts against a planet only beyond ~2 Jupiter radii (physics), not merely for being a giant
        # (which would just encode how rare giants are among Kepler's planets)
        np.log10(np.clip(aux[:, 8], 20.0, 300.0) / 20.0), np.clip(aux[:, 9], 0.2, 3.0),
        np.clip(aux[:, 10], -3.0, 15.0), np.clip(aux[:, 11], 0.0, 1.0), np.log10(np.clip(aux[:, 12], 0.1, 1000.0)),
    ])


HIDDEN = 12
MEMBERS = 7


def _fit_one(xt, yt, n_features: int, weight_decay: float, seed: int) -> dict:
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(n_features, HIDDEN), torch.nn.Tanh(), torch.nn.Linear(HIDDEN, len(CLASS_NAMES)))
    opt = torch.optim.Adam(net.parameters(), lr=0.02, weight_decay=weight_decay)
    for _ in range(400):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(net(xt), yt).backward()
        opt.step()
    lin1, lin2 = net[0], net[2]
    return {"w1": lin1.weight.detach().numpy().tolist(), "b1": lin1.bias.detach().numpy().tolist(),
            "w2": lin2.weight.detach().numpy().tolist(), "b2": lin2.bias.detach().numpy().tolist()}


def fit(probs: np.ndarray, aux: np.ndarray, y: np.ndarray, weight_decay: float = 0.01) -> dict:
    """An ensemble of small one-hidden-layer networks. The evidence interacts (a deep dip is a planet only if
    everything else is clean), and averaging several independently initialised members keeps the answer stable
    from one refit to the next."""
    x = features(probs, aux)
    mean, std = x.mean(0), x.std(0) + 1e-9
    xt = torch.tensor((x - mean) / std, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    return {"features": FEATURES, "mean": mean.tolist(), "std": std.tolist(), "n": int(len(y)),
            "members": [_fit_one(xt, yt, x.shape[1], weight_decay, seed) for seed in range(MEMBERS)]}


def save(params: dict) -> None:
    global _cached
    STACKER.write_text(json.dumps(params, indent=1))
    _cached = params


_cached: dict | None = None


def load() -> dict | None:
    global _cached
    if _cached is None and STACKER.exists():
        _cached = json.loads(STACKER.read_text())
    return _cached


def reset_cache() -> None:
    global _cached
    _cached = None


def predict(probs, aux, params: dict | None = None) -> np.ndarray | None:
    """Final class probabilities, or None when no stacker has been fitted (the CNN output is then used as is)."""
    params = params or load()
    if params is None:
        return None
    if "members" not in params or len(params["mean"]) != len(FEATURES):
        return None  # written by an older version: retrain to refresh
    x = (features(probs, aux) - np.array(params["mean"])) / np.array(params["std"])
    total = 0.0
    for m in params["members"]:
        z = np.tanh(x @ np.array(m["w1"]).T + np.array(m["b1"])) @ np.array(m["w2"]).T + np.array(m["b2"])
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        total = total + e / e.sum(1, keepdims=True)
    e = total
    return e / e.sum(1, keepdims=True)
