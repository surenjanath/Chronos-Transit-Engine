import json
import math

import numpy as np
import pytest

from cte import conditioning, physics, telemetry
from cte.bls import bls_search, fourier_comb_period
from cte.physics import StellarParameters, TransitAnalyticsEngine
from cte.synthetic import SyntheticSpec, overlap_fraction, synthesize


def test_median_normalization():
    f = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    norm, med = conditioning.median_normalize(f)
    assert med == 30.0 and np.median(norm) == 1.0


def test_sigma_clip_rejects_outliers_and_keeps_transits():
    rng = np.random.default_rng(0)
    f = 1 + rng.normal(0, 1e-3, 5000)
    f[2000:2080] -= 0.01          # an 80-cadence transit must survive
    f[[100, 900, 4000]] += 0.02   # single-cadence spikes must go
    keep, iterations = conditioning.sigma_clip(f, k=5.0, window=101)
    assert not keep[[100, 900, 4000]].any()
    assert keep[2000:2080].all()
    assert iterations >= 1


def test_savgol_detrend_removes_slow_trend():
    t = np.arange(0, 20, 2 / 1440)
    f = 1 + 0.01 * np.sin(2 * np.pi * t / 9.0)
    trend, window = conditioning.savgol_detrend(t, f, 1.5, 3)
    assert window % 2 == 1
    assert np.std(f / trend) < 1e-5


def test_physics_formulae():
    star = StellarParameters(5778.0, 1.0, 1.0)
    p = TransitAnalyticsEngine(star, 0.3).compute_all_parameters(3.524748, 0.0146, 0.0)
    k = math.sqrt(0.0146)
    assert p.radius_ratio == pytest.approx(k, abs=1e-6)
    assert p.radius_earth == pytest.approx(k * 109.076, abs=2e-3)        # PRD 2.4 A
    a = (3.524748 / 365.256) ** (2 / 3)
    assert p.semi_major_axis_au == pytest.approx(a, abs=1e-5)            # PRD 2.4 B
    rs_a = 6.957e8 / (a * 1.496e11)
    assert p.transit_duration_hours == pytest.approx(3.524748 / math.pi * math.asin(rs_a * (1 + k)) * 24, abs=2e-3)
    assert p.equilibrium_temp_k == pytest.approx(5778 * math.sqrt(rs_a / 2) * 0.7 ** 0.25, abs=0.02)


def test_earth_sun_sanity():
    out = physics.derive(StellarParameters(5772.0, 1.0, 1.0), 365.256, 8.4e-5)
    assert out["semi_major_axis_au"] == pytest.approx(1.0, abs=1e-4)
    assert out["radius_earth"] == pytest.approx(1.0, abs=0.01)
    assert out["equilibrium_temp_k"] == pytest.approx(254.8, abs=1.5)
    assert out["transit_duration_hours"] == pytest.approx(13.0, abs=0.3)


def test_phase_fold_and_binning():
    t = np.linspace(0, 30, 6000)
    phase, flux, _ = TransitAnalyticsEngine.phase_fold(t, np.ones_like(t), 3.0, 1.0)
    assert phase.min() >= -0.5 and phase.max() < 0.5 and np.all(np.diff(phase) >= 0)
    g = physics.global_view(phase, flux, 201)
    l = physics.local_view(phase, flux, 0.03, 61)
    assert g.shape == (201,) and l.shape == (61,) and np.allclose(g, 1) and np.allclose(l, 1)


def test_binned_median_matches_numpy():
    rng = np.random.default_rng(1)
    idx, vals = rng.integers(0, 7, 500), rng.normal(size=500)
    med, counts = physics.binned_median(idx, vals, 8)
    for b in range(7):
        assert med[b] == pytest.approx(np.median(vals[idx == b]))
    assert np.isnan(med[7]) and counts[7] == 0


def test_overlap_fraction_limits():
    assert overlap_fraction(np.array([0.0]), 0.1)[0] == pytest.approx(0.01)
    assert overlap_fraction(np.array([1.2]), 0.1)[0] == 0.0
    assert 0 < overlap_fraction(np.array([1.0]), 0.1)[0] < 0.01


def test_bls_recovers_period_and_depth():
    d = synthesize(SyntheticSpec(period_days=2.6158, depth=0.004, noise_ppm=500, impact_param=0.0,
                                 limb_darkening=False, seed=5))
    f, med = conditioning.median_normalize(d["flux"])
    keep, _ = conditioning.sigma_clip(f)
    t, f, e = d["time"][keep], f[keep], d["flux_err"][keep] / med
    trend, _ = conditioning.savgol_detrend(t, f)
    res, periods, sr = bls_search(t, f / trend, e)
    assert abs(res.period - 2.6158) / 2.6158 < 2e-3   # raw BLS; the pipeline refines this to < 1e-4
    assert 0.01 <= res.q <= 0.1
    assert res.sde > 6
    assert len(periods) == len(sr) and sr.max() > 0
    comb = fourier_comb_period(t, f / trend)
    assert min(abs(comb / 2.6158 - r) for r in (0.5, 1.0, 2.0)) < 0.02


def test_log_schema_validation_and_hash_chain(tmp_path):
    log = telemetry.RunLogger("abc", path=tmp_path / "abc.jsonl")
    log.info("first", "INGEST", value=np.float64(1.5), nan=float("nan"), arr=np.arange(3))
    log.warning("second", "BLS_SEARCH")
    lines = [json.loads(x) for x in (tmp_path / "abc.jsonl").read_text().splitlines()]
    assert lines[0]["telemetry"] == {"run_id": "abc", "stage": "INGEST", "value": 1.5, "nan": None, "arr": [0, 1, 2]}
    assert telemetry.verify_jsonl(tmp_path / "abc.jsonl") == {"records": 2, "valid": 2, "chain_ok": True, "errors": []}
    # tampering with a record breaks the chain
    lines[0]["message"] = "edited"
    (tmp_path / "abc.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    assert telemetry.verify_jsonl(tmp_path / "abc.jsonl")["chain_ok"] is False


@pytest.mark.parametrize("bad", [
    {"level": "INFO", "module": "m", "message": "no timestamp"},
    {"timestamp": "not-a-date", "level": "INFO", "module": "m", "message": "x"},
    {"timestamp": "2026-10-07T21:27:01+00:00", "level": "LOUD", "module": "m", "message": "x"},
    {"timestamp": "2026-10-07T21:27:01+00:00", "level": "INFO", "module": "m", "message": "x", "telemetry": [1]},
    {"timestamp": "2026-10-07T21:27:01+00:00", "level": "INFO", "module": "m", "message": "x", "extra": 1},
])
def test_invalid_records_are_rejected(bad):
    with pytest.raises(telemetry.LogIntegrityError):
        telemetry.validate_record(bad)


def test_prd_sample_records_validate():
    telemetry.validate_record({"timestamp": "2026-10-07T21:27:01.102341Z", "level": "INFO", "module": "transit_engine",
                               "message": "Initializing Chronos Transit Pipeline execution",
                               "telemetry": {"target": "TIC 261136679"}})


def test_diagnostic_sanity_suite():
    """Fundamental physics assertions: depth conservation, Kepler's third law, rejection of non-physical input."""
    from cte.physics import derive, validate_transit
    from cte.validation import physics_sanity
    one = derive(StellarParameters(5772.0, 1.0, 1.0), 365.256, 0.01)
    assert abs(one["radius_earth"] - 10.9) < 0.2
    assert abs(one["semi_major_axis_au"] - 1.0) < 0.001
    for period, duration in ((3.0, 3.0), (3.0, 5.0), (3.0, float("nan")), (3.0, 0.0), (-1.0, 0.1)):
        with pytest.raises(ValueError):
            validate_transit(period, duration)
    validate_transit(3.0, 0.12)
    assert physics_sanity()["status"] == "pass"


def test_injection_model_matches_batman_and_average_precision():
    from cte.injection import transit_model
    from cte.model import average_precision
    from cte.synthetic import transit_flux
    t = np.linspace(-0.2, 0.2, 800)
    m, engine = transit_model(t, 3.5, 0.0, 0.1, 9.0, 0.3, (0.4, 0.25), 1e-4)
    builtin = transit_flux(t, 3.5, 0.0, 0.1, 9.0, 0.3, (0.4, 0.25))
    assert engine.startswith("batman")
    assert (1 - m.min()) == pytest.approx(1 - builtin.min(), rel=0.02)   # depth agrees with the built-in model
    assert np.abs(m - builtin).max() < 2e-3  # small-planet approximation differs slightly in ingress
    assert average_precision(np.array([0.9, 0.8, 0.3, 0.1]), np.array([True, True, False, False])) == 1.0
    assert average_precision(np.array([0.9, 0.8, 0.3, 0.1]), np.array([False, True, False, True])) == pytest.approx(0.5)


def test_galactic_coordinates_and_placement():
    from cte import galaxy
    l, b = galaxy.equatorial_to_galactic(266.40499, -28.93617)  # Sagittarius A*
    assert min(l, 360 - l) < 0.1 and abs(b) < 0.1
    assert abs(galaxy.equatorial_to_galactic(192.85948, 27.12825)[1] - 90.0) < 1e-3  # north galactic pole
    ra, dec = galaxy.galactic_to_equatorial(*galaxy.equatorial_to_galactic(101.3, -16.7))
    assert abs(ra - 101.3) < 1e-6 and abs(dec + 16.7) < 1e-6
    rows = [{"pl_name": "A b", "hostname": "A", "ra": 266.40499, "dec": -28.93617, "sy_dist": 10.0, "pl_orbper": 3.0},
            {"pl_name": "A c", "hostname": "A", "ra": 266.40499, "dec": -28.93617, "sy_dist": 10.0, "pl_orbper": 1.0},
            {"pl_name": "B b", "hostname": "B", "ra": 10.0, "dec": 5.0, "sy_dist": None}]
    cat = galaxy.build_systems(rows)
    assert cat["n_systems"] == 1 and cat["n_planets"] == 2 and cat["unplaced_planets"] == 1  # no distance: left off, never guessed
    a = cat["systems"][0]
    assert abs(a["x"] - 32.616) < 0.01 and abs(a["y"]) < 0.01 and abs(a["z"]) < 0.01      # towards the Galactic centre
    assert [p["name"] for p in a["planets"]] == ["A c", "A b"]
    assert galaxy.locate(["a"], catalogue=cat)["name"] == "A" and galaxy.locate([], 266.405, -28.936, cat)["name"] == "A"
    assert galaxy.locate(["nothing"], 0.0, 0.0, cat) is None
    t = galaxy.travel(4.2465, 2026)
    assert 74000 < t["voyager1_years"] < 76000 and abs(t["light_left_year"] - 2021.75) < 0.01


def test_survey_matches_catalogued_periods_and_their_aliases():
    from cte.survey import match
    listed = [(9.0823, "Kepler-1301 b (CONFIRMED)"), (2.7533, "TCE 7908893-1 (NTP)")]
    assert match(9.0825, listed) == "Kepler-1301 b (CONFIRMED)"
    assert match(4.5412, listed) == "Kepler-1301 b (CONFIRMED) at 0.5x its period"
    assert match(5.5066, listed) == "TCE 7908893-1 (NTP) at 2x its period"
    assert match(28.464, listed) is None and match(3.0, []) is None
    assert match(5.9844, [(29.9218, "Kepler-1676 b (CONFIRMED)")]) == "Kepler-1676 b (CONFIRMED) at 1/5 of its period"
    assert match(6.05, [(29.9218, "Kepler-1676 b (CONFIRMED)")]) is None


def test_transit_timing_finds_a_wave_and_only_a_real_one():
    from cte import ttv
    rng = np.random.default_rng(1)
    t = np.arange(0, 400, 0.0204)
    period, t0 = 5.0, 2.5
    n = np.round((t - t0) / period)
    moved = t0 + n * period + (20 / 1440) * np.sin(2 * np.pi * n / 20)  # each transit up to 20 minutes early or late
    wavy = ttv.transit_times(t, 1 + ttv.template(t - moved, 0.12, 0.02, 0.003) + rng.normal(0, 0.0004, len(t)), period, t0, 2.88, 0.48, 3000)
    assert wavy["measurable"] and wavy["detected"] and abs(wavy["wave"]["amplitude_minutes"] - 20) < 3
    steady = 1 + ttv.template((t - t0 + period / 2) % period - period / 2, 0.12, 0.02, 0.003) + rng.normal(0, 0.0004, len(t))
    quiet = ttv.transit_times(t, steady, period, t0, 2.88, 0.48, 3000)
    assert quiet["measurable"] and not quiet["detected"] and quiet["scatter_minutes"] < 6
    shallow = ttv.transit_times(t, steady, period, t0, 2.88, 0.48, 100)  # the same data read as a 100 ppm signal: not timeable
    assert not shallow["measurable"] and "too shallow" in shallow["reason"]
