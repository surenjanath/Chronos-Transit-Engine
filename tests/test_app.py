"""End-to-end tests through the Django views: a full pipeline run per signal class, plus every page and action."""
import io
import os

import numpy as np
import pytest
from django.urls import reverse

from cte import model
from cte.synthetic import SyntheticSpec, synthesize
from engine import services
from engine.models import LogEvent, Run, Setting

pytestmark = [pytest.mark.django_db, pytest.mark.skipif(not model.CHECKPOINT.exists(), reason="no trained CNN checkpoint")]

PLANET = {"source": "synthetic", "target_name": "T planet", "kind": "planet", "period_days": 3.524748, "depth_ppm": 14600,
          "impact_param_syn": 0.1, "noise_ppm": 1000, "variability_amp": 0.002, "baseline_days": 27.4, "cadence_min": 2, "seed": 7}


def submit(client, **fields):
    resp = client.post(reverse("new"), fields)
    assert resp.status_code == 302, resp.context["form"].errors if resp.context else resp.content[:400]
    return Run.objects.select_related("target").get(id=resp.url.rstrip("/").rsplit("/", 1)[-1])


@pytest.fixture
def planet(client):
    return submit(client, **PLANET)


def test_planet_run_end_to_end(client, planet):
    assert planet.status == "completed", planet.error
    res = planet.results
    assert planet.prediction == "PLANET_CANDIDATE" and planet.disposition == "CONFIRMED_VETTING_QUEUE"
    assert res["truth"]["period_rel_error"] < 1e-4
    assert sum(res["classification"]["probabilities"].values()) == pytest.approx(1.0, abs=2e-3)
    assert len(res["tensors"]["global"]) == 201 and len(res["tensors"]["local"]) == 61
    assert set(res["parameters"]) >= {"period_days", "transit_depth_ppm", "radius_ratio", "radius_earth",
                                      "semi_major_axis_au", "transit_duration_hours", "equilibrium_temp_k"}
    assert set(res["archive"]) == {"lightcurve.parquet", "periodogram.parquet", "folded.parquet", "tensors.parquet"}

    v, shape = res["visuals"], res["shape"]
    assert len(v["observed"]) == len(v["model_fit"]) == 61 and len(v["stack"]) >= 5 and len(v["transit_depths"]) >= 5
    assert v["odd"] is not None and v["even"] is not None and v["residual_rms_ppm"] < 2000
    assert shape["valid"] and shape["a_over_rstar"] == pytest.approx(res["parameters"]["a_over_rstar"], rel=0.35)
    assert len(res["explain"]["local"]) == 61 and len(res["explain"]["global"]) == 201
    assert int(np.argmax(res["explain"]["local"])) in range(20, 41)  # the network looks at the transit itself

    refined = res["refined"]  # the limb-darkened fit removes the bias of delta = (Rp/R*)^2
    true_radius = (0.0146 ** 0.5) * 109.076
    assert refined["converged"] and refined["radius_earth"] == pytest.approx(true_radius, rel=0.03)
    assert abs(refined["radius_earth"] - true_radius) < abs(res["parameters"]["radius_earth"] - true_radius)
    assert refined["radius_earth_err"] > 0 and refined["impact_param"] < 0.5
    assert res["additional_signals"] == []

    page = client.get(reverse("run", args=[planet.id]))
    assert page.status_code == 200 and "Visual model" in page.content.decode() and "limb-darkened fit" in page.content.decode()
    html = page.content.decode()
    assert "T planet" in html and "Derived parameters" in html and "Vetting queue" in html

    stages = list(planet.logs.order_by("seq").values_list("stage", flat=True))
    for stage in ("INGEST", "NORMALIZATION", "OUTLIER_REJECTION", "DETRENDING", "BLS_SEARCH", "PHASE_FOLDING",
                  "PHYSICS_DERIVATION", "VETTING_DIAGNOSTICS", "CNN_INFERENCE", "ARCHIVE"):
        assert stage in stages
    assert list(planet.logs.order_by("seq").values_list("seq", flat=True)) == list(range(1, len(stages) + 1))


def test_series_downloads_and_integrity(client, planet):
    series = client.get(reverse("run_series", args=[planet.id])).json()
    assert len(series["time"]) == len(series["flux_flat"]) > 1000
    assert len(series["global_view"]) == 201 and len(series["local_x"]) == 61
    assert client.get(reverse("run_status", args=[planet.id])).json()["status"] == "completed"
    parquet = b"".join(client.get(reverse("run_download", args=[planet.id, "lightcurve.parquet"])).streaming_content)
    assert parquet[:4] == b"PAR1"
    assert client.get(reverse("run_download", args=[planet.id, "audit.jsonl"])).status_code == 200
    assert client.get(reverse("run_download", args=[planet.id, "nope.parquet"])).status_code == 404
    resp = client.post(reverse("run_verify", args=[planet.id]), follow=True)
    assert "Audit trail intact" in resp.content.decode()


def test_false_positives_are_not_exported(client, planet):
    eb = submit(client, source="synthetic", kind="eclipsing_binary", period_days=5.318, depth_ppm=50000, impact_param_syn=1.1, seed=15)
    blend = submit(client, source="synthetic", kind="blend", period_days=4.1057, depth_ppm=30000, noise_ppm=300, seed=16)
    noise = submit(client, source="synthetic", kind="noise", noise_ppm=900, seed=17)
    assert eb.disposition.startswith("FALSE_POSITIVE")
    assert "CENTROID_OFFSET" in [f["code"] for f in blend.results["flags"]]
    assert blend.disposition.startswith("FALSE_POSITIVE")
    assert noise.disposition in ("NO_SIGNIFICANT_SIGNAL", "REJECTED_NOISE")
    export = client.get(reverse("export_candidates")).content.decode()
    assert planet.id in export
    for run in (eb, blend, noise):
        assert run.id not in export


def test_csv_upload_run(client):
    d = synthesize(SyntheticSpec(period_days=2.2, depth=0.006, noise_ppm=600, seed=3))
    buf = io.StringIO()
    np.savetxt(buf, np.c_[d["time"], d["flux"], d["flux_err"]], delimiter=",", header="time,flux,flux_err", comments="")
    upload = io.BytesIO(buf.getvalue().encode())
    upload.name = "lc.csv"
    run = submit(client, source="upload", file=upload, target_name="T upload", t_eff_k=5200, r_star_solar=0.85, m_star_solar=0.88)
    assert run.status == "completed", run.error
    assert run.period_days == pytest.approx(2.2, rel=1e-3)
    assert run.results["stellar"]["r_star_solar"] == 0.85 and run.target.mission == "Upload"

    bad = io.BytesIO(b"a,b\n1,2\n")
    bad.name = "bad.csv"
    resp = client.post(reverse("new"), {"source": "upload", "file": bad})
    assert resp.status_code == 200 and "file" in resp.context["form"].errors


def test_vetting_and_export(client, planet):
    assert planet.id in client.get(reverse("vetting")).content.decode()
    client.post(reverse("run_vet", args=[planet.id]), {"status": "false_positive", "note": "test note"})
    planet.refresh_from_db()
    assert planet.vetting_status == "false_positive" and planet.vetting_note == "test note" and planet.vetted_at
    assert planet.id not in client.get(reverse("export_candidates")).content.decode()
    assert planet.id in client.get(reverse("vetting") + "?tab=false_positive").content.decode()
    resp = client.post(reverse("run_vet", args=[planet.id]), {"status": "confirmed", "next": "https://evil.example/"})
    assert resp.url == reverse("run", args=[planet.id])  # off-site redirects are refused
    assert planet.id in client.get(reverse("export_candidates")).content.decode()
    assert planet.logs.filter(stage="HUMAN_VETTING").count() == 2
    assert client.post(reverse("run_vet", args=[planet.id]), {"status": "bogus"}, follow=True).status_code == 200
    assert "T planet" in client.get(reverse("export_runs")).content.decode()


def test_pages_filters_and_logs(client, planet):
    for name in ("overview", "targets", "runs", "vetting", "atlas", "benchmark", "automation", "new", "calculator", "validation", "logs",
                 "settings", "method"):
        assert client.get(reverse(name)).status_code == 200, name
    assert client.get(reverse("target", args=[planet.target_id])).status_code == 200
    assert "T planet" in client.get(reverse("runs"), {"q": "T planet", "status": "completed"}).content.decode()
    assert "T planet" not in client.get(reverse("runs"), {"disposition": "REJECTED_NOISE"}).content.decode()
    assert "T planet" in client.get(reverse("targets"), {"mission": "Synthetic"}).content.decode()
    assert "Derived physical exoplanetary parameters" in client.get(reverse("logs"), {"stage": "PHYSICS_DERIVATION"}).content.decode()
    jsonl = client.get(reverse("export_logs")).content.decode()
    assert jsonl.count("\n") == LogEvent.objects.count()
    live = client.get(reverse("live")).json()
    assert live["active"] == [] and live["pending"] == 1
    assert client.get(reverse("log_tail")).json()["lines"]
    assert client.get(reverse("new"), {"source": "synthetic"}).status_code == 200
    o = services.overview()
    assert o["n_done"] == 1 and o["n_candidates"] == 1 and o["yield_pct"] == 100


def test_settings_and_calculator(client):
    values = dict(services.pipeline_settings(), sigma_k=4.5)
    assert client.post(reverse("settings"), values).status_code == 302
    assert services.pipeline_settings()["sigma_k"] == 4.5
    bad = client.post(reverse("settings"), dict(values, period_min=40))
    assert bad.status_code == 200 and bad.context["form"].errors
    assert client.post(reverse("settings"), {"reset": "1"}).status_code == 302
    assert services.pipeline_settings()["sigma_k"] == 5.0 and not Setting.objects.exists()

    q = {"t_eff_k": 5778, "r_star_solar": 1, "m_star_solar": 1, "period_days": 3.524748, "depth_ppm": 14600,
         "impact_param": 0, "bond_albedo": 0.3}
    out = client.get(reverse("calculator_compute"), q).json()
    assert out["parameters"]["radius_earth"] == pytest.approx(13.18, abs=0.01)
    assert 1 - min(out["curve"]["flux"]) == pytest.approx(0.0146, rel=1e-3)
    assert client.get(reverse("calculator_compute"), dict(q, depth_ppm=-1)).status_code == 422
    page = client.get(reverse("calculator"), dict(q, record=1), follow=True)
    assert "13.18" in page.content.decode()
    assert LogEvent.objects.filter(stage="PHYSICS_DERIVATION", telemetry__origin="interactive_vetting_tool").exists()


def test_form_validation_rerun_delete_and_failure(client, planet):
    resp = client.post(reverse("new"), {"source": "mast", "mast_target": ""})
    assert resp.status_code == 200 and "mast_target" in resp.context["form"].errors
    resp = client.post(reverse("new"), dict(PLANET, period_days=999))
    assert resp.status_code == 200 and "period_days" in resp.context["form"].errors
    assert client.get(reverse("run", args=["doesnotexist"])).status_code == 404
    assert client.get(reverse("run_rerun", args=[planet.id])).status_code == 405  # state changes are POST-only

    resp = client.post(reverse("run_rerun", args=[planet.id]))
    new_id = resp.url.rstrip("/").rsplit("/", 1)[-1]
    assert new_id != planet.id and Run.objects.get(id=new_id).status == "completed"
    assert client.post(reverse("run_delete", args=[new_id])).status_code == 302
    assert not Run.objects.filter(id=new_id).exists()

    failed_id = services.submit_run({"source": "upload", "upload_id": "0" * 32, "target_name": "T missing"})
    failed = Run.objects.get(id=failed_id)
    assert failed.status == "failed" and "not found" in failed.error
    assert failed.logs.filter(level="ERROR").exists()
    assert "Run failed" in client.get(reverse("run", args=[failed_id])).content.decode()


def test_batch_submission_validation(client):
    resp = client.post(reverse("new"), {"source": "mast", "mast_target": ", ".join(f"T{i}" for i in range(13))})
    assert resp.status_code == 200 and "mast_target" in resp.context["form"].errors


def test_shape_geometry_and_crossmatch_logic(monkeypatch):
    import math

    from cte import archive
    from cte.physics import shape_geometry

    k, a, period, b = 0.1, 9.0, 3.5, 0.4
    t14 = period / math.pi * math.asin(math.sqrt((1 + k) ** 2 - b * b) / a)
    t23 = period / math.pi * math.asin(math.sqrt((1 - k) ** 2 - b * b) / a)
    g = shape_geometry(period, k * k, t14, (t14 - t23) / 2)
    assert g["valid"] and g["impact_param"] == pytest.approx(b, abs=2e-3) and g["a_over_rstar"] == pytest.approx(a, rel=5e-3)
    assert not shape_geometry(period, k * k, 0.1, 0.06)["valid"]  # ingress longer than half the transit: V-shaped

    known = [{"name": "X b", "status": "Confirmed planet", "period_days": 4.0, "radius_earth": 10.0, "depth_ppm": 9000.0,
              "catalogue": "Planetary Systems"}]
    monkeypatch.setattr(archive, "known_objects", lambda name: known)
    assert archive.crossmatch("TIC 1", 4.00001, 11.0, 9000)["match"]["alias"] == "same period"
    assert archive.crossmatch("TIC 1", 2.0, 11.0, 9000)["match"]["alias"] == "half the catalogued period"
    assert archive.crossmatch("TIC 1", 3.1, 11.0, 9000)["match"] is None
    assert archive.known_objects.__name__ and archive.crossmatch("TIC 1", 4.0, 11.0, 9000)["match"]["radius_rel_error"] == pytest.approx(0.1)


def test_startup_only_fails_orphaned_jobs(client, planet):
    import os

    from engine.models import ValidationRun
    mine = Run.objects.create(id="liveowned0001", target=planet.target, source="synthetic", status="running", config={"_pid": os.getpid()})
    orphan = Run.objects.create(id="orphaned00001", target=planet.target, source="synthetic", status="running", config={"_pid": 2 ** 22 + 12345})
    legacy = Run.objects.create(id="legacynopid01", target=planet.target, source="synthetic", status="queued", config={})
    v_live = ValidationRun.objects.create(status="running", results={"pid": os.getpid()})
    v_dead = ValidationRun.objects.create(status="running", results={"pid": 2 ** 22 + 12345})
    services._started = False
    services.startup()
    for obj in (mine, orphan, legacy, v_live, v_dead):
        obj.refresh_from_db()
    assert mine.status == "running" and orphan.status == "failed" and legacy.status == "failed"
    assert v_live.status == "running" and v_dead.status == "failed"


def test_second_signal_is_found():
    import logging
    import pathlib

    from cte import ingest
    from cte.config import DEFAULT_PIPELINE
    from cte.core import SOLAR, process
    from cte.synthetic import a_over_rstar, transit_flux
    from cte.telemetry import RunLogger

    logging.getLogger("CTE-Core").setLevel(logging.ERROR)
    d = ingest.make_synthetic(vars(SyntheticSpec(seed=21, period_days=3.1, depth=0.004, noise_ppm=400)), SOLAR, "two planets")
    d.flux = d.flux * transit_flux(d.time, 7.37, d.time[0] + 2.2, 0.045, a_over_rstar(7.37, 1, 1), 0.2)
    res, _ = process(d, dict(DEFAULT_PIPELINE), RunLogger(None, path=pathlib.Path(os.devnull)))
    assert res["bls"]["period"] == pytest.approx(3.1, rel=1e-3)
    assert len(res["additional_signals"]) == 1 and res["additional_signals"][0]["period_days"] == pytest.approx(7.37, rel=2e-3)


def test_autopilot_cycle_and_settings(client, monkeypatch):
    from engine import autopilot
    from engine.models import AutomationCycle

    assert client.post(reverse("automation"), {"interval_minutes": 5, "tois_per_cycle": 1, "stars_per_cycle": 0,
                                               "retrain_after_stars": 60}, follow=True).status_code == 200
    assert autopilot.config()["interval_minutes"] == 360  # out-of-range value rejected
    client.post(reverse("automation"), {"enabled": "on", "interval_minutes": 30, "tois_per_cycle": 1, "stars_per_cycle": 0,
                                        "retrain_after_stars": 60})
    cfg = autopilot.config()
    assert cfg["enabled"] and cfg["interval_minutes"] == 30 and autopilot.next_due() is not None

    # one cycle with the archive and MAST stubbed: a new TOI is analysed, a TOI without photometry is skipped
    monkeypatch.setattr(autopilot, "newest_tois", lambda limit=150: [
        {"toi": "9001.01", "tid": 111, "tfopwg_disp": "PC", "pl_orbper": 3.52, "toi_created": "2026-08-01 00:00:00"},
        {"toi": "9002.01", "tid": 222, "tfopwg_disp": None, "pl_orbper": 5.0, "toi_created": "2026-07-01 00:00:00"}])
    monkeypatch.setattr(autopilot.ingest, "search_mast", lambda name, mission: [] if name == "TIC 111" else
                        [{"label": "TESS Sector 90", "sequence": 90, "exptime": 120.0, "author": "SPOC", "target_name": "222", "year": 2026}])
    synth = lambda *a, **k: autopilot.ingest.make_synthetic({"kind": "planet", "seed": 7}, {"t_eff_k": 5778.0, "r_star_solar": 1.0, "m_star_solar": 1.0}, "TIC 222")
    monkeypatch.setattr(services.ingest, "fetch_mast", synth)
    monkeypatch.setattr(services.archive, "crossmatch", lambda *a, **k: {"objects": [], "match": None})
    monkeypatch.setattr(autopilot.kepler, "dataset_summary", lambda: None)
    resp = client.post(reverse("automation_run"), follow=True)
    cycle = AutomationCycle.objects.first()
    assert cycle.status == "completed", cycle.summary
    runs = cycle.summary["targets"]["runs"]
    assert len(runs) == 1 and runs[0]["toi"] == "9002.01" and runs[0]["status"] == "completed"
    assert cycle.summary["targets"]["skipped"][0]["toi"] == "9001.01"
    assert cycle.summary["training"]["retrained"] is False
    assert Run.objects.get(id=runs[0]["run"]).config["origin"] == "autopilot"
    assert "TOI-9002.01" in resp.content.decode()
    # the same TOIs are not pulled twice
    client.post(reverse("automation_run"))
    assert AutomationCycle.objects.first().summary["targets"]["runs"] == []
    assert LogEvent.objects.filter(stage="AUTOPILOT", message="Autopilot cycle finished").count() == 2


def test_map_places_real_positions_only(client, tmp_path, monkeypatch):
    import json

    from cte import galaxy
    rows = [{"pl_name": "Near b", "hostname": "Near", "ra": 266.40499, "dec": -28.93617, "sy_dist": 2.0, "pl_orbper": 5.0,
             "pl_rade": 1.1, "pl_orbsmax": 0.05, "discoverymethod": "Transit", "disc_year": 2020, "sy_vmag": 5.0,
             "pl_tranmid": 2455000.5, "pl_orbpererr1": 2e-6, "pl_orbpererr2": -3e-6},
            {"pl_name": "Lost b", "hostname": "Lost", "ra": 1.0, "dec": 2.0, "sy_dist": None},
            {"pl_name": "Parallax b", "hostname": "Parallax", "ra": 10.0, "dec": 20.0, "sy_dist": None, "sy_plx": 100.0}]
    monkeypatch.setattr(galaxy, "SYSTEMS", tmp_path / "systems.json")
    galaxy.SYSTEMS.write_text(json.dumps(galaxy.build_systems(rows)))
    galaxy._cache.clear()
    page = client.get(reverse("map"))
    assert page.status_code == 200
    html = page.content.decode()
    assert "1 planet with no measured distance is left off" in html and "Near" in html and "6.52 ly" in html
    data = client.get(reverse("map_data")).json()
    assert [h["name"] for h in data["systems"]] == ["Near", "Parallax"]  # the star without a distance is not given one
    assert abs(data["systems"][0]["d"] - 2.0 * galaxy.LY_PER_PC) < 1e-3
    assert abs(data["systems"][1]["d"] - 10.0 * galaxy.LY_PER_PC) < 1e-3  # a parallax of 100 mas is 10 parsecs
    near = data["systems"][0]["planets"][0]                               # what the map needs to place the planet today
    assert near["t0"] == 2455000.5 and near["Pe"] == 3e-6 and data["systems"][1]["planets"][0]["t0"] is None
    assert 160 < page.context["ctx"]["voyager1"]["au"] < 200 and page.context["ctx"]["radio_ly"] > 100
    galaxy._cache.clear()


def test_signal_check_page_api_and_batch(client):
    from cte import signalcheck
    if not signalcheck.available():
        assert "has not been trained" in client.get(reverse("signal_check")).content.decode()
        assert client.get(reverse("signal_check_score"), {"period_days": 3, "depth_ppm": 500, "duration_hours": 2}).status_code == 503
        return
    binary = {"period_days": 5.318, "depth_ppm": 85000, "duration_hours": 4.6, "snr": 900, "n_transits": 250, "t_eff_k": 5900,
              "r_star_solar": 1.1, "m_star_solar": 1.05, "odd_even_sigma": 14, "secondary_sigma": 40, "secondary_depth_ppm": 21000}
    small = {"period_days": 12.3, "depth_ppm": 420, "duration_hours": 3.9, "snr": 28, "n_transits": 110, "t_eff_k": 5600,
             "r_star_solar": 0.95, "m_star_solar": 0.93, "odd_even_sigma": 0.6, "secondary_sigma": 0.3, "secondary_depth_ppm": 5, "centroid_sigma": 0.5}
    out = client.get(reverse("signal_check_score"), binary).json()
    assert out["prediction"] == "ECLIPSING_BINARY" and out["p_planet"] < 0.05 and out["notes"]
    assert sum(out["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    out = client.get(reverse("signal_check_score"), small).json()
    assert out["prediction"] == "PLANET_CANDIDATE" and out["physics"]["radius_earth"] == pytest.approx(2.12, abs=0.05)
    three = client.get(reverse("signal_check_score"), {"period_days": 9.5, "depth_ppm": 2200, "duration_hours": 2.8}).json()
    assert not three["moved"] and "the star" in three["missing"] and not three["physics"]
    assert client.get(reverse("signal_check_score"), {"period_days": 3}).status_code == 422
    assert client.get(reverse("signal_check_score"), dict(small, m_star_solar="")).status_code == 422  # radius without mass
    page = client.get(reverse("signal_check"), dict(binary, record=1), follow=True).content.decode()
    assert "likely eclipsing binary" in page and "Jupiter radii" in page
    assert LogEvent.objects.filter(stage="SIGNAL_CHECK", telemetry__origin="signal_check").exists()
    table = io.BytesIO(b"# from the archive\nkepoi_name,koi_period,koi_depth,koi_duration,koi_srad,koi_smass\nK1,12.3,420,3.9,0.95,0.93\nK2,5,-4,2,,\n")
    table.name = "koi.csv"
    rows = client.post(reverse("signal_check_batch"), {"table": table}).content.decode().strip().splitlines()
    assert len(rows) == 3 and rows[1].startswith("1,K1,12.3,420.0,3.9") and "depth_ppm" in rows[2].split(",")[-1]
    assert client.post(reverse("signal_check_batch"), {}).status_code == 302
