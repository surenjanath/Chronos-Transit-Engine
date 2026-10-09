# Chronos Transit Engine

Finds planets in starlight. Chronos takes a star's brightness over time from NASA's TESS, Kepler or K2 missions,
searches it for the small, regular dips a planet makes when it crosses its star, measures the planet, and decides
whether the dip is a planet, an eclipsing binary star, or noise. It then shows you where that star is, to scale,
on a map that runs from Earth's orbit to the whole Galaxy.

One Django project: server-rendered pages, the science pipeline, and the archive. Built to [`prd.md`](prd.md).

![The map, zoomed to the stars within 25 light-years, with TRAPPIST-1 selected](docs/screenshots/map-neighbourhood.png)

## Contents

- [What it does](#what-it-does) · [Screenshots](#screenshots) · [Run it](#run) · [Measured results](#measured-results)
- [Real data](#real-data) · [Map](#map) · [Automation](#automation) · [How the analysis works](#analysis-beyond-the-prd-formulae)
- [Layout](#layout) · [Pipeline](#pipeline) · [Configuration](#configuration) · [Known limits](#known-limits) · [Notes on the PRD](#notes-on-the-prd)

## What it does

| | |
|---|---|
| **Ingest** | Any TESS / Kepler / K2 target from MAST by name, a CSV upload, or a simulated star. |
| **Clean** | Median normalisation, rolling sigma clip, Savitzky-Golay detrend with the transit masked. |
| **Search** | Box Least Squares over 0.5–30 days on all CPU cores, with period-alias correction. |
| **Measure** | Period, depth, duration, planet radius (with a limb-darkened fit and 1σ errors), orbit size, temperature. |
| **Vet** | Odd/even depth, secondary eclipse at any orbital phase, centroid motion, V-shape, radius, duration. |
| **Classify** | A five-network ensemble of dual-branch 1D-CNNs plus a second stage that weighs the vetting evidence. |
| **Look further** | Removes each signal and the star's own variability, then searches again for more planets. |
| **Check itself** | Blind benchmark on held-out real stars, injection-and-recovery, golden targets, PRD acceptance criteria. |
| **Locate** | A zoomable map of 4,800 real planetary systems, with distances and travel times from Earth. |
| **Record** | Hash-chained JSONL audit log per run, Parquet archives, SQLite or PostgreSQL. |

## Screenshots

**Map: the Solar System to scale, with Voyager 1's path** (scroll out from here to the whole Galaxy)

![Solar System](docs/screenshots/map-solar-system.png)

**Map: every known planetary system in the Galaxy.** The spike is Kepler's field of view; the line to the right is
microlensing discoveries towards the Galactic centre.

![Galaxy](docs/screenshots/map-galaxy.png)

**Map: TRAPPIST-1's seven orbits with its liquid-water zone.** The whole system would fit well inside Mercury's orbit.

![TRAPPIST-1 orbits](docs/screenshots/map-trappist-orbits.png)

**Map: the sky as seen from Earth**

![Sky](docs/screenshots/map-sky.png)

**A run: WASP-12 b from TESS photometry** (derived parameters, diagnostics, visual model, light curve, periodogram, audit trail)

![Run page](docs/screenshots/run.png)

| Overview | Blind benchmark |
|---|---|
| ![Overview](docs/screenshots/overview.png) | ![Benchmark](docs/screenshots/benchmark.png) |
| **Verification** | **Atlas: detections among known planets** |
| ![Validation](docs/screenshots/validation.png) | ![Atlas](docs/screenshots/atlas.png) |

## Run

Requires Python 3.11 or newer. The `data/` directory (database, trained model, downloaded photometry) is not in the
repository; the first start builds what it needs.

```bash
git clone https://github.com/surenjanath/Chronos-Transit-Engine.git
cd Chronos-Transit-Engine
./run.sh            # http://127.0.0.1:8765
```

The first start creates a virtualenv, applies migrations and trains the classifier if no checkpoint exists
(simulated stars only, about 10 minutes). To reproduce the numbers below, also build the real dataset and retrain:

```bash
.venv/bin/python manage.py build_kepler_dataset --per-class 330   # ~1.5 GB of Kepler photometry from MAST
.venv/bin/python manage.py train_model                             # five networks, ~15 minutes on 10 cores
```

All commands:

```bash
.venv/bin/python manage.py seed                    # demonstration runs: 4 MAST targets + 8 synthetic injections
.venv/bin/python manage.py build_kepler_dataset    # real labelled data: Kepler TCEs from MAST (--per-class N)
.venv/bin/python manage.py train_model             # train on simulations + the real dataset, score on held-out stars
.venv/bin/python manage.py benchmark               # blind end-to-end runs on held-out real stars (--per-class N)
.venv/bin/python manage.py injection_study         # injection and recovery against S/N (--trials N)
.venv/bin/python manage.py autopilot               # one automation cycle (--loop to keep going)
.venv/bin/python manage.py verify                  # PRD §6 criteria + physics, golden targets, injections
.venv/bin/python manage.py reset_runs --yes        # clear runs, logs and archives
.venv/bin/python -m pytest                         # 33 tests; set CTE_TEST_DATABASE_URL to run them on PostgreSQL
```

## Measured results

Everything here was measured on real photometry; nothing is simulated except where it says so.

**Classifier, on 188 held-out real Kepler stars** (never trained on; labels from the Kepler Science Office)

| | AUC (planet vs rest) | PR-AUC | 3-class accuracy |
|---|---|---|---|
| CNN ensemble alone | 0.982 | 0.966 | 90% |
| With the second stage (what a run reports) | 0.985 | 0.973 | 91% |

PRD floor: AUC 0.96, PR-AUC 0.88. Training set: 30,000 simulated signals plus 802 real stars (3,201 real examples:
each training star is seen through three independent stretches of the four-year mission and once whole).

**Blind benchmark, 45 held-out stars** (the pipeline gets only the photometry and must find the period itself)

| | |
|---|---|
| Period found exactly | 27 of 30 (plus 1 at an alias) |
| Real planets called planets | 14 of 15 |
| False positives sent to the vetting queue | 1 of 30 |
| Class agrees with the catalogue label | 84% |

**Injection and recovery** (Mandel-Agol transits from `batman` injected into real stars)

| | |
|---|---|
| Quiet stars, injected S/N ≥ 10 | 98% recovered |
| Quiet stars, at the 7.1 detection threshold | 50% |
| Quiet stars, below S/N 4.5 | 4% (the false-alarm floor) |
| Variable stars, second planet behind the star's own signal | 46 of 48 (42 of the 44 detectable in principle) |
| Variable stars, nothing injected: stars given a spurious extra signal | 2 of 45 |

**PRD acceptance criteria** (`manage.py verify`): period error 4.6 × 10⁻⁵ (limit 10⁻⁴), depth 3.25% (limit 5%), radius
3.78% (limit 5%), log integrity 100%. **Golden targets**: 11 of 12 assertions hold, including all six TRAPPIST-1
planets b–g; the one miss is listed under [Known limits](#known-limits).

**Has it found a planet NASA does not have?** No. Every transit signal it reported on the 49 real stars analysed here
is already in NASA's catalogues. One weak unlisted signal (KIC 4149450, 28.35 d, S/N 7.6, the minimum for detection)
is far more likely noise than a planet. An unmatched signal would in any case be a *candidate*: confirming a planet
takes independent follow-up.

## Real data

- **MAST runs.** Any TESS / Kepler / K2 target by name or id; several at once with commas. Each run is cross-matched
  against the NASA Exoplanet Archive (confirmed planets, TOIs, KOIs) and the page shows whether the detection agrees.
  Every archive call has a timeout and one retry.
- **Kepler TCE benchmark.** `build_kepler_dataset` downloads long-cadence photometry (all seventeen quarters, about four
  years, 16 on average per star) for a class-balanced sample from the Kepler DR24 TCE table (Q1–Q17) with the Science Office labels PC / AFP / NTP - the
  ground truth AstroNet was scored on. Planets and binaries are included only when detectable (expected S/N ≥ 10 in six
  quarters). A quarter the archive fails to answer for is asked for again, never recorded as "not observed". A star's train / held-out split is fixed by its KIC number.
- **Training.** `train_model` mixes the real training stars into the simulated population and reports AUC and PR-AUC
  on the held-out real stars. Every training run is recorded in `data/models/history.json`.
- **Blind benchmark.** `benchmark` gives the pipeline only the photometry of held-out stars; it must find the period,
  derive the parameters and classify. The Benchmark page scores the result against the labels.
- **Injection and recovery.** `injection_study` injects Mandel-Agol transits (`batman`) into real NTP stars and
  measures the fraction recovered against injected S/N.

## Automation

The Automation page (or `manage.py autopilot --loop`) runs cycles on a schedule. Each cycle:

1. pulls the newest TESS Objects of Interest not yet in the catalogue and analyses them;
2. downloads more labelled Kepler stars into the dataset;
3. retrains once enough new stars have arrived, and activates the new model only if it scores no worse than the
   current one on the fixed held-out real stars (champion / challenger).

It is off by default. Each cycle and each training run is logged and listed on the page.

## Analysis beyond the PRD formulae

- **Limb-darkened fit.** R_p/R★, a/R★ and impact parameter are fitted with a quadratic limb-darkened model and
  reported with 1σ errors. The PRD's δ = (R_p/R★)² overestimates the radius by a few percent because limb darkening
  deepens mid-transit; both values are shown.
- **Further signals.** Each detected signal is removed and the remainder searched again (two further signals by
  default). The strongest few peaks of each search are judged: a residue of something already found, the star's own
  variability (a smooth wave, a "dip" longer than any transit of that star could last, or one no deeper than the
  star's wiggles at other phases), or a further transit-like signal. Only the last is reported; the others are
  divided out and the search repeats, so rotation or pulsation can neither use up the budget nor hide a planet.
  A further signal must stand out against the star's correlated noise, not just white noise. The page lists every
  peak examined and what was decided. Measured on fresh variable stars: 46 of 48 injected planets recovered (the
  first version of this search recovered about half), with a spurious signal on 2 of 45 stars when nothing is injected.
- **Second dip at any phase.** A model-shift test scans the whole orbit for a second dip, as the Kepler Robovetter
  does; an eccentric binary's secondary eclipse need not sit at phase 0.5. It counts against a planet only when it
  is too bright for a planet to produce (albedo above 1), so a hot Jupiter's real occultation is not held against
  it. On the labelled stars it fires on 168 of 309 false positives and 4 of 330 planets.
- **Ensemble and honest stacking.** Five networks are trained, each holding out a different fifth of the real
  training stars. The second stage is fitted on predictions for stars the predicting network never saw; fitted on
  memorised stars it would learn to trust the network too much. The local CNN branch sees four channels: all
  transits, odd, even, and the window half an orbit later.
- **Second-stage classifier.** An ensemble of small networks combines the CNN probabilities with the vetting
  evidence (odd/even, secondary eclipse too deep for a planet, centroid, radius beyond 2 R_J, duration); it is fitted
  on the real training stars plus simulated giant planets and binaries, and its answer is what every run reports.
- **K2.** Systematics-corrected community light curves (EVEREST, K2SFF) are preferred over the raw pipeline product.
- **Shape-derived geometry and stellar density**, compared with the catalogued star.
- **Vetting flags.** Odd/even depth (3σ and a 10% depth difference), secondary eclipse at phase 0.5, V-shape,
  radius, duration, and centroid motion. The centroid flag needs both significance, measured against the scatter at
  other orbital phases, and an implied source at least half a pixel away: on real Kepler data a significance-only
  rule flagged confirmed planets with deep on-target transits.

## Watching a run

While a run is in progress its page says what is happening in plain words ("Trying tens of thousands of possible
orbital periods, looking for a dip that repeats on schedule"), shows the eleven stages, the time elapsed, and a
running list of what has been found so far (measurements read, glitches removed, the period, the radius, the
verdict). During the search for further planets it reports each pass and what was just removed. A diagram of the
classifier sits idle until its stage arrives, then animates; on the finished page the same diagram is drawn with
that run's own folded light curve as input and its own probabilities as output.

## Visual diagnostics on every run

Animated system model (sky view, orbit from above, light curve, all to scale), expected-against-observed overlay with
residuals, odd/even comparison, phase-0.5 window, depth of each transit, every transit stacked, CNN occlusion saliency,
shape-derived impact parameter and stellar density, light curve with trend, BLS periodogram and both CNN input tensors.

## Layout

| Path | Contents |
|---|---|
| `cte/` | Framework-free science: galactic placement for the map (`galaxy.py`), conditioning (§2.1–2.2), BLS (§2.3), physics (§2.4), synthetic injection, dual-branch CNN, vetting diagnostics, visual-diagnostic arrays, MAST/CSV ingest, Kepler KOI dataset, archive cross-match, JSONL telemetry, verification suite |
| `engine/` | Django app: models, services (run execution, jobs, aggregates), forms, views, management commands |
| `templates/`, `static/` | Django templates, one stylesheet, small page scripts (canvas plots, live status) |
| `docs/screenshots/` | Images used in this README |
| `data/` | SQLite database, CNN checkpoint, JSONL audit logs, Parquet archives, MAST cache (git-ignored) |

## Pipeline

1. Median normalization, iterative rolling sigma clip (k = 5, 101 cadences)
2. Savitzky-Golay detrend (p = 3), repeated with the transit masked
3. Box Least Squares over P ∈ [0.5, 30] d, q ∈ [0.01, 0.1]; ephemeris refined against the fitted trapezoid
4. Phase fold; global (201-bin median) and local (61-bin) tensors
5. R_p, a, T_dur, T_eq from the PRD formulae
6. False-positive diagnostics: odd/even depth, secondary eclipse, centroid shift, V-shape, radius, duration
7. Dual-branch 1D-CNN softmax: planet / eclipsing binary / noise
8. JSONL audit record per stage (schema-validated, SHA-256 hash chained) and Parquet archive

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `CTE_DATABASE_URL` | SQLite in `data/` | `postgresql://user:pass@host:port/db` (or `...@/db?host=/socket/dir`) to archive in PostgreSQL; `docker-compose.yml` provides one |
| `CTE_DATA_DIR` | `./data` | Where the database, logs, archives and checkpoints live |
| `CTE_PORT` | `8765` | Port used by `run.sh` |
| `CTE_MAX_WORKERS` | `2` | Concurrent pipeline runs |
| `CTE_BLS_WORKERS` | cores − 2, at most 8 | Worker processes for the period search; `1` disables parallelism |
| `CTE_DEBUG` | `1` | Set `0` outside local use, with `DJANGO_SECRET_KEY` and `CTE_ALLOWED_HOSTS` |

## Map

`/map/` is a pannable, zoomable map of real space: scroll from Earth's orbit (to scale) out past Voyager 1, the
nearest stars and the Kepler field to the whole Galaxy. Every one of the ~4,800 host stars is placed from its
measured sky position and distance in the NASA Exoplanet Archive (`cte/galaxy.py`, cached in
`data/catalog/systems.json`, refreshed from the page). Nothing is invented: planets with no measured distance are
counted and left off; orbits are drawn to scale in size but face-on, because their orientation and each planet's
position today are not in the catalogue.

- Three viewpoints: the Galaxy from above, edge-on, and the sky as seen from Earth.
- Select a star for its distance, the year its light left, travel times (light, Voyager 1, Parker Solar Probe, an
  airliner), its planets with sizes to scale, and its orbits beside Earth's orbit and the liquid-water zone.
- "Travel there from Earth" animates the whole way out. Search finds any star or planet by name.
- Stars analysed by this engine are highlighted, and target and run pages link to their place on the map.

## Period aliases

With several quarters of data the trial-period grid is capped and too coarse for long periods, which lets P/2 or P/4
win. Two checks correct this: the best peak and its simple multiples are re-examined on a locally fine grid, and if
only every second or third epoch shows a transit the period is doubled or tripled. On the 45-star blind benchmark
this raised exact period recovery from 24 to 27 of 30.

## Performance

The period search splits its trial periods across CPU cores using a pool of worker processes that is started on the
first run and reused. The primary search of a six-quarter Kepler star (22,000 cadences, 60,000 trial periods) takes about 7 s,
against 43 s single-threaded; the periodogram is bit-for-bit identical. `CTE_BLS_WORKERS=1` forces the serial path.

Two experimental options exist and are off by default because they measured no benefit (17 of 36 injected planets
recovered on variable Kepler stars either way): `adaptive_detrend` (shorten the Savitzky-Golay window until the
residuals are white) and `remove_periodic_variability` (divide out a non-transit periodic waveform, then search again).

## Known limits

- **WASP-12 b scores 0.942, under the 0.95 golden-target bar** (it is still sent to the vetting queue, which needs
  0.90). Its transit is long for its orbit and the shape-only network gives it 0.59; the second stage lifts it.
- **Kepler-8 b sits in manual review at 0.899, a hair under the 0.90 queue threshold.** Among real Kepler signals
  deeper than 0.5% the sample holds 159 false positives and 22 planets, so a deep, high-impact transit is
  ambiguous on shape alone (the network gives it 0.25). Review is the calibrated answer, not a bug.
- **KOI-143.01 reaches the vetting queue although the archive lists it as a false positive.** The archive rejects
  it for a secondary eclipse. In all seventeen quarters this engine measures none (−0.9σ at phase 0.5, no
  significant dip at any other phase) and equal odd and even depths (0.2σ), so the photometry it has cannot
  reject it.
- **Variable stars.** Two of 48 injected planets are still missed, and 2 of 45 variable stars get a spurious extra
  signal. Additional signals are candidates for inspection, not classified detections.
- **Speed.** A four-year Kepler star (60,000 cadences) takes 40–190 s end to end, most of it in the further-signal
  search on variable stars. A 27-day TESS sector takes a few seconds.
- **The map.** Orbits are drawn face-on and planets are not placed along them: orientation and present position are
  not in the catalogue. 28 planets with no measured distance are left off.

## Notes on the PRD

- **R⊕ conversion.** §2.4 and the §5 sample use R☉/R⊕ = 109.076; the §4 listing's constant gives 109.198. The engine uses 109.076.
- **§5 sample values.** The sample `semi_major_axis_au` (0.04712) and `equilibrium_temp_k` (1398.24) do not follow from the §2.4
  formulae for a 1 M☉ star at P = 3.524748 d; the formulae give 0.04533 AU and 1197 K, and the engine follows the formulae.
- **Transit depth.** §2.4 defines δ = 1 − min(F). The minimum of noisy bins is biased deep, so the engine adopts the
  flat-bottom depth of a trapezoid fit and logs the min-bin and BLS depths beside it.
- **Classifier benchmark.** The AstroNet TFRecords are not bundled; the classifier is scored on held-out stars from
  the same Kepler DR24 TCE table with the same labels, using all available quarters of photometry per star.
- **TRAPPIST-1.** Observed only by K2. The pipeline uses the systematics-corrected EVEREST light curve and recovers
  planets b–g from campaign 12 when searched with shorter duty cycles (q from 0.004) than the PRD default, because
  its transits last under an hour.
- **TRAPPIST-1 e.** Its radius benchmark is analytic (archive depth → engine radius); WASP-12 b is also run end to end from TESS photometry.
