# Chronos Transit Engine

Finds planets in starlight. Chronos takes a star's brightness over time from NASA's TESS, Kepler or K2 missions,
searches it for the small, regular dips a planet makes when it crosses its star, measures the planet, and decides
whether the dip is a planet, an eclipsing binary star, or noise. It then shows you where that star is, to scale,
on a map that runs from Earth's orbit to the whole Galaxy.

One Django project: server-rendered pages, the science pipeline, and the archive. Built to [`prd.md`](prd.md).

**Paper:** [how the classifier works and how it was measured](docs/paper.md) (AUC 0.997 on 1,515 held-out Kepler stars).

![The map, zoomed to the stars within 25 light-years, with TRAPPIST-1 selected](docs/screenshots/map-neighbourhood.png)

## Contents

- [What it does](#what-it-does) · [Screenshots](#screenshots) · [Run it](#run) · [Measured results](#measured-results)
- [Signal check](#signal-check) · [Getting the trained model](#getting-the-trained-model) · [Real data](#real-data) · [Map](#map) · [Automation](#automation) · [How the analysis works](#analysis-beyond-the-prd-formulae)
- [Layout](#layout) · [Pipeline](#pipeline) · [Configuration](#configuration) · [Known limits](#known-limits) · [Notes on the PRD](#notes-on-the-prd)

## What it does

| | |
|---|---|
| **Ingest** | Any TESS / Kepler / K2 target from MAST by name, a CSV upload, or a simulated star. |
| **Clean** | Median normalisation, rolling sigma clip, Savitzky-Golay detrend with the transit masked. |
| **Search** | Box Least Squares over 0.5–30 days on all CPU cores, with period-alias correction. |
| **Measure** | Period, depth, duration, planet radius (with a limb-darkened fit and 1σ errors), orbit size, temperature. |
| **Vet** | Odd/even depth, secondary eclipse at any orbital phase, centroid motion, V-shape, radius, duration. |
| **Classify** | A five-network ensemble of dual-branch 1D-CNNs plus a second stage of boosted trees that weighs the vetting evidence. |
| **Signal check** | Type in a period, depth and duration (and whatever else is known) and get a verdict without a light curve; or upload a table of signals. |
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
.venv/bin/python -m pytest                         # 34 tests; set CTE_TEST_DATABASE_URL to run them on PostgreSQL
```

## Measured results

Everything here was measured on real photometry; nothing is simulated except where it says so.

**Classifier, on 1,515 held-out real Kepler stars** (never trained on; labels from the Kepler Science Office,
updated where NASA has since confirmed a planet or ruled a signal a false positive)

| | AUC (planet vs rest) | PR-AUC | 3-class accuracy |
|---|---|---|---|
| CNN ensemble alone | 0.973 | 0.918 | 78% |
| With the second stage (what a run reports) | 0.997 | 0.993 | 83% |

| With the second stage | |
|---|---|
| Planets called planets | 411 of 421 (97.6%) |
| ...of which NASA-confirmed planets | 330 of 331 (99.7%) |
| Planets sent straight to the vetting queue (P ≥ 0.9) | 393 of 421 (93.3%) |
| Other signals kept out of the vetting queue | 1,087 of 1,094 (99.4%) |

Nine of the ten planets not called planets are candidates NASA has not confirmed; three of the seven other signals
that reach the queue are ones NASA now lists as candidates. The second stage also uses Kepler's pixel data: the
position of the dimming on the detector during transit, compared with the position of the star.

Most three-class disagreements are between "binary" and "noise" for weak signals; both are "not a planet".
PRD floor: AUC 0.96, PR-AUC 0.88. Training set: 30,000 simulated signals plus 5,981 real stars (37,850 real examples:
every labelled signal on a star, each seen through three independent stretches of the four-year mission and once whole).

**Blind benchmark, 90 held-out stars, current classifier** (the pipeline gets only the photometry and must find the period itself)

| | |
|---|---|
| Period found exactly | 54 of 60 |
| Real planets called planets | 30 of 30 |
| Real planets sent straight to the vetting queue | 25 of 30 |
| Other signals sent to the vetting queue | 2 of 60 |
| Planet-or-not decision right | 87 of 90 (97%) |
| Class agrees with the catalogue label | 83% |
| Planet-vs-rest AUC | 0.980 |

One of the two other signals in the queue, TCE 9811915-1, carries a 2015 "noise" label but is listed by NASA today as a
candidate. An earlier 90-star run with the previous classifier scored 29 of 30 planets called, 2 of 60 in the queue,
55 of 60 periods exact (plus 2 at an alias), 86% class agreement and AUC 0.982: the new one calls one more planet, and
is slightly lower on period recovery, class agreement and AUC.

**Injection and recovery** (Mandel-Agol transits from `batman` injected into real stars)

| | |
|---|---|
| Quiet stars, injected S/N ≥ 10 | 98% recovered |
| Quiet stars, at the 7.1 detection threshold | 50% |
| Quiet stars, below S/N 4.5 | 4% (the false-alarm floor) |
| Variable stars, second planet behind the star's own signal | 46 of 48 (42 of the 44 detectable in principle) |
| Variable stars, nothing injected: stars given a spurious extra signal | 2 of 45 |

**PRD acceptance criteria** (`manage.py verify`): period error 4.6 × 10⁻⁵ (limit 10⁻⁴), depth 3.25% (limit 5%), radius
3.78% (limit 5%), log integrity 100%. **Golden targets**: 12 of 12 assertions hold, including all six TRAPPIST-1
planets b–g.

**Has it found a planet NASA does not have?** No confirmed new planet. Across the 96 real stars analysed:
every signal it sent to the vetting queue is a NASA-confirmed planet except KOI-143.01 (see Known limits). One star
it called a planet against its training label (TCE 6851425-1, labelled a false positive in the 2015 DR24 set) is
listed today as the confirmed planet Kepler-477 b. It also reported six further signals on five stars that no NASA
table lists. Four of those stars are known eclipsing binaries or false positives, where leftover stellar signal is
the likely cause. The fifth is the confirmed host Kepler-1772: a 28.35-day, 47 ppm dip at S/N 7.6, barely above the
detection minimum. That one is a candidate worth a second look, nothing more: confirming a planet takes
independent follow-up.

## Signal check

`/check/` answers "is it a planet?" for a signal you only have numbers for. Enter a period, depth and duration;
add the signal-to-noise, the number of transits, the star, the odd/even difference, the secondary eclipse and the
centroid shift if you know them. The page returns the three class probabilities, plain-language warnings (a body
too large to be a planet, a secondary eclipse too deep, a dip that lasts too long for the star), what each
optional input did to the answer, and the size and orbit the numbers imply.

![Signal check](docs/screenshots/signal-check.png)

- **Its own model.** Boosted trees fitted on the same Kepler training stars as the main classifier
  (`manage.py train_signal_check`), with optional inputs withheld at random during fitting so a blank field is
  something it has seen.
- **Measured on the same 1,515 held-out stars.** Planet-or-not is right for 95.6% with every input given (AUC 0.989),
  90.5% with period, depth, duration and the star, and 87.3% from period, depth and duration alone. The page shows this table.
- **Many at once.** Upload a CSV (NASA's KOI and TCE table exports work as they are) and get it back with a verdict per row.
- **JSON.** `GET /check/score.json?period_days=…&depth_ppm=…&duration_hours=…`
- It is a first look. It has seen Kepler signals only, and the full analysis of the light curve is more accurate.

## Getting the trained model

The trained model files are not in this repository. To request them, open an issue. Everything needed to train
your own is here: `build_kepler_dataset`, `train_model`, `train_signal_check` (see the [paper](docs/paper.md)).

## Real data

- **MAST runs.** Any TESS / Kepler / K2 target by name or id; several at once with commas. Each run is cross-matched
  against the NASA Exoplanet Archive (confirmed planets, TOIs, KOIs) and the page shows whether the detection agrees.
  Every archive call has a timeout and one retry.
- **Kepler TCE benchmark.** `build_kepler_dataset` downloads long-cadence photometry (all seventeen quarters, about four
  years, 16 on average per star) for a class-balanced sample from the Kepler DR24 TCE table (Q1–Q17) with the Science Office labels PC / AFP / NTP - the
  ground truth AstroNet was scored on - plus every other labelled star whose photometry is already on disk. Every labelled
  signal on a star is used, and a 2015 label is replaced where NASA's current KOI table has since confirmed the planet or
  ruled the signal a false positive. Planets and binaries are included only when detectable (expected S/N ≥ 10 in six
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
- **Second-stage classifier.** Gradient-boosted trees (five seeds, averaged) combine the CNN probabilities with the
  vetting evidence (odd/even, secondary eclipse too deep for a planet, centroid, radius beyond 2 R_J, duration) and
  forty further measurements of the light curve: brightness changes in step with the orbit (reflection, beaming,
  ellipsoidal distortion), depth from one observing season to the next and from transit to transit, transit shape and
  symmetry, and the star's temperature, size and density. It is fitted on the real training stars plus simulated giant
  planets and binaries, with the three classes weighted equally, and its answer is what every run reports.
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
counted and left off; orbits are drawn to scale in size, seen from above, because their orientation on the sky is not in the catalogue.

- Three viewpoints: the Galaxy from above, edge-on, and the sky as seen from Earth.
- Select a star for its distance, the year its light left, travel times (light, Voyager 1, Parker Solar Probe, an
  airliner), its planets with sizes to scale, and its orbits beside Earth's orbit and the liquid-water zone.
- "Travel there from Earth" animates the whole way out. Search finds any star or planet by name.
- Stars analysed by this engine are highlighted, and target and run pages link to their place on the map.
- A dot on each orbit shows where the planet is today: the Solar System from mean motions, other systems by counting
  orbits since a measured transit, when the planet stood on the line from its star to the Sun.

## Period aliases

With several quarters of data the trial-period grid is capped and too coarse for long periods, which lets P/2 or P/4
win. Two checks correct this: the best peak and its simple multiples are re-examined on a locally fine grid, and if
only every second or third epoch shows a transit the period is doubled or tripled. On the 45-star blind benchmark
this raised exact period recovery from 24 to 27 of 30.

## Performance

The period search splits its trial periods across CPU cores using a pool of worker processes that is started on the
first run and reused. The primary search of a six-quarter Kepler star (22,000 cadences, 60,000 trial periods) takes about 7 s,
against 43 s single-threaded; the periodogram is bit-for-bit identical. (Those timings predate a rewrite of the inner loop that made
both paths about 2.9 times faster, again with an identical periodogram.) `CTE_BLS_WORKERS=1` forces the serial path.

Two experimental options exist and are off by default because they measured no benefit (17 of 36 injected planets
recovered on variable Kepler stars either way): `adaptive_detrend` (shorten the Savitzky-Golay window until the
residuals are white) and `remove_periodic_variability` (divide out a non-transit periodic waveform, then search again).

## Known limits

- **Variable stars.** Two of 48 injected planets are still missed, and 2 of 45 variable stars get a spurious extra
  signal. Both misses have a signal-to-noise near 10 against the star's own correlated noise; in one the right period
  was examined and judged not significant. Loosening that test to catch them would raise the spurious rate, so the
  balance is left where it is. Additional signals are candidates for inspection, not classified detections.
- **Speed.** A four-year Kepler star (60,000 cadences) takes about half a minute, longer on variable stars where the
  further-signal search makes several passes. The period search is 2.9 times faster than it was, with a bit-for-bit
  identical periodogram (Kepler-8 end to end: 44 s before, 32 s after, on a machine busy with other work). A Kepler
  run also spends 5–10 s fetching pixel files. A 27-day TESS sector takes a few seconds.
- **The map.** 4,202 of 6,418 planets are drawn where they are on their orbit today, from a measured transit time and
  period; the rest have no transit time, or a period too uncertain to say where they are by now. Which way an orbit is
  turned on the sky is not in the catalogue. 27 planets with neither a distance nor a parallax are left off.
- **Kepler only for the pixel check and the accuracy figures.** TESS and K2 runs work and pass the golden targets, but
  the classifier's accuracy there is not measured on a labelled set.

**Cleared in this version**

- **Kepler-8 b** sat in manual review at 0.899. It now scores 0.987 and goes to the vetting queue (0.984 from a model
  that never saw the star in training).
- **KOI-143.01**, which the archive lists as a false positive, used to reach the vetting queue. It is a held-out star
  and is now called an eclipsing binary, with P(planet) = 0.03.
- **WASP-12 b** scored 0.942, under the 0.95 golden-target bar. It now scores 0.988.
- **The map** drew no planets on their orbits and left 28 planets off. See above.

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
