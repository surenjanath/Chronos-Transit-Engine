# Chronos Transit Engine: transit search, vetting and classification at AUC 0.997 on held-out Kepler stars

**Surenjanath Singh**

10 October 2026 · code: <https://github.com/surenjanath/Chronos-Transit-Engine>

## Abstract

Chronos Transit Engine is an open pipeline that takes a star's light curve from Kepler, K2 or TESS, searches it
for transits, measures the transiting body, and decides whether the signal is a planet, an eclipsing binary, or
noise. This paper describes its classifier and measures it on 1,515 Kepler stars that were held out of all
training. The classifier is a five-network convolutional ensemble that reads the folded light curve, followed by
gradient-boosted trees that weigh the network's opinion against 57 vetting measurements, including a pixel-level
difference-image check of where on the detector the dimming occurs. On the held-out stars it separates planets
from everything else with an area under the ROC curve of 0.997. It calls 330 of 331 NASA-confirmed planets a
planet (99.7%) and keeps 1,087 of 1,094 non-planet signals (99.4%) out of the high-confidence vetting queue.
Counting unconfirmed candidates as planets as well, it calls 411 of 421 (97.6%); nine of the ten it rejects are
candidates NASA has not confirmed. A learning curve and a comparison of alternative second-stage learners show
the result has stopped improving with more data or a different model: what remains is mostly uncertainty in the
labels. The network alone reaches AUC 0.973; nearly all of the gain comes from the measurements and the second
stage, not from the network.

## 1. Introduction

A transit survey produces far more periodic dips than planets. Kepler's pipeline flagged tens of thousands of
Threshold Crossing Events (TCEs); most are eclipsing binaries, light from neighbouring stars, stellar
variability, or instrument artefacts. Sorting them was first done by people, then by rule-based systems (the
Robovetter; Coughlin et al. 2016, Thompson et al. 2018), then by learned classifiers: the Autovetter random
forest (McCauliff et al. 2015), the AstroNet convolutional network on folded light curves (Shallue & Vanderburg
2018), and later models that add centroid and stellar information (Ansdell et al. 2018) or reproduce the full
set of vetting diagnostics as network inputs (ExoMiner; Valizadegan et al. 2022).

This work asks a practical question: how well can a small system, built and run on one laptop, do the whole job
from raw photometry to a verdict, and where does its accuracy actually come from? The answers are measured
rather than assumed. Every figure in this paper comes from stars the classifier never saw in training, every
model choice was made by cross-validation on training stars only, and results that got worse are reported
alongside those that got better.

## 2. Data and labels

**Signals.** The labelled set is the Kepler DR24 TCE table (Q1–Q17) with the Kepler Science Office training
labels: planet candidate (PC), astrophysical false positive (AFP), and non-transiting phenomenon (NTP)
(Catanzarite 2015). These are the labels AstroNet was trained and scored on. Planets and binaries are included
when they are detectable (expected S/N of at least 10 in six quarters).

**Label updates.** The DR24 labels date from 2015. Each TCE is matched to NASA's current cumulative KOI table by
star and period (within 0.2%). Where NASA now lists the signal as CONFIRMED the label becomes planet; where it
lists FALSE POSITIVE the label becomes binary, or noise if the "not transit-like" flag is set. A signal still
listed as an unresolved CANDIDATE keeps its 2015 label. This changed 125 labels. For a like-for-like comparison
with earlier versions, results are also reported on the original 913 held-out stars with their unmodified 2015
labels.

**Photometry.** Long-cadence PDCSAP light curves for all seventeen quarters (about four years, 16 quarters per
star on average) from MAST. For the pixel-level check, target pixel files for quarters 3–10.

**Split.** A star is held out if its KIC number is divisible by five. This fixes the split by star, never by
signal, so no star contributes to both sides.

| | Stars | Planet | Binary | Noise |
|---|---|---|---|---|
| Training | 5,981 | 1,650 | 3,114 | 1,217 |
| Held out | 1,515 | 421 | 806 | 288 |

Class counts are for the strongest signal on each star. Training also uses every other labelled signal on a
training star (with stronger signals masked first) and three sub-records of each (independent stretches of the
mission), for 37,850 real training examples. Held-out stars are scored once, on their whole record and strongest
signal. All but two of the labelled planet-host stars that meet the detectability cut are in the set, so there
is no further planet data to add from this table.

## 3. Method

### 3.1 From light curve to signal

The light curve is median-normalised, sigma-clipped (5σ in a rolling window of 101 cadences), and detrended with
a Savitzky–Golay filter that is refitted with the transit masked. Box Least Squares (Kovács et al. 2002)
searches periods from 0.5 to 30 days by default, with a check that corrects half- and third-period aliases. A
trapezoid fit refines the ephemeris, and a limb-darkened fit (Mandel & Agol 2002) gives the radius ratio, scaled
semi-major axis and impact parameter with 1σ errors. Each detected signal is removed and the remainder searched
again.

### 3.2 The network

Following AstroNet, the folded curve is binned into a 201-bin global view of the whole orbit and a 61-bin local
view of the transit. The local view has four channels: all transits, odd transits, even transits, and the window
half an orbit later, so the network can see an odd/even difference or a secondary eclipse directly. Two 1D
convolutional branches (three blocks for the global view, two for the local) feed a dense head with three
outputs. Five such networks are trained, each on 30,000 simulated signals plus the real training stars with a
different fifth of those stars held out.

### 3.3 Measurements

Thirteen vetting measurements are taken from the light curve: depth, V-shape, odd/even difference, secondary
eclipse significance and relative depth, S/N, centroid shift and implied separation, radius beyond two Jupiter
radii, duration against that expected for the star, and three outputs of a model-shift test that scans the whole
orbit for a second dip and asks whether a planet could be bright enough to cause it.

Forty further measurements were added in this work: the ephemeris and the star's temperature, radius, mass and
density; out-of-transit brightness changes at the orbital period and at half of it (reflection, beaming,
ellipsoidal distortion), in sigma and relative to the depth; depth consistency between Kepler's four observing
seasons, between quarters, and between individual transits; the share of the detection carried by the single
strongest event; transit asymmetry, the level of the wings, ingress fraction; red-noise level; and S/N per
transit. Anything that cannot be measured is left missing rather than filled in.

### 3.4 Pixel-level check

A dip caused by a neighbouring star moves the centre of light in the photometric aperture. The flux-weighted
centroid from the light-curve file sees this only weakly. So for each star the target pixel files of eight
quarters are read, and for each quarter an out-of-transit image and an in-transit image are averaged over that
quarter's transits. Their difference is an image of where the light was lost. Four numbers are kept: the offset
between the centroid of the difference image and the centroid of the star out of transit, that offset's
significance from the quarter-to-quarter scatter, the distance of the brightest difference pixel from the star,
and the number of quarters measured. This follows the difference-imaging method of Bryson et al. (2013) in a
much reduced form. It could be measured for 97% of signals.

### 3.5 Second stage

Gradient-boosted trees (scikit-learn's histogram implementation; 400 iterations, learning rate 0.05, at most 15
leaves, five random seeds averaged) take the network's two log-odds and the 57 measurements and return the three
class probabilities that a run reports. Classes are weighted equally, so the answer does not drift with how many
binaries happen to be in the training set. Trees suit this evidence: each measurement matters only in
combination with others, and several are missing for some stars.

### 3.6 Keeping the evaluation honest

The second stage must be fitted on network outputs for stars the network has not memorised, or it learns to
trust the network too much. Each training star is therefore scored by the one network that held it out. Model
choices (learner, settings, which measurements to include, class weights) were made by five-fold
cross-validation over training stars, with folds assigned by star. The held-out stars were used only to report.

## 4. Results

### 4.1 Held-out stars

| | AUC (planet vs rest) | PR-AUC | Three-class agreement |
|---|---|---|---|
| Network ensemble alone | 0.973 | 0.918 | 78% |
| With the second stage | **0.997** | **0.993** | 83% |

| With the second stage, 1,515 held-out stars | | |
|---|---|---|
| NASA-confirmed planets called planets | 330 of 331 | **99.7%** |
| Non-planet signals kept out of the vetting queue (P < 0.9) | 1,087 of 1,094 | **99.4%** |
| All labelled planets called planets | 411 of 421 | 97.6% |
| Planets sent straight to the vetting queue (P ≥ 0.9) | 393 of 421 | 93.3% |
| Planet-or-not decision correct | 1,489 of 1,515 | 98.3% |

Bootstrap 95% intervals over held-out stars: AUC 0.994–0.999; all labelled planets called 95–99%; non-planets
kept out of the queue 98.8–99.8%. On the 805 held-out signals whose label NASA has settled either way (confirmed
planet or certified false positive), the AUC is 0.999.

Three-class agreement is limited by weak signals that could be called either "binary" or "noise": 228 of the
254 three-class disagreements are between those two classes, and both mean "not a planet".

Shallue & Vanderburg (2018) report AUC 0.988 for AstroNet on its test set drawn from the same label table. The
two numbers are not directly comparable: their set includes every TCE at any S/N and uses the 2015 labels,
while this one is restricted to detectable signals, scores the strongest signal per star, and uses updated
labels.

### 4.2 Where the accuracy comes from

On the original 913 held-out stars with unmodified 2015 labels:

| Version | AUC | Planets called | Kept out of queue | Planets queued |
|---|---|---|---|---|
| Small neural second stage, 13 measurements | 0.986 | 95.6% | 99.2% | 73.9% |
| Boosted trees, 53 measurements, 5,981 training stars | 0.991 | 95.3% | 99.4% | 90.8% |
| Plus pixel-level check | 0.993 | 95.3% | 99.4% | 91.5% |

The pixel-level check, by cross-validation on training stars: AUC 0.9955 → 0.9961, planets called 96.5% →
97.1%, planets queued 90.6% → 92.7%, non-planets kept out of the queue 99.01% → 98.96% (two stars
in 4,331).

The network matters less than expected. Retraining it with five different recipes and a wider architecture
moved its own AUC between 0.963 and 0.973 and the final result by less than 0.001. Removing the network's
outputs from the second stage altogether costs about 0.0003 AUC. The measurements carry the result.

### 4.3 Is there more to gain?

**More data.** Fitting the second stage on a random share of the training stars (cross-validated, two draws
each):

| Share of training stars | AUC | Planet-or-not correct |
|---|---|---|
| 12.5% | 0.991 | 96.5% |
| 25% | 0.994 | 97.2% |
| 50% | 0.9955 | 97.5% |
| 75% | 0.9959 | 97.6% |
| 100% | 0.9961 | 97.7% |

The curve is flat from half the data upward. The roughly 1,000 labelled stars not yet used are almost all
binaries.

**A different model.** Extremely randomised trees reach AUC 0.9949, a random forest 0.9952, an average of all
three 0.9955, shallower boosted trees 0.9958, and three times as many boosting rounds at a lower rate 0.9960,
against 0.9961 for the model used. Every variant lands between 0.995 and 0.996.

### 4.4 The errors that remain

Of the ten held-out planets not called planets, nine are DR24 candidates that NASA has never confirmed; one is
a confirmed planet. Of the seven non-planets that reach the vetting queue, three carry a 2015 false-positive or
noise label but are listed by NASA today as candidates. The remaining disagreements are therefore
mostly on signals whose true nature is not known, and no classifier can be scored above the accuracy of its
labels.

### 4.5 Choosing where to draw the line

A signal is called a planet when planet is the most probable of the three classes. Lowering the bar trades one
error for the other. On the held-out stars:

| Call a planet when P(planet) is at least | Planets called | Non-planets called planets | Planet-or-not correct |
|---|---|---|---|
| 0.9 (vetting queue) | 93.3% | 0.6% | 97.7% |
| 0.5 | 97.1% | 1.5% | 98.2% |
| 0.3 | 99.0% | 1.7% | 98.5% |
| 0.1 | 99.3% | 2.9% | 97.7% |

The thresholds in use (0.9 for the queue, most-probable class for the label) were fixed beforehand and are not
tuned on these stars; the table shows what a survey that cares more about completeness could choose.

### 4.6 Blind benchmark

In the blind benchmark the pipeline receives only the photometry of held-out stars and must find the period
itself before classifying. On the 43 stars run with the previous version of the classifier (before the
pixel-level check): 13 of 14 planets called planets, none of 29 other signals sent to the vetting queue, the
period found exactly for 19 of 28 periodic signals and at half the period for 3 more.

### 4.7 Detection limits

Mandel–Agol transits injected into real quiet Kepler stars are recovered 98% of the time at S/N of 10 or more,
50% at the 7.1 detection threshold by construction, and 4% below S/N 4.5 (the false-alarm floor). No classifier
can restore a signal the search does not find, so "99% of all planets" is not attainable at low S/N.

## 5. Signal check: a verdict without a light curve

The system also includes a second, deliberately simpler model for a signal known only by its catalogue
parameters. It takes a period, depth and duration, and optionally S/N, number of transits, the star's
temperature, radius and mass, the odd/even difference, the secondary eclipse, and the centroid shift. It is
fitted on the same training stars, with optional inputs withheld at random so that a blank field is something
it has seen. On the same 1,515 held-out stars:

| Inputs given | Planet-or-not correct | AUC |
|---|---|---|
| All twelve | 95.6% | 0.989 |
| No vetting measurements | 90.2% | 0.967 |
| Period, depth, duration and the star | 90.5% | 0.967 |
| Period, depth and duration only | 87.3% | 0.947 |

It is a first look, not a substitute for analysing the photometry, and it has seen Kepler signals only.

## 6. Limitations

- **Kepler only.** The classifier is trained and scored on Kepler long-cadence data. It runs on TESS and K2
  light curves, and passes spot checks on well-known TESS planets, but its accuracy there is not measured.
- **Detectable signals.** Planets and binaries below an expected S/N of 10 are outside the labelled set.
- **Labels.** About a fifth of held-out "planets" are unconfirmed candidates. Scores against them are scores
  against a best guess.
- **Pixel check.** Eight of seventeen quarters, a centroid rather than a fitted point-spread function, and no
  catalogue of neighbouring stars. A full difference-image analysis would do better on crowded fields.
- **Not a confirmation.** A high probability means a signal looks like the planets in the training set.
  Confirming a planet takes independent follow-up.

## 7. Reproducing this

```bash
python manage.py build_kepler_dataset --per-class 1500   # downloads photometry; many hours and tens of GB
python manage.py train_model                              # five networks, then the second stage
python manage.py train_signal_check
python manage.py verify                                   # acceptance checks
python manage.py benchmark --per-class 30                 # blind benchmark
```

The trained model files are not in the repository. To request them, open an issue at
<https://github.com/surenjanath/Chronos-Transit-Engine/issues>.

## Acknowledgements

This work uses data collected by the Kepler mission and obtained from the Mikulski Archive for Space Telescopes
(MAST), and the NASA Exoplanet Archive. Software: NumPy, SciPy, PyTorch, scikit-learn, Astropy, Lightkurve,
batman, Django. The code and experiments were developed with the assistance of Claude (Anthropic).

## References

- Ansdell, M., et al. 2018, ApJL, 869, L7. Scientific domain knowledge improves exoplanet transit classification with deep learning.
- Bryson, S. T., et al. 2013, PASP, 125, 889. Identification of background false positives from Kepler data.
- Catanzarite, J. H. 2015, Autovetter Planet Candidate Catalog for Q1–Q17 Data Release 24, KSCI-19091-001.
- Coughlin, J. L., et al. 2016, ApJS, 224, 12. Planetary candidates observed by Kepler. VII.
- Kovács, G., Zucker, S., & Mazeh, T. 2002, A&A, 391, 369. A box-fitting algorithm in the search for periodic transits.
- Kreidberg, L. 2015, PASP, 127, 1161. batman: BAsic Transit Model cAlculatioN in Python.
- Mandel, K., & Agol, E. 2002, ApJL, 580, L171. Analytic light curves for planetary transit searches.
- McCauliff, S. D., et al. 2015, ApJ, 806, 6. Automatic classification of Kepler planetary transit candidates.
- Shallue, C. J., & Vanderburg, A. 2018, AJ, 155, 94. Identifying exoplanets with deep learning.
- Thompson, S. E., et al. 2018, ApJS, 235, 38. Planetary candidates observed by Kepler. VIII.
- Valizadegan, H., et al. 2022, ApJ, 926, 120. ExoMiner: a highly accurate and explainable deep learning classifier.
