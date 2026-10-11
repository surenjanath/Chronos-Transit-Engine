# The trained model

These are the files the engine was measured with in the [paper](../docs/paper.md). `./run.sh` copies them into
`data/models/` on first start; to do it by hand, `mkdir -p data/models && cp models/* data/models/`.

| File | What it is |
|---|---|
| `cte_cnn.pt` | The five neural networks (PyTorch state dictionaries) |
| `stacker.joblib` | The second stage: five gradient-boosted tree models on the networks' output and 57 measurements |
| `signalcheck.joblib` | The parameters-only model behind the Signal check page |
| `metrics.json` | Scores on the 1,515 held-out Kepler stars, with the prediction for each star |
| `signalcheck.json` | The Signal check model's scores on the same stars |
| `history.json` | The training run that produced the networks |

**Scores on held-out stars:** AUC 0.997; 330 of 331 NASA-confirmed planets called planets; 1,087 of 1,094 other
signals kept out of the vetting queue.

**Trained on** 5,981 Kepler stars (37,850 examples) plus simulated signals. Kepler long-cadence data only: its
accuracy on TESS and K2 is not measured.

**Versions.** The `.joblib` files are Python pickles made with scikit-learn 1.9.1. A different scikit-learn may
refuse to load them or warn; install the same version, or retrain with `manage.py train_model` and
`manage.py train_signal_check`. As with any pickle, load these files only from a source you trust.

**Licence:** MIT, like the code.
