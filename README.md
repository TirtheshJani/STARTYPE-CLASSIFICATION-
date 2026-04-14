# Star Type Classification

Classify stars into one of six broad types from their physical properties
using a scikit-learn pipeline. The project exposes a reusable Python
package, command-line scripts for training and inference, and the
original exploratory notebook.

## Dataset

`Stars.csv` contains 240 stars with the following columns:

| Column          | Description                                         |
|-----------------|-----------------------------------------------------|
| `Temperature`   | Surface temperature in Kelvin                       |
| `L`             | Luminosity relative to the Sun (L / L\_\u2609)       |
| `R`             | Radius relative to the Sun (R / R\_\u2609)           |
| `A_M`           | Absolute magnitude (M\_v)                           |
| `Color`         | General color of the spectrum                       |
| `Spectral_Class`| Spectral class (O, B, A, F, G, K, M)                |
| `Type`          | Target label — 0–5 (see below)                      |

**Target labels**

| Code | Star type     |
|------|---------------|
| 0    | Red Dwarf     |
| 1    | Brown Dwarf   |
| 2    | White Dwarf   |
| 3    | Main Sequence |
| 4    | Super Giant   |
| 5    | Hyper Giant   |

## Repository layout

```
.
├─ Stars.csv                 # Raw dataset
├─ StartypeclassificationprojectTirtheshjani.ipynb  # Exploratory analysis
├─ requirements.txt
├─ src/
│  ├─ data_utils.py          # Loading + categorical normalisation
│  ├─ star_classifier.py     # sklearn pipeline + training/inference
│  └─ visualize.py           # HR diagram / confusion matrix helpers
├─ scripts/
│  ├─ train.py               # CLI: train and persist the pipeline
│  └─ predict.py             # CLI: classify new rows
└─ tests/
   └─ test_data_utils.py
```

## Quick start

```bash
git clone https://github.com/TirtheshJani/STARTYPE-CLASSIFICATION-.git
cd STARTYPE-CLASSIFICATION-
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Train the classifier

```bash
python -m scripts.train --dataset Stars.csv --model models/star_classifier.joblib
```

This prints test-set accuracy, 5-fold cross-validation scores and a
per-class classification report, then saves the fitted pipeline to
`models/star_classifier.joblib`.

### Predict on new data

```bash
python -m scripts.predict \
    --model models/star_classifier.joblib \
    --input new_stars.csv \
    --output predictions.csv
```

The input CSV must provide the six feature columns
(`Temperature`, `L`, `R`, `A_M`, `Color`, `Spectral_Class`). The output
adds `predicted_type` (0–5) and `predicted_label` (human-readable name).

### Run the tests

```bash
pip install pytest
pytest
```

## Modelling notes

* Numeric features are standard-scaled; categorical features are one-hot
  encoded via a `ColumnTransformer`, so the whole flow is reproducible
  through a single `Pipeline`.
* The default classifier is `RandomForestClassifier(n_estimators=200)`
  with a fixed `random_state`. Train / test splits are stratified on the
  target so every star type is represented in both folds.
* `src/data_utils.normalize_color` collapses the free-form `Color`
  column (e.g. "Blue White", "blue-white", "Whitish") into canonical
  labels, which measurably reduces the one-hot feature space.

## License

This project is released for educational purposes.
