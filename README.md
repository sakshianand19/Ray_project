# Distributed KNN Classification with Ray

**Python · Ray · scikit-learn · Pandas · NumPy · Jupyter**

This project compares a standard **K-Nearest Neighbors (KNN)** classifier with a **distributed
version built on [Ray](https://www.ray.io/)**. It measures how parallel training affects accuracy and
run time across datasets of different sizes, from 150 rows to 100,000 rows.

## How it works

| Version | Approach |
| --- | --- |
| **Plain KNN** | One scikit-learn `KNeighborsClassifier` trained on the full training set. |
| **Ray KNN** | The training set is split into **5 shards**. Each shard trains its own KNN model in parallel as a Ray remote task (`@ray.remote`), every model predicts the full test set, and the 5 predictions are combined by **majority vote**. |

```
                 ┌─► worker 1: KNN on shard 1 ─┐
 training data ──┼─► worker 2: KNN on shard 2 ─┤
  (5 shards)     ├─►            ...            ├─► majority vote ─► final prediction
                 └─► worker 5: KNN on shard 5 ─┘
```

Both versions use an 80/20 train/test split (`random_state=42`) and time training plus
prediction.

## Results

| Dataset | Rows | Plain KNN | Ray KNN (5 workers) |
| --- | --- | --- | --- |
| Synthetic dataset 1 (10 features, 2 classes) | 100,000 | 84.25% · 4.83 s | **86.23%** · 14.63 s |
| Synthetic dataset 2 | – | 82.62% · 5.84 s | **85.63%** · 11.60 s |
| Iris (3 classes) | 150 | 100% · 0.007 s | 100% · 5.88 s |
| US COVID-19 county data | – | 51.47% · 9.34 s | 44.57% · 32.89 s |

Measured on a single local machine (Ray 2.39, Python 3.10). Times include training and
prediction; the Ray times also include task scheduling and data transfer to the workers.

## Findings

- **Accuracy:** the sharded ensemble was **2–3 percentage points more accurate** on both large
  synthetic datasets. Combining several models trained on different data subsets smooths out
  noisy neighbors, much like bagging.
- **Speed:** the Ray version was **slower on every dataset**. On one machine, the cost of starting
  tasks and copying data to workers outweighs the parallel speed-up at these sizes (most visible
  on Iris, where the actual work takes milliseconds). Ray's benefits would show on much larger
  datasets or when spread across a multi-machine cluster.
- **COVID-19 data:** this notebook predicts the number of *confirmed cases*, a count with many
  distinct values, using a classifier, which explains the low accuracy. It would be better
  modeled as regression (for example `KNeighborsRegressor`) or by grouping case counts into ranges.

## Data preprocessing (COVID-19 dataset)

The real-world COVID-19 county data needed cleaning before modeling:

1. Dropped rows missing county (`Admin2`) or state, and filled other missing values.
2. Removed identifier columns that carry no signal (`UID`, `iso2`, `iso3`, `FIPS`, and so on).
3. Split the `Date` column into year, month and day features.
4. Label-encoded the county and state names.
5. Scaled the features with `StandardScaler` (fit on the training set only).

## Project structure

```
├── DATASET 1- 10 features and 1,00,000 instances/
│   ├── KNN_python_Synthetic1.ipynb       Plain KNN
│   ├── KNN_RAY_VERSION_Synthetic1.ipynb  Ray KNN
│   └── classification_dataset.csv        100,000 rows × 10 features, 2 classes
├── IRIS DATASET/
│   ├── KNN_python_Iris.ipynb
│   ├── KNN_RAY_VERSION_Iris.ipynb
│   └── irisNum.csv                       150 rows, 3 classes
├── covid-19 dataset/
│   ├── KNN_python_Synthetic_Covid.ipynb  Includes the preprocessing steps
│   └── KNN_RAY_Version_Covid.ipynb
├── KNN_python_Synthetic2.ipynb           Synthetic dataset 2
├── KNN_RAY_VERSION_Synthetic2.ipynb
└── Spotify dataset/
    └── Spotify_songs_preprocessed_dataset.csv   Preprocessed data (not used by a notebook yet)
```

> **Data not included:** the Synthetic dataset 2 file (`classification_dataset1.csv`) and the
> COVID-19 file (`usa_county_wise.csv`, from the Johns Hopkins CSSE COVID-19 time series) are not
> in this repository. Place them next to their notebooks to re-run those experiments.

## Running the notebooks

```bash
pip install "ray==2.39.0" scikit-learn pandas numpy scipy jupyter
jupyter notebook
```

Open any `KNN_python_*` notebook for the plain version or `KNN_RAY_*` for the distributed one and
run all cells. The Ray notebooks call `ray.init()`, which starts a local Ray instance
automatically.

## Author

**Sakshi Anand** · [GitHub](https://github.com/sakshianand19) · [LinkedIn](https://www.linkedin.com/in/sakshi-anand-731464216/)
