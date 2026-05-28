# Heart Stroke Prediction

## Overview

A data preparation and classification project focused on handling real-world data challenges: missing values, feature scaling, and severe class imbalance. Using the Stroke Prediction Dataset, the workflow covers full EDA, imputation strategies, scaling comparison, and the effect of random under-sampling on minority-class recall.

**Data source:** [Stroke Prediction Dataset — Kaggle (fedesoriano)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

---

## Dataset

| Property | Detail |
|---|---|
| File | `healthcare-dataset-stroke-data.csv` |
| Rows | 5,110 |
| Columns | 12 |
| Target column | `stroke` (0 = no stroke, 1 = stroke) |
| Class balance | ~95.1% negative / ~4.9% positive (severely imbalanced) |


---

## Workflow

### Part 1: Data Description & Missing Data Preparation

**Inspection**
- Loaded dataset into `df`; confirmed shape (5110, 12).
- Printed `df.head()`, `df.info()`, `df.describe(include='all')`.
- Split columns into `numeric_cols` and `categorical_cols` via `select_dtypes`.

**Numeric columns:** `id`, `age`, `hypertension`, `heart_disease`, `avg_glucose_level`, `bmi`, `stroke`

**Categorical columns:** `gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`

**Correlation**
- Computed and printed the full numeric correlation matrix.
- Notable finding: `age` and `bmi` show the strongest correlation (r ≈ 0.33), suggesting BMI tends to increase with age.

**Visualisations**
- Scatter plot: `age` vs `avg_glucose_level`, coloured by `stroke`.
- Box plot: `bmi` grouped by `stroke` (two boxes).

**Missing values & duplicates**
- `'Unknown'` in `smoking_status` replaced with `NaN` before reporting.

| Column | Missing Count | Missing % |
|---|---|---|
| `bmi` | 201 | ~3.93% |
| `smoking_status` | ~1,544 (post-replacement) | ~30.2% |
| All others | 0 | 0% |

- Duplicate rows: **0**, no removal needed.

**Deletion vs imputation comparison**

| Approach | Rows remaining |
|---|---|
| Listwise deletion (`df_listwise`) | 3,426 |
| Imputation (`df_imputed`) | 5,110 |

Imputation strategy: `bmi` → median; `smoking_status` → most frequent (via `SimpleImputer`). Zero missing values confirmed post-imputation. Cleaned data exported to `cleaned_stroke.csv`.

---

### Part 2: Scaling, Class Imbalance & Resampling

**Features & target**
- `y = df_imputed['stroke']`
- `X = df_imputed[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']]`
- 80/20 stratified split (`random_state=42`)

**Class distribution in training set**

| Class | Count | Proportion |
|---|---|---|
| 0 (no stroke) | 3,889 | 95.13% |
| 1 (stroke) | 199 | 4.87% |

**Feature scaling:** both scalers fitted on `X_train` only (no data leakage):

| Scaling method | Euclidean distance (rows 0 & 1 of X_test) |
|---|---|
| Before scaling (raw) | 21.760 |
| After StandardScaler | 0.925 |
| After MinMaxScaler | 0.248 |

**Baseline: DummyClassifier (`most_frequent`)**

| Metric | Value |
|---|---|
| Accuracy | 95.11% |
| Recall (stroke=1) | 0.00 |

> The dummy classifier achieves high accuracy simply by always predicting class 0, but recall for stroke is 0% — it never identifies a single stroke case. This demonstrates why accuracy alone is insufficient for imbalanced datasets.

**Logistic Regression (no resampling, StandardScaler)**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0 (no stroke) | 0.95 | 1.00 | 0.97 |
| 1 (stroke) | 0.00 | 0.00 | 0.00 |

Confusion matrix: all 50 stroke cases in the test set were misclassified as non-stroke.

**Resampling: RandomUnderSampler**

After under-sampling the training set, both classes were balanced at 199 samples each.

**Logistic Regression (after under-sampling)**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0 (no stroke) | 0.99 | 0.74 | 0.85 |
| 1 (stroke) | 0.14 | 0.80 | 0.23 |

**Recall comparison (stroke=1)**

| | Recall |
|---|---|
| Before resampling | 0.00 |
| After under-sampling | **0.80** |

> Under-sampling balanced the training classes (199 vs 199), which dramatically improved the model's ability to detect stroke cases — recall rose from 0% to 80%. This came at the cost of overall accuracy (95% → 74%), but for a medical use-case, correctly identifying strokes is the priority.

---

## Key Findings

- **Imputation vs deletion:** Listwise deletion removed ~33% of the dataset. Imputation preserved all 5,110 rows at the cost of introducing estimates for missing values, a worthwhile trade-off given the small missing proportions.
- **Scaling compresses distances:** Raw Euclidean distance between two test samples dropped from 21.76 (unscaled) to 0.93 (StandardScaler) and 0.25 (MinMaxScaler), illustrating how unscaled features with large ranges dominate distance-based calculations.
- **Class imbalance is deceptive:** A 95% accurate model can be completely useless in a medical context if it never predicts the minority (stroke) class. Recall for the minority class is the critical metric here.
- **Under-sampling works:** `RandomUnderSampler` balanced the training data and boosted stroke recall from 0% to 80%, enabling the model to be practically useful despite the trade-off in precision.

