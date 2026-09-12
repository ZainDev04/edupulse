# Model card: `math_score` (v20260912-060202)

**Task type:** regression  
**Description:** Cross-subject score prediction: estimate a student's math score from their background and literacy (reading/writing) scores.  
**Selected model:** `ridge` (Ridge)  
**Created:** 2026-09-12T06:02:32+00:00  
**Training data hash:** `1d32fe149100deae` · 800 train / 200 test rows  
**MLflow run:** `e69dc155921a43adab83a5d5cb2138e9` (experiment 1)  

## Intended use

Estimate an expected math score from the reading and writing scores and background. Useful for checking data entry, filling in a missing exam, or spotting a student whose math result is far below expectation.

## Inputs

- `gender`
- `race_ethnicity`
- `parental_level_of_education`
- `lunch`
- `test_preparation_course`
- `reading_score`
- `writing_score`

## Hold-out performance

| Metric | Value |
|---|---|
| n_test | 200.0000 |
| r2 | 0.8819 |
| rmse | 5.3616 |
| mae | 4.1790 |
| mape | 9.4010 |
| residual_std | 5.3548 |
| interval_coverage | 0.8850 |
| interval_nominal | 0.9000 |
| interval_width | 18.4212 |

Cross-validated r2 of the selected configuration: **0.8676**

## Prediction intervals (conformal)

Every prediction comes with a symmetric interval of half-width **9.24** points, calibrated for **90%** coverage on 800 out-of-fold residuals from the training split (cross-conformal, absolute out-of-fold residuals).
Empirical hold-out coverage: **88.5%** (mean width 18.4 points after clipping to 0-100).

## Model leaderboard (repeated stratified CV)

|   Rank | Model                  |   CV r2 (mean) |    std |
|-------:|:-----------------------|---------------:|-------:|
|      1 | ridge                  |         0.8675 | 0.0187 |
|      2 | gradient_boosting      |         0.8463 | 0.0247 |
|      3 | xgboost                |         0.8388 | 0.0268 |
|      4 | random_forest          |         0.8346 | 0.0268 |
|      5 | hist_gradient_boosting |         0.8341 | 0.0215 |
|      6 | lightgbm               |         0.8269 | 0.0257 |
|      7 | extra_trees            |         0.8261 | 0.0294 |
|      8 | svm_rbf                |         0.7829 | 0.0193 |
|      9 | knn                    |         0.7825 | 0.0178 |
|     10 | baseline_mean          |        -0.0075 | 0.0055 |

## Tuned hyper-parameters (Optuna)

```json
{
  "alpha": 2.3774410482619612
}
```

## Top features (mean |SHAP|)

| Feature | mean abs SHAP |
|---|---|
| writing_score | 8.6414 |
| gender | 6.4601 |
| reading_score | 3.1318 |
| race_ethnicity | 2.4101 |
| lunch | 1.6696 |
| test_preparation_course | 1.4930 |
| parental_level_of_education | 0.4992 |

## Fairness audit (hold-out subgroup gaps)

| Attribute | Metric | Gap |
|---|---|---|
| gender | mae_gap | 0.258 |
| lunch | mae_gap | 0.170 |
| parental_level_of_education | mae_gap | 1.571 |
| race_ethnicity | mae_gap | 2.153 |

## Limitations

- Trained on 1,000 anonymised records from one cohort. Results may not transfer to other institutions.
- Background attributes are proxies, not causes. Predictions are for prioritising support and must never be used to penalise a student.
- Sensitive attributes are model inputs. The fairness section measures subgroup disparities; it does not remove them.

## Environment

- python: 3.14.6
- sklearn: 1.9.1
- pandas: 3.0.5
- edupulse: 1.0.0
- platform: Windows-10-10.0.19045-SP0
