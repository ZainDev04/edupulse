# Model card: `performance_level` (v20260911-224015)

**Task type:** multiclass  
**Description:** Performance tiering (low, medium, high) from background and subject scores: the original coursework formulation.  
**Selected model:** `hist_gradient_boosting` (HistGradientBoostingClassifier)  
**Created:** 2026-09-11T22:40:15+00:00  
**Training data hash:** `1d32fe149100deae` · 800 train / 200 test rows  

> Leakage note: The target is a deterministic function of the three score inputs (average < 60 -> low, < 80 -> medium, else high). Near-perfect accuracy is therefore expected and says nothing about generalisation. The task is kept for continuity with the original report and as a leakage case study. See the 'at_risk' task for the background-only formulation.

## Intended use

Reproduce the original coursework tiering task. Its main use is as a demonstration of target leakage in model evaluation.

## Inputs

- `gender`
- `race_ethnicity`
- `parental_level_of_education`
- `lunch`
- `test_preparation_course`
- `math_score`
- `reading_score`
- `writing_score`

## Hold-out performance

| Metric | Value |
|---|---|
| n_test | 200.0000 |
| accuracy | 0.9850 |
| balanced_accuracy | 0.9877 |
| f1_macro | 0.9852 |
| f1_weighted | 0.9850 |
| roc_auc_ovr | 0.9996 |
| log_loss | 0.0504 |

Cross-validated f1_macro of the selected configuration: **0.9708**

## Model leaderboard (repeated stratified CV)

|   Rank | Model                  |   CV f1_macro (mean) |    std |
|-------:|:-----------------------|---------------------:|-------:|
|      1 | hist_gradient_boosting |               0.9674 | 0.0085 |
|      2 | logistic_regression    |               0.9668 | 0.0164 |
|      3 | gradient_boosting      |               0.9667 | 0.0140 |
|      4 | lightgbm               |               0.9664 | 0.0152 |
|      5 | xgboost                |               0.9633 | 0.0132 |
|      6 | random_forest          |               0.9611 | 0.0153 |
|      7 | extra_trees            |               0.9535 | 0.0212 |
|      8 | svm_rbf                |               0.9473 | 0.0115 |
|      9 | knn                    |               0.8670 | 0.0291 |
|     10 | baseline_majority      |               0.2273 | 0.0007 |

## Tuned hyper-parameters (Optuna)

```json
{
  "learning_rate": 0.018659959624904916,
  "max_iter": 175,
  "max_leaf_nodes": 36,
  "min_samples_leaf": 29,
  "l2_regularization": 0.0028585493941961923
}
```

## Top features (mean |SHAP|)

| Feature | mean abs SHAP |
|---|---|
| reading_score | 1.2501 |
| math_score | 0.8720 |
| writing_score | 0.8415 |
| race_ethnicity | 0.0385 |
| parental_level_of_education | 0.0354 |
| lunch | 0.0263 |
| test_preparation_course | 0.0031 |
| gender | 0.0028 |

## Fairness audit (hold-out subgroup gaps)

| Attribute | Metric | Gap |
|---|---|---|
| gender | accuracy_gap | 0.012 |
| lunch | accuracy_gap | 0.001 |
| parental_level_of_education | accuracy_gap | 0.049 |
| race_ethnicity | accuracy_gap | 0.053 |

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
