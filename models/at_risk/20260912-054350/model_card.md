# Model card: `at_risk` (v20260912-054350)

**Task type:** binary  
**Description:** Early-warning system: flag students likely to average below 60 using ONLY background information available before any exam is taken.  
**Selected model:** `logistic_regression` (LogisticRegression)  
**Created:** 2026-09-12T05:44:25+00:00  
**Training data hash:** `1d32fe149100deae` · 800 train / 200 test rows  
**MLflow run:** `2f0b2779ac214d47b92a5cb74beb5a87` (experiment 1)  

## Intended use

Rank incoming students by probability of under-performing so that academic advisors can offer test preparation or tutoring before exams. The model is a triage tool. It is not a verdict on the student.

## Inputs

- `gender`
- `race_ethnicity`
- `parental_level_of_education`
- `lunch`
- `test_preparation_course`

## Hold-out performance

| Metric | Value |
|---|---|
| n_test | 200.0000 |
| threshold | 0.4138 |
| roc_auc | 0.6986 |
| average_precision | 0.5151 |
| brier | 0.2220 |
| log_loss | 0.6302 |
| accuracy | 0.5300 |
| balanced_accuracy | 0.5922 |
| precision | 0.3471 |
| recall | 0.7368 |
| f1 | 0.4719 |
| f2 | 0.6017 |
| positive_rate | 0.2850 |
| flagged_rate | 0.6050 |

Cross-validated roc_auc of the selected configuration: **0.7429**

Decision threshold: **0.414** (highest threshold that keeps out-of-fold recall ≥ target recall).

## Model leaderboard (repeated stratified CV)

|   Rank | Model                  |   CV roc_auc (mean) |    std |
|-------:|:-----------------------|--------------------:|-------:|
|      1 | logistic_regression    |              0.7420 | 0.0342 |
|      2 | svm_rbf                |              0.7115 | 0.0371 |
|      3 | gradient_boosting      |              0.6982 | 0.0356 |
|      4 | xgboost                |              0.6835 | 0.0353 |
|      5 | hist_gradient_boosting |              0.6738 | 0.0361 |
|      6 | lightgbm               |              0.6690 | 0.0341 |
|      7 | random_forest          |              0.6376 | 0.0253 |
|      8 | knn                    |              0.6334 | 0.0333 |
|      9 | extra_trees            |              0.6254 | 0.0329 |
|     10 | baseline_majority      |              0.5000 | 0.0000 |

## Tuned hyper-parameters (Optuna)

```json
{
  "C": 0.1413261476102576
}
```

## Top features (mean |SHAP|)

| Feature | mean abs SHAP |
|---|---|
| lunch | 0.6315 |
| test_preparation_course | 0.4405 |
| race_ethnicity | 0.3018 |
| parental_level_of_education | 0.2838 |
| gender | 0.2736 |

## Fairness audit (hold-out subgroup gaps)

| Attribute | Metric | Gap |
|---|---|---|
| gender | selection_rate_gap | 0.262 |
| gender | tpr_gap | 0.215 |
| gender | fpr_gap | 0.261 |
| gender | precision_gap | 0.057 |
| gender | disparate_impact_ratio | 0.641 |
| lunch | selection_rate_gap | 0.569 |
| lunch | tpr_gap | 0.447 |
| lunch | fpr_gap | 0.604 |
| lunch | precision_gap | 0.034 |
| lunch | disparate_impact_ratio | 0.400 |
| parental_level_of_education | selection_rate_gap | 0.556 |
| parental_level_of_education | tpr_gap | 0.455 |
| parental_level_of_education | fpr_gap | 0.603 |
| parental_level_of_education | precision_gap | 0.281 |
| parental_level_of_education | disparate_impact_ratio | 0.339 |
| race_ethnicity | selection_rate_gap | 0.565 |
| race_ethnicity | tpr_gap | 0.600 |
| race_ethnicity | fpr_gap | 0.486 |
| race_ethnicity | precision_gap | 0.278 |
| race_ethnicity | disparate_impact_ratio | 0.355 |

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
