# Assignment 1: End-to-End Machine Learning Project

Predicts whether a borrower will default on a loan, using a synthetic credit-risk dataset.

- **Dataset:** 10,000 rows × 21 columns
- **Target:** `target_default_risk` (0 = No Default, 1 = Default)
- **Submitted by:** Vijay

## Results.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **XGBoost** | **95.60%** | 96.44% | 94.93% | 95.68% |
| Random Forest | 94.30% | 96.06% | 92.69% | 94.35% |
| Logistic Regression | 93.95% | 96.13% | 91.91% | 93.97% |
| Decision Tree | 92.80% | 93.15% | 92.79% | 92.97% |
| SVM | 92.70% | 95.45% | 90.06% | 92.68% |

**Summary:** XGBoost was the best model. Tuning improved it only slightly (95.6% → 95.8%), and Random Forest stayed at 94.3%. Income and `debt_to_income` were the features most related to default.

## What was done

1. **EDA:** data types, missing values, class balance, distributions, outliers, correlations.
2. **Cleaning and features:** fixed the "Bachlors" typo, filled missing values with the median, capped outliers, added `income_per_dependent` and `recency_days`, one-hot encoded categories, scaled the data.
3. **Models:** Logistic Regression, Decision Tree, SVM, Random Forest, XGBoost, evaluated with accuracy, precision, recall, F1 and confusion matrix.
4. **Tuning:** GridSearchCV on Random Forest and XGBoost.

## Tech stack

Python 3, pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn

## Notes

- The dataset is synthetic and includes missing values, typos and outliers on purpose.
- This is a single Jupyter Notebook project, for educational purposes.

**Author:** Vijay
