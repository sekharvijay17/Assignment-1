
]
**Assignment 1: End-to-End Machine Learning Project**

## 📌 Short Description

A machine learning project that predicts whether a borrower will default on a loan, using a synthetic but realistic 10,000-row credit-risk dataset. The project covers the full pipeline — exploratory data analysis, data cleaning, feature engineering, and training/evaluating five classification models (Logistic Regression, Decision Tree, SVM, Random Forest, XGBoost) with hyperparameter tuning to compare their performance on identifying default risk.

**Dataset:** Synthetic Credit Risk Dataset (10,000 rows × 21 columns)
**Target Variable:** `target_default_risk` (0 = No Default, 1 = Default)
**Submitted by:** Vijay

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| XGBoost (Tuned) | 0.9580 | 0.9682 | 0.9493 | 0.9587 |
| XGBoost | 0.9560 | 0.9644 | 0.9493 | 0.9568 |
| Random Forest (Tuned) | 0.9430 | 0.9606 | 0.9269 | 0.9435 |
| Random Forest | 0.9430 | 0.9606 | 0.9269 | 0.9435 |
| Logistic Regression | 0.9395 | 0.9613 | 0.9191 | 0.9397 |
| Decision Tree | 0.9280 | 0.9315 | 0.9279 | 0.9297 |
| SVM | 0.9270 | 0.9545 | 0.9006 | 0.9268 |

*(Ranked by accuracy, highest to lowest.)*

**Key takeaway:** XGBoost performed best, even before tuning — tuning gave only a small additional boost (95.6% → 95.8%). Random Forest and Logistic Regression followed closely, suggesting most of the predictive signal (especially from `debt_to_income` and `credit_score`) is captured well by both linear and tree-based models. SVM and Decision Tree trailed slightly behind.

---

## 🔍 Workflow

### Step 1: EDA
- Data types, missing values, unique categorical values
- Target class balance
- Numeric distributions (histograms)
- Outlier detection (boxplots)
- Categorical variable counts
- Correlation heatmap

### Step 2: Feature Engineering
- Fixed typo in `education` ("Bachlors" → "Bachelors")
- Imputed missing values (median) in `income`, `savings`, `monthly_expenses`, `credit_score`
- Capped outliers using IQR
- Created `income_per_dependent` and `recency_days`
- Dropped `signup_date`, `customer_id`
- One-hot encoded categorical columns
- Train/test split + feature scaling

### Step 3: Model Evaluation
Trained and evaluated 5 models — accuracy, precision, recall, F1-score, confusion matrix:
- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

### Step 4: Hyperparameter Tuning
- GridSearchCV applied to Random Forest and XGBoost
- Compared baseline vs tuned performance

---

## 🛠️ Tech Stack
- Python 3
- pandas, numpy
- scikit-learn
- xgboost
- matplotlib, seaborn

---

## 📝 Notes
- Dataset is synthetic, built to mimic real-world data quality issues (missing values, typos, outliers) for learning purposes.
- This is a Jupyter Notebook project (single notebook), not a multi-script pipeline.

---

## 📄 License
This project is for educational purposes.

---

### ✍️ Author
**Vijay**
