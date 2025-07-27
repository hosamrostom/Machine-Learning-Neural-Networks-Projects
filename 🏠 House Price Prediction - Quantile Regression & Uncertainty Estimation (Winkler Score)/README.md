# 🏠 House Price Prediction using Multi-Model Regressor

This project is part of a Kaggle-style competition to predict house prices using ensemble regression techniques. The core idea is to combine multiple models and evaluate performance using the **Winkler Score**, which measures both prediction accuracy and confidence interval quality.

---

## 📌 Project Overview

- **Task**: Predict house prices with confidence intervals.
- **Input**: `X_train`, `y_train`, and `X_test`.
- **Models**: Multiple regressors (e.g., Gradient Boosting, Random Forest, SVR, etc.) trained using K-Fold cross-validation.
- **Metric**: Custom Winkler Score to assess interval prediction accuracy.

---

## 📊 Workflow

1. **Data Preprocessing**  
   - Clean and scale data  
   - Handle missing values and encode features if needed

2. **Model Training**  
   - K-Fold cross-validation (e.g., K=5)  
   - Train multiple regressors  
   - Collect predictions and standard deviations

3. **Interval Estimation**  
   - Calculate upper and lower prediction bounds  
   - Use standard deviation and Z-value for confidence intervals

4. **Scoring**  
   - Evaluate using Winkler Score

---

## 📈 Evaluation Metric: Winkler Score

The **Winkler Score** penalizes both large interval widths and when the true value lies outside the predicted range. It is computed as:

```python
def winkler_score(y_true, lower, upper, alpha=0.1):
    width = upper - lower
    penalty = (2 / alpha) * ((lower - y_true) * (y_true < lower) +
                             (y_true - upper) * (y_true > upper))
    return width + penalty





🧠 Author
Developed by: [Hossam Mustafa ]
Passionate about ML, NLP, and Uncertainty Estimation.
Feel free to connect or contribute!

