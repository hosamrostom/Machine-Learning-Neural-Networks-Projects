# 🎯 Titanic Classification with Multiple ML Models

This project focuses on predicting survival outcomes of Titanic passengers using various supervised machine learning models. The dataset is preprocessed, and multiple classifiers are trained and evaluated using accuracy as a metric. Hyperparameter tuning was applied using `GridSearchCV` where applicable.

---

## 📊 Models & Accuracy Scores

| Model                        | Accuracy   |
|-----------------------------|------------|
| KNeighborsClassifier         | 85.65%     |
| DecisionTreeClassifier       | 88.76%     |
| RandomForestClassifier       | 88.04%     |
| GaussianNB                   | 89.71%     |
| SVC                          | 97.13%     |
| ExtraTreeClassifier          | 88.52%     |
| GradientBoostingClassifier   | 94.74%     |
| AdaBoostClassifier           | **98.09%** |
| ExtraTreesClassifier         | 89.00%     |

---

## ⚙️ Technologies Used

- Python 3
- scikit-learn
- pandas
- NumPy
- GridSearchCV

---

## 🧠 Objective

Compare and evaluate the performance of different classification models on the Titanic dataset to find the most accurate model.

---


## ✅ Best Performing Model

> **AdaBoostClassifier** achieved the highest accuracy: **98.09%**

---

## 📌 Notes

- All models were tuned using `GridSearchCV` with 5-fold cross-validation.
- Accuracy metric was used as the scoring parameter.
- This project can serve as a baseline for ensemble learning exploration.

---

## 📷 Optional Visualization

You may add a bar chart using matplotlib/seaborn to visualize the model comparison:

```python
import matplotlib.pyplot as plt

models = ['KNN', 'DecisionTree', 'RandomForest', 'GaussianNB', 'SVC', 'ExtraTree', 'GradientBoosting', 'AdaBoost', 'ExtraTrees']
scores = [85.65, 88.76, 88.04, 89.71, 97.13, 88.52, 94.74, 98.09, 89.00]

plt.figure(figsize=(10,6))
plt.barh(models, scores, color='skyblue')
plt.xlabel('Accuracy (%)')
plt.title('Model Comparison on Titanic Dataset')
plt.xlim(80, 100)
plt.grid(True)
plt.show()
