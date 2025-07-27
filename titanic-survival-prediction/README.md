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


🧠 Author
Developed by: [Hossam Mustafa ]
Passionate about ML, NLP, and Uncertainty Estimation.
Feel free to connect or contribute!


