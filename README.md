# Predicting Admission Chances
This project leverages machine learning techniques to predict the chances of a student being admitted to a university based on several academic and non-academic factors. It includes comparisons between multiple regression and ensemble learning models to identify the most effective model.

## Project Overview

### Dataset:
The dataset contains 500 samples, each with the following features:
- **GRE Score**: Graduate Record Examination score.
- **TOEFL Score**: Test of English as a Foreign Language score.
- **University Rating**: University ranking on a scale of 1 to 5.
- **SOP**: Statement of Purpose strength on a scale of 1 to 5.
- **LOR**: Letter of Recommendation strength on a scale of 1 to 5.
- **CGPA**: Undergraduate Cumulative Grade Point Average.
- **Research**: Binary variable indicating research experience (1 = Yes, 0 = No).
- **Chance of Admit**: The dependent variable representing the probability of admission.

---

### Objective:
The objective of this project is to:
1. Develop multiple machine learning models to predict the probability of admission.
2. Identify the features with the highest impact on admission chances.
3. Compare the performance of regression-based and ensemble learning models.

---

### Tools and Libraries:
- **Python**: Core programming language.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computing.
- **Scikit-learn**: Machine learning algorithms and metrics.
- **Matplotlib** and **Seaborn**: Data visualization.

---

### Models Implemented:
1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**
4. **Gradient Boosting Regressor**
5. **Support Vector Regressor**
6. **K-Nearest Neighbors (KNN) Regressor**
7. **Lasso and Ridge Regressors**
8. **AdaBoost Regressor**
9. **ElasticNet Regressor**

---

### Evaluation Metrics:
The models were evaluated based on:
- **RMSE (Root Mean Square Error)**: Measures the average error magnitude.
- **R-Squared Score**: Indicates the proportion of variance in the dependent variable explained by the model.

---

## Key Results:

### Feature Importance:
The Random Forest Regressor highlighted the following features as most important:
1. **CGPA**: 85.36% importance.
2. **GRE Score**: 9.06% importance.
3. **TOEFL Score**: 5.58% importance.

---

### Model Comparisons:
| Model                     | RMSE Score | R-Squared Score |
|---------------------------|------------|-----------------|
| Linear Regression         | 0.064      | 0.763          |
| Decision Tree Regressor   | 0.087      | 0.565          |
| Random Forest Regressor   | 0.067      | 0.743          |
| Gradient Boosting Regressor | 0.068   | 0.736          |
| Support Vector Regressor  | 0.079      | 0.641          |
| KNN Regressor             | 0.090      | 0.537          |
| AdaBoost Regressor        | 0.067      | 0.741          |
| Ridge Regression          | 0.064      | 0.763          |

---

### Insights:
1. **CGPA** is the most significant predictor of admission chances.
2. **Linear Regression** and **Ridge Regression** demonstrated the best performance with the highest R-squared score.
3. Ensemble methods like **Random Forest** and **Gradient Boosting** also performed well.

---

## Additional Analysis:
An annotated version of the project notebook is available. It includes a step-by-step explanation of model training, feature importance analysis, and comparisons.

---

## References:
- Dataset source: Provided as part of coursework.
- Python libraries: [Scikit-learn](https://scikit-learn.org/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/).

---

## Acknowledgments:
This project was completed as part of coursework in **Principles of Machine Learning** at Indiana University Bloomington. Special thanks to the instructors for guidance.
