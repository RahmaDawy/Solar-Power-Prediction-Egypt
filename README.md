# Solar Photovoltaic Power Forecasting in Aswan, Egypt
### Advanced Predictive Modeling using Meteorological Data

---

## 🚀 Project Overview
This repository contains a comprehensive Machine Learning framework designed to predict Solar Photovoltaic (PV) power output in the Aswan region of Egypt. The project addresses the challenge of predicting renewable energy generation using only low-cost, standard meteorological variables (Temperature, Humidity, Wind Speed, and Pressure) rather than expensive irradiance sensors.

The study implements both **Classification** (predicting output levels) and **Regression** (predicting continuous power values), utilizing a robust pipeline of dimensionality reduction and non-linear modeling.

---

## 👤 Author
* **Rahma Asem Dawi**
* *Artificial Intelligence and Data Science Program (AID)*

---

## 🏗️ Technical Pipeline

### 1. Data Engineering & Statistical Validation
* **Preprocessing:** Automated handling of missing values, datetime feature extraction (seasonality), and target binning for categorical classification.
* **Statistical Profiling:** Rigorous analysis of **Skewness** and **Kurtosis** to identify and handle data distribution irregularities.
* **Hypothesis Testing:** Utilization of **ANOVA** and **T-Tests** to statistically prove the influence of weather features on solar power variance.

### 2. Feature Reduction (Dimensionality Reduction)
To optimize performance and address multicollinearity, four distinct reduction techniques were implemented and compared:
* **PCA (Principal Component Analysis):** Linear variance maximization.
* **Kernel PCA:** Non-linear mapping using RBF kernels for complex weather patterns.
* **SVD (Singular Value Decomposition):** Matrix factorization for latent feature extraction.
* **LDA (Linear Discriminant Analysis):** Supervised dimensionality reduction for maximum class separability.

### 3. Machine Learning Architectures
A diverse suite of algorithms was deployed to identify the optimal "Right Fit" model:
* **Neural Networks (MLP):** Multi-layer Perceptron for high-dimensional non-linear mapping.
* **Decision Trees:** Entropy-based logic for interpretable decision boundaries.
* **K-Nearest Neighbors (KNN):** Comparison of Manhattan and Euclidean distance metrics.
* **Naive Bayes & LDA:** Probability-based and discriminant-based classification.

---

## 📊 Performance Benchmarks

### **Classification Results**
| Model | Accuracy | Error Rate | F1-Score | Fit Status |
| :--- | :--- | :--- | :--- | :--- |
| **Neural Network (MLP)** | **73.37%** | **0.266** | **0.73** | **Right Fit (Champion)** |
| Decision Tree (Entropy) | 70.12% | 0.298 | 0.69 | Robust |
| LDA Classifier | 64.45% | 0.355 | 0.62 | Stable |
| PCA + KNN | 56.25% | 0.437 | 0.54 | Underfitting |

### **Regression Accuracy**
The models were evaluated against strict meteorological benchmarks to ensure precision in actual energy output prediction:
* **Willmott Index of Agreement:** **0.9053** (High precision)
* **Nash–Sutcliffe Efficiency (NSE):** **0.7241**
* **Legates–McCabe Index:** **0.6810**
* **$R^2$ Score:** **0.7045**

---

## 📈 Visualizations Included
The project generates high-fidelity visualizations for result interpretation:
* **Confusion Matrix Grids:** Identifying misclassifications between Low, Med, and High generation days.
* **Fit Diagnosis Charts:** Visual proof of model generalization vs. overfitting.
* **Error Analysis:** Bar charts documenting the error rate per model as required by the study rubric.



---

## 📂 Repository Structure
* `AswanData_weatherdata.csv`: The primary dataset for Aswan, Egypt.
* `Solar_Prediction_Final.ipynb`: Complete Jupyter Notebook with code, comments, and visualizations.
* `Presentation.pptx`: Official project presentation.
* `README.md`: Project documentation and technical overview.

---

## 📖 References

1. **Allam, G. H., El-Shimy, M. E., & El-Metwally, M. M.** (2021). Solar Power Forecasting in Ismailia City, Egypt using Machine Learning. *International Journal of Scientific and Research Publications (IJSRP)*, 11(12).
2. **Hassan, A. A., Atia, D. M., & El-Madany, H. T.** (2024). Machine Learning-Based Medium-Term Power Forecasting of a Grid-Tied Photovoltaic Plant. *Smart Grid and Renewable Energy*, 15(12), 289-306.
3. **Ibrahim, H. M., et al.** (2020). Performance Analysis of Solar Radiation Models for Upper Egypt. *Energy Reports*, 6, 234-241.
4. **Willmott, C. J.** (1981). On the validation of models. *Physical Geography*, 2(2), 184-194.
5. **Nash, J. E., & Sutcliffe, J. V.** (1970). River flow forecasting through conceptual models part I — A discussion of principles. *Journal of Hydrology*, 10(3), 282-290.
6. **Legates, D. R., & McCabe, G. J.** (1999). Evaluating the use of "goodness-of-fit" measures in hydrologic and hydroclimatic model validation. *Water Resources Research*, 35(1), 233-241.