# Solar Photovoltaic Power Forecasting in Aswan, Egypt

### Advanced Predictive Modeling using Meteorological Data

---

## 🚀 Project Overview

This repository contains a comprehensive Machine Learning framework designed to predict Solar Photovoltaic (PV) power output in the Aswan region of Egypt. The project addresses the challenge of predicting renewable energy generation using only low-cost, standard meteorological variables (**Temperature, Humidity, Wind Speed, and Pressure**) rather than expensive irradiance sensors.

The study implements both **Classification** (predicting output levels) and **Regression** (predicting continuous power values), utilizing a robust pipeline of dimensionality reduction and non-linear modeling.

---

## 👤 Author

* **Rahma Asem Dawi**
* *Artificial Intelligence and Data Science Program (AID)*
* *E-JUST (Egypt-Japan University of Science and Technology)*

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
| --- | --- | --- | --- | --- |
| **Neural Network (MLP)** | **73.37%** | **0.266** | **0.73** | **Right Fit (Champion)** |
| Decision Tree (Entropy) | 70.12% | 0.298 | 0.69 | Robust |
| LDA Classifier | 64.45% | 0.355 | 0.62 | Stable |
| PCA + KNN | 56.25% | 0.437 | 0.54 | Underfitting |

### **Regression Accuracy**

The models were evaluated against strict meteorological benchmarks to ensure precision in actual energy output prediction:

* **Willmott Index of Agreement:** **0.9053** (High precision)
* **Nash–Sutcliffe Efficiency (NSE):** **0.7241**
* **Legates–McCabe Index:** **0.6810**
* ** Score:** **0.7045**

---

## 📈 Visualizations Included

The project generates high-fidelity visualizations for result interpretation:

* **Confusion Matrix Grids:** Identifying misclassifications between Low, Med, and High generation days.
* **ROC/AUC Curves:** Measuring the true positive rate across multiclass outputs.
* **Fit Diagnosis Charts:** Visual proof of model generalization vs. overfitting.

---

## 📂 Repository Structure

* `AswanData_weatherdata.csv`: The primary dataset for Aswan, Egypt.
* `Solar_Prediction_Final.ipynb`: Complete Jupyter Notebook with code, comments, and visualizations.
* `Presentation.pptx`: Official project presentation.
* `README.md`: Project documentation and technical overview.

---

## 📖 References

[1] Allam, G. H., Elnaghi, B. E., Abdelwahab, M. N., & Mohammed, R. H. (2021). Using Machine Learning to forecast Solar Power in Ismailia. *International Journal of Scientific and Research Publications (IJSRP)*, 11(12), 238-244.

[2] Hassan, A. A., Atia, D. M., & El-Madany, H. T. (2024). Machine Learning-Based Medium-Term Power Forecasting of a Grid-Tied Photovoltaic Plant. *Smart Grid and Renewable Energy*, 15(12), 289-306.

[3] Ibrahim, H. M. (2020). Estimation of global solar radiation on horizontal surfaces over Egypt: A review. *Energy Reports*, 6, 234-241.

[4] Louzazni, M., Khouya, A., & Mosalam, H. (2020). Comparison of power production for different photovoltaic technologies in Nile Delta, Egypt. *Procedia Manufacturing*, 46, 724-729.

[5] Abdelsattar, M., AbdelMoety, A., & Emad-Eldeen, A. (2025). Comparative analysis of machine learning techniques for temperature and humidity prediction in photovoltaic environments. *Scientific Reports*, 15(1), 15650.

[6] Iyengar, S., Sharma, N., Irwin, D., Shenoy, P., & Ramamritham, K. (2014, November). SolarCast: a cloud-based black box solar predictor for smart homes. In *Proceedings of the 1st ACM Conference on Embedded Systems for Energy-Efficient Buildings* (pp. 40-49).

[7] Abdellatif, S. O., et al. (2022). Assessment of different machine learning algorithms for short-term solar power forecasting in Egypt. *International Journal of Renewable Energy Research (IJRER)*, 12(1), 125-135.

[8] Sharabati, A. M. (2025). *A Comparative Machine Learning Approach to Forecasting Solar Power Across Diverse Climate Conditions* [Master’s dissertation, Arab American University].

[9] Chaaban, K., & Alfadl, N. (2024). A Comparative Study of Machine Learning Approaches for Accurate Solar Energy Predictive Modeling. *International Journal of Power Electronics and Drive Systems (IJPEDS)*, 15(3), 1542-1551.

[10] Liceaga-Ortiz-De-La-Peña, J. M., et al. (2025). Advancing Smart Energy: A Review for Algorithms Enhancing Power Grid Reliability and Efficiency. *Energies*, 18(12), 3094.

[11] Willmott, C. J. (1981). On the validation of models. *Physical Geography*, 2(2), 184-194.

[12] Nash, J. E., & Sutcliffe, J. V. (1970). River flow forecasting through conceptual models part I—A discussion of principles. *Journal of Hydrology*, 10(3), 282-290.

[13] Legates, D. R., & McCabe Jr, G. J. (1999). Evaluating the use of “goodness-of-fit” measures in hydrologic and hydroclimatic model validation. *Water Resources Research*, 35(1), 233-241.

