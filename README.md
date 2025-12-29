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

[1] G. H. Allam, B. E. Elnaghi, M. N. Abdelwahab, and R. H. Mohammed, “Using machine learning to forecast solar power in Ismailia,” *International Journal of Scientific and Research Publications (IJSRP)*, vol. 11, no. 12, pp. 238–244, 2021.

[2] A. A. Hassan, D. M. Atia, and H. T. El-Madany, “Machine learning-based medium-term power forecasting of a grid-tied photovoltaic plant,” *Smart Grid and Renewable Energy*, vol. 15, no. 12, pp. 289–306, 2024.

[3] H. M. Ibrahim, “Estimation of global solar radiation on horizontal surfaces over Egypt: A review,” *Energy Reports*, vol. 6, pp. 234–241, 2020.

[4] M. Louzazni, A. Khouya, and H. Mosalam, “Comparison of power production for different photovoltaic technologies in Nile Delta, Egypt,” *Procedia Manufacturing*, vol. 46, pp. 724–729, 2020.

[5] M. Abdelsattar, A. AbdelMoety, and A. Emad-Eldeen, “Comparative analysis of machine learning techniques for temperature and humidity prediction in photovoltaic environments,” *Scientific Reports*, vol. 15, no. 1, p. 15650, 2025.

[6] S. Iyengar, N. Sharma, D. Irwin, P. Shenoy, and K. Ramamritham, “SolarCast: A cloud-based black box solar predictor for smart homes,” in *Proc. 1st ACM Conf. Embedded Systems for Energy-Efficient Buildings*, Nov. 2014, pp. 40–49.

[7] Taha, A., Makeen, P., & Nazih, N. (2025). Short-term and long-term solar irradiance forecasting with advanced machine learning techniques in Zafarana, Egypt. Scientific Reports, 15(1), 39553.

[8] A. M. Sharabati, “A comparative machine learning approach to forecasting solar power across diverse climate conditions,” Master’s thesis, Arab American University, 2025.

[9] Chaaban, A. K., & Alfadl, N. (2024). A comparative study of machine learning approaches for an accurate predictive modeling of solar energy generation. Energy Reports, 12, 1293-1302.

[10] J. M. Liceaga-Ortiz-De-La-Peña *et al*., “Advancing smart energy: A review for algorithms enhancing power grid reliability and efficiency,” *Energies*, vol. 18, no. 12, p. 3094, 2025.

[11] C. J. Willmott, “On the validation of models,” *Physical Geography*, vol. 2, no. 2, pp. 184–194, 1981.

[12] J. E. Nash and J. V. Sutcliffe, “River flow forecasting through conceptual models part I—A discussion of principles,” *Journal of Hydrology*, vol. 10, no. 3, pp. 282–290, 1970.

[13] D. R. Legates and G. J. McCabe Jr., “Evaluating the use of goodness-of-fit measures in hydrologic and hydroclimatic model validation,” *Water Resources Research*, vol. 35, no. 1, pp. 233–241, 1999.


