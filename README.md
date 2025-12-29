# Solar Photovoltaic Power Forecasting in Aswan, Egypt
### Advanced Predictive Modeling using Meteorological Data

---

## 🚀 Project Overview
This repository contains a comprehensive Machine Learning framework designed to predict Solar Photovoltaic (PV) power output in the Aswan region of Egypt. The project addresses the challenge of predicting renewable energy generation using only low-cost, standard meteorological variables (**Temperature, Humidity, Wind Speed, and Pressure**) rather than expensive irradiance sensors.

The study implements both **Classification** (predicting output levels) and **Regression** (predicting continuous power values), utilizing a robust pipeline of dimensionality reduction and non-linear modeling.

---

## 👤 Author & Supervision
* **Presenter:** Rahma Asem Dawi
* **Program:** Artificial Intelligence and Data Science (AID)
* **Institution:** E-JUST (Egypt-Japan University of Science and Technology)
* **Supervisors:** Dr. Ahmed Anter and Eng. Sama AlQasaby
* **Date:** December 2025

---

## 🏗️ Technical Pipeline & Methodology

### 1. Data Engineering & Statistical Validation
* **Data Cleaning:** The initial dataset of 421 observations was refined to **398 unique chronological records** by removing redundant data blocks (specifically duplicated April 2022 entries).
* **Preprocessing:** Automated handling of missing values, outlier removal (resulting in **312 usable samples**), and target binning for categorical classification (Low, Medium, and High generation).
* **Statistical Profiling:** Rigorous analysis of **Skewness** and **Kurtosis** to identify and handle data distribution irregularities.
* **Hypothesis Testing:** Utilization of **ANOVA** and **T-Tests** ($p < 0.05$) to statistically prove the influence of weather features, particularly Temperature, on solar power variance.



### 2. Feature Reduction (Dimensionality Reduction)
To optimize performance and address multicollinearity, three distinct reduction techniques were implemented and compared:
* **PCA (Principal Component Analysis):** Captured 94.2% of total variance with 2 components.
* **SVD (Singular Value Decomposition):** Matrix factorization for latent feature extraction.
* **LDA (Linear Discriminant Analysis):** Provided the best separability for the three binned classes.

### 3. Machine Learning Architectures
A diverse suite of algorithms was deployed to identify the optimal "Right Fit" model:
* **Neural Networks (MLP):** Multi-layer Perceptron for high-dimensional non-linear mapping.
* **Decision Trees:** Entropy-based logic for interpretable decision boundaries.
* **K-Nearest Neighbors (KNN):** Comparison of Manhattan and Euclidean distance metrics.
* **Naive Bayes & Logistic Regression:** Used as baselines to evaluate non-linear performance gains.



---

## 📊 Performance Benchmarks

### **5.1 Data Statistical Profile**
ANOVA tests confirmed that Temperature has the highest statistical impact on solar power output ($p < 0.05$).

| Metric | Temperature | Humidity | Wind Speed | Solar Power |
| :--- | :--- | :--- | :--- | :--- |
| **Mean** | 31.2 | 22.4 | 5.4 | 145.2 |
| **Variance** | 26.1 | 67.2 | 3.2 | 7814.5 |
| **Skewness** | 0.21 | -0.15 | 0.45 | 0.32 |
| **Kurtosis** | -1.1 | -0.8 | 0.12 | -1.05 |

### **5.2 Classification Results (80/20 Split)**
The MLP model demonstrated strong generalization, identified as the "Right Fit".

| Model | Accuracy | Error Rate | Precision | F-Measure | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Neural Network (MLP)** | **73.37%** | **0.266** | **0.73** | **0.73** | **Right Fit** |
| **Decision Tree** | 70.12% | 0.298 | 0.69 | 0.69 | Robust |
| **KNN (Manhattan)** | 67.50% | 0.325 | 0.67 | 0.67 | Stable |

### **5.3 Regression Precision Metrics**
The Multi-Layer Perceptron (MLP) significantly outperformed linear baselines in capturing non-linear meteorological relationships.

| Metric | Neural Network | Linear Regression |
| :--- | :--- | :--- |
| **Willmott Index** | **0.9053** | 0.7841 |
| **NSE Index** | **0.7241** | 0.5520 |
| **R² Score** | **0.7045** | 0.5130 |

---

## 📈 Visualizations Included
The project generates high-fidelity visualizations for result interpretation:
* **Confusion Matrix Grids:** Identifying misclassifications between generation categories.
* **Fit Diagnosis Charts:** Visual proof of model generalization vs. overfitting.
* **Error Analysis:** Comparative bar charts documenting the error rate per model.

---

## 📂 Repository Structure
* `AswanData_weatherdata.csv`: The primary dataset for Aswan, Egypt.
* `Solar_Prediction_Final.ipynb`: Complete Jupyter Notebook with code, comments, and visualizations.
* `Solar_Prediction_Final.ipynb`: Complete Jupyter Notebook with code, comments, and visualizations.
* `README.md`: Project documentation and technical overview.

---

## 📖 References (APA Style)
[1] Allam, G. H., Elnaghi, B. E., Abdelwahab, M. N., & Mohammed, R. H. (2021). Using machine learning to forecast solar power in Ismailia. *International Journal of Scientific and Research Publications (IJSRP)*, 11(12), 238–244.

[2] Hassan, A. A., Atia, D. M., & El-Madany, H. T. (2024). Machine learning-based medium-term power forecasting of a grid-tied photovoltaic plant. *Smart Grid and Renewable Energy*, 15(12), 289–306.

[3] Ibrahim, H. M. (2020). Estimation of global solar radiation on horizontal surfaces over Egypt: A review. *Energy Reports*, 6, 234–241.

[4] Louzazni, M., Khouya, A., & Mosalam, H. (2020). Comparison of power production for different photovoltaic technologies in Nile Delta, Egypt. *Procedia Manufacturing*, 46, 724–729.

[5] Abdelsattar, M., AbdelMoety, A., & Emad-Eldeen, A. (2025). Comparative analysis of machine learning techniques for temperature and humidity prediction in photovoltaic environments. *Scientific Reports*, 15(1), p. 15650.

[6] Iyengar, S., Sharma, N., Irwin, D., Shenoy, P., & Ramamritham, K. (2014). SolarCast: A cloud-based black box solar predictor for smart homes. In *Proc. 1st ACM Conf. Embedded Systems for Energy-Efficient Buildings*, pp. 40–49.

[7] Taha, A., Makeen, P., & Nazih, N. (2025). Short-term and long-term solar irradiance forecasting with advanced machine learning techniques in Zafarana, Egypt. *Scientific Reports*, 15(1), 39553.

[8] Sharabati, A. M. (2025). *A comparative machine learning approach to forecasting solar power across diverse climate conditions*. Master’s thesis, Arab American University.

[9] Chaaban, A. K., & Alfadl, N. (2024). A comparative study of machine learning approaches for an accurate predictive modeling of solar energy generation. *Energy Reports*, 12, 1293-1302.

[10] Liceaga-Ortiz-De-La-Peña, J. M., et al. (2025). Advancing smart energy: A review for algorithms enhancing power grid reliability and efficiency. *Energies*, 18(12), p. 3094.

[11] Willmott, C. J. (1981). On the validation of models. *Physical Geography*, 2(2), 184–194.

[12] Nash, J. E., & Sutcliffe, J. V. (1970). River flow forecasting through conceptual models part I—A discussion of principles. *Journal of Hydrology*, 10(3), 282–290.

[13] Legates, D. R., & McCabe Jr., G. J. (1999). Evaluating the use of goodness-of-fit measures in hydrologic and hydroclimatic model validation. *Water Resources Research*, 35(1), 233–241.