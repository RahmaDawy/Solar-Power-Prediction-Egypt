# Solar Photovoltaic Power Forecasting in Aswan, Egypt
### Advanced Predictive Modeling using Meteorological Data

---

## 🚀 Project Overview
[cite_start]This repository contains a comprehensive Machine Learning framework designed to predict Solar Photovoltaic (PV) power output in the Aswan region of Egypt—one of the world's most irradiance-rich areas[cite: 1, 6]. [cite_start]The project addresses the challenge of predicting renewable energy generation using only low-cost, standard meteorological variables (**Temperature, Humidity, Wind Speed, and Pressure**) rather than expensive irradiance sensors[cite: 16, 17].

[cite_start]The study implements both **Classification** (predicting output levels) and **Regression** (predicting continuous power values), utilizing a robust pipeline of dimensionality reduction and non-linear modeling[cite: 8, 9].

---

## 👤 Author & Supervision
* [cite_start]**Presenter:** Rahma Asem Dawi [cite: 2]
* **Program:** Artificial Intelligence and Data Science (AID)
* **Institution:** E-JUST (Egypt-Japan University of Science and Technology)
* **Supervisors:** Dr. Ahmed Anter and Eng. [cite_start]Sama AlQasaby [cite: 3]

---

## 🏗️ Technical Pipeline

### 1. Data Engineering & Statistical Validation
* [cite_start]**Data Cleaning:** The initial dataset was refined to **398 unique chronological records** by removing redundant data blocks (specifically duplicated April 2022 entries)[cite: 7, 48].
* [cite_start]**Preprocessing:** Automated handling of missing values, outlier removal (resulting in **312 usable samples**), and target binning for categorical classification[cite: 7, 41, 48].
* [cite_start]**Statistical Profiling:** Rigorous analysis of **Skewness** and **Kurtosis** to identify and handle data distribution irregularities[cite: 20, 54, 55].
* [cite_start]**Hypothesis Testing:** Utilization of **ANOVA** and **T-Tests** ($p < 0.05$) to statistically prove the influence of weather features, particularly Temperature, on solar power variance[cite: 20, 56].



### 2. Feature Reduction (Dimensionality Reduction)
[cite_start]To optimize performance and address multicollinearity, four distinct reduction techniques were implemented and compared[cite: 9, 21]:
* [cite_start]**PCA (Principal Component Analysis):** Captured 94% of the variance with 2 principal components[cite: 42, 58].
* [cite_start]**Kernel PCA:** Non-linear mapping using RBF kernels for complex weather patterns[cite: 21].
* [cite_start]**SVD (Singular Value Decomposition):** Matrix factorization for latent feature extraction[cite: 42, 59].
* [cite_start]**LDA (Linear Discriminant Analysis):** Supervised dimensionality reduction which provided the best class separability for binned data[cite: 42, 59].

### 3. Machine Learning Architectures
[cite_start]A diverse suite of algorithms was deployed to identify the optimal "Right Fit" model[cite: 22]:
* [cite_start]**Neural Networks (MLP):** Multi-layer Perceptron for high-dimensional non-linear mapping[cite: 22, 44].
* [cite_start]**Decision Trees:** Entropy-based logic for interpretable decision boundaries[cite: 22].
* [cite_start]**K-Nearest Neighbors (KNN):** Comparison of Manhattan and Euclidean distance metrics[cite: 22, 66].
* [cite_start]**Naive Bayes & Logistic Regression:** Probability-based and discriminant-based models[cite: 22].

---

## 📊 Performance Benchmarks

### **Classification Results (80/20 Split)**
[cite_start]The MLP model demonstrated strong generalization, identified as the "Right Fit"[cite: 61, 64].

| Model | Accuracy | Error Rate | F1-Score | Fit Status |
| :--- | :--- | :--- | :--- | :--- |
| **Neural Network (MLP)** | **73.37%** | **0.26** | **0.73** | **Right Fit (Champion)** |
| Decision Tree | 70.12% | 0.29 | 0.69 | Robust |
| KNN | 67.50% | 0.32 | 0.67 | Reliable |

### **Regression Accuracy**
[cite_start]The models were evaluated against strict meteorological benchmarks to ensure precision in energy output prediction[cite: 23, 70]:
* [cite_start]**Willmott Index of Agreement:** **0.9053** (High precision) [cite: 10, 73]
* [cite_start]**Nash–Sutcliffe Efficiency (NSE):** **0.7241** [cite: 74]
* [cite_start]**$R^2$ Score:** **0.7045** [cite: 75]

---

## 📈 Visualizations Included
[cite_start]The project generates high-fidelity visualizations for result interpretation[cite: 45]:
* [cite_start]**Confusion Matrix Grids:** Identifying misclassifications between Low, Med, and High generation days[cite: 45].
* **ROC/AUC Curves:** Measuring the true positive rate across multiclass outputs.
* [cite_start]**Fit Diagnosis Charts:** Visual proof of model generalization vs. overfitting[cite: 25].



---

## 📂 Repository Structure
* [cite_start]`AswanData_weatherdata.csv`: The primary dataset for Aswan, Egypt[cite: 7].
* `Solar_Prediction_Final.ipynb`: Complete Jupyter Notebook with code, comments, and visualizations.
* `Presentation.pptx`: Official project presentation.
* `README.md`: Project documentation and technical overview.

---

## 📖 References
[1] Allam, G. H., et al. (2021). Solar Power Forecasting in Ismailia City, Egypt using Machine Learning. [cite_start]*IJSRP*, 11(12)[cite: 84].

[2] Hassan, A. A., et al. (2024). Machine Learning-Based Medium-Term Power Forecasting. [cite_start]*Smart Grid and Renewable Energy*, 15(12)[cite: 85].

[3] Ibrahim, H. M., & Abdel-Wahab, A. (2020). Performance Analysis of Solar Radiation Models for Upper Egypt. [cite_start]*Energy Reports*, 6, 234-241[cite: 86, 87].

[4] Louzazni, M., et al. (2020). Comparison of Power Production for Different Photovoltaic Technologies in Nile Delta, Egypt. [cite_start]*Procedia Manufacturing*, 46, 724-729[cite: 88, 89].

[5] Abdelsattar, M., et al. (2025). Comparative analysis of deep learning architectures in solar power prediction. [cite_start]*Scientific Reports*, 15:31729[cite: 90].

[6] Singhal, et al. (2022). "Solar-Cast" machine learning-based framework for precise forecasting. [cite_start]*IRJET*, 12(5)[cite: 91].

[7] Taha, A., Makeen, P., & Nazih, N. (2025). Short-term and long-term solar irradiance forecasting with advanced machine learning techniques in Zafarana, Egypt. *Scientific Reports*, 15(1), 39553.

[8] Sharabati, A. M. (2023). A Comparative Machine Learning Approach to Forecasting Solar Power Across Diverse Climate Conditions. [cite_start]*AAUP Repository*[cite: 93, 94].

[9] Chaaban, A. K., & Alfadl, N. (2024). A comparative study of machine learning approaches for an accurate predictive modeling of solar energy generation. *Energy Reports*, 12, 1293-1302.

[10] Ortiz, et al. (2023). ML-based model using deep learning algorithms to forecast the power output of solar energy plants. [cite_start]*PLOS ONE*[cite: 97, 98].

[11] Willmott, C. J. (1981). On the validation of models. [cite_start]*Physical Geography*, 2(2), 184-194[cite: 99].

[12] Nash, J. E., & Sutcliffe, J. V. (1970). [cite_start]River flow forecasting through conceptual models part I. *Journal of Hydrology*, 10(3)[cite: 100].

[13] Legates, D. R., & McCabe Jr, G. J. (1999). Evaluating the use of goodness-of-fit measures in hydrologic and hydroclimatic model validation. *Water Resources Research*, 35(1), 233-241.