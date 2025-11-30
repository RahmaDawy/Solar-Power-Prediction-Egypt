# Solar Power Prediction in Egypt (Machine Learning Analysis)

## 📌 Project Overview
**Objective:** Predict the daily power output of solar panels (`Solar(PV)`) in Aswan, Egypt, using recent weather data.
**Context:** Climate change has altered weather patterns in the MENA region, making historical prediction models less reliable. This project implements a full Data Science pipeline to identify the best predictive model using standard meteorological features (Temperature, Wind, Humidity, Pressure).
**Best Model:** K-Nearest Neighbors (Euclidean) achieved **51.25% accuracy**, significantly outperforming random guessing (33%) and linear baselines.

---

## 📂 Dataset
* **Source:** `AswanData_weatherdata.csv`
* **Size:** 398 Daily Observations
* **Input Features:**
    * `AvgTemperture`: Average daily temperature.
    * `AverageDew`: Dew point (humidity indicator).
    * `Humidity`: Relative humidity (%).
    * `Wind`: Wind speed (mph).
    * `Pressure`: Atmospheric pressure.
* **Target Variable:** `Solar(PV)` (Binned into **Low**, **Medium**, **High** classes).

---

## ⚙️ Methodology

### Phase 1: Discovery & Preprocessing
* **Cleaning:** Handled date formats and verified 0 missing values.
* **Statistics:** Performed **T-tests** (proving Wind is a significant predictor, p=0.012) and **ANOVA** (showing Temperature alone is not a linear predictor, p=0.32).
* **Visualization:** Generated Heatmaps (to check multicollinearity) and Boxplots.

### Phase 2: Feature Reduction
* **PCA (Principal Component Analysis):** Reduced 5 features to 2 components, retaining **80% of variance**.
* **LDA (Linear Discriminant Analysis):** Visualized class separation, confirming the data is not linearly separable.

### Phase 3: Model Implementation
We implemented and compared 5 algorithms:
1.  **Naive Bayes (Gaussian):** Baseline probabilistic model.
2.  **Bayesian Belief Network (Concept):** Modeled causal dependencies (Wind → Efficiency).
3.  **Decision Tree (Entropy):** Captured non-linear decision rules.
4.  **LDA Classifier:** Tested linear boundaries.
5.  **K-Nearest Neighbors (K-NN):** Tested both Euclidean and Manhattan distances.

### Phase 4: Evaluation
* **Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC Curve.
* **Diagnostics:** Analyzed Overfitting vs. Underfitting using Training/Testing splits.

---

## 📊 Results & Analysis

| Model | Test Accuracy | Status | Insight |
| :--- | :--- | :--- | :--- |
| **K-NN (Euclidean)** | **51.25%** | **Best Fit** | Successfully captures local weather patterns. |
| **K-NN + PCA** | 46.25% | Good | Lost some signal due to dimensionality reduction. |
| **Naive Bayes** | 45.00% | Underfitting | Too simple; features are not independent. |
| **Decision Tree** | 45.00% | **Overfitting** | Memorized training noise (Train Acc: 70%). |
| **LDA** | 38.75% | Underfitting | Data is not linearly separable. |

### Comparison with State-of-the-Art
| Reference | Features Used | Accuracy | Why the difference? |
| :--- | :--- | :--- | :--- |
| **Zadeh et al. (2023)** | Temp, **Irradiance**, **Dust** | ~88% | They used expensive sensors for Sunlight & Dust. |
| **This Project** | Temp, Wind, Humidity | **~51%** | We proved prediction is possible using *only* basic weather reports, without expensive sensors. |

---

## 📷 Visualizations
* **Confusion Matrices:** Show that models struggle most with the "Low" vs "Medium" boundary.
* **ROC Curves:** K-NN achieves an AUC > 0.60, confirming predictive power.
* **Feature Reduction:** PCA plots reveal complex, overlapping clusters.

---

## 📚 References
1.  **Allam, G. H., et al.** (2021). "Using Machine Learning to forecast Solar Power in Ismailia." *IJSRP*.
2.  **Hassan, A. A., et al.** (2024). "Machine Learning-Based Medium-Term Power Forecasting." *Smart Grid and Renewable Energy*.
3.  **Galal, E. M., & Abdel-Mawgoud, A. S.** (2023). "Solar Modules Under Climate Conditions of El-Kharga Oasis." *Intl. Journal of Thin Film Science*.

---
*Project for Mathematics for Data Science (AID311), E-JUST University.*