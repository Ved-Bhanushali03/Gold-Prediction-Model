# Gold Price Prediction Using Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An industrial-grade predictive analytics system designed to model global macroeconomic factors and index benchmarks to determine gold prices with high regression precision. This project utilizes a **Random Forest Regressor** to evaluate non-linear correlations across interconnected financial markets based on 10 years of historical asset data.

---

## 📌 System Architecture & Workflow

The machine learning pipeline follows a strict production lifecycle:

1. **Data Ingestion:** Loads historical 10-year CSV market configurations into structured Pandas DataFrames.
2. **Data Engineering:** Drops non-predictive features (`Date`) and checks for null vectors to ensure structural integrity.
3. **Exploratory Data Analysis (EDA):** Uses Pearson correlation matrices and statistical density checks to capture asset patterns.
4. **Target Stratification:** Isolates independent variables (SPX, USO, SLV, EUR/USD) from the dependent target matrix (`GLD`).
5. **Deterministic Split:** Segregates arrays into an 80/20 train-test split configuration.
6. **Ensemble Optimization:** Trains a multi-decision-tree Random Forest Regression system (`n_estimators=100`).
7. **Performance Evaluation:** Validates convergence against test sets using the R² (Coefficient of Determination) metric.

---

## 📊 Dataset Specifications

The predictive engine runs over 2,290 historical financial market rows, evaluating dependencies across diverse asset sectors:

| Column Variant | Data Type | Description |
| :--- | :--- | :--- |
| **Date** | Object | Timestamp coordinate ($MM/DD/YYYY$) — *Omitted from mathematical execution*. |
| **SPX** | Float64 | S&P 500 Stock Capitalization Index (Tracks top 500 leading US public firms). |
| **USO** | Float64 | United States Oil Fund value index (Tracks global crude oil benchmarks). |
| **SLV** | Float64 | Silver Trust market spot values (Strong parallel commodity proxy). |
| **EUR/USD** | Float64 | European Union Currency to US Dollar foreign exchange pair ratio. |
| **GLD (Target)** | Float64 | **Gold Spot Price index value optimized for regression predictions.** |

### Analytics Insights
* **Multicollinearity Metrics:** Exploratory analysis highlights a profound direct positive Pearson relationship (~0.9) between Silver (`SLV`) and Gold (`GLD`) spot prices.
* **Density Spread:** The absolute majority of historical `GLD` asset distribution values display a strong concentration around the **120** index benchmark mark.

---

## ⚙️ Core Technology Stack

* **Programming Language:** Python 3.8+
* **Data Engineering & Linear Arrays:** `pandas`, `numpy`
* **Statistical Visualizations:** `matplotlib`, `seaborn`
* **Machine Learning Architecture:** `scikit-learn` (Ensemble Regressor Suites)
* **Execution Infrastructure:** Jupyter Notebook / Google Colaboratory

---

## 📈 Model Performance & Evaluation

The ensemble regression system yields robust validation parameters when evaluated against unseen evaluation splits:

* **Primary Regression Metric:** $R^2$ Score (Coefficient of Determination)
* **Achieved System Precision:** `0.989` (The trained architecture explains roughly 98.9% of the overall price variance).

### Performance Verification Plot
The line chart distribution below visualizes the high-fidelity alignment between real market entries and system predictions over the validation slice:

![Actual Price vs Predicted Price Trend](https://raw.githubusercontent.com/your-username/your-repo-name/main/actual_vs_predicted_placeholder.png) 
*(Note: Replace this path once you save your matplotlib trend image inside your repository assets)*

---

## 🚀 Local Installation & Deployment

### 1. Repository Assembly
```bash
git clone https://github.com/Ved-Bhanushali03/Gold-Prediction-Model.git

2. Environment Isolation
Bash
python3 -m venv env
source env/bin/activate  # On Windows terminal use: env\Scripts\activate

