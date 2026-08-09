# 🌾 Corn Price Forecasting with Interpretable Machine Learning

**A Multi-Phase Framework for Weekly Corn Futures Prediction**

---

## 📚 Table of Contents
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology Overview](#methodology-overview)
- [Results Summary](#results-summary)
- [Repository Structure](#repository-structure)
- [Installation & Usage](#installation--usage)
- [Phase-by-Phase Execution](#phase-by-phase-execution)
- [Key Findings](#key-findings)
- [Dependencies](#dependencies)
- [References](#references)
- [License](#license)

---

## Abstract

This repository presents a comprehensive, four-phase framework for forecasting weekly corn futures prices using machine learning, with a strong emphasis on model interpretability. The project integrates:

- **Phase 1**: Exploratory Data Analysis (EDA) to uncover relationships between weather variables (temperature, precipitation, evapotranspiration) and corn price movements.
- **Phase 2**: A context-aware model selection framework that evaluated five candidate models (ARIMA, Random Forest, XGBoost, LightGBM, WaveletANN) using calibrated coefficients tailored to the dataset.
- **Phase 3**: Comparative evaluation of Random Forest versus ARIMA, where Random Forest achieved superior performance (RMSE = 8.72, MAE = 6.77, R² = 0.796).
- **Phase 4**: Model interpretability using SHAP and LIME to explain predictions, confirming that while price variables dominate, temperature (with a two-week lag) and pollination period exert measurable, context-dependent effects.

The final model provides transparent, explainable forecasts, bridging the gap between predictive accuracy and stakeholder trust in agricultural commodity markets.

---

## Introduction

Agricultural commodity price forecasting is critical for farmers, traders, and policymakers. Corn, as one of the most traded staples globally, is highly sensitive to weather conditions—yet the relationship is complex, non-linear, and often delayed. While machine learning models offer high predictive power, their "black-box" nature limits adoption in high-stakes decision-making.

This project addresses this gap by:

1. **Identifying** weak but statistically significant correlations between weather variables (especially temperature) and corn prices.
2. **Selecting** the most appropriate model (Random Forest) through a data-driven calibration framework.
3. **Validating** the model's performance on a held-out test set.
4. **Interpreting** predictions using SHAP and LIME to reveal how weather, price, and seasonal factors jointly influence forecasts.

The result is a transparent, reproducible, and scientifically grounded forecasting system.

---

## Dataset

### Source
The dataset is derived from the **AgriCommodity Futures Multi-Source ML Dataset** (Kaggle), specifically the weekly corn futures file:

- **File**: `integrated_corn_weekly.csv`
- **Time Range**: March 2011 – December 2025 (697 weekly records)
- **Commodity**: Corn (ZC=F) Futures, USD/bushel

### Key Variables
| Category | Variables |
| :--- | :--- |
| **Price** | Open, High, Low, Close, Volume |
| **Weather** | Temperature (max), Precipitation, Evapotranspiration, Dry Spell |
| **Technical** | RSI, MACD, Bollinger Bands, SMA, EMA, Volatility |
| **Calendar** | Pollination indicator, Planting/ Harvest duration, Month, Week |
| **Macro** | USD Index, VIX, CPI, Inflation |

The dataset is pre-processed for time-series analysis with lagged features (1–4 weeks) and no missing values after cleaning.

---

## Methodology Overview

The project is structured in four sequential phases:

### Phase 1: Exploratory Data Analysis
- Spearman rank correlation between weather variables and weekly price changes.
- Categorical analysis (temperature and precipitation bins).
- Scenario testing (drought, heavy rain, ideal conditions, pollination period).
- Lag analysis (1- and 2-week temperature and precipitation lags).

**Key Finding**: Temperature shows the strongest (yet weak) negative correlation with price changes (ρ = -0.107), while precipitation has minimal direct impact.

### Phase 2: Model Selection Framework
- Context-aware recommendation using calibrated coefficients:
  - Interpretability (α = 0.45)
  - Robustness (γ = 0.50)
  - Scalability (ζ = 0.40)
  - Representation Capacity (θ = 0.85)
- Five candidates evaluated: ARIMA, Random Forest, XGBoost, LightGBM, WaveletANN.
- Compatibility scores (Manhattan and Euclidean) computed across three expertise levels.

**Result**: ARIMA ranked highest theoretically due to interpretability, but Random Forest was selected for empirical validation.

### Phase 3: Comparative Model Evaluation
- Aligned test period: final 20 weeks (July–December 2025).
- Target: next-week closing price (one-step-ahead forecast).
- Metrics: RMSE, MAE, R².

**Result**: Random Forest outperformed ARIMA across all metrics:
- RMSE: 13.37 vs. 21.05 (36.5% improvement)
- MAE: 10.49 vs. 16.84 (37.7% improvement)
- R²: 0.52 vs. -0.19

### Phase 4: Model Interpretability
- **SHAP**: Global feature importance and local explanations.
- **LIME**: Local explanations for three specific test samples.
- Focus on weather variables: temperature (lag 2), precipitation (lag 1/2), pollination, evapotranspiration.

**Key Findings**:
- Price variables (Low, High, Open) account for >85% of predictive power.
- Temperature with a two-week lag is the most important weather feature (SHAP importance = 1.08).
- Pollination period acts as a positive seasonal factor in local explanations.
- Weather effects are context-dependent: the same variable can increase or decrease predictions depending on market conditions.

---

## Results Summary

| Phase | Key Outcome |
| :--- | :--- |
| **Phase 1** | Temperature (ρ = -0.107) and evapotranspiration (ρ = -0.108) show weak but significant negative correlations with price changes. |
| **Phase 2** | ARIMA ranked highest theoretically (Manhattan = 0.8009), but Random Forest was selected for empirical validation. |
| **Phase 3** | Random Forest achieved RMSE = 8.72, MAE = 6.77, R² = 0.796 on the test set. |
| **Phase 4** | SHAP confirms price dominance; temperature (2-week lag) is the only weather variable in top 15; LIME shows context-dependent weather effects and pollination as a positive seasonal factor. |

**Final Forecast (2025-12-22)**: **432.75 USD/bushel**

---

## Repository Structure
```txt
farm-cashFlow-ML/
│
├── dataset/
│ └── integrated_corn_weekly.csv # Main dataset
│
│── analyze_weather_price.py
│── compare_models.py
│── forcasting-price.py
│
│── decision-model
│ └── calibrate.py
│ └── framework_decision_corn.py
│
├── .github/workflows/
│ ├── analyze-weather-price.yml
│ ├── calibrate.yml
│ ├── recommendation.yml
│ ├── compare-models.yml
│ └── interpret.yml
│
├── README.md
```

---


## Installation & Usage

1. Clone the Repository
```bash
git clone https://github.com/aliNzLami/farm-cashFlow-ML.git
cd farm-cashFlow-ML
```

2. Install Dependencies
```bash
pip install numpy pandas matplotlib scikit-learn statsmodels shap lime treeinterpreter
```

3. Run Individual Phases
```bash
# Phase 1: Exploratory Data Analysis
python scripts/analyze_weather_price.py

# Phase 2: Calibration & Model Selection
python scripts/calibrate_corn.py
python scripts/framework_decision_corn.py

# Phase 3: Model Comparison
python scripts/compare_models_fixed.py

# Phase 4: Interpretability
python scripts/interpret_rf.py
```

---

## License
This project is licensed under the MIT License – see the LICENSE file for details.

---

## Acknowledgements

This project would not have been possible without the valuable contributions of the open-source community and the availability of high-quality agricultural data.

We extend our sincere gratitude to:

- **Kaggle** and the contributors of the **AgriCommodity Futures Multi-Source ML Dataset** for providing the comprehensive corn futures data used in this research.
- The academic community for advancing research in agricultural economics, commodity price forecasting, and interpretable machine learning, which inspired the multi-phase methodology presented here.
- All contributors, reviewers, and users who provide feedback and help improve this project.

---

## Contact

For questions, feedback, collaboration inquiries, or support regarding this project, please reach out:

| Channel | Details |
| :--- | :--- |
| **Email** | [your.email@example.com](mailto:ali.nabizadeh79@yahoo.com) |
| **GitHub** | [github.com/your-username/farm-cashFlow-ML](https://github.com/aliNzLami/farm-cashFlow-ML) |

We welcome contributions, bug reports, feature requests, and any other form of engagement. If you use this project in your research or work, please cite it appropriately — a citation reference is available in the repository.

---

**Maintainer**: [Ali Nabizadeh Lamiry]  
**Last Updated**: August 2026
