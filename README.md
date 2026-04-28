# EDM — Model Evaluation

Repository of practical assignments for the course **Model Evaluation** (EDM/GCD - Reliable Models).

This repository contains **Python and R implementations** focused on advanced evaluation of Machine Learning models:
- **Cost-sensitive classification** and Open Set Recognition
- **Algorithmic fairness** analysis in criminal justice systems
- **Interpretability** of linear models and explainable AI (XAI)

---

##  Authors

| Name |
|---|
| Mario Álvarez Martínez |
| Marcos Carrasco Panadero |
| Sergio Samaniego Hernández |

---

##  Repository Structure

```
EDM/
│
├── Practice_1/
│   └── EDM_Pract1_2026.ipynb              # Practice 1 — Cost-Sensitive Classification & OSR (Python)
│
├── Task_2/
│   ├── Task_2.Rmd                         # Task 2 — Fairness Analysis (R Markdown)
│   └── compas-scores-two-years.csv        # COMPAS Dataset (ProPublica)
│
├── Task_3/
│   ├── TASK_3.Rmd                         # Task 3 — Interpretability of Linear Models (R Markdown)
│   ├── Report_Task3_Interpretability...   # Generated PDF Report
│   └── day.csv                            # Bike-Sharing Dataset (Capital Bikeshare)
│
└── README.md                              # This file
```

---

##  Course Overview

This course covers three critical aspects of **responsible AI and model evaluation**:

### **Practice 1: Cost-Sensitive Classification & Open Set Recognition**

**Language:** Python · **Environment:** Jupyter Notebook

#### Exercise 1 — Test Costs & Misclassification Costs

**Problem:** Designing a classifier that **minimizes total cost**

```
Total cost = Test cost (attributes used) + Misclassification cost
```

**Dataset:** BreastCancer (OpenML ID: 15) — medical diagnosis prediction

**Cost Matrix** (asymmetric):

| | Predicted: Benign | Predicted: Malignant |
|---|---|---|
| **Actual: Benign** | 0 | 4 (FP) |
| **Actual: Malignant** | 20 (FN) | 0 |

False negatives (missing cancer) cost 5× more than false positives!

**Feature Selection:** Greedy forward search avoiding exhaustive search.

**Cost Mitigation Strategies:**

| Strategy | Description |
|---|---|
| **Thresholding** | Optimal threshold: `th = C_FP / (C_FP + C_FN) ≈ 0.167` |
| **Class Weights** | Rebalancing proportional to cost ratio |
| **Oversampling** | Synthetic oversampling of the malignant class |

**Classifiers:** Decision Tree (max_depth=3) & Logistic Regression  
**Result:** Pareto-efficient trade-off plot (test cost vs. misclassification cost)

#### Exercise 2 — Open Set Recognition (OSR)

**Problem:** Detect both *known* and *unknown* classes

**Dataset:** MNIST (classes 0–7 known, 8–9 unknown)

**Scoring Strategies:**

| Strategy | Description |
|---|---|
| **Max Class Probability** | Confidence-based thresholding |
| **Distance to Nearest Centroid** | Prototype-based detection |

**Evaluation Metrics:**
- **AUROC** — Known vs. Unknown detection performance
- **OSCR** — Open Set Classification Rate (correctly accepts & classifies)
- **AUPR** — Area Under Precision-Recall (handles class imbalance)
- **FPR@95%TPR** — False positive rate at 95% true positive rate

---

### **Task 2: Fairness Analysis of the COMPAS System**

**Language:** R · **Format:** R Markdown  
**Dataset:** `compas-scores-two-years.csv` (ProPublica — Broward County, Florida)

Analyzes **algorithmic fairness** in the COMPAS risk assessment tool used by U.S. courts to predict recidivism.

#### 2.1 Sufficiency (Calibration)

Tests if `decile_score` has **equal meaning across racial groups**:

```
P(recidivism | decile_score = r, race = African-American) 
  ≈ P(recidivism | decile_score = r, race = Caucasian)
```

**Finding:** COMPAS approximately satisfies sufficiency, but imperfectly  
(e.g., decile_score=4 → 44% recidivism for Caucasians vs. 50% for African-Americans)

#### 2.2 Separation (Equality of Error Rates)

Compares **TPR** and **FPR** between racial groups via ROC curves.

Searches for threshold pairs `(t_aa, t_ca)` to equalize both metrics within 1% difference.

**Finding:** Separation is achievable with different thresholds per group, but creates trade-offs in **PPV** (Positive Predictive Value), illustrating the **impossibility theorem**: *sufficiency and separation cannot be simultaneously satisfied*.

#### 2.3 Risk Factors: Age Analysis

Age groups show dramatically different recidivism rates:

| Age Group | Recidivism Rate |
|---|---|
| ≤ 25 years | High |
| 26–49 years | Medium |
| ≥ 50 years | Low |

**Finding:** ~25 percentage point difference between youngest and oldest groups — age is a critical risk factor.

---

### **Task 3: Interpretability of Linear Models**

**Language:** R · **Format:** R Markdown  
**Dataset:** `day.csv` — Capital Bikeshare (Washington D.C., 2011-2012)

Comprehensive exploration of **linear regression interpretability** for predicting daily bicycle rentals.

#### Context & Motivation

Interpretability is essential in high-stakes domains (medicine, finance, public administration) where accuracy alone is insufficient. **Linear regression** offers an unmatched advantage: each coefficient has direct, quantifiable meaning.

**Research Question:** *What factors influence daily bike rentals, and what is the quantitative impact of each?*

#### Key Objectives

-  Understand interpretability principles in linear regression
-  Learn rigorous coefficient interpretation & communication
-  Validate interpretations through diagnostic checks
-  Create effective visualizations (weight plots, effect plots)
-  Identify inherent limitations of linear models

#### Dataset Overview

**Target Variable:**
- `cnt`: Total daily bicycle rentals (casual + registered users)

**Features:**
- `season`: Season (1: spring, 2: summer, 3: fall, 4: winter)
- `workingday`: Working day indicator (0/1)
- `holiday`: Public holiday indicator (0/1)
- `weathersit`: Weather (1: clear, 2: cloudy, 3: light rain, 4: heavy rain)
- `temp`: Normalized temperature (0–1)
- `atemp`: Normalized apparent temperature (0–1)
- `hum`: Normalized humidity (0–1)
- `windspeed`: Normalized wind speed (0–1)

**Statistics:**
- Total observations: 731 days
- Mean rentals: 4,504 (SD: 1,937)
- Range: 22–8,714 bikes/day

#### Initial Hypotheses (Before Modeling)

1.  **Temperature effect:** Warmer days → more rentals
2.  **Weather effect:** Rain/snow → fewer rentals
3.  **Day type effect:** Working days vs. weekends/holidays differ
4.  **Seasonal effect:** Usage varies by season (higher in summer)
5.  **Trend effect:** 2011 to 2012 growth as service matures

#### Methodology: Data Preprocessing

**One-Hot Encoding** (Categorical Features)
- `season1`: 1 if spring, 0 otherwise
- `season2`: 1 if summer, 0 otherwise
- `season3`: 1 if fall, 0 otherwise
- (winter is the implicit reference category)

**Binary Weather Features**
- `MISTY`: 1 if conditions cloudy/misty, 0 otherwise
- `RAIN`: 1 if light rain/snow or heavy rain/storm, 0 otherwise

**Denormalization** (Continuous Features)
To improve interpretability, normalized [0,1] values are converted to real-world units:

| Variable | Scale | Multiplier | Interpretation |
|---|---|---|---|
| `temp` | 0–1 → °C | ×41 | Degrees Celsius |
| `hum` | 0–1 → % | ×100 | Percentage humidity |
| `windspeed` | 0–1 → km/h | ×67 | Wind speed in km/h |

**Trend Feature**
- `days_since_2011`: Days elapsed since Jan 1, 2011 (captures organic growth)

#### Correlation Analysis

**Key Findings:**

| Feature | Correlation with `cnt` | Significance |
|---|---|---|
| `days_since_2011` | +0.63 | Strongest positive |
| `temp` | +0.63 | Strongest positive |
| `season1` (Spring) | -0.56 | Negative |
| `RAIN` | -0.24 | Negative |
| `temp` ↔ `season1` | -0.62 | Partial collinearity alert |

**Alert:** Collinearity between temperature and spring season may affect individual coefficient stability.

#### Linear Model Results

**Model Specification:**
```r
cnt ~ workingday + holiday + season1 + season2 + season3 
    + MISTY + RAIN + temp + hum + windspeed + days_since_2011
```

**Goodness of Fit:**
- **R² = 0.7936** — explains ~79% of variance in daily rentals
- **F-statistic:** Highly significant (p < 2.2e-16)
- **Residual SE:** 886.9 rentals

**Model Coefficients (Interpretation):**

| Variable | Coefficient | Std. Error | t-stat | p-value | Interpretation |
|---|---|---|---|---|---|
| (Intercept) | 1,939.37 | 275.93 | 7.03 | *** | Baseline rentals |
| `workingday` | +124.92 | 73.27 | 1.71 | 0.089 | Working day: +125 rentals (marginal) |
| `holiday` | -686.12 | 203.30 | -3.38 | *** | Holiday: -686 rentals (commuters absent) |
| `season1` (Spring) | -425.60 | 110.82 | -3.84 | *** | Spring: -426 rentals vs. winter |
| `season2` (Summer) | +473.72 | 109.95 | 4.31 | *** | Summer: +474 rentals (peak season) |
| `season3` (Fall) | -287.39 | 134.22 | -2.14 | * | Fall: -287 rentals |
| `MISTY` | -379.40 | 87.55 | -4.33 | *** | Cloudy: -379 rentals |
| `RAIN` | -1,901.54 | 223.64 | -8.50 | *** | **Rain: -1,902 rentals (strongest deterrent)** |
| `temp` | +126.91 | 8.07 | 15.72 | *** | **+1°C: +127 rentals (second strongest positive)** |
| `hum` | -17.38 | 3.17 | -5.48 | *** | +1% humidity: -17 rentals |
| `windspeed` | -42.51 | 6.89 | -6.17 | *** | +1 km/h wind: -43 rentals |
| `days_since_2011` | +4.93 | 0.17 | 28.51 | *** | **+1 day: +5 rentals (strongest positive trend)** |

**Key Insights:**

1. **Temperature** (+127/°C) and **days_since_2011** (+5/day) are the strongest positive drivers
2. **Rain** (-1,902) is the dominant negative factor — a heavy rain day loses >1,900 rentals!
3. **Holidays** (-686) show the service is commuter-dependent
4. **Collinearity note:** The negative spring effect partially reflects spring's cooler temperatures (captured by temp), not spring itself

#### Limitations of Linear Interpretation

 **Confounding:** `days_since_2011` absorbs both service growth AND seasonal patterns  
 **Collinearity:** temp and season1 correlation (-0.62) makes individual effects unstable  
 **Non-linearity:** Model assumes linear relationships; real effects may plateau at extremes  
 **No interactions:** Cannot capture how wind's effect varies by day type (weekday vs. weekend)  
 **Outliers:** Extreme observations can shift coefficients substantially

#### Visualization Tools

**Weight Plots:**
- **Unstandardized:** Shows raw coefficients but misleads (apples-to-oranges)
- **Standardized:** Coefficients for ±1 SD changes — enables fair comparisons

**Effect Plots:**
- Combines coefficient magnitude **AND** feature variance
- Shows practical impact range for each feature across the dataset
- Ordered by mean effect for business importance assessment

**Individual Prediction Decomposition:**
- Explains single predictions feature-by-feature
- Shows which factors pushed prediction above/below average
- Essential for stakeholder communication

#### Example: Predicting Day 6 (Jan 6, 2011)

**Actual:** 1,606 rentals  
**Predicted:** 1,571 rentals (error: -35, or 2%)  
**System average:** 4,504 rentals

**Why 1,571 (so far below average)?**

- 🔴 **`days_since_2011`** (day 5): Nearly zero trend effect — service just launched
- 🔴 **`temp`** (cold January): Lower tail of distribution — heavy negative contributor
- 🟡 **`hum` & `windspeed`**: Modest negative effects from winter weather
- ⚪ **Season, weather binary, day type**: Marginal contributions (~0)

**Conclusion:** Early winter launch day with minimal trend and cold temperatures → far below average. The effect decomposition makes this transparent and explainable to any stakeholder.

#### Business Implications

1. **Weather forecasting:** Invest in demand forecasting tied to weather predictions
2. **Fleet optimization:** Growing service supports infrastructure expansion
3. **Commuter-focused strategy:** Working days are revenue drivers; promotions on holidays could attract recreational users
4. **Model transparency:** Linear regression enables full auditability and communication of decisions
5.  **Model limitations:** For higher accuracy, explore flexible models; linear regression remains the interpretable baseline

---

## 🛠 Technologies & Libraries

### Python
- **scikit-learn** — classification, regression, metrics, feature selection
- **numpy, pandas** — numerical computing and data manipulation
- **matplotlib, seaborn** — static and interactive visualization

### R
- **dplyr** — data manipulation and wrangling
- **ggplot2** — modern data visualization
- **gridExtra** — combining multiple plots
- **corrplot** — correlation matrices
- **car, lmtest** — regression diagnostics
- **reshape2** — data reshaping for long/wide formats
- **plotly** — interactive visualization
- **fairness** — algorithmic fairness metrics

---

##  How to Run

### Practice 1 (Python — Cost-Sensitive Classification & OSR)

```bash
# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn jupyter

# Navigate to practice directory
cd Practice_1/

# Open and run the notebook
jupyter notebook EDM_Pract1_2026.ipynb
```

### Task 2 (R — COMPAS Fairness Analysis)

```r
# Install dependencies (run once)
install.packages(c("dplyr", "plotly", "fairness", "reshape2", "ggplot2"))

# Set working directory to Task_2/
setwd("Task_2/")

# Render the R Markdown report
rmarkdown::render("Task_2.Rmd", output_format = "pdf_document")
```

### Task 3 (R — Bike-Sharing Interpretability)

```r
# Install dependencies (run once)
install.packages(c(
  "dplyr", "ggplot2", "gridExtra", "corrplot", 
  "car", "lmtest", "reshape2"
))

# Set working directory to Task_3/
setwd("Task_3/")

# Ensure day.csv is in the working directory
# Then render the R Markdown report
rmarkdown::render("TASK_3.Rmd", output_format = "pdf_document")
```

---

##  References

### Practice 1
- **OpenML Dataset 15:** BreastCancer  
  https://www.openml.org/d/15
  
- **MNIST Dataset**  
  http://yann.lecun.com/exdb/mnist/
  
- Scheirer, W. J., et al. (2013). *Toward Open Set Recognition*. IEEE TPAMI.

### Task 2
- **ProPublica Machine Bias Investigation**  
  https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing
  
- Chouldechova, A. (2017). *Fair Prediction with Disparate Impact: A Study of Bias in Recidivism Prediction Instruments*. Big Data, 5(2), 153-163.
  
- Corbett-Davies, S., et al. (2017). *Algorithmic Fairness and the Impossibility Results*. arXiv.

### Task 3
- **Capital Bikeshare Dataset**  
  UCI Machine Learning Repository
  
- Fanaee-T, H., & Ghaderi, J. (2013). *Event labeling combining ensemble detectors and background knowledge*. Progress in Artificial Intelligence, 2(2-3), 113-126.
  
- Molnar, C. (2020). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*. https://christophm.github.io/interpretable-ml-book/
  
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. KDD.

---

##  Summary Table

| Assignment | Topic | Language | Dataset | Techniques |
|---|---|---|---|---|
| **Practice 1** | Cost-Sensitive Classification & OSR | Python | BreastCancer, MNIST | Greedy FS, thresholding, distance metrics, AUROC, OSCR |
| **Task 2** | Fairness Analysis | R | COMPAS | Sufficiency, separation, fairness metrics, ROC curves |
| **Task 3** | Linear Model Interpretability | R | Bike-Sharing | Coefficient interpretation, weight plots, effect plots, feature engineering |

---

##  Key Takeaways

1. **Cost-sensitive learning matters:** Different error types have different costs; optimize accordingly
2. **Open set recognition is realistic:** Real-world classifiers must detect unknown classes
3. **Fairness is multidimensional:** Sufficiency and separation are fundamentally incompatible (impossibility theorem)
4. **Interpretability enables trust:** Linear models provide full transparency for high-stakes decisions
5. **Feature engineering improves interpretability:** Denormalization and meaningful representations aid communication

---

##  License

Educational material for the EDM (Model Evaluation) course.

---

##  Contact

For questions or issues, please contact:
- Mario Álvarez Martínez
- Marcos Carrasco Panadero
- Sergio Samaniego Hernández

---

**Last updated:** April 2026  
**Status:** Complete with all three assignments