# EDM — Model Evaluation

Repository of practical assignments for the course **Model Evaluation** (EDM/GCD - Reliable Models).

This repository contains **Python and R implementations** focused on advanced evaluation of Machine Learning models:
- **Cost-sensitive classification** and Open Set Recognition
- **Algorithmic fairness** analysis in criminal justice systems
- **Explainable AI (XAI)** — Partial Dependence Plots for interpreting complex models (Random Forests)

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
│   ├── TaskXAI3.Rmd                       # Task 3 — Explainable AI (XAI) & Partial Dependence Plots (R Markdown)
│   ├── DataInsight_Explainability_Report.pdf  # Comprehensive PDP Analysis Report (Bike Rentals & House Prices)
│   ├── day.csv                            # Bike-Sharing Dataset (Capital Bikeshare)
│   ├── kc_house_data.csv                  # King County House Sales Dataset (for price prediction)
│   └── [Additional Model Outputs & Visualizations]
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

### **Task 3: Explainable AI (XAI) & Partial Dependence Plots**

**Language:** R · **Format:** R Markdown  
**Datasets:** 
- `day.csv` — Capital Bikeshare (Washington D.C., 2011-2012) — **Bike Rental Demand Forecasting**
- `kc_house_data.csv` — King County House Sales — **Residential House Price Prediction**

Comprehensive exploration of **model-agnostic explainability techniques** using Partial Dependence Plots (PDPs) to interpret complex machine learning models (Random Forests) across two real-world use cases.

#### Context & Motivation

Modern machine learning often sacrifices interpretability for accuracy. **Explainable AI (XAI)** bridges this gap using **model-agnostic methods** that work with ANY model, regardless of complexity.

**Challenge:** How do we explain and communicate the behavior of "black-box" models like Random Forests to stakeholders without deep ML knowledge?

**Solution:** **Partial Dependence Plots (PDPs)** — a powerful, intuitive visualization technique that reveals the marginal effect of each feature on predictions while holding all other variables constant.

#### Theory: Model Classification Framework

**By Interpretability Type:**
| Category | Examples | Trait |
|---|---|---|
| **Interpretable Models** | Linear Regression, Decision Trees, Rule-based systems | Direct, transparent internal logic |
| **Black-Box Models** | Neural Networks, SVM, Gradient Boosting, Random Forests | Complex, opaque decision logic |

**By Explanation Method:**
| Category | Examples | Coverage |
|---|---|---|
| **Model-Specific Methods** | Regression coefficients (linear only), tree splits | Limited to one model type |
| **Model-Agnostic Methods** | PDP, SHAP, LIME | Apply to ANY model; only need inputs & outputs |

#### Key Objectives

- Master model-agnostic explainability and why it matters for business
- Apply 1D and 2D Partial Dependence Plots to interpret Random Forest predictions
- Identify feature interactions and non-linear relationships
- Compare explainability across different domains (demand forecasting vs. price prediction)
- Create actionable insights for stakeholders and decision-makers
- Understand limitations and best practices of PDP methodology

---

#### **Use Case 1: Bike Rental Demand Forecasting (Bike-Sharing Dataset)**

**Objective:** Understand how weather, time, and operational factors influence bike rental demand using Random Forest predictions.

**Model:** Random Forest Regressor (500 trees, R² = 88.48%)

**1D PDP Analysis — Key Findings:**

| Feature | Pattern | Business Insight |
|---|---|---|
| **Temperature** | Non-linear ∩ shape; optimal 20–30°C | Warm days boost demand; extreme heat reduces rentals |
| **Humidity** | Negative relationship; steep drop >70% | High humidity is strong demand deterrent |
| **Wind Speed** | Negative linear decay | Each km/h wind decreases predicted rentals |
| **Days since 2011** | Strong upward trend → plateau | Service shows healthy organic growth, then stabilizes |

**2D PDP Analysis — Temperature × Humidity Interaction:**

The 2D heatmap reveals how these variables combine:
- **Best conditions:** High temp (20–30°C) + low humidity → peak demand (~5,000+ rentals)
- **Worst conditions:** Low temp + high humidity → minimum demand (~3,000 rentals)
- **Temperature dominates:** Rental demand increases sharply with warmth regardless of humidity
- **Humidity dampens:** High humidity partially offsets temperature's positive effect

**Key Business Recommendations:**
1. Integrate weather forecasts into fleet availability planning
2. Prepare for demand peaks on warm, dry days
3. Adjust staffing/pricing for low-demand humid periods
4. Use historical growth trends to justify expansion investments

---

#### **Use Case 2: Residential House Price Prediction (King County Sales)**

**Objective:** Identify which structural property features drive house prices using Random Forest predictions.

**Model:** Random Forest Regressor (500 trees, R² = 56.54%)

**1D PDP Analysis — Key Findings:**

| Feature | Pattern | Business Insight |
|---|---|---|
| **Living Area (sqft_living)** | Strong positive linear (most important) | Size is the #1 price driver — every sqft counts |
| **Bathrooms** | Consistent positive effect | Each bathroom adds meaningful value; accelerating returns |
| **Bedrooms** | Non-linear ∩ shape; peak at 2 bedrooms | More bedrooms don't always increase value; diminishing returns |
| **Year Built** | Positive trend; newer homes command premiums | Age/condition affects buyer preferences |
| **Floors** | Staircase pattern; moderate positive effect | Multi-story homes valued higher, but gains plateau |
| **Lot Size (sqft_lot)** | Moderate positive effect | Land size matters but less than living space |

**Non-Linear Surprise: Bedroom Paradox**
The model reveals homes with ~2 bedrooms command highest prices, while additional bedrooms reduce value. This may reflect:
- Trade-off: More bedrooms = less living space per room
- Market segmentation: Large-bedroom properties cluster in lower-value segments

**Key Business Recommendations:**
1. **Valuation:** Prioritize living area in all pricing models — it dominates predictions
2. **Acquisition:** Weight bathroom count heavily; bathrooms signal quality to buyers
3. **Development:** Avoid assuming linear bedroom-price relationship
4. **Marketing:** Lead with square footage and bathroom count in listings

---

#### PDP Methodology: Why This Technique?

**Advantages of PDPs:**
✓ **Model-agnostic:** Works with any model (linear, tree-based, neural networks, etc.)  
✓ **Intuitive:** Clear visualizations that stakeholders understand without ML expertise  
✓ **Causal-like interpretation:** Shows marginal effect of changing one feature while holding others constant  
✓ **Captures non-linearity:** Reveals curves, plateaus, and thresholds that linear models miss  
✓ **Supports 2D interaction analysis:** Visualizes how two features combine to influence predictions  

**Limitations to Consider:**
⚠ **Assumes independence:** Features may correlate (e.g., larger homes have more bathrooms)  
⚠ **Extrapolation risk:** Predictions outside data range can be unreliable  
⚠ **Ignores individual heterogeneity:** Shows averages; doesn't explain why one specific prediction differs  
⚠ **Computational cost:** 2D PDPs require more computation; use random sampling for large datasets  

#### Technical Implementation

**Data Processing Pipeline:**
1. **Feature Engineering:** Denormalize scaled variables to real-world units for interpretability
2. **Categorical Encoding:** One-hot encode categorical features (seasons, weather conditions)
3. **Train-Test Split:** Random Forest trained on full/sample data
4. **Grid Resolution:** 25–30 grid points for smooth PDP curves

**R Libraries Used:**
```r
library(randomForest)  # Model training
library(pdp)          # Partial Dependence Plot computation
library(ggplot2)      # Visualization
library(patchwork)    # Multi-plot layouts
```

#### Business Implications: From Explanation to Action

**For Bike-Sharing Operations:**
- Deploy predictive demand dashboards using weather feeds
- Proactively rebalance fleet before forecasted high-demand periods
- Launch targeted promotions during low-demand humid/windy days
- Validate long-term expansion plans against 5-year growth trends

**For Real Estate Valuation:**
- Implement automated valuation models (AVMs) anchored on living area
- Train sales teams on non-linear relationships (bedroom paradox)
- Adjust acquisition strategies based on bathroom/square-footage ROI
- Monitor market shifts by comparing PDP curves across time periods

---

##  Technologies & Libraries

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

### Task 3 (R — Explainable AI & Partial Dependence Plots)

```r
# Install dependencies (run once)
install.packages(c(
  "dplyr", "ggplot2", "patchwork",
  "randomForest", "pdp"
))

# Set working directory to Task_3/
setwd("Task_3/")

# Ensure both datasets are present:
# - day.csv (Bike-Sharing Dataset)
# - kc_house_data.csv (King County House Sales)

# Then render the R Markdown report
rmarkdown::render("TaskXAI3.Rmd", output_format = "html_document")
# Or to PDF:
# rmarkdown::render("TaskXAI3.Rmd", output_format = "pdf_document")
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
  
- **King County House Sales Dataset**  
  UCI Machine Learning Repository & Kaggle
  
- Molnar, C. (2020). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*. https://christophm.github.io/interpretable-ml-book/
  - Chapter 4: Partial Dependence Plot (PDP)
  - Chapter 5: Individual Conditional Expectation (ICE)
  
- Friedman, J. H. (2001). *Greedy function approximation: A gradient boosting machine*. Annals of Statistics, 29(5), 1189-1232.
  - Original introduction of Partial Dependence concept
  
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. KDD.
  - LIME — complementary model-agnostic explanation technique
  
- Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. NIPS.
  - SHAP — advanced model-agnostic explanation framework

---

##  Summary Table

| Assignment | Topic | Language | Datasets | Techniques |
|---|---|---|---|---|
| **Practice 1** | Cost-Sensitive Classification & OSR | Python | BreastCancer, MNIST | Greedy FS, thresholding, distance metrics, AUROC, OSCR |
| **Task 2** | Fairness Analysis | R | COMPAS | Sufficiency, separation, fairness metrics, ROC curves |
| **Task 3** | Explainable AI & Partial Dependence Plots | R | Bike-Sharing, King County Houses | PDP (1D & 2D), Random Forest interpretation, feature interactions, non-linearity analysis |

---

##  Key Takeaways

1. **Cost-sensitive learning matters:** Different error types have different costs; optimize accordingly
2. **Open set recognition is realistic:** Real-world classifiers must detect unknown classes
3. **Fairness is multidimensional:** Sufficiency and separation are fundamentally incompatible (impossibility theorem)
4. **Explainability bridges accuracy and trust:** Model-agnostic methods (PDP, SHAP) work with any model and enhance interpretability
5. **Non-linearity matters:** Partial Dependence Plots reveal curves, interactions, and thresholds that simpler models miss
6. **Business context drives interpretation:** The same model outputs different insights in demand forecasting vs. valuation contexts
7. **Feature interactions are crucial:** 2D PDPs show how variables combine — temperature + humidity jointly shape bike rental demand
8. **Stakeholder communication is paramount:** Clear visualizations (PDPs) enable non-technical stakeholders to understand and act on model predictions

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
