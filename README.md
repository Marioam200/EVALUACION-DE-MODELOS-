# EDM — Model Evaluation

> Repository of practical assignments for the course **Model Evaluation** (EDM).  
> It contains Python and R implementations focused on advanced evaluation of Machine Learning models: cost-sensitive classification, open set recognition, and algorithmic fairness analysis.

---

## *Authors*

| Name |
|---|
| Mario Álvarez Martínez |
| Marcos Carrasco Panadero |
| Sergio Samaniego Hernández |

---

## *Repository Structure*

```
EDM/
│
├── Practica_1/
│   └── EDM_Pract1_2026.ipynb       # Practice 1 — Python (Jupyter Notebook)
│
├── Task_2/
│   ├── Task_2.Rmd                  # Task 2 — R Markdown
│   └── compas-scores-two-years.csv # COMPAS Dataset (ProPublica)
│
└── README.md
```

---

## *Practice 1 — Cost-Sensitive Classification and Open Set Recognition*

**Language:** Python · **Environment:** Jupyter Notebook

### *Exercise 1 — Test Costs & Misclassification Costs*

Dataset: **BreastCancer** (OpenML ID: 15). The objective is to design a classifier that **minimizes the total cost**, defined as:

```
Total cost = Test cost (attributes used) + Misclassification cost
```

The classification cost matrix is asymmetric:

| | Predicted: Benign | Predicted: Malignant |
|---|---|---|
| **Actual: Benign** | 0 | 4 (FP) |
| **Actual: Malignant** | 20 (FN) | 0 |

#### *Feature Selection Strategy*

A **greedy forward search** is implemented over the feature subset space, avoiding exhaustive search. At each step, the feature that produces the greatest reduction in total cost is added.

#### *Strategies for Handling Asymmetric Costs*

Each greedy step evaluates three strategies and selects the one with the lowest cost:

| Strategy | Description |
|---|---|
| **Thresholding** | Optimal theoretical threshold: `th = C_FP / (C_FP + C_FN) ≈ 0.167` |
| **Rebalancing (weights)** | Class weights proportional to the cost ratio |
| **Rebalancing (oversampling)** | Oversampling of the malignant class according to the cost ratio |

The classifiers used are **Decision Tree** (`max_depth=3`) and **Logistic Regression**.

#### *Result*

A plot is generated showing the trade-off between test cost and misclassification cost, highlighting the **Pareto-efficient** options.

---

### *Exercise 2 — Open Set Recognition (OSR)*

Dataset: **MNIST** (classes 0–7 as *known*, classes 8–9 as *unknown*).

The objective is to determine whether a sample belongs to a known class or is an *unknown*, and to correctly classify it when it is known.

#### *Scoring Strategies*

| Strategy | Description |
|---|---|
| **Max Class Probability** | Score = maximum predicted class probability (Confidence Thresholding) |
| **Distance to Nearest Centroid** | Score = negative distance to the nearest class centroid |

#### *Evaluation Metrics*

- **AUROC** — Area Under the ROC Curve (Known vs Unknown detection)
- **OSCR** — Open Set Classification Rate: accepts *and* classifies correctly
- **AUPR** — Area Under the Precision–Recall Curve (useful with class imbalance)
- **FPR@95%TPR** — False positive rate when accepting 95% of known samples

---

## *Task 2 — Fairness Analysis of the COMPAS System*

**Language:** R · **Format:** R Markdown (`Task_2.Rmd`)  
**Dataset:** `compas-scores-two-years.csv` (ProPublica — Broward County, Florida)

This task analyzes the **algorithmic fairness** of the COMPAS system, a tool used in the U.S. judicial system to estimate the risk of recidivism.

### *2.1 Sufficiency (Calibration)*

It is tested whether the `decile_score` has the same probabilistic meaning across racial groups:

```
P(is_recid = 1 | decile_score = r, race = African-American)
    ≈ P(is_recid = 1 | decile_score = r, race = Caucasian)
```

**Result:** COMPAS satisfies sufficiency *approximately*, but not perfectly (e.g., for `decile_score = 4`, the recidivism rate is ~44% for Caucasians and ~50% for African-Americans).

---

### *2.2 Separation (Equality of Error Rates)*

**TPR** and **FPR** are compared between groups using ROC curves by race. Pairs of thresholds `(t_aa, t_ca)` are searched to equalize both metrics with a difference < 1%.

**Result:** It is possible to approximate *separation* using different thresholds for each group (e.g., `t_aa = 7`, `t_ca = 5`), but this introduces differences in **PPV** (Positive Predictive Value), reflecting the **impossibility of simultaneously satisfying** *sufficiency* and *separation*.

---

### *2.3 Risk Factors — Age*

The relationship between age and the recidivism rate is analyzed in three groups:

| Age group | Recidivism rate |
|---|---|
| ≤ 25 years | High |
| 26–49 years | Medium |
| ≥ 50 years | Low |

**Result:** Age is a relevant risk factor, with a difference of about **25 percentage points** between the youngest and oldest groups.

---

## *Technologies and Libraries*

### Python
- `scikit-learn` — classifiers, metrics, feature selection
- `numpy`, `pandas` — data manipulation
- `matplotlib` — visualization

### R
- `dplyr` — data manipulation
- `plotly` — interactive visualization
- `fairness` — algorithmic fairness metrics
- `reshape2` — data transformation

---

## *How to Run*

### *Practice 1 (Python)*

```bash
# Install dependencies
pip install numpy pandas scikit-learn matplotlib

# Open the notebook
jupyter notebook EDM_Pract1_2026.ipynb
```

### *Task 2 (R)*

```r
# Install dependencies
install.packages(c("dplyr", "plotly", "fairness", "reshape2"))

# Render the report
rmarkdown::render("Task_2/Task_2.Rmd")
```

---

## *References*

- ProPublica — Machine Bias (COMPAS dataset)  
https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing

- OpenML — BreastCancer dataset (ID: 15)  
https://www.openml.org/d/15

- MNIST dataset  
http://yann.lecun.com/exdb/mnist/

- Scheirer, W. J. et al. (2013). *Toward Open Set Recognition*. IEEE TPAMI.

- Chouldechova, A. (2017). *Fair prediction with disparate impact*. Big Data.
