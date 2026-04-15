#  EDM — Evaluación de Modelos

> Repositorio de prácticas de la asignatura **Evaluación de Modelos** (EDM).  
> Contiene implementaciones en Python y R sobre evaluación avanzada de modelos de Machine Learning: clasificación sensible a costes, reconocimiento de conjuntos abiertos y análisis de equidad algorítmica.

---

##  Autores

| Nombre | 
|---|
| Mario Álvarez Martínez |
| Marcos Carrasco Panadero |
| Sergio Samaniego Hernández |

---

##  Estructura del repositorio

```
EDM/
│
├── Practica_1/
│   └── EDM_Pract1_2026.ipynb       # Práctica 1 — Python (Jupyter Notebook)
│
├── Task_2/
│   ├── Task_2.Rmd                  # Task 2 — R Markdown
│   └── compas-scores-two-years.csv # Dataset COMPAS (ProPublica)
│
└── README.md
```

---

##  Práctica 1 — Clasificación Sensible a Costes y Open Set Recognition

**Lenguaje:** Python · **Entorno:** Jupyter Notebook

### Ejercicio 1 — Test Costs & Misclassification Costs

Dataset: **BreastCancer** (OpenML ID: 15). El objetivo es diseñar un clasificador que **minimice el coste global**, definido como:

```
Coste global = Coste de tests (atributos usados) + Coste de mala clasificación
```

La matriz de costes de clasificación es asimétrica:

| | Predicho: Benigno | Predicho: Maligno |
|---|---|---|
| **Real: Benigno** | 0 | 4 (FP) |
| **Real: Maligno** | 20 (FN) | 0 |

#### Estrategia de selección de features
Se implementa una **búsqueda greedy forward** sobre el espacio de subconjuntos de atributos, evitando la búsqueda exhaustiva. En cada paso se incorpora la feature que mayor reducción de coste global produce.

#### Estrategias para manejar costes asimétricos
Cada paso greedy evalúa tres estrategias y selecciona la de menor coste:

| Estrategia | Descripción |
|---|---|
| **Thresholding** | Umbral teórico óptimo: `th = C_FP / (C_FP + C_FN) ≈ 0.167` |
| **Rebalancing (pesos)** | Pesos de clase proporcionales al ratio de costes |
| **Rebalancing (oversampling)** | Sobremuestreo de la clase maligna según el ratio de costes |

Los clasificadores utilizados son **Decision Tree** (`max_depth=3`) y **Regresión Logística`.

#### Resultado
Se genera un gráfico del trade-off entre coste de tests y coste de mala clasificación, con las opciones **Pareto-eficientes** destacadas.

---

### Ejercicio 2 — Open Set Recognition (OSR)

Dataset: **MNIST** (clases 0–7 como *known*, clases 8–9 como *unknown*).

El objetivo es identificar si una muestra pertenece a una clase conocida o es un *unknown*, además de clasificarla correctamente cuando sea conocida.

#### Estrategias de scoring

| Estrategia | Descripción |
|---|---|
| **Max Class Probability** | Score = máxima probabilidad de clase predicha (Confidence Thresholding) |
| **Distancia al centroide más cercano** | Score = distancia negativa al centroide de clase más cercano |

#### Métricas de evaluación

- **AUROC** — Área bajo la curva ROC (Known vs Unknown detection)
- **OSCR** — Open Set Classification Rate: acepta *y* clasifica correctamente
- **AUPR** — Área bajo la curva Precisión-Recall (útil con desbalance)
- **FPR@95%TPR** — Falsos positivos cuando se aceptan el 95% de los knowns

---

##  Task 2 — Análisis de Equidad del sistema COMPAS

**Lenguaje:** R · **Formato:** R Markdown (`Task_2.Rmd`)  
**Dataset:** `compas-scores-two-years.csv` (ProPublica — Condado de Broward, Florida)

Se analiza la **equidad algorítmica** del sistema COMPAS, una herramienta usada en el sistema judicial estadounidense para estimar el riesgo de reincidencia.

### 2.1 Sufficiency (Calibración)

Se verifica si el `decile_score` tiene el mismo significado probabilístico entre grupos raciales:

```
P(is_recid = 1 | decile_score = r, race = African-American)
    ≈ P(is_recid = 1 | decile_score = r, race = Caucasian)
```

**Resultado:** COMPAS cumple la suficiencia de forma *aproximada*, pero no perfecta (e.g. para `decile_score = 4`, la tasa de reincidencia es ~44% en caucásicos y ~50% en afroamericanos).

### 2.2 Separation (Igualdad de tasas de error)

Se comparan **TPR** y **FPR** entre grupos mediante curvas ROC por raza. Se buscan pares de thresholds `(t_aa, t_ca)` que igualen ambas métricas con diferencia < 1%.

**Resultado:** Es posible aproximar *separation* con thresholds distintos por grupo (e.g. `t_aa = 7`, `t_ca = 5`), pero esto introduce diferencias en la **PPV** (Positive Predictive Value), lo que refleja la **imposibilidad de satisfacer simultáneamente** *sufficiency* y *separation*.

### 2.3 Factores de riesgo — Edad

Se analiza la relación entre la edad y la tasa de reincidencia en tres grupos:

| Grupo de edad | Tasa de reincidencia |
|---|---|
| ≤ 25 años | Alta |
| 26–49 años | Media |
| ≥ 50 años | Baja |

**Resultado:** La edad es un factor de riesgo relevante, con una diferencia de ~25 puntos porcentuales entre el grupo más joven y el de mayor edad.

---

##  Tecnologías y librerías

### Python
- `scikit-learn` — clasificadores, métricas, selección de features
- `numpy`, `pandas` — manipulación de datos
- `matplotlib` — visualización

### R
- `dplyr` — manipulación de datos
- `plotly` — visualización interactiva
- `fairness` — métricas de equidad algorítmica
- `reshape2` — transformación de datos

---

## Cómo ejecutar

### Práctica 1 (Python)
```bash
# Instalar dependencias
pip install numpy pandas scikit-learn matplotlib

# Abrir el notebook
jupyter notebook EDM_Pract1_2026.ipynb
```

### Task 2 (R)
```r
# Instalar dependencias
install.packages(c("dplyr", "plotly", "fairness", "reshape2"))

# Renderizar el informe
rmarkdown::render("Task_2/Task_2.Rmd")

---

## Referencias

- [ProPublica — Machine Bias (COMPAS dataset)](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)
- [OpenML — BreastCancer dataset (ID: 15)](https://www.openml.org/d/15)
- [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
- Scheirer, W. J. et al. (2013). *Toward Open Set Recognition*. IEEE TPAMI.
- Chouldechova, A. (2017). *Fair prediction with disparate impact*. Big Data.
