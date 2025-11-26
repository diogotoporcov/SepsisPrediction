# Early Diagnosis of Sepsis through Artificial Intelligence

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.x-ffcc00.svg)](https://catboost.ai/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-yellow.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

Research project developed for my undergraduate scientific initiation at the **Federal Institute of São Paulo (IFSP)**, supported by the **Institutional Scientific and Technological Initiation Scholarship Program (PIBIFSP)**.

Sepsis is a severe and rapidly progressing clinical condition triggered by a dysregulated immune response to infection, often leading to multiple organ failure and high mortality. Early symptoms, such as fever, altered heart rate, and subtle physiological changes, are nonspecific and overlap with other medical conditions, making early detection challenging.

Advances in Artificial Intelligence have driven significant progress in medical applications, including tumor classification, image-based diagnostics, and prediction of acute clinical events.

This project focuses on developing an AI predictive model capable of identifying early signs of sepsis based on vital signs and laboratory test data, aiming to support faster and more effective clinical decision-making.

> **Note:** This project is conducted in collaboration with the Clinical Hospital of the University of São Paulo Medical School (HCFMUSP).
---

## Summary

1. [Project Overview](#project-overview)
2. [Objectives](#objectives)
3. [Dataset and Features](#dataset-and-features)
4. [Methodology](#methodology)
5. [Research Context](#research-context)

---

## Project Overview

> **Note:** _To do_

---

## Objectives

> **Note:** _To do_

---

## Dataset and Features

The dataset used in this project comes from the **“PhysioNet/Computing in Cardiology Challenge 2019 – Early Prediction of Sepsis from Clinical Data”**, publicly available at:
[https://physionet.org/content/challenge-2019/1.0.0/](https://physionet.org/content/challenge-2019/1.0.0/)

This dataset provides time-series clinical data collected from ICU patients, containing vital signs, laboratory test results, demographic information, and a binary outcome label indicating sepsis onset. Its structure enables the development of AI systems for early detection, as the sepsis label identifies windows up to six hours before clinical diagnosis.

The dataset consists of the following feature groups:

### Vital Signs (columns 1–8)

| Feature   | Description                           |
|-----------|---------------------------------------|
| **HR**    | Heart rate (beats per minute)         |
| **O2Sat** | Peripheral oxygen saturation (%)      |
| **Temp**  | Body temperature (°C)                 |
| **SBP**   | Systolic blood pressure (mm Hg)       |
| **MAP**   | Mean arterial pressure (mm Hg)        |
| **DBP**   | Diastolic blood pressure (mm Hg)      |
| **Resp**  | Respiration rate (breaths per minute) |
| **EtCO₂** | End-tidal carbon dioxide (mm Hg)      |

### Laboratory Values (columns 9–34)

| Feature              | Description                              |
|----------------------|------------------------------------------|
| **BaseExcess**       | Excess bicarbonate (mmol/L)              |
| **HCO₃**             | Bicarbonate (mmol/L)                     |
| **FiO₂**             | Fraction of inspired oxygen (%)          |
| **pH**               | Blood acidity level                      |
| **PaCO₂**            | Arterial partial pressure of CO₂ (mm Hg) |
| **SaO₂**             | Arterial oxygen saturation (%)           |
| **AST**              | Aspartate transaminase (IU/L)            |
| **BUN**              | Blood urea nitrogen (mg/dL)              |
| **Alkalinephos**     | Alkaline phosphatase (IU/L)              |
| **Calcium**          | Serum calcium (mg/dL)                    |
| **Chloride**         | Serum chloride (mmol/L)                  |
| **Creatinine**       | Serum creatinine (mg/dL)                 |
| **Bilirubin_direct** | Direct bilirubin (mg/dL)                 |
| **Glucose**          | Serum glucose (mg/dL)                    |
| **Lactate**          | Lactic acid (mg/dL)                      |
| **Magnesium**        | Serum magnesium (mmol/dL)                |
| **Phosphate**        | Serum phosphate (mg/dL)                  |
| **Potassium**        | Serum potassium (mmol/L)                 |
| **Bilirubin_total**  | Total bilirubin (mg/dL)                  |
| **TroponinI**        | Troponin I (ng/mL)                       |
| **Hct**              | Hematocrit (%)                           |
| **Hgb**              | Hemoglobin (g/dL)                        |
| **PTT**              | Partial thromboplastin time (s)          |
| **WBC**              | Leukocyte count (10³/µL)                 |
| **Fibrinogen**       | Fibrinogen (mg/dL)                       |
| **Platelets**        | Platelet count (10³/µL)                  |

### Demographics and Administrative Data (columns 35–40)

| Feature         | Description                                        |
|-----------------|----------------------------------------------------|
| **Age**         | Age in years (values ≥90 masked as 100)            |
| **Gender**      | Female (0) or Male (1)                             |
| **Unit1**       | ICU unit identifier (MICU)                         |
| **Unit2**       | ICU unit identifier (SICU)                         |
| **HospAdmTime** | Hours between hospital admission and ICU admission |
| **ICULOS**      | ICU length of stay (hours since ICU admit)         |

### Outcome Label (column 41)

| Feature         | Description                                                                                                |
|-----------------|------------------------------------------------------------------------------------------------------------|
| **SepsisLabel** | For sepsis patients: 1 if *t ≥ t<sub>sepsis</sub> – 6* and 0 otherwise. For non-sepsis patients: always 0. |

---

## Methodology

The methodological approach is divided into two major components:

1. [Preprocessing Pipeline](#preprocessing-pipeline), responsible for transforming raw ICU time-series data into a structured machine-learning-ready representation.
2. [Model Training](#model-training), where a CatBoost classifier is trained, tuned, validated, and evaluated on the processed dataset.

---

## Preprocessing Pipeline

The preprocessing and data preparation pipeline was designed to transform raw ICU time-series data into a structured feature set suitable for machine learning models focused on early sepsis prediction.

### 1. Data Loading and Initial Organization

The dataset is loaded and reorganized to ensure a consistent structure.
Key patient identifiers and static variables (such as Age and Gender) are placed first, followed by all dynamic clinical measurements.

### 2. Temporal Sorting

All records are sorted by Patient_ID and ICU length-of-stay (ICULOS), ensuring proper chronological order for each patient’s time series.

### 3. Log Transformations

Several laboratory values are right-skewed or heavy-tailed. Selected biochemical markers undergo a log1p transformation to stabilize distributions and improve downstream model behavior.

### 4. Missingness Modeling

The dataset contains substantial and informative missingness. Instead of simple imputation, missingness is explicitly modeled through additional feature channels:

1. **Binary missingness mask** – 0/1 indicators denoting whether each variable is missing at each hour.
2. **Time-since-last-measurement (TSLM)** – hours elapsed since each dynamic variable was last observed.
3. **Measurement frequency** – rolling counts of observations in the past 6 and 12 hours.

These features capture clinically meaningful patterns such as irregular monitoring and potential deterioration.

### 5. Patient-Level Train/Validation/Test Split

Splitting is performed at the patient level, ensuring that no individual appears in more than one subset.
Stratified sampling preserves the ratio of septic and non-septic patients across Train, Validation, and Test sets.

### 6. Sparse Variable Handling

Dynamic variables with extreme sparsity (more than 95% missing in the training set) are excluded from numeric channels and preserved only through their missingness-related features, preventing instability and noise in model training.

### 7. Imputation Strategy

Remaining dense dynamic variables undergo a two-step imputation:

1. **Forward fill** within each patient to maintain chronological consistency.
2. **Median-based neutral fill** (computed *only from the training set*) for values missing prior to a variable’s first appearance.

### 8. Rolling Statistical Features

Rolling windows of 6 and 12 hours are used to compute statistical summaries for each dynamic variable:

* Mean
* Minimum / Maximum
* Standard deviation
* Median
* IQR (interquartile range)

These derived features encode short-term physiological trends relevant to sepsis progression.

### 9. Train, Validation, and Test Splits

After all preprocessing steps, the dataset is divided into Train, Validation, and Test splits using patient-level stratification to ensure a balanced representation of sepsis cases.

### Final Preprocessed Dataset

The final dataset consists of 466 columns, combining the original variables with all additional features created during preprocessing, including temporal statistics, missingness indicators, time-since-last-measurement features, and measurement-frequency signals.

---

## Model Training

The model architecture selected for this study is CatBoost, chosen for its strong performance on high-dimensional tabular data, its ability to handle heterogeneous feature distributions, and its native support for categorical variables without requiring one-hot encoding.

### 1. Feature Construction and Categorical Encoding

All columns except Patient_ID, ICULOS, and SepsisLabel are used as input features.
The Gender column is treated as a categorical feature and encoded internally by CatBoost.

### 2. Class Imbalance Handling

To address the inherent imbalance between septic and non-septic patients, a positive-class weight is computed based on the ratio of negative to positive samples in the training set.
This weight is passed to CatBoost through its `class_weights` parameter.

### 3. Model Configuration

The final model uses a tuned set of hyperparameters obtained through a grid-search optimization process.

### 4. Model Fitting

The model is trained using:

* Training dataset (`train_pool`)
* Validation dataset (`val_pool`) for early stopping and model selection

### 5. Model Evaluation

Evaluation is conducted on the held-out test set, using metrics including:

* Classification report (precision, recall, F1-score)
* ROC-AUC
* Average Precision (AUPRC)
* Confusion matrix

Graphical diagnostics include prediction-evolution plots, sepsis-onset prediction analysis, and patient-level evaluation visualizations focused specifically on cases where the model made incorrect predictions, allowing detailed inspection of each patient’s clinical trajectory to understand potential reasons for misclassification.

### 6. PhysioNet Challenge Score

Predictions and labels for the test set are exported in the PhysioNet Challenge format to allow calculation of the 2019 Challenge score.
While the official scoring dataset is private, this enables approximate compatibility with the challenge framework.

---

# Research Context

> **Note:** _To do_

---

## License

This project is licensed under the MIT License - See [LICENSE](LICENSE) for more information.