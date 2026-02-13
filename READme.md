# Explainable AI IDS – Initial Phase

## Overview

This repository contains the **initial research and preprocessing phase** of an
**Explainable AI–based Intrusion Detection System (IDS)** built using the **CICIDS2017 dataset**.

The goal of this phase is to:

- Prepare a **clean, research-ready dataset**
- Perform **Exploratory Data Analysis (EDA)**
- Establish a **reproducible preprocessing pipeline**
- Lay the foundation for **ML modeling and XAI (SHAP, LIME, Lens-XAI)** in later stages

---

## Dataset

**Dataset Used:** CICIDS2017
**Format:** Flow-based CSV files generated via CICFlowMeter

Each CSV represents:

- A specific **day of network traffic**
- A mix of **benign and attack behaviors**
- ~**78 numerical features + 1 label column (79 total columns)**

---

## Repository Structure

```
Data/
 ├── CICIDS_Raw/              # Original CICIDS2017 CSV files
 └── CICIDS2017_combined.csv  # Merged dataset (generated)

Notebook/
 └── 01_EDA.ipynb             # Exploratory Data Analysis

Pipeline/
 └── data_combination.py      # Script to merge raw CSV files

README.md
```

---

## Work Completed in Initial Phase

### 1. Data Merging

- Combined all CICIDS2017 CSV files using **Python + Pandas**
- Ensured:
  - Correct **column count (79)**
  - Proper **label preservation**
  - Reproducible **script-based pipeline**

---

### 2. Exploratory Data Analysis (EDA)

EDA includes:

- Dataset **shape, columns, statistics**
- **Label distribution** and class imbalance visualization
- Detection of:
  - **Missing values**
  - **Infinite values**

- Initial **data quality validation**

---

### 3. Data Cleaning Strategy

Research-standard preprocessing decisions:

- Strip whitespace from column names
- Replace **±∞ → NaN**
- Drop rows containing **NaN values**
- Prepare dataset for:
  - Feature scaling
  - Label encoding
  - Class balancing
  - ML training

---

## Next Steps (Upcoming Phases)

### Phase 2 – Preprocessing & Modeling

- Feature scaling
- Train/test split
- Class imbalance handling (**SMOTE / ADASYN**)
- Train baseline models:
  - Random Forest
  - XGBoost
  - Neural Network

---

### Phase 3 – Explainable AI

- SHAP global & local explanations
- LIME comparison
- **Lens-XAI integration (research contribution)**

---

## Research Goal

To build a **publication-ready Explainable IDS framework** that:

- Achieves **strong detection performance**
- Provides **interpretable security insights**
- Contributes toward **modern XAI-driven cybersecurity research**

---

## Author

**Santanu Ojha**
B.Tech – Internet of Things
University School of Automation and Robotics

---

## Status

**Current Stage:** Initial EDA & preprocessing
**Next Milestone:** ML modeling + SHAP explainability
