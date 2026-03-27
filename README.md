# Feasibility of Track-Based Photon Identification for the HL-LHC Trigger

Senior Honours Project by **Tom Greenwood**  
Supervisor: **Dr. Liza Mijovic**  
March 2026

## Overview

This repository contains the code used for the Senior Honours Project:

**“Feasibility of Track-Based Photon Identification for the HL-LHC Trigger”**

The project investigates whether **track-based observables** can be used to improve **photon–jet discrimination** in a way that remains compatible with the computational constraints of the **fast trigger** at the **High-Luminosity Large Hadron Collider (HL-LHC)**.

The study compares:

- a simple **baseline classifier** based solely on nearby track multiplicity
- a **physics-motivated isolation classifier**
- three **machine-learning models**:
  - feed-forward neural network
  - histogram-based gradient boosting
  - XGBoost

Classifier performance is evaluated using:

- **ROC curves**
- **AUC**
- **jet fake rate** at fixed photon efficiencies of **80%, 90%, and 95%**

The main result of the project is that **track-based information provides strong photon–jet discrimination**, and that a simple **isolation-based classifier** outperforms the tested machine-learning models while also remaining computationally inexpensive relative to full track reconstruction.

---

## Repository Structure

```text
.
├── models/
│   ├── baseline_classifier/
│   ├── isolation_cut_classifier/
│   ├── nn_classifier/
│   ├── treehgb_classifier/
│   └── xgboost_classifier/
├── extra_analysis/
├── final_outputs/
├── .gitignore
└── README.md