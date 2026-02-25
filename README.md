# 🚦 Accident Zone Risk System  
###  Risk Analysis & Accident Hotspot Detection using 300K+ Records


Project Overview

This project analyzes 300,000+ real-world road accident records to identify accident-prone zones and assign a risk score to each zone.

The primary objective is spatial risk analysis — identifying high-frequency and high-severity accident areas.  
Predictive modeling is used as a supporting tool to understand accident severity patterns.

---
Objectives

- Identify accident-prone zones based on historical data
- Assign a risk score to each zone
- Visualize accident hotspots using an interactive map
- Predict accident severity using machine learning models

---

Zone Risk Analysis

Accident locations were grouped into zones based on geographic segmentation.

For each zone, the following were calculated:

- Total accident count
- Severity distribution
- Composite risk score

Zones with higher accident frequency and higher severity concentration receive higher risk scores.

An interactive map was built where:

- Each zone is clickable
- Risk score is displayed on click
- Accident distribution is visualized spatially

This enables intuitive identification of accident hotspots.

---

Risk Scoring Logic

Each zone is assigned a risk score based on:

- Accident frequency
- Severity concentration

The score reflects relative accident risk within the dataset.

Higher score → Higher accident risk.

---
 Machine Learning Component

To support severity analysis, the following models were implemented:

- Logistic Regression (Baseline)
- Logistic Regression (Class-weight balanced)
- Random Forest Classifier

 Model Performance

- Overall accuracy: ~85%
- Special focus was placed on handling class imbalance
- Emphasis on recall for severe accident cases

Accuracy was treated as a secondary metric, since the primary goal of the project is spatial risk analysis.

---

  Dataset

- 300,000+ accident records
- Includes:
  - Location (latitude/longitude)
  - Date & time
  - Road type
  - Environmental conditions
  - Vehicle information
  - Casualties

> Raw dataset not included due to size constraints.


📁 Project Structure


