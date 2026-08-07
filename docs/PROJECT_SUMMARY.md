# Project Summary

## Title

AI-Based Road Accident Detection and Zone Risk Analysis System

## One-Line Summary

An end-to-end machine learning and geospatial analytics project that identifies accident-prone road zones from 300K+ historical accident records.

## Problem

Raw accident records are difficult to use directly for safety planning because they describe individual events rather than high-risk areas. This project converts individual accident data into zone-level risk insights that are easier to interpret and act on.

## My Contribution

- Built the complete Python data pipeline from ingestion to visualization.
- Cleaned and standardized raw accident records.
- Engineered time, location, casualty, road, weather, lighting, and vehicle features.
- Trained an XGBoost multiclass severity classifier.
- Handled class imbalance using balanced sample weighting.
- Created zone-level historical risk scores based on accident count and severity.
- Combined historical severity risk with model-predicted severe-risk probabilities.
- Generated an interactive Folium map with heatmap, markers, popups, boundaries, legend, and minimap.

## Tech Stack

Python, Pandas, Scikit-learn, XGBoost, Folium, Joblib, HTML

## Results

- Dataset size: about 307K accident records
- Training rows: 246,377
- Test rows: 61,595
- Model accuracy: 61.76%
- Fatal recall: 45.76%
- Serious recall: 39.37%
- Slight recall: 65.47%

## Why This Project Matters

The project focuses on interpretable safety insight rather than only model accuracy. The final output helps identify high-risk road zones, which can support traffic safety review, emergency-response planning, and future alert-system development.
