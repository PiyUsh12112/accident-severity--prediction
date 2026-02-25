🚦 Accident Zone Risk Analysis & Severity Prediction

Project Overview

Road accidents are not randomly distributed — certain zones consistently experience higher accident frequency and higher severity levels.

This project analyzes 300,000+ real-world accident records to:

- Identify accident-prone zones
- Detect high-risk areas
- Predict accident severity
- Analyze patterns over time

The objective is to combine data analysis and machine learning to understand **where accidents occur most frequently and how severe they are.

Problem Statement
 
Can we:
1. Identify zones with high accident frequency?
2. Detect high-risk accident hotspots?
3. Predict accident severity based on environmental and vehicle-related features?

This project addresses both spatial risk analysis and predictive modeling.


Dataset Information

- 300K+ real-world accident records
- Features include:
  - Location / Zone
  - Date & Time
  - Road type
  - Weather conditions
  - Vehicle information
  - Casualties

> Raw dataset not included due to size constraints.

Zone Risk Analysis

  Accident Frequency by Zone
- Aggregated accident counts by geographic zones
- Identified high-frequency accident regions
- Visualized accident density distribution

 High-Risk Zone Detection
- Calculated accident severity concentration per zone
- Identified zones with high severe accident ratios
- Compared frequency vs severity trade-offs


 🧠 Machine Learning Approach

 1️⃣ Data Cleaning
- Handled missing values
- Removed redundant columns

 2️⃣ Feature Engineering
- Extracted Year, Month, Hour from datetime
- Encoded categorical variables
- Created meaningful ratio-based features

 3️⃣ Handling Class Imbalance
- Analyzed severity class distribution
- Applied class-weight balancing


 Models Used

- Logistic Regression (Baseline)
- Logistic Regression (Balanced)
- Random Forest Classifier



 Evaluation Focus

Rather than relying only on accuracy, evaluation emphasized:

- Recall for severe accidents
- F1-score
- Performance comparison between balanced and unbalanced models
- Minority class detection

