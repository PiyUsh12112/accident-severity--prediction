# Accident Zone Risk System

Machine learning and geospatial analytics project for detecting accident-prone road zones from 300K+ historical road accident records.

The system cleans raw accident data, engineers model-ready features, trains an XGBoost severity classifier, scores geographic zones by accident frequency and severity, and renders an interactive Folium map for risk exploration.

## Project Highlights

- Processed about 307K road accident records.
- Built an end-to-end Python pipeline for data cleaning, feature engineering, training, scoring, and visualization.
- Trained an XGBoost multiclass classifier to estimate accident severity risk.
- Used balanced sample weighting to improve learning on imbalanced severity classes.
- Converted individual accident records into latitude/longitude grid zones.
- Combined historical risk and predicted severe-risk probability into a zone-level danger score.
- Generated an interactive HTML map with heatmap, markers, zone boundaries, popups, and risk-grade legend.

## Problem Statement

Road accident datasets are usually record-level and difficult to use directly for safety planning. A single prediction about one accident is less useful than identifying locations where accidents repeatedly occur or where severe accidents are more likely.

This project turns raw accident records into interpretable zone-level risk insights so high-risk areas can be reviewed for road safety planning, traffic monitoring, and emergency-response prioritization.

## Tech Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- Folium
- Joblib
- HTML map output

## Dataset

The project is built around 300K+ road accident records. The local processed dataset contains about 307K rows.

Typical fields include:

- Accident latitude and longitude
- Accident date and time
- Road type and speed limit
- Weather, lighting, and road-surface conditions
- Junction and local authority information
- Number of casualties and vehicles
- Accident severity: Fatal, Serious, or Slight

Large raw and processed data files may be excluded from GitHub depending on file size and data-sharing restrictions. The code expects the raw file at:

```text
data/raw/Raw Road Accident Data.xlsx
```

## Methodology

1. Data ingestion
   - Loads the raw Excel dataset.
   - Standardizes raw column names into canonical snake_case fields.

2. Data cleaning
   - Removes duplicate and empty rows.
   - Parses accident date and time values.
   - Converts numeric columns such as latitude, longitude, casualties, vehicles, and speed limit.
   - Normalizes severity labels into Fatal, Serious, and Slight.

3. Feature engineering
   - Extracts year, month, day, and hour.
   - Creates `casualties_per_vehicle`.
   - Builds numeric and categorical feature sets for modeling.

4. Machine learning
   - Trains an XGBoost multiclass classifier.
   - Uses median imputation for numeric features.
   - Uses most-frequent imputation and one-hot encoding for categorical features.
   - Applies balanced sample weighting to address class imbalance.

5. Zone risk scoring
   - Groups accidents into geographic grid zones.
   - Calculates accident count and severity counts for each zone.
   - Scores historical risk using severity weights:
     - Fatal: 5
     - Serious: 3
     - Slight: 1
   - Adds predicted fatal, serious, and severe risk probabilities.
   - Builds a combined risk score using historical risk and predicted severe-risk signal.

6. Visualization
   - Renders an interactive Folium map.
   - Includes heatmap, clustered markers, zone boundaries, popups, minimap, and legend.
   - Saves the output as `zone_risk_map.html`.

## Model Performance

Current saved evaluation metrics:

| Metric | Value |
| --- | ---: |
| Model | XGBoost |
| Accuracy | 61.76% |
| Fatal recall | 45.76% |
| Serious recall | 39.37% |
| Slight recall | 65.47% |
| Training rows | 246,377 |
| Test rows | 61,595 |

Accuracy is not the only success metric for this project because the severity classes are imbalanced. Recall for Fatal and Serious cases is especially important because the goal is to identify safety-critical risk, not just maximize overall classification accuracy.

## Repository Structure

```text
.
├── data/
│   ├── raw/                     # Raw source data, if available locally
│   └── processed/               # Generated cleaned/features/zone-score files
├── docs/
│   ├── DATA_DICTIONARY.md       # Human-readable field documentation
│   └── data_dictionary.csv      # Data dictionary in CSV format
├── models/                      # Generated model artifacts
├── reports/
│   └── training_metrics.json    # Saved training and evaluation metrics
├── src/
│   ├── ingest.py                # Raw data loading and cleaned-data export
│   ├── preprocessing.py         # Cleaning and feature engineering
│   ├── modeling.py              # Model pipeline, training, prediction
│   ├── zones.py                 # Zone-level risk scoring logic
│   ├── score_zones.py           # Standalone zone scoring script
│   ├── render_zone_map.py       # Folium map rendering
│   ├── pipeline.py              # End-to-end project pipeline
│   └── evaluate.py              # Metrics display helper
├── requirements.txt
├── README.md
└── zone_risk_map.html           # Generated interactive map, when available
```

## How To Run

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python -m src.pipeline
```

Or run steps separately:

```bash
python -m src.ingest
python -m src.train
python -m src.score_zones
python -m src.render_zone_map
python -m src.evaluate
```

Open the generated map:

```bash
open zone_risk_map.html
```

## Key Outputs

- `data/processed/cleaned_accident_data.csv`
- `data/processed/model_features.csv`
- `data/processed/zone_risk_scores.csv`
- `reports/training_metrics.json`
- `models/accident_model_bundle.pkl`
- `zone_risk_map.html`

## Example Use Cases

- Identify accident-prone geographic zones.
- Compare historical accident frequency with predicted severe-risk probability.
- Support traffic safety reviews using interpretable zone scores.
- Visualize high-risk locations on an interactive map.
- Build a foundation for future emergency alert or routing systems.

## Future Improvements

- Add live accident reporting or emergency alert integration.
- Add weather and traffic APIs for real-time risk scoring.
- Tune model hyperparameters with cross-validation.
- Add SHAP-based feature explanations.
- Deploy the map as a lightweight web dashboard.
- Add automated tests for preprocessing, scoring, and model-output contracts.

## Note

This is a portfolio-style machine learning project focused on interpretable road safety insight. The model supports the analysis, but the main deliverable is the zone-level accident-risk system and interactive risk map.
