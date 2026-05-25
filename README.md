# Accident Zone Risk System

Risk analysis and accident hotspot detection using 300K+ road accident records.

This project groups accident locations into geographic zones, scores each zone by historical accident severity and frequency, and renders a map that highlights the areas with the highest combined danger signal. A machine learning model is used as a supporting layer to estimate severity risk, while the main product focus remains zone-level spatial analysis.

## What It Does

- Cleans and standardizes road accident records.
- Builds model-ready features from location, time, road, weather, vehicle, and casualty fields.
- Groups accident points into latitude/longitude grid zones.
- Calculates per-zone accident counts, severity counts, and risk scores.
- Adds predicted fatal/serious severity probabilities to each zone.
- Produces an interactive Folium map for high-risk zones.

## Dataset

The project is built around 300K+ road accident records. The processed local dataset currently contains about 307K accident rows.

Typical fields include:

- Latitude and longitude
- Accident date and time
- Road type and speed limit
- Weather, lighting, and road-surface conditions
- Vehicle information
- Number of casualties and vehicles
- Severity label: Fatal, Serious, or Slight

Large raw and processed data files should be handled carefully and may be excluded from GitHub when they are too large or sensitive.

## Zone Risk Logic

Each accident is assigned to a geographic grid zone. For each zone, the pipeline calculates:

- Accident count
- Fatal accident count
- Serious accident count
- Slight accident count
- Average zone latitude and longitude
- Historical risk score
- Predicted fatal risk
- Predicted serious risk
- Predicted severe risk
- Combined risk score and danger level

The higher the combined score, the more important the zone is for safety review and accident-prevention planning.

## Machine Learning

The current source code uses an `XGBoost` multiclass classifier with balanced sample weighting to handle class imbalance across accident severities.

Current saved training metrics:

- Accuracy: 61.76%
- Fatal recall: 45.76%
- Serious recall: 39.37%
- Slight recall: 65.47%
- Training rows: 246,377
- Test rows: 61,595

Accuracy is not the only success metric here because the dataset is imbalanced. Recall for severe classes is especially important when the goal is risk detection.

## Project Structure

```text
.
├── data/
│   ├── raw/                 # Original dataset, when available locally
│   └── processed/           # Cleaned data, model features, zone scores
├── docs/                    # Data dictionary and documentation
├── models/                  # Trained model artifacts, usually ignored
├── reports/                 # Training metrics and evaluation output
├── src/                     # Pipeline, preprocessing, modeling, scoring, map rendering
├── README.md
└── zone_risk_map.html       # Generated interactive map, when built locally
```

## How To Run Locally

Install the project dependencies in your Python environment, then run the pipeline scripts from the repository root.

```bash
python -m src.pipeline
python -m src.train
python -m src.score_zones
python -m src.render_zone_map
```

The generated map is saved as `zone_risk_map.html`.

## Outputs

- `data/processed/cleaned_accident_data.csv`
- `data/processed/model_features.csv`
- `data/processed/zone_risk_scores.csv`
- `reports/training_metrics.json`
- `models/accident_model_bundle.pkl`
- `zone_risk_map.html`

## Notes

This is a portfolio-style data science project focused on spatial safety insight. The model helps enrich the risk score, but the most important deliverable is the interpretable zone-level accident-risk output.
