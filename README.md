# AI-driven Dynamic Pricing Optimization System

## Overview
This project builds an end-to-end AI system that predicts travel demand from a large real-world taxi dataset and recommends an optimal price using a rule-based pricing engine.

## Why this is valuable
- realistic spatiotemporal demand forecasting
- decision intelligence on top of ML
- clear business logic and measurable revenue uplift
- strong fit for data scientist / ML engineer / AI engineer portfolios

## Dataset
Kaggle NYC Yellow Taxi Trip Data (January 2015 file)

## Pipeline
raw taxi trips → preprocessing → demand panel → feature engineering → model training → diagnostics → pricing optimization → final report

## Main outputs
- demand prediction model
- residual analysis
- train/validation/test comparison
- feature importance
- error analysis by hour / zone / peak hours
- calibration plot
- dynamic pricing recommendations
- batch simulation summary

## Recommended run order
1. `python src/preprocessing.py`
2. `python src/visualization.py`
3. `python src/model.py`
4. `python src/main.py`
5. `streamlit run app/app.py`

## Outputs
- `outputs/data/`
- `outputs/plots/`
- `outputs/artifacts/`
- `outputs/predictions/`
- `outputs/final/`

## Notes
The model is trained with a chronological split to avoid leakage. The pricing layer is rule-based and uses the model’s predicted demand as input.
