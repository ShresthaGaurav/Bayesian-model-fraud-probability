# Fraud Detection Framework

This project implements an adaptive fraud detection system combining Bayesian methods (PyMC) and machine learning (Scikit-learn), tailored for Nepal's financial context. It includes data processing, modeling, evaluation, deployment, and monitoring phases.

## Setup
1. Install Python 3.13.
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python src/main.py`

## Features
- Synthetic dataset generation (1,000 transactions).
- Bayesian modeling for probabilistic fraud risk.
- Hybrid ML model (Random Forest) with 50:50 weighted average.
- Evaluation with AUC-ROC, F1-score, and visualizations.
- REST API with FastAPI for real-time inference.
- Apache Spark for batch processing.
- Monitoring and adaptation with MLflow.

## Outputs
- `data/synthetic_transactions.csv`, `data/processed_transactions.csv`
- `figures/eda_plots.png`, `confusion_matrix.png`, `roc_curve.png`, etc.
- `mlruns/` for model tracking

## Deployment
- Start API: `uvicorn api.app:app --host 0.0.0.0 --port 8000`
- Replace `<run_id>` in `api/app.py` with the MLflow run ID.

## Monitoring
- Simulate drift and adaptation in `src/main.py`.
- Update with new data and feedback manually.
