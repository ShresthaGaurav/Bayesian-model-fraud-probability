
import os
import sys
from data_processing import generate_synthetic_data, preprocess_data, engineer_features, apply_differential_privacy
from modeling import bayesian_model, train_ml_model
from evaluation import evaluate_model
import uvicorn
from monitoring import monitor_drift, adapt_model, update_with_feedback
import mlflow
import subprocess

if __name__ == "__main__":
    # Setup directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("mlruns", exist_ok=True)
    
    # Data processing
    data = generate_synthetic_data()
    data = preprocess_data(data)
    data = engineer_features(data)
    data = apply_differential_privacy(data)
    
    # Modeling
    data, baseline_trace = bayesian_model(data)
    rf, X_test, y_test, y_prob_ml, y_prob_hybrid, run_id = train_ml_model(data)
    
    # Evaluation
    metrics = evaluate_model(rf, X_test, y_test, y_prob_ml, y_prob_hybrid)
    print("Initial Metrics:", metrics)
    
    # Baseline metrics
    baseline_metrics = metrics.copy()
    
    # Log metrics
    mlflow.set_tracking_uri("file:./mlruns")
    with mlflow.start_run():
        mlflow.log_metrics(metrics)
    
    # Monitoring and Adaptation (simulated)
    new_data = data.copy()  # Simulate new data
    new_data["fraud"][:10] = 1  # Simulate feedback
    updated_data, updated_rf, new_trace = adapt_model(new_data, baseline_trace)
    new_metrics = evaluate_model(updated_rf, X_test, y_test, y_prob_ml, y_prob_hybrid)
    print("Updated Metrics:", new_metrics)
    
    if monitor_drift(new_metrics, baseline_metrics):
        print("Model drift detected, adaptation triggered.")
    
    # Start API in a separate process
    print(f"Starting API with run_id: {run_id}")
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"],
    env={**os.environ, "MLFLOW_RUN_ID": run_id}
    )
