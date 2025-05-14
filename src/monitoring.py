
import mlflow
from modeling import bayesian_model, train_ml_model
import pandas as pd

def monitor_drift(current_metrics, baseline_metrics):
    drift = {k: abs(current_metrics[k] - baseline_metrics[k]) for k in current_metrics}
    return any(v > 0.05 for v in drift.values())

def adapt_model(new_data, baseline_trace):
    """Adapt the model to new data using the baseline trace."""
    # Preprocess and engineer features for new data
    # Assuming new_data is already preprocessed similarly to the original pipeline
    updated_data = new_data.copy()
    
    # Retrain the ML model on the updated data
    rf, _, _, _, _, _ = train_ml_model(updated_data)  # Now unpacking all 6 values
    
    # Update Bayesian model using the baseline trace
    updated_data, new_trace = bayesian_model(updated_data)
    
    return updated_data, rf, new_trace

def update_with_feedback(data, feedback):
    feedback_df = pd.DataFrame(feedback)
    data = pd.concat([data, feedback_df]).reset_index(drop=True)
    return data
