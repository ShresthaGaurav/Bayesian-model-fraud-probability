import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax
import jax.numpy as jnp

# Enable parallel chains on CPU
numpyro.set_host_device_count(2)

def bayesian_model(data):
    """Estimate fraud probability using NumPyro."""
    # Convert Pandas Series to NumPy arrays for JAX compatibility
    amount_zscore = data["amount_zscore"].values
    is_night = data["is_night"].values
    fraud = data["fraud"].values

    def model():
        fraud_prior = numpyro.sample("fraud_prior", dist.Beta(1, 99))
        amount_effect = numpyro.sample("amount_effect", dist.Normal(0, 1))
        night_effect = numpyro.sample("night_effect", dist.Normal(0, 1))
        fraud_prob = numpyro.deterministic(
            "fraud_prob",
            1 / (1 + jnp.exp(-(fraud_prior + amount_effect * amount_zscore + night_effect * is_night)))
        )
        numpyro.sample("observed", dist.Bernoulli(fraud_prob), obs=fraud)
    
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=200, num_warmup=200, num_chains=2)
    mcmc.run(rng_key=jax.random.PRNGKey(0))
    samples = mcmc.get_samples()
    fraud_probs = samples["fraud_prob"].mean(axis=0)
    data["bayesian_fraud_prob"] = fraud_probs  # Assign back to Pandas DataFrame
    return data, samples

def train_ml_model(data):
    """Train Random Forest with Bayesian probabilities and compute hybrid score."""
    features = ["amount_zscore", "rolling_avg_amount", "dr_cr_ratio", "is_night", "hour", "day_of_week", "bayesian_fraud_prob"]
    X = data[features]  # Keep as DataFrame
    y = data["fraud"]
    
    # Store original indices before SMOTE
    original_indices = X.index
    
    # Check for number of classes in y
    unique_classes = np.unique(y)
    print(f"Unique classes in y before SMOTE: {unique_classes}")  # Debug print
    
    # Apply SMOTE only if there are at least 2 classes
    if len(unique_classes) > 1:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        # Create a mapping of resampled indices to original indices
        # For synthetic samples, map to the nearest original sample (SMOTE uses existing indices for non-synthetic samples)
        resampled_indices = []
        for i in range(len(X_res)):
            if i < len(original_indices):
                resampled_indices.append(original_indices[i])
            else:
                # Synthetic sample: map to the last original index (simplified approach)
                # In practice, SMOTE uses nearest neighbors; this is a simplification
                resampled_indices.append(original_indices[-1])
        
        # Convert resampled data back to DataFrame with new indices
        X_res = pd.DataFrame(X_res, columns=features, index=range(len(X_res)))
        y_res = pd.Series(y_res, name="fraud", index=range(len(X_res)))
        # Store the mapping as a Series
        index_mapping = pd.Series(resampled_indices, index=range(len(X_res)), name="original_index")
    else:
        print("Warning: Only one class found in target variable. Skipping SMOTE.")
        X_res, y_res = X, y
        index_mapping = pd.Series(X_res.index, index=X_res.index, name="original_index")
    
    # Split the data while preserving indices
    train_idx, test_idx = train_test_split(X_res.index, test_size=0.3, random_state=42)
    X_train = X_res.loc[train_idx]
    X_test = X_res.loc[test_idx]
    y_train = y_res.loc[train_idx]
    y_test = y_res.loc[test_idx]
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_prob_ml = rf.predict_proba(X_test)[:, 1]
    
    with mlflow.start_run() as run:
        mlflow.log_params({"n_estimators": 100, "smote": len(unique_classes) > 1})
        # Add input example to suppress MLflow warning
        input_example = X_train.iloc[:5]
        mlflow.sklearn.log_model(rf, "random_forest_model", input_example=input_example)
        run_id = run.info.run_id  # Capture the run_id
    
    # Hybrid score (50:50 weighted average)
    # Map test indices back to original indices using the index mapping
    original_test_indices = index_mapping[test_idx]
    y_prob_bayesian = data.loc[original_test_indices, "bayesian_fraud_prob"].values
    y_prob_hybrid = 0.5 * y_prob_bayesian + 0.5 * y_prob_ml
    
    return rf, X_test, y_test, y_prob_ml, y_prob_hybrid, run_id