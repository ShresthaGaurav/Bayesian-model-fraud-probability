import pandas as pd
import numpy as np

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic transaction data mimicking Nepal's fraud patterns."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="h")
    date_series = pd.Series(dates).sample(n_samples, replace=True, random_state=42).reset_index(drop=True)
    amounts = np.random.lognormal(mean=8, sigma=1.5, size=n_samples)
    types = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    fraud = np.zeros(n_samples)
    
    for i in range(n_samples):
        hour = date_series[i].hour
        # Increase fraud likelihood for high amounts during night hours
        if types[i] == 0 and amounts[i] > 20000 and hour in [0, 1, 2, 3, 22, 23]:
            fraud[i] = 1 if np.random.rand() < 0.9 else 0
        # Increase general fraud probability
        elif np.random.rand() < 0.05:
            fraud[i] = 1
    
    # Ensure at least 10% fraud cases by forcing some fraud labels
    fraud_count = int(np.sum(fraud))  # Cast to int to avoid float
    min_fraud = int(0.1 * n_samples)  # Target at least 10% fraud
    if fraud_count < min_fraud:
        additional_fraud = int(min_fraud - fraud_count)  # Ensure integer
        indices = np.random.choice(np.where(fraud == 0)[0], size=additional_fraud, replace=False)
        fraud[indices] = 1

    print(f"Generated dataset: {int(np.sum(fraud))} fraud cases out of {n_samples} samples")  # Debug print

    data = pd.DataFrame({
        "date": date_series,
        "amount": amounts,
        "type": types,
        "fraud": fraud.astype(np.float64)
    })
    data.to_csv("data/synthetic_transactions.csv", index=False)
    return data

def preprocess_data(data):
    """Preprocess transaction data."""
    data["date"] = pd.to_datetime(data["date"])
    data["hour"] = data["date"].dt.hour
    data["day_of_week"] = data["date"].dt.dayofweek
    data["amount"] = data["amount"].clip(lower=100, upper=100000)
    data["type"] = data["type"].astype(int)
    data.fillna({"amount": data["amount"].median()}, inplace=True)
    return data

def engineer_features(data):
    """Create temporal, amount-based, and type-based features."""
    data["amount_zscore"] = (data["amount"] - data["amount"].mean()) / data["amount"].std()
    data["rolling_avg_amount"] = data["amount"].rolling(window=5, min_periods=1).mean()
    data["dr_cr_ratio"] = data["type"].rolling(window=10, min_periods=1).mean()
    data["is_night"] = data["hour"].apply(lambda x: 1 if x in [0, 1, 2, 3, 22, 23] else 0).astype(np.float64)
    return data

def apply_differential_privacy(data, epsilon=1.0):
    """Add Laplace noise to amount-based features for privacy."""
    noise = np.random.laplace(0, 1/epsilon, size=len(data))
    data["amount_zscore"] = data["amount_zscore"] + noise
    noise = np.random.laplace(0, 1/epsilon, size=len(data))
    data["rolling_avg_amount"] = data["rolling_avg_amount"] + noise
    return data